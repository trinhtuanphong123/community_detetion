
# ============================================================
# Cell 11: Cycle matcher, diagnostic only
# ============================================================

BIDIRECTIONAL_CYCLE_THRESHOLD = 999  # retained for stats compatibility; cycles use forward DFS only

def passes_pairwise_amount_ratio(
    edges: List[EdgeRecord],
    ratio_min: Optional[float],
    ratio_max: Optional[float],
) -> bool:
    if ratio_min is None and ratio_max is None:
        return True
    for i in range(len(edges)):
        e_in = edges[i]
        e_out = edges[(i + 1) % len(edges)]
        if e_in.amount <= 0:
            return False
        ratio = e_out.amount / e_in.amount
        if ratio_min is not None and ratio < ratio_min:
            return False
        if ratio_max is not None and ratio > ratio_max:
            return False
    return True


def remaining_hops_feasible(
    current_step: int,
    anchor_step: int,
    max_duration: int,
    remaining_hops: int,
    delta_hop: Optional[int],
) -> bool:
    elapsed = current_step - anchor_step
    if elapsed > max_duration:
        return False
    time_budget = max_duration - elapsed
    # With strict time order, step must increase by at least 1 per hop
    if remaining_hops > time_budget:
        return False
    return True


def compute_amount_consistency(edges: List[EdgeRecord]) -> float:
    """
    Measure amount consistency inside one cycle instance.
    """
    amounts = [float(e.amount) for e in edges if float(e.amount) > 0]

    if not amounts:
        return 0.0

    max_amount = max(amounts)
    if max_amount <= 0:
        return 0.0

    return float(min(amounts) / max_amount)


class CycleKMatcher:
    """
    Generalized matcher for path_cycle motifs.

    Pattern:
        v1 -> v2 -> ... -> vk -> v1

    Strategy:
        1. Anchor on primary edge e1: v1 -> v2.
        2. If cycle length is small, use forward DFS.
        3. If cycle length is large, dispatch to bidirectional search.
        4. Enforce strict temporal order.
        5. Enforce distinct intermediate nodes.
        6. Enforce total duration.
        7. Use canonical anchor to reduce duplicate cycle rotations.
        8. Store amount_consistency for post-hoc ranking.
    """

    def __init__(
        self,
        max_branching: int = 5,
        branching_schedule: Optional[Dict[int, int]] = None,
        max_instances_per_window: int = 20_000,
        amount_min: Optional[float] = None,
        use_amount_ratio: bool = False,
        candidate_policy: str = "hybrid",
        delta_hop: Optional[int] = DELTA_HOP,
        max_anchors_per_window: Optional[int] = None,
    ):
        self.max_branching = int(max_branching)
        self.branching_schedule = branching_schedule or {}
        self.max_instances_per_window = max_instances_per_window
        self.amount_min = amount_min
        self.use_amount_ratio = use_amount_ratio
        self.candidate_policy = candidate_policy
        self.delta_hop = delta_hop
        self.max_anchors_per_window = max_anchors_per_window

    def _get_cycle_branching(self, k: int) -> int:
        return int(self.branching_schedule.get(k, self.max_branching))

    def _filter_candidates(
        self,
        edges: List[EdgeRecord],
        pattern: MotifPattern,
    ) -> List[EdgeRecord]:
        if self.amount_min is not None:
            edges = [e for e in edges if e.amount >= self.amount_min]

        k = len(pattern.edges)
        branching = self._get_cycle_branching(k)

        return cap_edges(
            edges,
            branching,
            policy=self.candidate_policy,
        )

    def _get_primary_edges(self, df_primary: pl.DataFrame) -> List[EdgeRecord]:
        edge_ids = df_primary["edge_id"].to_list()
        srcs = df_primary["src"].to_list()
        dsts = df_primary["dst"].to_list()
        steps = df_primary["step"].to_list()
        amounts = df_primary["amount"].to_list()
        is_sars = df_primary["is_sar"].to_list()

        primary_edges = [
            EdgeRecord(
                edge_id=int(edge_id),
                src=int(src),
                dst=int(dst),
                step=int(step),
                amount=float(amount),
                is_sar=int(is_sar),
            )
            for edge_id, src, dst, step, amount, is_sar in zip(
                edge_ids, srcs, dsts, steps, amounts, is_sars
            )
        ]

        primary_edges = sorted(primary_edges, key=lambda e: (e.step, e.edge_id))

        if self.max_anchors_per_window is not None:
            primary_edges = primary_edges[: self.max_anchors_per_window]

        return primary_edges

    def _emit_cycle_instance(
        self,
        output_buffer: MatcherOutputBuffer,
        window: WindowSpec,
        pattern: MotifPattern,
        edges: List[EdgeRecord],
        anchor_edge: EdgeRecord,
        k: int,
        index: Optional[TemporalIndex] = None,
    ) -> bool:
        try:
            motif_row = make_motif_instance_row(
                window_id=window.window_id,
                pattern=pattern,
                edges=edges,
                anchor_edge_id=anchor_edge.edge_id,
                index=index,
            )
            member_rows = make_edge_motif_membership_rows(
                motif_row=motif_row,
                pattern=pattern,
                edges=edges,
            )
            output_buffer.add(motif_row, member_rows)
            return True
        except Exception:
            return False

    def match(
        self,
        df_primary: pl.DataFrame,
        df_extended: pl.DataFrame,
        index: TemporalIndex,
        window: WindowSpec,
        pattern: MotifPattern,
        output_buffer: MatcherOutputBuffer,
    ) -> Dict[str, Any]:

        if pattern.matcher_type != "path_cycle":
            raise ValueError(
                f"CycleKMatcher expects matcher_type='path_cycle'. "
                f"Got {pattern.matcher_type}"
            )

        k = len(pattern.edges)

        if k < 3:
            raise ValueError(f"Cycle pattern must have at least 3 edges. Got {k}.")

        if len(pattern.nodes) != k:
            raise ValueError(
                f"Cycle pattern {pattern.name} should have same number of nodes and edges. "
                f"Got nodes={len(pattern.nodes)}, edges={len(pattern.edges)}."
            )

        # Always use the forward DFS path. The bidirectional variant is
        # disabled because its temporal merge logic is incorrect for cycles.
        use_bidirectional = False

        start_time = time.time()
        num_instances_emitted = 0
        num_membership_rows_emitted = 0

        num_anchor_edges = 0
        num_dfs_expansions = 0
        num_close_queries = 0
        num_paths_checked = 0

        num_rejected_nodes = 0
        num_rejected_duration = 0
        num_rejected_anchor = 0
        num_rejected_amount = 0
        num_rejected_duplicate_edge = 0
        num_rejected_canonical = 0

        num_pruned_by_hop_budget = 0
        num_pruned_by_return_feasibility = 0
        num_pruned_by_canonical_rotation = 0

        primary_edges = self._get_primary_edges(df_primary)
        cycle_branching = self._get_cycle_branching(k)

        for anchor_edge in primary_edges:
            num_anchor_edges += 1

            if not is_edge_in_primary_window(anchor_edge, window):
                num_rejected_anchor += 1
                continue

            if self.amount_min is not None and anchor_edge.amount < self.amount_min:
                num_rejected_amount += 1
                continue

            start_node = int(anchor_edge.src)
            second_node = int(anchor_edge.dst)

            if start_node == second_node:
                num_rejected_nodes += 1
                continue

            stack = [
                (
                    [anchor_edge],
                    [start_node, second_node],
                )
            ]

            while stack:
                path_edges, path_nodes = stack.pop()

                if num_instances_emitted >= self.max_instances_per_window:
                    break

                anchor_key = (anchor_edge.step, anchor_edge.edge_id)

                if any((e.step, e.edge_id) < anchor_key for e in path_edges[1:]):
                    num_rejected_canonical += 1
                    num_pruned_by_canonical_rotation += 1
                    continue

                remaining_hops = k - len(path_edges)

                if not remaining_hops_feasible(
                    current_step=path_edges[-1].step,
                    anchor_step=path_edges[0].step,
                    max_duration=pattern.max_duration,
                    remaining_hops=remaining_hops,
                    delta_hop=self.delta_hop,
                ):
                    num_rejected_duration += 1
                    num_pruned_by_hop_budget += 1
                    continue

                current_edge = path_edges[-1]
                current_node = path_nodes[-1]

                if len(path_edges) == k - 1:
                    t_min = current_edge.step

                    if self.delta_hop is not None:
                        t_max = min(
                            current_edge.step + self.delta_hop,
                            path_edges[0].step + pattern.max_duration,
                        )
                    else:
                        t_max = path_edges[0].step + pattern.max_duration

                    close_candidates = index.pair(
                        src=current_node,
                        dst=start_node,
                        t_min=t_min,
                        t_max=t_max,
                        include_left=False,
                    )

                    close_candidates = self._filter_candidates(
                        close_candidates,
                        pattern=pattern,
                    )
                    num_close_queries += 1

                    for close_edge in close_candidates:
                        if num_instances_emitted >= self.max_instances_per_window:
                            break

                        edges = path_edges + [close_edge]
                        num_paths_checked += 1

                        if not has_unique_edge_ids(edges):
                            num_rejected_duplicate_edge += 1
                            continue

                        if not is_within_total_duration(edges, pattern.max_duration):
                            num_rejected_duration += 1
                            continue

                        if not has_distinct_nodes(path_nodes):
                            num_rejected_nodes += 1
                            continue

                        earliest_edge = min(edges, key=lambda e: (e.step, e.edge_id))

                        if earliest_edge.edge_id != anchor_edge.edge_id:
                            num_rejected_canonical += 1
                            continue

                        if not is_edge_in_primary_window(anchor_edge, window):
                            num_rejected_anchor += 1
                            continue

                        if self.use_amount_ratio:
                            if not passes_pairwise_amount_ratio(
                                edges,
                                pattern.amount_ratio_min,
                                pattern.amount_ratio_max,
                            ):
                                num_rejected_amount += 1
                                continue

                        emitted = self._emit_cycle_instance(
                            output_buffer=output_buffer,
                            window=window,
                            pattern=pattern,
                            edges=edges,
                            anchor_edge=anchor_edge,
                            k=k,
                            index=index,
                        )
                        if emitted:
                            num_instances_emitted += 1
                            num_membership_rows_emitted += k

                    continue

                t_min = current_edge.step

                if self.delta_hop is not None:
                    t_max = min(
                        current_edge.step + self.delta_hop,
                        path_edges[0].step + pattern.max_duration,
                    )
                else:
                    t_max = path_edges[0].step + pattern.max_duration

                next_candidates = index.outgoing(
                    src=current_node,
                    t_min=t_min,
                    t_max=t_max,
                    include_left=False,
                )

                next_candidates = self._filter_candidates(
                    next_candidates,
                    pattern=pattern,
                )

                for next_edge in reversed(next_candidates):
                    if num_instances_emitted >= self.max_instances_per_window:
                        break

                    next_node = int(next_edge.dst)

                    if next_node in path_nodes:
                        num_rejected_nodes += 1
                        continue

                    if next_node == start_node:
                        num_rejected_nodes += 1
                        continue

                    new_edges = path_edges + [next_edge]

                    if not has_unique_edge_ids(new_edges):
                        num_rejected_duplicate_edge += 1
                        continue

                    if not is_within_total_duration(new_edges, pattern.max_duration):
                        num_rejected_duration += 1
                        continue

                    if len(new_edges) >= k - 1:
                        time_budget = (
                            path_edges[0].step + pattern.max_duration
                            - next_edge.step
                        )

                        if time_budget < 0:
                            num_rejected_duration += 1
                            continue

                        if self.delta_hop is not None:
                            max_return_step = next_edge.step + min(
                                self.delta_hop,
                                time_budget,
                            )
                        else:
                            max_return_step = next_edge.step + time_budget

                        if not index.has_any_pair(
                            src=next_node,
                            dst=start_node,
                            t_min=next_edge.step,
                            t_max=max_return_step,
                        ):
                            num_rejected_nodes += 1
                            num_pruned_by_return_feasibility += 1
                            continue

                    new_nodes = path_nodes + [next_node]

                    stack.append((new_edges, new_nodes))
                    num_dfs_expansions += 1

            if num_instances_emitted >= self.max_instances_per_window:
                break

        elapsed = time.time() - start_time

        stats = {
            "window_id": int(window.window_id),
            "motif_type": pattern.name,
            "matcher_type": pattern.matcher_type,
            "cycle_length": int(k),
            "cycle_branching": int(cycle_branching),
            "use_bidirectional": int(use_bidirectional),
            "bidirectional_threshold": int(BIDIRECTIONAL_CYCLE_THRESHOLD),

            "primary_start": int(window.primary_start),
            "primary_end": int(window.primary_end),
            "extended_start": int(window.extended_start),
            "extended_end": int(window.extended_end),
            "primary_num_edges": int(df_primary.height),
            "extended_num_edges": int(df_extended.height),

            "num_anchor_edges": int(num_anchor_edges),
            "num_dfs_expansions": int(num_dfs_expansions),
            "num_close_queries": int(num_close_queries),
            "num_paths_checked": int(num_paths_checked),

            "num_instances": int(num_instances_emitted),
            "num_membership_rows": int(num_membership_rows_emitted),

            "num_rejected_nodes": int(num_rejected_nodes),
            "num_rejected_duration": int(num_rejected_duration),
            "num_rejected_anchor": int(num_rejected_anchor),
            "num_rejected_amount": int(num_rejected_amount),
            "num_rejected_duplicate_edge": int(num_rejected_duplicate_edge),
            "num_rejected_canonical": int(num_rejected_canonical),

            "num_pruned_by_hop_budget": int(num_pruned_by_hop_budget),
            "num_pruned_by_return_feasibility": int(num_pruned_by_return_feasibility),
            "num_pruned_by_canonical_rotation": int(num_pruned_by_canonical_rotation),

            "hit_max_instances_per_window": int(
                num_instances_emitted >= self.max_instances_per_window
            ),
            "elapsed_seconds": float(elapsed),
        }

        return stats


@dataclass
class FrontierState:
    endpoint:       int
    min_step:       int
    max_step:       int
    used_nodes:     frozenset
    used_edge_ids:  frozenset
    edges:          List[EdgeRecord]


class BidirectionalCycleSearch:
    def __init__(
        self,
        max_branching: int,
        delta_hop: Optional[int],
        candidate_policy: str,
        amount_min: Optional[float],
        max_instances_per_window: int,
    ):
        self.max_branching = max_branching
        self.delta_hop = delta_hop
        self.candidate_policy = candidate_policy
        self.amount_min = amount_min
        self.max_instances_per_window = max_instances_per_window

    def _filter_candidates(
        self,
        edges: List[EdgeRecord],
    ) -> List[EdgeRecord]:
        if self.amount_min is not None:
            edges = [e for e in edges if e.amount >= self.amount_min]
        return cap_edges(
            edges,
            self.max_branching,
            policy=self.candidate_policy,
        )

    def _grow_forward(
        self,
        state: FrontierState,
        index: TemporalIndex,
        k: int,
        max_duration: int,
        anchor_step: int,
        start_node: int,
    ) -> List[FrontierState]:
        depth = len(state.edges)
        limit = k // 2

        if depth >= limit:
            return [state]

        t_min = state.edges[-1].step
        time_used = t_min - anchor_step
        time_budget = max_duration - time_used

        if time_budget < 0:
            return []

        remaining_hops = k - depth

        if not remaining_hops_feasible(
            current_step=t_min,
            anchor_step=anchor_step,
            max_duration=max_duration,
            remaining_hops=remaining_hops,
            delta_hop=self.delta_hop,
        ):
            return []

        if self.delta_hop is not None:
            t_max = min(t_min + self.delta_hop, anchor_step + max_duration)
        else:
            t_max = anchor_step + max_duration

        candidates = index.outgoing(src=state.endpoint, t_min=t_min, t_max=t_max, include_left=False)
        candidates = self._filter_candidates(candidates)

        results = []

        for e in candidates:
            nxt = int(e.dst)

            if nxt in state.used_nodes:
                continue

            if nxt == start_node:
                continue

            if e.edge_id in state.used_edge_ids:
                continue

            new_state = FrontierState(
                endpoint=nxt,
                min_step=state.min_step,
                max_step=int(e.step),
                used_nodes=state.used_nodes | {nxt},
                used_edge_ids=state.used_edge_ids | {e.edge_id},
                edges=state.edges + [e],
            )

            results.extend(
                self._grow_forward(
                    state=new_state,
                    index=index,
                    k=k,
                    max_duration=max_duration,
                    anchor_step=anchor_step,
                    start_node=start_node,
                )
            )

        return results

    def _grow_backward(
        self,
        state: FrontierState,
        index: TemporalIndex,
        k: int,
        max_duration: int,
        anchor_step: int,
        start_node: int,
        second_node: int,
    ) -> List[FrontierState]:
        depth = len(state.edges)
        limit = k - (k // 2)

        if depth >= limit:
            return [state]

        t_max = state.edges[0].step if state.edges else anchor_step

        if self.delta_hop is not None:
            t_min = max(t_max - self.delta_hop, anchor_step - max_duration)
        else:
            t_min = anchor_step - max_duration

        candidates = index.incoming(dst=state.endpoint, t_min=t_min, t_max=t_max, include_left=False)
        candidates = self._filter_candidates(candidates)

        results = []

        for e in candidates:
            prev_node = int(e.src)

            if prev_node in state.used_nodes:
                continue

            if prev_node == start_node:
                continue

            if prev_node == second_node:
                continue

            if e.edge_id in state.used_edge_ids:
                continue

            new_state = FrontierState(
                endpoint=prev_node,
                min_step=int(e.step),
                max_step=state.max_step,
                used_nodes=state.used_nodes | {prev_node},
                used_edge_ids=state.used_edge_ids | {e.edge_id},
                edges=[e] + state.edges,
            )

            results.extend(
                self._grow_backward(
                    state=new_state,
                    index=index,
                    k=k,
                    max_duration=max_duration,
                    anchor_step=anchor_step,
                    start_node=start_node,
                    second_node=second_node,
                )
            )

        return results

    def search_all_windows(
        self,
        df_primary: pl.DataFrame,
        index: TemporalIndex,
        pattern: MotifPattern,
        window: WindowSpec,
    ) -> List[List[EdgeRecord]]:
        k = len(pattern.edges)
        results = []

        primary_edges = []
        for row in df_primary.iter_rows(named=True):
            primary_edges.append(
                EdgeRecord(
                    edge_id=int(row["edge_id"]),
                    src=int(row["src"]),
                    dst=int(row["dst"]),
                    step=int(row["step"]),
                    amount=float(row["amount"]),
                    is_sar=int(row["is_sar"]),
                )
            )
        primary_edges = sorted(primary_edges, key=lambda e: (e.step, e.edge_id))

        seen_canonical_keys = set()

        for anchor_edge in primary_edges:
            if not is_edge_in_primary_window(anchor_edge, window):
                continue

            if self.amount_min is not None and anchor_edge.amount < self.amount_min:
                continue

            start_node = int(anchor_edge.src)
            second_node = int(anchor_edge.dst)

            if start_node == second_node:
                continue

            # Forward search starting from anchor_edge.
            fwd_init = FrontierState(
                endpoint=second_node,
                min_step=int(anchor_edge.step),
                max_step=int(anchor_edge.step),
                used_nodes=frozenset({start_node, second_node}),
                used_edge_ids=frozenset({anchor_edge.edge_id}),
                edges=[anchor_edge],
            )

            fwd_states = self._grow_forward(
                state=fwd_init,
                index=index,
                k=k,
                max_duration=pattern.max_duration,
                anchor_step=anchor_edge.step,
                start_node=start_node,
            )

            if not fwd_states:
                continue

            # Backward search starting at start_node, going back to start_node.
            bwd_init = FrontierState(
                endpoint=start_node,
                min_step=int(anchor_edge.step),
                max_step=int(anchor_edge.step),
                used_nodes=frozenset({start_node}),
                used_edge_ids=frozenset(),
                edges=[],
            )

            bwd_states = self._grow_backward(
                state=bwd_init,
                index=index,
                k=k,
                max_duration=pattern.max_duration,
                anchor_step=anchor_edge.step,
                start_node=start_node,
                second_node=second_node,
            )

            if not bwd_states:
                continue

            bwd_by_endpoint: Dict[int, List[FrontierState]] = defaultdict(list)
            for bwd in bwd_states:
                bwd_by_endpoint[bwd.endpoint].append(bwd)

            for fwd in fwd_states:
                for bwd in bwd_by_endpoint.get(fwd.endpoint, []):
                    if fwd.max_step >= bwd.min_step:
                        continue

                    if self.delta_hop is not None:
                        if bwd.min_step - fwd.max_step > self.delta_hop:
                            continue

                    shared = fwd.used_nodes & bwd.used_nodes
                    if shared != {anchor_edge.src}:
                        continue

                    if fwd.used_edge_ids & bwd.used_edge_ids:
                        continue

                    cycle_edges = fwd.edges + bwd.edges

                    if len(cycle_edges) != k:
                        continue

                    if not is_within_total_duration(cycle_edges, pattern.max_duration):
                        continue

                    earliest = min(cycle_edges, key=lambda e: (e.step, e.edge_id))
                    if earliest.edge_id != anchor_edge.edge_id:
                        continue

                    ck = make_canonical_key(
                        pattern.name,
                        [e.edge_id for e in cycle_edges],
                        ordered=True,
                    )

                    if ck in seen_canonical_keys:
                        continue

                    seen_canonical_keys.add(ck)
                    results.append(cycle_edges)

                    if len(results) >= self.max_instances_per_window:
                        return results

        return results


# Matcher registration moved to Cell 12 in code_matcher_2.md
print("Cell 11 completed.")

