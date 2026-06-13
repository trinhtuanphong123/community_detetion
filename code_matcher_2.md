# ============================================================
# Cell 12: Common Helpers for Complex Motif Matchers
# ============================================================

from itertools import product


def is_strictly_after(e_next: EdgeRecord, e_prev: EdgeRecord) -> bool:
    return e_next.step > e_prev.step


def is_within_delta_hop(e_next: EdgeRecord, e_prev: EdgeRecord, delta_hop: int) -> bool:
    return 0 < (e_next.step - e_prev.step) <= delta_hop


def is_within_total_duration(edges: List[EdgeRecord], max_duration: int) -> bool:
    steps = [e.step for e in edges]
    return (max(steps) - min(steps)) <= max_duration


def has_unique_edge_ids(edges: List[EdgeRecord]) -> bool:
    edge_ids = [e.edge_id for e in edges]
    return len(edge_ids) == len(set(edge_ids))


def has_distinct_nodes(node_values: List[int]) -> bool:
    return len(node_values) == len(set(node_values))


def passes_pairwise_amount_ratio(
    edges: List[EdgeRecord],
    ratio_min: Optional[float],
    ratio_max: Optional[float],
) -> bool:
    """
    Check adjacent amount ratios in the given edge order.

    This is optional and should be used carefully.
    For initial exploration, matchers below allow turning this off.
    """
    if ratio_min is None or ratio_max is None:
        return True

    for i in range(len(edges) - 1):
        prev_amount = edges[i].amount
        curr_amount = edges[i + 1].amount

        if prev_amount <= 0:
            return False

        ratio = curr_amount / prev_amount

        if ratio < ratio_min or ratio > ratio_max:
            return False

    return True
def passes_consecutive_step_gap(
    edges: List[EdgeRecord],
    max_gap: Optional[int],
) -> bool:
    """
    Check whether consecutive edges after sorting are close enough in time.
    """
    if max_gap is None:
        return True

    if len(edges) <= 1:
        return True

    edges_sorted = sorted(edges, key=lambda e: (e.step, e.edge_id))
    steps = [int(e.step) for e in edges_sorted]

    for i in range(len(steps) - 1):
        if steps[i + 1] - steps[i] > max_gap:
            return False

    return True


def phase_gap_ok(
    earlier_edges: List[EdgeRecord],
    later_edges: List[EdgeRecord],
    max_gap: Optional[int],
) -> bool:
    """
    Check whether the later phase starts soon after the earlier phase ends.
    """
    if max_gap is None:
        return True

    if len(earlier_edges) == 0 or len(later_edges) == 0:
        return False

    earlier_end = max(int(e.step) for e in earlier_edges)
    later_start = min(int(e.step) for e in later_edges)

    return 0 < (later_start - earlier_end) <= max_gap

def select_edges_by_hybrid_policy(
    edges: List[EdgeRecord],
    max_candidates: int,
    early_ratio: float = 0.5,
) -> List[EdgeRecord]:
    """
    Hybrid candidate selection:
        keep part earliest edges and part largest-amount edges.

    This is safer than earliest-only for AML because it keeps both temporal
    coverage and high-value transactions.
    """
    if max_candidates is None or len(edges) <= max_candidates:
        return sorted(edges, key=lambda e: (e.step, e.edge_id))

    k_early = int(max_candidates * early_ratio)
    k_amount = max_candidates - k_early

    earliest_edges = sorted(edges, key=lambda e: (e.step, e.edge_id))[:k_early]
    top_amount_edges = sorted(edges, key=lambda e: (-e.amount, e.step, e.edge_id))[:k_amount]

    selected = {}
    for e in earliest_edges + top_amount_edges:
        selected[e.edge_id] = e

    if len(selected) < max_candidates:
        for e in sorted(edges, key=lambda e: (e.step, e.edge_id)):
            selected[e.edge_id] = e
            if len(selected) >= max_candidates:
                break

    return sorted(selected.values(), key=lambda e: (e.step, e.edge_id))


def cap_edges(
    edges: List[EdgeRecord],
    max_candidates: int,
    policy: str = "hybrid",
) -> List[EdgeRecord]:
    """
    Candidate cap wrapper.

    Supported policies:
        earliest
        top_amount
        hybrid
    """
    if max_candidates is None or len(edges) <= max_candidates:
        return sorted(edges, key=lambda e: (e.step, e.edge_id))

    if policy == "earliest":
        return sorted(edges, key=lambda e: (e.step, e.edge_id))[:max_candidates]

    if policy == "top_amount":
        selected = sorted(edges, key=lambda e: (-e.amount, e.step, e.edge_id))[:max_candidates]
        return sorted(selected, key=lambda e: (e.step, e.edge_id))

    if policy == "hybrid":
        return select_edges_by_hybrid_policy(edges, max_candidates=max_candidates, early_ratio=0.5)

    raise ValueError(f"Unknown candidate selection policy: {policy}")


def create_empty_matcher_stats(
    window: WindowSpec,
    pattern: MotifPattern,
    df_primary: pl.DataFrame,
    df_extended: pl.DataFrame,
    start_time: float,
) -> Dict[str, Any]:
    elapsed = time.time() - start_time

    return {
        "window_id": int(window.window_id),
        "motif_type": pattern.name,
        "matcher_type": pattern.matcher_type,
        "primary_start": int(window.primary_start),
        "primary_end": int(window.primary_end),
        "extended_start": int(window.extended_start),
        "extended_end": int(window.extended_end),
        "primary_num_edges": int(df_primary.height),
        "extended_num_edges": int(df_extended.height),
        "num_instances": 0,
        "num_membership_rows": 0,
        "hit_max_instances_per_window": 0,
        "elapsed_seconds": float(elapsed),
    }

def remaining_hops_feasible(
    current_step:    int,
    anchor_step:     int,
    max_duration:    int,
    remaining_hops:  int,
    delta_hop:       Optional[int],
) -> bool:
    """
    Guard A for cycle DFS: check whether the remaining hops can still fit
    within the time budget.

    time_used    = current_step - anchor_step
    time_budget  = max_duration - time_used

    If delta_hop is set, each hop costs at least 1 step.
    So we need time_budget >= remaining_hops (minimum 1 step per hop).

    If delta_hop is None, only max_duration applies; this check is trivially True.
    """
    if delta_hop is None:
        return True
    time_used   = current_step - anchor_step
    time_budget = max_duration - time_used
    return time_budget >= remaining_hops

print("Cell 12 completed.")







# ============================================================
# Cell 13: CycleKMatcher
# ============================================================
# General matcher for directed temporal cycles:
#   cycle_k: v1 -> v2 -> ... -> vk -> v1
#
# Supports:
#   cycle_5, cycle_6, ..., cycle_12
#
# Dispatch policy:
#   - For k < BIDIRECTIONAL_CYCLE_THRESHOLD: use DFS with Guards A, B, C.
#   - For k >= BIDIRECTIONAL_CYCLE_THRESHOLD: use BidirectionalCycleSearch.
#
# Changes in this version:
#   1. max_branching is selected by cycle length k using CYCLE_MAX_BRANCHING.
#   2. amount_consistency is computed as min(amount) / max(amount)
#      and stored in each motif_row.
#
# Important:
#   Long cycles are much more expensive than cycle_5.
#   Use tight max_branching and max_instances_per_window.


def compute_amount_consistency(edges: List[EdgeRecord]) -> float:
    """
    Measure amount consistency inside one cycle instance.

    Definition:
        amount_consistency = min_positive_amount / max_positive_amount

    Interpretation:
        - close to 1.0: amounts are highly consistent
        - close to 0.0: amounts are very different

    This is NOT used as a hard filter.
    It is stored in motif_row for later ranking or analysis.
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
        # Fallback branching value if k is not found in branching_schedule.
        self.max_branching = int(max_branching)

        # k -> max_branching, for example:
        # {5: 8, 6: 6, 7: 5, 8: 4, 9: 4, 10: 4}
        self.branching_schedule = branching_schedule or {}

        self.max_instances_per_window = max_instances_per_window
        self.amount_min = amount_min
        self.use_amount_ratio = use_amount_ratio
        self.candidate_policy = candidate_policy
        self.delta_hop = delta_hop
        self.max_anchors_per_window = max_anchors_per_window

    def _get_cycle_branching(self, k: int) -> int:
        """
        Get max_branching according to cycle length k.

        If k is not present in branching_schedule, fall back to self.max_branching.
        """
        return int(self.branching_schedule.get(k, self.max_branching))

    def _filter_candidates(
        self,
        edges: List[EdgeRecord],
        pattern: MotifPattern,
    ) -> List[EdgeRecord]:
        """
        Apply amount filter and candidate cap.

        Important:
            pattern is passed in so this method can read k = len(pattern.edges).
        """
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
        """
        Convert primary window edges to EdgeRecord list.
        """
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

        if self.max_anchors_per_window is not None:
            primary_edges = primary_edges[: self.max_anchors_per_window]

        return primary_edges

    def _emit_cycle_instance(
        self,
        motif_rows: List[Dict[str, Any]],
        membership_rows: List[Dict[str, Any]],
        window: WindowSpec,
        pattern: MotifPattern,
        edges: List[EdgeRecord],
        anchor_edge: EdgeRecord,
        k: int,
        validate: bool = True,
    ) -> bool:
        """
        Build node_map, role_map, motif row, and membership rows for one cycle.

        Returns:
            True if an instance is emitted.
            False if output construction fails.
        """
        try:
            # Cycle edge order:
            #   edges[0]: v1 -> v2
            #   edges[1]: v2 -> v3
            #   ...
            #   edges[k-1]: vk -> v1
            #
            # Therefore node sequence is:
            #   v1 = edges[0].src
            #   v2 = edges[0].dst
            #   v3 = edges[1].dst
            #   ...
            #   vk = edges[k-2].dst
            path_nodes = [int(edges[0].src)]

            for e in edges[:-1]:
                path_nodes.append(int(e.dst))

            if len(path_nodes) != k:
                return False

            node_map = {
                f"v{i + 1}": int(node_id)
                for i, node_id in enumerate(path_nodes)
            }

            role_map = {
                pattern.edges[i].role: int(edges[i].edge_id)
                for i in range(k)
            }

            motif_row = make_motif_instance_row(
                window_id=window.window_id,
                pattern=pattern,
                edges=edges,
                node_map=node_map,
                role_map=role_map,
                anchor_edge_id=anchor_edge.edge_id,
                validate=validate,
            )

            # New field:
            # Store amount consistency for later ranking or filtering.
            motif_row["amount_consistency"] = compute_amount_consistency(edges)

            membership = make_edge_motif_membership_rows(
                motif_row=motif_row,
                pattern=pattern,
                edges=edges,
            )

            motif_rows.append(motif_row)
            membership_rows.extend(membership)

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
        write_output: bool = False,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, Any]]:

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

        # ------------------------------------------------------------
        # Dispatch decision
        # ------------------------------------------------------------
        use_bidirectional = (k >= BIDIRECTIONAL_CYCLE_THRESHOLD)

        start_time = time.time()

        motif_rows = []
        membership_rows = []

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

        # Select max_branching for this specific cycle length k.
        cycle_branching = self._get_cycle_branching(k)

        # ============================================================
        # Branch 1: Bidirectional search for long cycles
        # ============================================================
        if use_bidirectional:
            bidi_searcher = BidirectionalCycleSearch(
                max_branching=cycle_branching,
                delta_hop=self.delta_hop,
                candidate_policy=self.candidate_policy,
                amount_min=self.amount_min,
                max_instances_per_window=self.max_instances_per_window,
            )

            all_cycle_edge_lists = bidi_searcher.search_all_windows(
                df_primary=df_primary,
                index=index,
                pattern=pattern,
                window=window,
            )

            # Approximate anchor count if BidirectionalCycleSearch does not expose stats.
            num_anchor_edges = int(df_primary.height)

            for cycle_edges in all_cycle_edge_lists:
                if len(motif_rows) >= self.max_instances_per_window:
                    break

                edges = list(cycle_edges)
                num_paths_checked += 1

                if len(edges) != k:
                    num_rejected_nodes += 1
                    continue

                if not has_unique_edge_ids(edges):
                    num_rejected_duplicate_edge += 1
                    continue

                if not is_within_total_duration(edges, pattern.max_duration):
                    num_rejected_duration += 1
                    continue

                anchor_edge = edges[0]

                if self.amount_min is not None:
                    if any(e.amount < self.amount_min for e in edges):
                        num_rejected_amount += 1
                        continue

                if not is_edge_in_primary_window(anchor_edge, window):
                    num_rejected_anchor += 1
                    continue

                earliest_edge = min(edges, key=lambda e: (e.step, e.edge_id))
                if earliest_edge.edge_id != anchor_edge.edge_id:
                    num_rejected_canonical += 1
                    continue

                # Reconstruct path nodes:
                # edges[0]: v1 -> v2
                # edges[1]: v2 -> v3
                # ...
                # edges[-1]: vk -> v1
                path_nodes = [int(edges[0].src)]
                for e in edges[:-1]:
                    path_nodes.append(int(e.dst))

                if not has_distinct_nodes(path_nodes):
                    num_rejected_nodes += 1
                    continue

                # Closing edge must return to start node.
                if int(edges[-1].dst) != int(edges[0].src):
                    num_rejected_nodes += 1
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
                    motif_rows=motif_rows,
                    membership_rows=membership_rows,
                    window=window,
                    pattern=pattern,
                    edges=edges,
                    anchor_edge=anchor_edge,
                    k=k,
                    validate=True,
                )

                if not emitted:
                    continue

        # ============================================================
        # Branch 2: Existing DFS search for short cycles
        # ============================================================
        else:
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

                # Current partial path:
                #   edges: [v1 -> v2]
                #   nodes: [v1, v2]
                stack = [
                    (
                        [anchor_edge],
                        [start_node, second_node],
                    )
                ]

                while stack:
                    path_edges, path_nodes = stack.pop()

                    if len(motif_rows) >= self.max_instances_per_window:
                        break

                    # ------------------------------------------------
                    # Guard C — early canonical rotation pruning.
                    # If any edge in the current partial path is earlier
                    # than the anchor, this path cannot produce a canonical cycle.
                    # ------------------------------------------------
                    anchor_key = (anchor_edge.step, anchor_edge.edge_id)

                    if any((e.step, e.edge_id) < anchor_key for e in path_edges[1:]):
                        num_rejected_canonical += 1
                        num_pruned_by_canonical_rotation += 1
                        continue

                    # ------------------------------------------------
                    # Guard A — remaining-hop time feasibility.
                    # remaining_hops means number of edges still needed
                    # to complete the cycle, including the final closing edge.
                    # ------------------------------------------------
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

                    # ------------------------------------------------
                    # If we already have k-1 edges, close the cycle.
                    # Need final edge: current_node -> start_node
                    # ------------------------------------------------
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
                            if len(motif_rows) >= self.max_instances_per_window:
                                break

                            edges = path_edges + [close_edge]
                            num_paths_checked += 1

                            if not has_unique_edge_ids(edges):
                                num_rejected_duplicate_edge += 1
                                continue

                            if not is_within_total_duration(edges, pattern.max_duration):
                                num_rejected_duration += 1
                                continue

                            # Distinct cycle nodes only.
                            # Closing returns to start_node, so path_nodes should already be distinct.
                            if not has_distinct_nodes(path_nodes):
                                num_rejected_nodes += 1
                                continue

                            # Canonical anchor rule:
                            # keep only cycles whose anchor is the earliest edge.
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
                                motif_rows=motif_rows,
                                membership_rows=membership_rows,
                                window=window,
                                pattern=pattern,
                                edges=edges,
                                anchor_edge=anchor_edge,
                                k=k,
                                validate=True,
                            )

                            if not emitted:
                                continue

                        continue

                    # ------------------------------------------------
                    # Otherwise extend forward:
                    #   current_node -> next_node
                    # ------------------------------------------------
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
                        # reversed because stack is LIFO; this keeps earlier edges explored first.
                        if len(motif_rows) >= self.max_instances_per_window:
                            break

                        next_node = int(next_edge.dst)

                        # Do not revisit nodes before final closing edge.
                        if next_node in path_nodes:
                            num_rejected_nodes += 1
                            continue

                        # Do not close to start node too early.
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

                        # ------------------------------------------------
                        # Guard B — return-to-start reachability.
                        #
                        # Apply only when the new path already has k-1 edges.
                        # At that depth, the next step must be the closing edge:
                        #   next_node -> start_node
                        #
                        # If no such closing edge exists in the temporal index,
                        # do not push this partial path into the stack.
                        # ------------------------------------------------
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

                if len(motif_rows) >= self.max_instances_per_window:
                    break

        motif_df = motif_instance_rows_to_polars(motif_rows)
        membership_df = membership_rows_to_polars(membership_rows)

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

            "num_instances": int(motif_df.height),
            "num_membership_rows": int(membership_df.height),

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
                len(motif_rows) >= self.max_instances_per_window
            ),
            "elapsed_seconds": float(elapsed),
        }

        if write_output:
            motif_path, membership_path = write_motif_outputs(
                motif_rows=motif_rows,
                membership_rows=membership_rows,
                window_id=window.window_id,
                motif_type=pattern.name,
            )

            stats["motif_path"] = motif_path
            stats["membership_path"] = membership_path

        return motif_df, membership_df, stats


# ============================================================
# Bidirectional cycle search for k >= BIDIRECTIONAL_CYCLE_THRESHOLD
# ============================================================

@dataclass
class FrontierState:
    """
    One partial path state in bidirectional cycle search.

    forward=True:  path starts from anchor edge, grows forward.
    forward=False: path starts from anchor.src, grows backward in time.
    """
    endpoint:       int
    min_step:       int
    max_step:       int
    used_nodes:     frozenset
    used_edge_ids:  frozenset
    edges:          List[EdgeRecord]


class BidirectionalCycleSearch:
    """
    Bidirectional DFS for directed temporal cycles of length k >= 7.

    Forward frontier:
        paths of length floor(k/2) from anchor_edge forward.

    Backward frontier:
        paths of length ceil(k/2) starting at anchor.src,
        traversed by following incoming edges backward in time.

    Join condition:
        forward.endpoint == backward.endpoint
        AND temporal compatibility
        AND disjoint node sets except anchor.src
        AND disjoint edge sets.
    """

    def __init__(
        self,
        max_branching: int,
        delta_hop: Optional[int],
        candidate_policy: str,
        amount_min: Optional[float],
        max_instances_per_window: int,
    ):
        self.max_branching = int(max_branching)
        self.delta_hop = delta_hop
        self.candidate_policy = candidate_policy
        self.amount_min = amount_min
        self.max_instances_per_window = max_instances_per_window

    def _filter(self, edges: List[EdgeRecord]) -> List[EdgeRecord]:
        if self.amount_min is not None:
            edges = [e for e in edges if e.amount >= self.amount_min]

        return cap_edges(
            edges,
            self.max_branching,
            policy=self.candidate_policy,
        )

    def _build_forward_states(
        self,
        anchor_edge: EdgeRecord,
        depth: int,
        index: TemporalIndex,
        pattern: MotifPattern,
    ) -> List[FrontierState]:
        """
        Build all forward partial paths of exactly depth edges,
        starting from anchor_edge.
        """
        k = len(pattern.edges)

        initial = FrontierState(
            endpoint=anchor_edge.dst,
            min_step=anchor_edge.step,
            max_step=anchor_edge.step,
            used_nodes=frozenset([anchor_edge.src, anchor_edge.dst]),
            used_edge_ids=frozenset([anchor_edge.edge_id]),
            edges=[anchor_edge],
        )

        frontier = [initial]

        for _ in range(depth - 1):
            next_frontier = []

            for state in frontier:
                remaining = k - len(state.edges)

                t_max = min(
                    state.max_step + (self.delta_hop or pattern.max_duration),
                    anchor_edge.step + pattern.max_duration - remaining * 1,
                )

                candidates = self._filter(
                    index.outgoing(
                        src=state.endpoint,
                        t_min=state.max_step,
                        t_max=t_max,
                        include_left=False,
                    )
                )

                for e in candidates:
                    if e.edge_id in state.used_edge_ids:
                        continue

                    next_node = e.dst
                    if next_node in state.used_nodes:
                        continue

                    next_frontier.append(
                        FrontierState(
                            endpoint=next_node,
                            min_step=state.min_step,
                            max_step=e.step,
                            used_nodes=state.used_nodes | {next_node},
                            used_edge_ids=state.used_edge_ids | {e.edge_id},
                            edges=state.edges + [e],
                        )
                    )

            frontier = next_frontier

            if not frontier:
                break

        return frontier

    def _build_backward_states(
        self,
        anchor_edge: EdgeRecord,
        depth: int,
        index: TemporalIndex,
        pattern: MotifPattern,
    ) -> List[FrontierState]:
        """
        Build all backward partial paths of exactly depth edges,
        ending at anchor_edge.src by traversing incoming edges in reverse time.
        """
        initial = FrontierState(
            endpoint=anchor_edge.src,
            min_step=anchor_edge.step,
            max_step=anchor_edge.step,
            used_nodes=frozenset([anchor_edge.src]),
            used_edge_ids=frozenset([anchor_edge.edge_id]),
            edges=[],
        )

        frontier = [initial]

        for _ in range(depth):
            next_frontier = []

            for state in frontier:
                # Walk backward: find edges arriving at state.endpoint
                # with step < state.min_step.
                t_min = anchor_edge.step - pattern.max_duration
                t_max = state.min_step - 1

                if t_max < t_min:
                    continue

                candidates = self._filter(
                    index.incoming(
                        dst=state.endpoint,
                        t_min=t_min,
                        t_max=t_max,
                        include_left=True,
                    )
                )

                for e in reversed(candidates):
                    if e.edge_id in state.used_edge_ids:
                        continue

                    prev_node = e.src
                    if prev_node in state.used_nodes:
                        continue

                    next_frontier.append(
                        FrontierState(
                            endpoint=prev_node,
                            min_step=e.step,
                            max_step=state.max_step,
                            used_nodes=state.used_nodes | {prev_node},
                            used_edge_ids=state.used_edge_ids | {e.edge_id},
                            edges=[e] + state.edges,
                        )
                    )

            frontier = next_frontier

            if not frontier:
                break

        return frontier

    def search_all_windows(
        self,
        df_primary: pl.DataFrame,
        index: TemporalIndex,
        pattern: MotifPattern,
        window: WindowSpec,
    ) -> List[List[EdgeRecord]]:
        """
        Run bidirectional search over all anchor edges in the primary window.

        Returns:
            list of complete cycle edge lists.
        """
        k = len(pattern.edges)
        half_fwd = k // 2
        half_bwd = k - half_fwd

        results: List[List[EdgeRecord]] = []
        seen_canonical_keys: set = set()

        for row in df_primary.iter_rows(named=True):
            anchor_edge = EdgeRecord(
                edge_id=int(row["edge_id"]),
                src=int(row["src"]),
                dst=int(row["dst"]),
                step=int(row["step"]),
                amount=float(row["amount"]),
                is_sar=int(row["is_sar"]),
            )

            if not is_edge_in_primary_window(anchor_edge, window):
                continue

            fwd_states = self._build_forward_states(
                anchor_edge=anchor_edge,
                depth=half_fwd,
                index=index,
                pattern=pattern,
            )

            bwd_states = self._build_backward_states(
                anchor_edge=anchor_edge,
                depth=half_bwd,
                index=index,
                pattern=pattern,
            )

            # Index backward states by endpoint for O(1) join lookup.
            bwd_by_endpoint: Dict[int, List[FrontierState]] = defaultdict(list)

            for bwd in bwd_states:
                bwd_by_endpoint[bwd.endpoint].append(bwd)

            for fwd in fwd_states:
                for bwd in bwd_by_endpoint.get(fwd.endpoint, []):
                    # Temporal order compatibility.
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

                    # Canonical deduplication.
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


# Register cycle matcher.
cycle_k_matcher = CycleKMatcher(
    max_branching=4,  # fallback if k is not found in CYCLE_MAX_BRANCHING
    branching_schedule=CYCLE_MAX_BRANCHING,
    max_instances_per_window=20_000,
    amount_min=None,
    use_amount_ratio=False,  # amount_consistency is stored, not used as hard filter
    candidate_policy="hybrid",
    delta_hop=DELTA_HOP,
    max_anchors_per_window=None,
)

MATCHER_REGISTRY["path_cycle"] = cycle_k_matcher

print("CycleKMatcher (per-k branching + amount_consistency) registered.")
print("Branching schedule:", CYCLE_MAX_BRANCHING)
print("MATCHER_REGISTRY keys:", list(MATCHER_REGISTRY.keys()))
print("Cell 13 completed.")




# ============================================================
# Cell 14: SplitMergeMatcher and CenterInOutMatcher
# ============================================================

def _build_split_merge_roles(
    split_edges: List[EdgeRecord],
    merge_edges_tuple: tuple,
    intermediates: List[int],
    n_branches: int,
) -> Tuple[List[EdgeRecord], List[EdgeRecord], Dict[str, int]]:
    """
    Build role_map while preserving the correct mid_i -> merge_i mapping.

    Important:
        merge_edges_tuple[i] is the selected merge edge from intermediates[i] -> sink.

    Therefore:
        merge_{i+1} must refer to merge_edges_tuple[i],
        not to the i-th edge after sorting by step.

    Returns:
        merge_edges_by_intermediate:
            Merge edges in the same order as intermediates.

        merge_edges_sorted:
            Merge edges sorted by temporal order.
            Use this only for temporal validation.

        role_map:
            Correct pattern-role to real edge_id mapping.
    """
    merge_edges_by_intermediate = list(merge_edges_tuple)

    if len(split_edges) != n_branches:
        raise ValueError(
            f"Expected {n_branches} split edges, got {len(split_edges)}"
        )

    if len(merge_edges_by_intermediate) != n_branches:
        raise ValueError(
            f"Expected {n_branches} merge edges, got {len(merge_edges_by_intermediate)}"
        )

    if len(intermediates) != n_branches:
        raise ValueError(
            f"Expected {n_branches} intermediates, got {len(intermediates)}"
        )

    role_map: Dict[str, int] = {}

    for i in range(n_branches):
        role_map[f"split_{i + 1}"] = int(split_edges[i].edge_id)
        role_map[f"merge_{i + 1}"] = int(merge_edges_by_intermediate[i].edge_id)

    merge_edges_sorted = sorted(
        merge_edges_by_intermediate,
        key=lambda e: (e.step, e.edge_id),
    )

    return merge_edges_by_intermediate, merge_edges_sorted, role_map


def _build_split_merge_node_map(
    source: int,
    intermediates: List[int],
    sink: int,
) -> Dict[str, int]:
    """
    Build node_map while preserving the correct intermediate index.

    mid_{i+1} corresponds to intermediates[i].
    """
    node_map = {
        "src": int(source),
        "sink": int(sink),
    }

    node_map.update({
        f"mid_{i + 1}": int(intermediates[i])
        for i in range(len(intermediates))
    })

    return node_map


class SplitMergeMatcher:
    """
    Matcher for split_merge motifs.

    Pattern:
        src -> mid_1
        src -> mid_2
        ...
        src -> mid_n

        mid_1 -> sink
        mid_2 -> sink
        ...
        mid_n -> sink

    Strategy:
        1. Choose source src.
        2. Select n outgoing split edges from src.
        3. For intermediate nodes mid_1..mid_n, find common sink.
        4. Pick one merge edge from each intermediate to that common sink.
        5. Preserve role mapping:
             split_i = src -> mid_i
             merge_i = mid_i -> sink
    """

    def __init__(
        self,
        max_out_candidates: int = 12,
        max_merge_candidates_per_intermediate: int = 3,
        max_instances_per_window: int = 30_000,
        amount_min: Optional[float] = None,
        use_amount_ratio: bool = True,
        candidate_policy: str = "hybrid",
        split_phase_delta: Optional[int] = DELTA_HOP,
        merge_phase_delta: Optional[int] = DELTA_HOP,
        split_to_merge_delta: Optional[int] = DELTA_HOP,
        max_instances_per_source: Optional[int] = 2_000,
    ):
        self.max_out_candidates = max_out_candidates
        self.max_merge_candidates_per_intermediate = max_merge_candidates_per_intermediate
        self.max_instances_per_window = max_instances_per_window
        self.amount_min = amount_min
        self.use_amount_ratio = use_amount_ratio
        self.candidate_policy = candidate_policy
        self.split_phase_delta = split_phase_delta
        self.merge_phase_delta = merge_phase_delta
        self.split_to_merge_delta = split_to_merge_delta
        self.max_instances_per_source = max_instances_per_source

    def match(
        self,
        df_primary: pl.DataFrame,
        df_extended: pl.DataFrame,
        index: TemporalIndex,
        window: WindowSpec,
        pattern: MotifPattern,
        write_output: bool = False,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, Any]]:

        if pattern.matcher_type != "split_merge":
            raise ValueError(
                f"SplitMergeMatcher expects matcher_type='split_merge'. "
                f"Got {pattern.matcher_type}"
            )

        start_time = time.time()

        motif_rows = []
        membership_rows = []

        num_sources_scanned = 0
        num_split_combinations = 0
        num_merge_products_checked = 0

        num_rejected_duration = 0
        num_rejected_nodes = 0
        num_rejected_anchor = 0
        num_rejected_amount = 0
        num_rejected_split_phase_delta = 0
        num_rejected_merge_phase_delta = 0
        num_rejected_phase_gap = 0
        num_rejected_source_cap = 0

        # Pattern has n split edges and n merge edges.
        n_branches = len(pattern.edges) // 2

        for source, out_edges in index.out_edges.items():
            source_instance_count = 0
            num_sources_scanned += 1

            split_candidates = cap_edges(
                out_edges,
                max_candidates=self.max_out_candidates,
                policy=self.candidate_policy,
            )

            if self.amount_min is not None:
                split_candidates = [
                    e for e in split_candidates
                    if e.amount >= self.amount_min
                ]

            if len(split_candidates) < n_branches:
                continue

            for split_combo in combinations(split_candidates, n_branches):
                split_edges = sorted(
                    list(split_combo),
                    key=lambda e: (e.step, e.edge_id),
                )

                num_split_combinations += 1

                if not passes_consecutive_step_gap(
                    split_edges,
                    self.split_phase_delta,
                ):
                    num_rejected_split_phase_delta += 1
                    continue

                if not has_unique_edge_ids(split_edges):
                    continue

                a = split_edges[0].src

                if any(e.src != a for e in split_edges):
                    continue

                intermediates = [e.dst for e in split_edges]

                if not has_distinct_nodes([a] + intermediates):
                    num_rejected_nodes += 1
                    continue

                split_end_step = max(e.step for e in split_edges)

                max_end_step = min(
                    split_edges[0].step + pattern.max_duration,
                    split_end_step + pattern.max_duration,
                )

                # ----------------------------------------------------
                # Build merge candidates for each intermediate.
                # merge_by_intermediate[i] contains candidates from
                # intermediates[i] -> some sink.
                # ----------------------------------------------------
                merge_by_intermediate = []

                for mid in intermediates:
                    cands = index.outgoing(
                        src=mid,
                        t_min=split_end_step,
                        t_max=max_end_step,
                        include_left=False,
                        max_candidates=None,
                    )

                    if self.amount_min is not None:
                        cands = [
                            e for e in cands
                            if e.amount >= self.amount_min
                        ]

                    cands = cap_edges(
                        cands,
                        max_candidates=self.max_merge_candidates_per_intermediate,
                        policy=self.candidate_policy,
                    )

                    merge_by_intermediate.append(cands)

                if any(len(x) == 0 for x in merge_by_intermediate):
                    continue

                # ----------------------------------------------------
                # Group merge candidates by sink for each intermediate.
                # sink_maps[i][sink] = list of edges:
                #     intermediates[i] -> sink
                # ----------------------------------------------------
                sink_maps = []

                for cands in merge_by_intermediate:
                    sink_map = defaultdict(list)

                    for e in cands:
                        sink_map[e.dst].append(e)

                    sink_maps.append(sink_map)

                common_sinks = set(sink_maps[0].keys())

                for sm in sink_maps[1:]:
                    common_sinks &= set(sm.keys())

                if len(common_sinks) == 0:
                    continue

                for sink in common_sinks:
                    e_node = sink

                    if e_node in [a] + intermediates:
                        num_rejected_nodes += 1
                        continue

                    # lists[i] contains candidate edges from intermediates[i] -> e_node.
                    lists = [sm[e_node] for sm in sink_maps]

                    for merge_edges_tuple in product(*lists):
                        num_merge_products_checked += 1

                        try:
                            (
                                merge_edges_by_intermediate,
                                merge_edges_sorted,
                                role_map,
                            ) = _build_split_merge_roles(
                                split_edges=split_edges,
                                merge_edges_tuple=merge_edges_tuple,
                                intermediates=intermediates,
                                n_branches=n_branches,
                            )
                        except Exception:
                            continue

                        # For validation, use temporal order.
                        edges_for_validation = split_edges + merge_edges_sorted

                        # For role semantics, use mapping by intermediate.
                        # We still pass edges_for_validation to the row builder
                        # because duration, anchor, and membership should use
                        # the actual edge set in temporal-friendly order.
                        if not passes_consecutive_step_gap(
                            merge_edges_sorted,
                            self.merge_phase_delta,
                        ):
                            num_rejected_merge_phase_delta += 1
                            continue

                        if not phase_gap_ok(
                            split_edges,
                            merge_edges_sorted,
                            self.split_to_merge_delta,
                        ):
                            num_rejected_phase_gap += 1
                            continue

                        if not has_unique_edge_ids(edges_for_validation):
                            continue

                        if not is_within_total_duration(
                            edges_for_validation,
                            pattern.max_duration,
                        ):
                            num_rejected_duration += 1
                            continue

                        anchor = get_anchor_edge(edges_for_validation)

                        if not is_edge_in_primary_window(anchor, window):
                            num_rejected_anchor += 1
                            continue

                        if self.use_amount_ratio:
                            split_sum = sum(e.amount for e in split_edges)
                            merge_sum = sum(e.amount for e in merge_edges_by_intermediate)

                            if split_sum <= 0:
                                num_rejected_amount += 1
                                continue

                            ratio = merge_sum / split_sum

                            if (
                                pattern.amount_ratio_min is not None
                                and ratio < pattern.amount_ratio_min
                            ):
                                num_rejected_amount += 1
                                continue

                            if (
                                pattern.amount_ratio_max is not None
                                and ratio > pattern.amount_ratio_max
                            ):
                                num_rejected_amount += 1
                                continue

                        node_map = _build_split_merge_node_map(
                            source=a,
                            intermediates=intermediates,
                            sink=e_node,
                        )

                        try:
                            motif_row = make_motif_instance_row(
                                window_id=window.window_id,
                                pattern=pattern,
                                edges=edges_for_validation,
                                node_map=node_map,
                                role_map=role_map,
                                anchor_edge_id=anchor.edge_id,
                                validate=True,
                            )
                        except Exception:
                            continue

                        membership = make_edge_motif_membership_rows(
                            motif_row=motif_row,
                            pattern=pattern,
                            edges=edges_for_validation,
                        )

                        motif_rows.append(motif_row)
                        membership_rows.extend(membership)

                        source_instance_count += 1

                        if (
                            self.max_instances_per_source is not None
                            and source_instance_count >= self.max_instances_per_source
                        ):
                            num_rejected_source_cap += 1
                            break

                        if len(motif_rows) >= self.max_instances_per_window:
                            break

                    if (
                        self.max_instances_per_source is not None
                        and source_instance_count >= self.max_instances_per_source
                    ):
                        break

                    if len(motif_rows) >= self.max_instances_per_window:
                        break

                if (
                    self.max_instances_per_source is not None
                    and source_instance_count >= self.max_instances_per_source
                ):
                    break

                if len(motif_rows) >= self.max_instances_per_window:
                    break

            if len(motif_rows) >= self.max_instances_per_window:
                break

        motif_df = motif_instance_rows_to_polars(motif_rows)
        membership_df = membership_rows_to_polars(membership_rows)

        elapsed = time.time() - start_time

        stats = {
            "window_id": int(window.window_id),
            "motif_type": pattern.name,
            "matcher_type": pattern.matcher_type,

            "primary_num_edges": int(df_primary.height),
            "extended_num_edges": int(df_extended.height),

            "num_sources_scanned": int(num_sources_scanned),
            "num_split_combinations": int(num_split_combinations),
            "num_merge_products_checked": int(num_merge_products_checked),

            "num_instances": int(motif_df.height),
            "num_membership_rows": int(membership_df.height),

            "num_rejected_duration": int(num_rejected_duration),
            "num_rejected_nodes": int(num_rejected_nodes),
            "num_rejected_anchor": int(num_rejected_anchor),
            "num_rejected_amount": int(num_rejected_amount),
            "num_rejected_split_phase_delta": int(num_rejected_split_phase_delta),
            "num_rejected_merge_phase_delta": int(num_rejected_merge_phase_delta),
            "num_rejected_phase_gap": int(num_rejected_phase_gap),
            "num_rejected_source_cap": int(num_rejected_source_cap),

            "hit_max_instances_per_window": int(
                len(motif_rows) >= self.max_instances_per_window
            ),
            "elapsed_seconds": float(elapsed),
        }

        if write_output:
            motif_path, membership_path = write_motif_outputs(
                motif_rows=motif_rows,
                membership_rows=membership_rows,
                window_id=window.window_id,
                motif_type=pattern.name,
            )

            stats["motif_path"] = motif_path
            stats["membership_path"] = membership_path

        return motif_df, membership_df, stats



# ============================================================
# Cell 14: CenterInOutMatcher
# ============================================================

def _compute_center_inout_flow_features(
    in_edges: List[EdgeRecord],
    out_edges: List[EdgeRecord],
) -> Dict[str, float]:
    """
    Compute flow features for one center-in-out motif instance.

    Features:
        in_sum:
            Total amount flowing into the center.

        out_sum:
            Total amount flowing out from the center.

        flow_ratio:
            out_sum / in_sum.
            Close to 1.0 means the center redistributes almost all received money.

        handoff_gap:
            min(out_step) - max(in_step).
            Smaller gap means faster handoff from incoming phase to outgoing phase.
    """
    in_sum = float(sum(e.amount for e in in_edges))
    out_sum = float(sum(e.amount for e in out_edges))

    flow_ratio = out_sum / in_sum if in_sum > 0 else 0.0

    in_end = max(int(e.step) for e in in_edges)
    out_start = min(int(e.step) for e in out_edges)

    handoff_gap = int(out_start - in_end)

    return {
        "in_sum": in_sum,
        "out_sum": out_sum,
        "flow_ratio": float(flow_ratio),
        "handoff_gap": float(handoff_gap),
    }


class CenterInOutMatcher:
    """
    Matcher for center-in-out motifs.

    General pattern:
        in_1 -> center
        in_2 -> center
        ...
        in_n -> center

        center -> out_1
        center -> out_2
        ...
        center -> out_m

    AML interpretation:
        The center account collects money from multiple sources,
        then redistributes it to multiple destinations.

    Key improvement in this version:
        Outgoing candidates are queried after each incoming group is known.

        Instead of:
            get all outgoing edges of center
            combine them with every incoming group

        We do:
            choose incoming group
            compute in_end
            query outgoing in (in_end, in_end + center_handoff_delta]
            combine only temporally compatible outgoing edges
    """

    def __init__(
        self,
        max_in_candidates: int = 30,
        max_out_candidates: int = 20,
        max_instances_per_window: int = 50_000,
        amount_min: Optional[float] = None,
        use_amount_ratio: bool = False,
        candidate_policy: str = "hybrid",
        incoming_phase_delta: Optional[int] = DELTA_HOP,
        outgoing_phase_delta: Optional[int] = DELTA_HOP,
        center_handoff_delta: Optional[int] = DELTA_HOP,
        max_instances_per_center: Optional[int] = 2_000,
    ):
        self.max_in_candidates = max_in_candidates
        self.max_out_candidates = max_out_candidates
        self.max_instances_per_window = max_instances_per_window
        self.amount_min = amount_min
        self.use_amount_ratio = use_amount_ratio
        self.candidate_policy = candidate_policy
        self.incoming_phase_delta = incoming_phase_delta
        self.outgoing_phase_delta = outgoing_phase_delta
        self.center_handoff_delta = center_handoff_delta
        self.max_instances_per_center = max_instances_per_center

    def match(
        self,
        df_primary: pl.DataFrame,
        df_extended: pl.DataFrame,
        index: TemporalIndex,
        window: WindowSpec,
        pattern: MotifPattern,
        write_output: bool = False,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, Any]]:

        if pattern.matcher_type != "center_in_out":
            raise ValueError(
                f"CenterInOutMatcher expects matcher_type='center_in_out'. "
                f"Got {pattern.matcher_type}"
            )

        start_time = time.time()

        motif_rows = []
        membership_rows = []

        num_centers_scanned = 0
        num_in_combinations_checked = 0
        num_products_checked = 0

        num_rejected_time = 0
        num_rejected_nodes = 0
        num_rejected_anchor = 0
        num_rejected_amount = 0
        num_rejected_incoming_phase_delta = 0
        num_rejected_outgoing_phase_delta = 0
        num_rejected_handoff_delta = 0
        num_rejected_center_cap = 0
        num_rejected_low_outgoing_for_group = 0

        # Generalize by reading pattern structure.
        n_in = sum(1 for e in pattern.edges if e.dst == "center")
        n_out = sum(1 for e in pattern.edges if e.src == "center")

        candidate_centers = set(index.in_edges.keys()) & set(index.out_edges.keys())

        for center in candidate_centers:
            center_instance_count = 0
            num_centers_scanned += 1

            # Fast degree check before candidate selection.
            if len(index.in_edges.get(center, [])) < n_in:
                continue

            if len(index.out_edges.get(center, [])) < n_out:
                continue

            # Only cap incoming globally.
            # Outgoing will be queried per incoming group using in_end.
            incoming = cap_edges(
                index.in_edges.get(center, []),
                max_candidates=self.max_in_candidates,
                policy=self.candidate_policy,
            )

            if self.amount_min is not None:
                incoming = [
                    e for e in incoming
                    if e.amount >= self.amount_min
                ]

            if len(incoming) < n_in:
                continue

            for in_combo in combinations(incoming, n_in):
                num_in_combinations_checked += 1

                in_edges = sorted(
                    list(in_combo),
                    key=lambda e: (e.step, e.edge_id),
                )

                if not passes_consecutive_step_gap(
                    in_edges,
                    self.incoming_phase_delta,
                ):
                    num_rejected_incoming_phase_delta += 1
                    continue

                src_nodes = [e.src for e in in_edges]

                if not has_distinct_nodes(src_nodes + [center]):
                    num_rejected_nodes += 1
                    continue

                in_end = max(int(e.step) for e in in_edges)

                # ----------------------------------------------------
                # Group-aware outgoing query:
                # only query outgoing edges after this incoming group.
                # ----------------------------------------------------
                if self.center_handoff_delta is not None:
                    t_out_max = min(
                        in_end + self.center_handoff_delta,
                        in_edges[0].step + pattern.max_duration,
                    )
                else:
                    t_out_max = in_edges[0].step + pattern.max_duration

                outgoing_for_group = index.outgoing(
                    src=center,
                    t_min=in_end,
                    t_max=t_out_max,
                    include_left=False,
                )

                if self.amount_min is not None:
                    outgoing_for_group = [
                        e for e in outgoing_for_group
                        if e.amount >= self.amount_min
                    ]

                outgoing_for_group = cap_edges(
                    outgoing_for_group,
                    max_candidates=self.max_out_candidates,
                    policy=self.candidate_policy,
                )

                if len(outgoing_for_group) < n_out:
                    num_rejected_low_outgoing_for_group += 1
                    continue

                for out_combo in combinations(outgoing_for_group, n_out):
                    num_products_checked += 1

                    out_edges = sorted(
                        list(out_combo),
                        key=lambda e: (e.step, e.edge_id),
                    )

                    if not passes_consecutive_step_gap(
                        out_edges,
                        self.outgoing_phase_delta,
                    ):
                        num_rejected_outgoing_phase_delta += 1
                        continue

                    dst_nodes = [e.dst for e in out_edges]

                    if not has_distinct_nodes(src_nodes + [center] + dst_nodes):
                        num_rejected_nodes += 1
                        continue

                    out_start = min(int(e.step) for e in out_edges)

                    # Incoming must happen before outgoing.
                    if out_start <= in_end:
                        num_rejected_time += 1
                        continue

                    if not phase_gap_ok(
                        in_edges,
                        out_edges,
                        self.center_handoff_delta,
                    ):
                        num_rejected_handoff_delta += 1
                        continue

                    edges = in_edges + out_edges

                    if not has_unique_edge_ids(edges):
                        continue

                    if not is_within_total_duration(
                        edges,
                        pattern.max_duration,
                    ):
                        num_rejected_time += 1
                        continue

                    anchor = get_anchor_edge(edges)

                    if not is_edge_in_primary_window(anchor, window):
                        num_rejected_anchor += 1
                        continue

                    flow_features = _compute_center_inout_flow_features(
                        in_edges=in_edges,
                        out_edges=out_edges,
                    )

                    if self.use_amount_ratio:
                        ratio = flow_features["flow_ratio"]

                        if ratio <= 0:
                            num_rejected_amount += 1
                            continue

                        if (
                            pattern.amount_ratio_min is not None
                            and ratio < pattern.amount_ratio_min
                        ):
                            num_rejected_amount += 1
                            continue

                        if (
                            pattern.amount_ratio_max is not None
                            and ratio > pattern.amount_ratio_max
                        ):
                            num_rejected_amount += 1
                            continue

                    # General node_map for any n_in, n_out.
                    node_map = {"center": int(center)}

                    node_map.update({
                        f"in_{i + 1}": int(in_edges[i].src)
                        for i in range(n_in)
                    })

                    node_map.update({
                        f"out_{i + 1}": int(out_edges[i].dst)
                        for i in range(n_out)
                    })

                    role_map = {}

                    for i in range(n_in):
                        role_map[f"incoming_{i + 1}"] = int(in_edges[i].edge_id)

                    for i in range(n_out):
                        role_map[f"outgoing_{i + 1}"] = int(out_edges[i].edge_id)

                    try:
                        motif_row = make_motif_instance_row(
                            window_id=window.window_id,
                            pattern=pattern,
                            edges=edges,
                            node_map=node_map,
                            role_map=role_map,
                            anchor_edge_id=anchor.edge_id,
                            validate=True,
                        )

                        # Add flow features into motif row.
                        motif_row.update(flow_features)

                    except Exception:
                        continue

                    membership = make_edge_motif_membership_rows(
                        motif_row=motif_row,
                        pattern=pattern,
                        edges=edges,
                    )

                    motif_rows.append(motif_row)
                    membership_rows.extend(membership)

                    center_instance_count += 1

                    if (
                        self.max_instances_per_center is not None
                        and center_instance_count >= self.max_instances_per_center
                    ):
                        num_rejected_center_cap += 1
                        break

                    if len(motif_rows) >= self.max_instances_per_window:
                        break

                if (
                    self.max_instances_per_center is not None
                    and center_instance_count >= self.max_instances_per_center
                ):
                    break

                if len(motif_rows) >= self.max_instances_per_window:
                    break

            if len(motif_rows) >= self.max_instances_per_window:
                break

        motif_df = motif_instance_rows_to_polars(motif_rows)
        membership_df = membership_rows_to_polars(membership_rows)

        elapsed = time.time() - start_time

        stats = {
            "window_id": int(window.window_id),
            "motif_type": pattern.name,
            "matcher_type": pattern.matcher_type,

            "primary_num_edges": int(df_primary.height),
            "extended_num_edges": int(df_extended.height),

            "n_in": int(n_in),
            "n_out": int(n_out),

            "num_centers_scanned": int(num_centers_scanned),
            "num_in_combinations_checked": int(num_in_combinations_checked),
            "num_products_checked": int(num_products_checked),

            "num_instances": int(motif_df.height),
            "num_membership_rows": int(membership_df.height),

            "num_rejected_time": int(num_rejected_time),
            "num_rejected_nodes": int(num_rejected_nodes),
            "num_rejected_anchor": int(num_rejected_anchor),
            "num_rejected_amount": int(num_rejected_amount),
            "num_rejected_incoming_phase_delta": int(num_rejected_incoming_phase_delta),
            "num_rejected_outgoing_phase_delta": int(num_rejected_outgoing_phase_delta),
            "num_rejected_handoff_delta": int(num_rejected_handoff_delta),
            "num_rejected_center_cap": int(num_rejected_center_cap),
            "num_rejected_low_outgoing_for_group": int(num_rejected_low_outgoing_for_group),

            "hit_max_instances_per_window": int(
                len(motif_rows) >= self.max_instances_per_window
            ),
            "elapsed_seconds": float(elapsed),
        }

        if write_output:
            motif_path, membership_path = write_motif_outputs(
                motif_rows=motif_rows,
                membership_rows=membership_rows,
                window_id=window.window_id,
                motif_type=pattern.name,
            )

            stats["motif_path"] = motif_path
            stats["membership_path"] = membership_path

        return motif_df, membership_df, stats




# Register split-merge matcher.
split_merge_matcher = SplitMergeMatcher(
    max_out_candidates=12,
    max_merge_candidates_per_intermediate=3,
    max_instances_per_window=30_000,
    amount_min=None,
    use_amount_ratio=True,
    candidate_policy="hybrid",
    split_phase_delta=DELTA_HOP,
    merge_phase_delta=DELTA_HOP,
    split_to_merge_delta=DELTA_HOP,
    max_instances_per_source=2_000,
)

MATCHER_REGISTRY["split_merge"] = split_merge_matcher

print("SplitMergeMatcher (role mapping fixed) registered.")
print("MATCHER_REGISTRY keys:", list(MATCHER_REGISTRY.keys()))


center_inout_matcher = CenterInOutMatcher(
    max_in_candidates=15,
    max_out_candidates=8,
    max_instances_per_window=30_000,
    amount_min=None,
    use_amount_ratio=True,
    candidate_policy="hybrid",
    incoming_phase_delta=DELTA_HOP,
    outgoing_phase_delta=DELTA_HOP,
    center_handoff_delta=DELTA_HOP,
    max_instances_per_center=2_000,
)

MATCHER_REGISTRY["center_in_out"] = center_inout_matcher

print("CenterInOutMatcher (group-aware outgoing + flow features) registered.")
print("MATCHER_REGISTRY keys:", list(MATCHER_REGISTRY.keys()))
print("Cell 14 completed.")



# ============================================================
# Cell 15: Post-Run Aggregation
# ============================================================
# Run this cell AFTER a full pipeline run (Cell 17) completes.

from pathlib import Path


def merge_and_deduplicate_motif_shards(
    motif_instance_dir: str,
    output_path:        str,
) -> pl.DataFrame:
    """
    Merge all per-window motif instance shards and deduplicate by canonical_key.
    Keeps the instance from the earliest window when duplicates exist.
    """
    paths  = sorted(Path(motif_instance_dir).glob("*.parquet"))
    shards = [pl.read_parquet(p) for p in paths if p.stat().st_size > 0]

    if not shards:
        print("No motif instance shards found.")
        return pl.DataFrame()

    df_all = pl.concat(shards)

    df_deduped = (
        df_all
        .sort(["canonical_key", "window_id"])
        .unique(subset=["canonical_key"], keep="first")
    )

    df_deduped.write_parquet(output_path)
    print(f"Deduplicated motif instances: {df_deduped.height} (from {df_all.height} raw)")
    return df_deduped


def build_edge_participation_summary(
    membership_dir: str,
    df_edges:       pl.DataFrame,
) -> pl.DataFrame:
    """
    For each edge_id, aggregate all motif memberships into summary statistics.

    Output columns:
        edge_id, num_motif_instances, num_motif_types,
        motif_types_list, roles_list,
        + all original df_edges columns (joined)
    """
    paths  = sorted(Path(membership_dir).glob("*.parquet"))
    shards = [pl.read_parquet(p) for p in paths if p.stat().st_size > 0]

    if not shards:
        print("No membership shards found.")
        return df_edges

    df_membership = pl.concat(shards)

    # Deduplicate membership rows by (edge_id, motif_instance_id).
    df_membership = df_membership.unique(subset=["edge_id", "motif_instance_id"])

    summary = (
        df_membership
        .group_by("edge_id")
        .agg([
            pl.col("motif_instance_id").n_unique().alias("num_motif_instances"),
            pl.col("motif_type").n_unique().alias("num_motif_types"),
            pl.col("motif_type").unique().alias("motif_types_list"),
            pl.col("role_in_motif").unique().alias("roles_list"),
        ])
    )

    result = df_edges.join(summary, on="edge_id", how="left")
    result = result.with_columns([
        pl.col("num_motif_instances").fill_null(0),
        pl.col("num_motif_types").fill_null(0),
    ])

    return result


def compute_structural_recall(
    edge_participation: pl.DataFrame,
) -> Dict[str, Any]:
    """
    POST-HOC EVALUATION ONLY. Not used during mining.

    Computes what fraction of edges labelled is_sar=1 appear in
    at least one matched motif instance.

    This tells us whether our purely structural miner happens to
    recover the same edges human analysts flagged as suspicious.

    Low recall means: look at the topology of uncovered SAR edges
    and decide whether new motif families are needed.
    """
    if "is_sar" not in edge_participation.columns:
        return {"error": "is_sar column not present"}

    total_edges = edge_participation.height
    sar_edges   = edge_participation.filter(pl.col("is_sar") == 1)
    total_sar   = sar_edges.height

    if total_sar == 0:
        return {"total_sar": 0, "structural_recall": None}

    covered = sar_edges.filter(pl.col("num_motif_instances") > 0)
    n_covered = covered.height

    # Breakdown by motif type for covered SAR edges.
    if "motif_types_list" in covered.columns:
        type_counts = (
            covered
            .explode("motif_types_list")
            .group_by("motif_types_list")
            .len()
            .sort("len", descending=True)
        )
    else:
        type_counts = pl.DataFrame()

    return {
        "total_edges":                  total_edges,
        "total_sar_edges":              total_sar,
        "sar_in_at_least_one_motif":    n_covered,
        "structural_recall":            round(n_covered / total_sar, 4),
        "sar_in_no_motif":              total_sar - n_covered,
        "motif_type_breakdown":         type_counts,
    }


# ============================================================
# Run aggregation (uncomment after a full Cell 17 run)
# ============================================================

# DEDUPED_MOTIF_PATH = f"{OUTPUT_DIR}/all_motif_instances_deduped.parquet"

# df_motif_deduped = merge_and_deduplicate_motif_shards(
#     motif_instance_dir = MOTIF_INSTANCE_DIR,
#     output_path        = DEDUPED_MOTIF_PATH,
# )

# edge_participation = build_edge_participation_summary(
#     membership_dir = MEMBERSHIP_DIR,
#     df_edges       = df_edges,
# )

# recall_result = compute_structural_recall(edge_participation)

# print("\nStructural recall (post-hoc evaluation):")
# for k, v in recall_result.items():
#     if isinstance(v, pl.DataFrame):
#         print(f"\n  {k}:")
#         display(v)
#     else:
#         print(f"  {k}: {v}")

# EDGE_FEATURE_PATH = f"{FEATURE_DIR}/edge_participation_summary.parquet"
# edge_participation.write_parquet(EDGE_FEATURE_PATH)
# print(f"\nEdge participation summary saved to {EDGE_FEATURE_PATH}")

print("Cell 15 (aggregation) loaded. Uncomment run block after full pipeline completes.")
print("Cell 15 completed.")



# ============================================================
# Cell 16: Validation Run — one window, all families
# ============================================================
"""
Purpose:
    Run exactly 1 window for each matcher type to confirm:
    - generalized n-branch fan patterns work for all sizes
    - cycle bidirectional search activates correctly for k >= 7
    - split_merge and center_inout work for all configured sizes
    - stacked_bipartite produces output without error

    Check that num_instances > 0 for at least one pattern per family.
    Check that no matcher errors or crashes.
    Check that output schemas match the expected parquet structure.
"""

VALIDATION_PATTERNS = [
    # One representative from each family
    fan_in_patterns[0],     # fan_in_4 (n=3)
    fan_in_patterns[-1],    # fan_in_8 (n=7) — largest
    fan_out_patterns[0],
    fan_out_patterns[-1],
    cycle_patterns[0],      # cycle_5
    cycle_patterns[2],      # cycle_7 — triggers bidirectional
    split_merge_patterns[0],
    split_merge_patterns[-1],
    center_inout_patterns[0],
    center_inout_patterns[-1],
    # stacked_bipartite_patterns[0],
]

required_types = sorted(set(p.matcher_type for p in VALIDATION_PATTERNS))
missing = [t for t in required_types if t not in MATCHER_REGISTRY]
if missing:
    raise ValueError(f"Missing matchers: {missing}")

validation_stats_df = run_matchers_over_windows(
    df_edges              = df_edges,
    windows               = windows,
    patterns_to_run       = VALIDATION_PATTERNS,
    matcher_registry      = MATCHER_REGISTRY,
    max_windows           = 1,
    skip_existing         = False,
    write_empty_outputs   = True,
)

print("\nValidation run stats:")
display(validation_stats_df)

# Check for any zero-instance patterns (might indicate bugs, worth inspecting).
zero_instance = validation_stats_df.filter(
    (pl.col("status") == "completed") & (pl.col("num_instances") == 0)
)
if zero_instance.height > 0:
    print("\nPatterns with 0 instances in validation window:")
    display(zero_instance.select(["motif_type", "matcher_type", "primary_num_edges"]))
    print("(This may be expected if the first window has sparse matching topology.)")
else:
    print("\nAll patterns found at least one instance in the validation window.")

print("\nCell 16 completed.")



# ============================================================
# Cell 17: Full Production Run — All Patterns, All Windows
# ============================================================
"""
Purpose:
    Run the complete retained motif set across all temporal windows.

Recommended workflow:
    1. Run Cell 16 (validation) first.
    2. Confirm no matcher errors and at least some instances found.
    3. Confirm runtime per window is acceptable (< 5 min per window total).
    4. Set MAX_WINDOWS_TO_RUN_ALL = None and run this cell.
"""

PATTERNS_TO_RUN_ALL = (
    fan_in_patterns          # fan_in_4 .. fan_in_8
    + fan_out_patterns       # fan_out_4 .. fan_out_8
    + cycle_patterns         # cycle_5 .. cycle_10
    + split_merge_patterns   # split_merge_6, _8, _10
    + center_inout_patterns  # all CENTER_INOUT_CONFIGS
    # + stacked_bipartite_patterns  # all STACKED_BIPARTITE_CONFIGS
)

required_matcher_types = sorted(set(p.matcher_type for p in PATTERNS_TO_RUN_ALL))
missing_matchers = [t for t in required_matcher_types if t not in MATCHER_REGISTRY]

if missing_matchers:
    raise ValueError(
        f"Missing matchers in MATCHER_REGISTRY: {missing_matchers}. "
        f"Available: {list(MATCHER_REGISTRY.keys())}"
    )

print(f"Full run: {len(PATTERNS_TO_RUN_ALL)} patterns across {len(windows)} windows.")
print("Patterns:", [p.name for p in PATTERNS_TO_RUN_ALL])

MAX_WINDOWS_TO_RUN_ALL    = None    # set to small int (e.g. 4) for test run
SKIP_EXISTING_OUTPUTS_ALL = True    # resume safely if interrupted
WRITE_EMPTY_OUTPUTS_ALL   = True

all_run_stats_df = run_matchers_over_windows(
    df_edges            = df_edges,
    windows             = windows,
    patterns_to_run     = PATTERNS_TO_RUN_ALL,
    matcher_registry    = MATCHER_REGISTRY,
    max_windows         = MAX_WINDOWS_TO_RUN_ALL,
    skip_existing       = SKIP_EXISTING_OUTPUTS_ALL,
    write_empty_outputs = WRITE_EMPTY_OUTPUTS_ALL,
)

print("\nFull run stats:")
display(all_run_stats_df)

# Aggregate summary.
if all_run_stats_df.height > 0 and "status" in all_run_stats_df.columns:
    completed = all_run_stats_df.filter(pl.col("status") == "completed")

    if completed.height > 0:
        summary = (
            completed
            .group_by("motif_type")
            .agg([
                pl.col("num_instances").sum().alias("total_instances"),
                pl.col("num_membership_rows").sum().alias("total_membership_rows"),
                pl.col("elapsed_seconds").sum().alias("total_match_seconds"),
                pl.col("hit_max_instances_per_window").sum().alias("windows_hit_cap"),
                pl.col("window_id").n_unique().alias("windows_completed"),
            ])
            .sort("total_instances", descending=True)
        )

        print("\nAggregate summary by motif type:")
        display(summary)

        capped = summary.filter(pl.col("windows_hit_cap") > 0)
        if capped.height > 0:
            print("\nWarning: these patterns hit max_instances_per_window cap:")
            display(capped.select(["motif_type", "windows_hit_cap",
                                   "total_instances", "windows_completed"]))
            print("Consider reducing candidate caps or window size for these families.")

print("\nCell 17 completed.")