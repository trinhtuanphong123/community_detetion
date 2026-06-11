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
# Important:
#   Long cycles are much more expensive than cycle_5.
#   Use tight max_branching and max_instances_per_window.

class CycleKMatcher:
    """
    Generalized matcher for path_cycle motifs.

    Pattern:
        v1 -> v2 -> ... -> vk -> v1

    Strategy:
        1. Anchor on primary edge e1: v1 -> v2.
        2. DFS forward from current node.
        3. At depth k-1, close cycle by querying current_node -> v1.
        4. Enforce strict temporal order.
        5. Enforce distinct intermediate nodes.
        6. Enforce total duration.
        7. Use canonical anchor to reduce duplicate cycle rotations.
    """

    def __init__(
        self,
        max_branching: int = 5,
        max_instances_per_window: int = 20_000,
        amount_min: Optional[float] = None,
        use_amount_ratio: bool = False,
        candidate_policy: str = "hybrid",
        delta_hop: Optional[int] = DELTA_HOP,
        max_anchors_per_window: Optional[int] = None,
    ):
        self.max_branching = max_branching
        self.max_instances_per_window = max_instances_per_window
        self.amount_min = amount_min
        self.use_amount_ratio = use_amount_ratio
        self.candidate_policy = candidate_policy
        self.delta_hop = delta_hop
        self.max_anchors_per_window = max_anchors_per_window

    def _filter_candidates(self, edges: List[EdgeRecord]) -> List[EdgeRecord]:
        """
        Apply amount filter and candidate cap.
        """
        if self.amount_min is not None:
            edges = [e for e in edges if e.amount >= self.amount_min]

        return cap_edges(
            edges,
            max_candidates=self.max_branching,
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

        primary_edges = self._get_primary_edges(df_primary)

        for anchor_edge in primary_edges:
            num_anchor_edges += 1

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

                    close_candidates = self._filter_candidates(close_candidates)
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

                        # Distinct cycle nodes only. Closing returns to start_node,
                        # so path_nodes should already be distinct.
                        if not has_distinct_nodes(path_nodes):
                            num_rejected_nodes += 1
                            continue

                        # Canonical anchor rule:
                        # keep only cycles whose anchor is the earliest edge.
                        # This reduces duplicate rotations across different anchors.
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

                        node_map = {
                            f"v{i + 1}": int(node_id)
                            for i, node_id in enumerate(path_nodes)
                        }

                        role_map = {
                            pattern.edges[i].role: int(edges[i].edge_id)
                            for i in range(k)
                        }

                        try:
                            motif_row = make_motif_instance_row(
                                window_id=window.window_id,
                                pattern=pattern,
                                edges=edges,
                                node_map=node_map,
                                role_map=role_map,
                                anchor_edge_id=anchor_edge.edge_id,
                                validate=True,
                            )
                        except Exception:
                            continue

                        membership = make_edge_motif_membership_rows(
                            motif_row=motif_row,
                            pattern=pattern,
                            edges=edges,
                        )

                        motif_rows.append(motif_row)
                        membership_rows.extend(membership)

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

                next_candidates = self._filter_candidates(next_candidates)

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


# ------------------------------------------------------------
# Register generalized cycle matcher
# ------------------------------------------------------------

cycle_k_matcher = CycleKMatcher(
    max_branching=5,
    max_instances_per_window=20_000,
    amount_min=None,
    use_amount_ratio=False,
    candidate_policy="hybrid",
    delta_hop=DELTA_HOP,
    max_anchors_per_window=None,
)

MATCHER_REGISTRY["path_cycle"] = cycle_k_matcher

print("CycleKMatcher registered.")
print("MATCHER_REGISTRY keys:", list(MATCHER_REGISTRY.keys()))
print("Cell 13 completed.")


# ============================================================
# Cell 14: SplitMergeMatcher and CenterInOutMatcher
# ============================================================

class SplitMergeMatcher:
    """
    Matcher for split_merge_5.

    Pattern:
        a -> b
        a -> c
        a -> d
        b -> e
        c -> e
        d -> e

    Strategy:
        1. Choose source a.
        2. Select 3 outgoing split edges from a.
        3. For intermediate nodes b,c,d, find common sink e.
        4. Pick one merge edge from each intermediate to e.
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
            raise ValueError(f"SplitMergeMatcher expects matcher_type='split_merge'. Got {pattern.matcher_type}")

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

        for source, out_edges in index.out_edges.items():
            source_instance_count = 0
            num_sources_scanned += 1

            split_candidates = cap_edges(
                out_edges,
                max_candidates=self.max_out_candidates,
                policy=self.candidate_policy,
            )

            if self.amount_min is not None:
                split_candidates = [e for e in split_candidates if e.amount >= self.amount_min]

            if len(split_candidates) < 3:
                continue

            for split_triplet in combinations(split_candidates, 3):
                split_edges = sorted(list(split_triplet), key=lambda e: (e.step, e.edge_id))
                num_split_combinations += 1

                if not passes_consecutive_step_gap(split_edges, self.split_phase_delta):
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
                max_end_step = min(split_edges[0].step + pattern.max_duration, split_end_step + pattern.max_duration)

                # Build possible merge edges by common sink.
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
                        cands = [e for e in cands if e.amount >= self.amount_min]

                    cands = cap_edges(
                        cands,
                        max_candidates=self.max_merge_candidates_per_intermediate,
                        policy=self.candidate_policy,
                    )

                    merge_by_intermediate.append(cands)

                if any(len(x) == 0 for x in merge_by_intermediate):
                    continue

                # Group merge candidates by sink for each intermediate.
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

                    lists = [sm[e_node] for sm in sink_maps]

                    for merge_edges_tuple in product(*lists):
                        merge_edges = sorted(list(merge_edges_tuple), key=lambda e: (e.step, e.edge_id))
                        edges = split_edges + merge_edges
                        num_merge_products_checked += 1

                        if not passes_consecutive_step_gap(merge_edges, self.merge_phase_delta):
                            num_rejected_merge_phase_delta += 1
                            continue

                        if not phase_gap_ok(split_edges, merge_edges, self.split_to_merge_delta):
                            num_rejected_phase_gap += 1
                            continue

                        if not has_unique_edge_ids(edges):
                            continue

                        if not is_within_total_duration(edges, pattern.max_duration):
                            num_rejected_duration += 1
                            continue

                        anchor = get_anchor_edge(edges)
                        if not is_edge_in_primary_window(anchor, window):
                            num_rejected_anchor += 1
                            continue

                        if self.use_amount_ratio:
                            split_sum = sum(e.amount for e in split_edges)
                            merge_sum = sum(e.amount for e in merge_edges)

                            if split_sum <= 0:
                                num_rejected_amount += 1
                                continue

                            ratio = merge_sum / split_sum

                            if pattern.amount_ratio_min is not None and ratio < pattern.amount_ratio_min:
                                num_rejected_amount += 1
                                continue

                            if pattern.amount_ratio_max is not None and ratio > pattern.amount_ratio_max:
                                num_rejected_amount += 1
                                continue

                        node_map = {
                            "a": int(a),
                            "b": int(intermediates[0]),
                            "c": int(intermediates[1]),
                            "d": int(intermediates[2]),
                            "e": int(e_node),
                        }

                        role_map = {
                            "split_1": int(split_edges[0].edge_id),
                            "split_2": int(split_edges[1].edge_id),
                            "split_3": int(split_edges[2].edge_id),
                            "merge_1": int(merge_edges[0].edge_id),
                            "merge_2": int(merge_edges[1].edge_id),
                            "merge_3": int(merge_edges[2].edge_id),
                        }

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
                        except Exception:
                            continue

                        membership = make_edge_motif_membership_rows(
                            motif_row=motif_row,
                            pattern=pattern,
                            edges=edges,
                        )

                        motif_rows.append(motif_row)
                        membership_rows.extend(membership)

                        source_instance_count += 1

                        if (self.max_instances_per_source is not None and source_instance_count >= self.max_instances_per_source):
                            # num_rejected_source_cap += 1
                            break

                        if len(motif_rows) >= self.max_instances_per_window:
                            break

                    if len(motif_rows) >= self.max_instances_per_window:
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
            "hit_max_instances_per_window": int(len(motif_rows) >= self.max_instances_per_window),
            "elapsed_seconds": float(elapsed),
            "num_rejected_split_phase_delta": int(num_rejected_split_phase_delta),
            "num_rejected_merge_phase_delta": int(num_rejected_merge_phase_delta),
            "num_rejected_phase_gap": int(num_rejected_phase_gap),
            "num_rejected_source_cap": int(num_rejected_source_cap),
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


class CenterInOutMatcher:
    """
    Matcher for fanin_fanout_6.

    Pattern:
        a -> d
        b -> d
        c -> d
        d -> e
        d -> f
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
            raise ValueError(f"CenterInOutMatcher expects matcher_type='center_in_out'. Got {pattern.matcher_type}")

        start_time = time.time()

        motif_rows = []
        membership_rows = []

        num_centers_scanned = 0
        num_products_checked = 0
        num_rejected_time = 0
        num_rejected_nodes = 0
        num_rejected_anchor = 0
        num_rejected_amount = 0
        num_rejected_incoming_phase_delta = 0
        num_rejected_outgoing_phase_delta = 0
        num_rejected_handoff_delta = 0
        num_rejected_center_cap = 0

        candidate_centers = set(index.in_edges.keys()) & set(index.out_edges.keys())

        for center in candidate_centers:
            center_instance_count = 0
            num_centers_scanned += 1

            incoming = cap_edges(
                index.in_edges.get(center, []),
                self.max_in_candidates,
                policy=self.candidate_policy,
            )
            outgoing = cap_edges(
                index.out_edges.get(center, []),
                self.max_out_candidates,
                policy=self.candidate_policy,
            )

            if self.amount_min is not None:
                incoming = [e for e in incoming if e.amount >= self.amount_min]
                outgoing = [e for e in outgoing if e.amount >= self.amount_min]

            if len(incoming) < 3 or len(outgoing) < 2:
                continue

            for in_triplet in combinations(incoming, 3):
                in_edges = sorted(list(in_triplet), key=lambda e: (e.step, e.edge_id))

                if not passes_consecutive_step_gap(in_edges, self.incoming_phase_delta):
                    num_rejected_incoming_phase_delta += 1
                    continue

                src_nodes = [e.src for e in in_edges]
                if not has_distinct_nodes(src_nodes + [center]):
                    num_rejected_nodes += 1
                    continue

                in_end = max(e.step for e in in_edges)

                for out_pair in combinations(outgoing, 2):
                    out_edges = sorted(list(out_pair), key=lambda e: (e.step, e.edge_id))
                    num_products_checked += 1
                    if not passes_consecutive_step_gap(out_edges, self.outgoing_phase_delta):
                        num_rejected_outgoing_phase_delta += 1
                        continue

                    dst_nodes = [e.dst for e in out_edges]

                    if not has_distinct_nodes(src_nodes + [center] + dst_nodes):
                        num_rejected_nodes += 1
                        continue

                    out_start = min(e.step for e in out_edges)

                    # Incoming must happen before outgoing.
                    if out_start <= in_end:
                        num_rejected_time += 1
                        continue

                    if not phase_gap_ok(in_edges, out_edges, self.center_handoff_delta):
                        num_rejected_handoff_delta += 1
                        continue

                    edges = in_edges + out_edges

                    if not has_unique_edge_ids(edges):
                        continue

                    if not is_within_total_duration(edges, pattern.max_duration):
                        num_rejected_time += 1
                        continue

                    anchor = get_anchor_edge(edges)
                    if not is_edge_in_primary_window(anchor, window):
                        num_rejected_anchor += 1
                        continue

                    if self.use_amount_ratio:
                        in_sum = sum(e.amount for e in in_edges)
                        out_sum = sum(e.amount for e in out_edges)

                        if in_sum <= 0:
                            num_rejected_amount += 1
                            continue

                        ratio = out_sum / in_sum

                        if pattern.amount_ratio_min is not None and ratio < pattern.amount_ratio_min:
                            num_rejected_amount += 1
                            continue

                        if pattern.amount_ratio_max is not None and ratio > pattern.amount_ratio_max:
                            num_rejected_amount += 1
                            continue

                    node_map = {
                        "a": int(in_edges[0].src),
                        "b": int(in_edges[1].src),
                        "c": int(in_edges[2].src),
                        "d": int(center),
                        "e": int(out_edges[0].dst),
                        "f": int(out_edges[1].dst),
                    }

                    role_map = {
                        "incoming_1": int(in_edges[0].edge_id),
                        "incoming_2": int(in_edges[1].edge_id),
                        "incoming_3": int(in_edges[2].edge_id),
                        "outgoing_1": int(out_edges[0].edge_id),
                        "outgoing_2": int(out_edges[1].edge_id),
                    }

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
                        # num_rejected_center_cap += 1
                        break

                    if len(motif_rows) >= self.max_instances_per_window:
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
            "num_centers_scanned": int(num_centers_scanned),
            "num_products_checked": int(num_products_checked),
            "num_instances": int(motif_df.height),
            "num_membership_rows": int(membership_df.height),
            "num_rejected_time": int(num_rejected_time),
            "num_rejected_nodes": int(num_rejected_nodes),
            "num_rejected_anchor": int(num_rejected_anchor),
            "num_rejected_amount": int(num_rejected_amount),
            "hit_max_instances_per_window": int(len(motif_rows) >= self.max_instances_per_window),
            "elapsed_seconds": float(elapsed),
            "num_rejected_incoming_phase_delta": int(num_rejected_incoming_phase_delta),
            "num_rejected_outgoing_phase_delta": int(num_rejected_outgoing_phase_delta),
            "num_rejected_handoff_delta": int(num_rejected_handoff_delta),
            "num_rejected_center_cap": int(num_rejected_center_cap),
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


MATCHER_REGISTRY["split_merge"] = split_merge_matcher
MATCHER_REGISTRY["center_in_out"] = center_inout_matcher

print("SplitMergeMatcher and CenterInOutMatcher registered.")
print("MATCHER_REGISTRY keys:", list(MATCHER_REGISTRY.keys()))
print("Cell 14 completed.")



# ============================================================
# Cell 16: Run Complex Motif Matchers
# ============================================================
"""
Purpose:
Run only the complex motif patterns that are currently kept:
1. cycle_5
2. split_merge_5
3. fanin_fanout_6

Note:
two_stage_split_8 is intentionally excluded because it is expensive
and still requires separate debugging.
"""

# Giả định các biến pattern đã được định nghĩa trước đó
PATTERNS_TO_RUN_COMPLEX = [
    cycle_5,
    split_merge_5,
    fanin_fanout_6,
]

# ------------------------------------------------------------
# Validate matcher registry
# ------------------------------------------------------------
required_matcher_types = sorted(set(p.matcher_type for p in PATTERNS_TO_RUN_COMPLEX))
missing_matchers = [
    matcher_type
    for matcher_type in required_matcher_types
    if matcher_type not in MATCHER_REGISTRY
]

if missing_matchers:
    raise ValueError(
        f"Missing matchers in MATCHER_REGISTRY: {missing_matchers}. "
        f"Available keys: {list(MATCHER_REGISTRY.keys())}"
    )

print("Cell 16 complex motif setup ready.")
print("Patterns to run:", [p.name for p in PATTERNS_TO_RUN_COMPLEX])
print("Required matcher types:", required_matcher_types)
print("Matcher registry:", list(MATCHER_REGISTRY.keys()))

# ------------------------------------------------------------
# Run mode
# ------------------------------------------------------------
# Recommended workflow:
# 1. Keep MAX_WINDOWS_TO_RUN_COMPLEX = 1 for debugging.
# 2. After checking runtime/output size, set to None for all windows.
MAX_WINDOWS_TO_RUN_COMPLEX = 1
SKIP_EXISTING_OUTPUTS_COMPLEX = False
WRITE_EMPTY_OUTPUTS_COMPLEX = True

# ------------------------------------------------------------
# Run complex motifs
# ------------------------------------------------------------
complex_run_stats_df = run_matchers_over_windows(
    df_edges=df_edges,
    windows=windows,
    patterns_to_run=PATTERNS_TO_RUN_COMPLEX,
    matcher_registry=MATCHER_REGISTRY,
    max_windows=MAX_WINDOWS_TO_RUN_COMPLEX,
    skip_existing=SKIP_EXISTING_OUTPUTS_COMPLEX,
    write_empty_outputs=WRITE_EMPTY_OUTPUTS_COMPLEX,
)

print("\nComplex motif run stats:")
display(complex_run_stats_df)

# ------------------------------------------------------------
# Aggregate summary
# ------------------------------------------------------------
if complex_run_stats_df.height > 0 and "status" in complex_run_stats_df.columns:
    complex_completed_stats = complex_run_stats_df.filter(pl.col("status") == "completed")

    if complex_completed_stats.height > 0:
        complex_summary = (
            complex_completed_stats
            .group_by("motif_type")
            .agg([
                pl.col("num_instances").sum().alias("total_instances"),
                pl.col("num_membership_rows").sum().alias("total_membership_rows"),
                pl.col("elapsed_seconds").sum().alias("total_match_seconds"),
                pl.col("total_pattern_seconds").sum().alias("total_pattern_seconds"),
                pl.col("hit_max_instances_per_window").sum().alias("num_windows_hit_cap"),
                pl.col("window_id").n_unique().alias("num_windows_completed"),
            ])
            .sort("motif_type")
        )

        print("\nComplex motif aggregate summary:")
        display(complex_summary)

        hit_cap_summary = complex_summary.filter(pl.col("num_windows_hit_cap") > 0)

        if hit_cap_summary.height > 0:
            print("\nWarning: Some complex motif runs hit max_instances_per_window.")
            print("These outputs are capped and should not be treated as full enumeration.")
            display(hit_cap_summary)
        else:
            print("\nNo complex motif run hit max_instances_per_window.")
    else:
        print("\nNo completed complex motif rows found.")
else:
    print("\ncomplex_run_stats_df is empty or missing status column.")

print("\nCell 16 completed.")


# ============================================================
# Cell 17: Run All Selected Motif Types Together
# ============================================================
"""
Purpose:
Run the final selected motif set:
1. fan_in_4
2. fan_out_4
3. cycle_5
4. split_merge_5
5. fanin_fanout_6

Note:
two_stage_split_8 is intentionally excluded from the main run.
"""

PATTERNS_TO_RUN_ALL = [
    fan_in_4,
    fan_out_4,
    cycle_5,
    split_merge_5,
    fanin_fanout_6,
]

# ------------------------------------------------------------
# Validate matcher registry
# ------------------------------------------------------------
required_matcher_types = sorted(set(p.matcher_type for p in PATTERNS_TO_RUN_ALL))
missing_matchers = [
    matcher_type
    for matcher_type in required_matcher_types
    if matcher_type not in MATCHER_REGISTRY
]

if missing_matchers:
    raise ValueError(
        f"Missing matchers in MATCHER_REGISTRY: {missing_matchers}. "
        f"Available keys: {list(MATCHER_REGISTRY.keys())}"
    )

print("Cell 17 all motif setup ready.")
print("Patterns to run:", [p.name for p in PATTERNS_TO_RUN_ALL])
print("Required matcher types:", required_matcher_types)
print("Matcher registry:", list(MATCHER_REGISTRY.keys()))

# ------------------------------------------------------------
# Run mode
# ------------------------------------------------------------
# Recommended workflow:
# 1. First run with MAX_WINDOWS_TO_RUN_ALL = 1.
# 2. If no motif hits cap and runtime is acceptable, set to None.
# For full dataset: MAX_WINDOWS_TO_RUN_ALL = None

MAX_WINDOWS_TO_RUN_ALL = None
SKIP_EXISTING_OUTPUTS_ALL = False
WRITE_EMPTY_OUTPUTS_ALL = True

# ------------------------------------------------------------
# Run all selected motifs
# ------------------------------------------------------------
all_run_stats_df = run_matchers_over_windows(
    df_edges=df_edges,
    windows=windows,
    patterns_to_run=PATTERNS_TO_RUN_ALL,
    matcher_registry=MATCHER_REGISTRY,
    max_windows=MAX_WINDOWS_TO_RUN_ALL,
    skip_existing=SKIP_EXISTING_OUTPUTS_ALL,
    write_empty_outputs=WRITE_EMPTY_OUTPUTS_ALL,
)

print("\nAll motif run stats:")
display(all_run_stats_df)

# ------------------------------------------------------------
# Aggregate summary
# ------------------------------------------------------------
if all_run_stats_df.height > 0 and "status" in all_run_stats_df.columns:
    all_completed_stats = all_run_stats_df.filter(pl.col("status") == "completed")

    if all_completed_stats.height > 0:
        all_summary = (
            all_completed_stats
            .group_by("motif_type")
            .agg([
                pl.col("num_instances").sum().alias("total_instances"),
                pl.col("num_membership_rows").sum().alias("total_membership_rows"),
                pl.col("elapsed_seconds").sum().alias("total_match_seconds"),
                pl.col("total_pattern_seconds").sum().alias("total_pattern_seconds"),
                pl.col("hit_max_instances_per_window").sum().alias("num_windows_hit_cap"),
                pl.col("window_id").n_unique().alias("num_windows_completed"),
            ])
            .sort("motif_type")
        )

        print("\nAll motif aggregate summary:")
        display(all_summary)

        hit_cap_summary = all_summary.filter(pl.col("num_windows_hit_cap") > 0)

        if hit_cap_summary.height > 0:
            print("\nWarning: Some motif runs hit max_instances_per_window.")
            print("These outputs are capped and should not be treated as full enumeration.")
            display(hit_cap_summary)
        else:
            print("\nNo motif run hit max_instances_per_window.")
    else:
        print("\nNo completed motif rows found.")
else:
    print("\nall_run_stats_df is empty or missing status column.")

print("\nCell 17 completed.")