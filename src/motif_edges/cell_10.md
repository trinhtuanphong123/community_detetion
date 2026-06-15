

# ============================================================
# Cell 10: Flow motif matchers
# ============================================================

from collections import defaultdict
from itertools import product

def _build_split_merge_roles(
    split_edges: List[EdgeRecord],
    merge_edges_tuple: tuple,
    intermediates: List[int],
    n_branches: int,
) -> Tuple[List[EdgeRecord], List[EdgeRecord], Dict[str, int]]:
    """
    Build role_map while preserving the correct mid_i -> merge_i mapping.
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


def _compute_center_inout_flow_features(
    in_edges: List[EdgeRecord],
    out_edges: List[EdgeRecord],
) -> Dict[str, float]:
    """
    Compute flow features for one center-in-out motif instance.
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


class SplitMergeMatcher:
    """
    Matcher for split_merge motifs.
    Implements a score-then-rank emission strategy per source node to prevent supernode domination.
    """

    def __init__(
        self,
        max_out_candidates: int = 10,
        max_merge_candidates_per_intermediate: int = 2,
        max_instances_per_window: int = 20_000,
        amount_min: Optional[float] = None,
        use_amount_ratio: bool = True,
        candidate_policy: str = "hybrid",
        split_phase_delta: Optional[int] = DELTA_HOP,
        merge_phase_delta: Optional[int] = DELTA_HOP,
        split_to_merge_delta: Optional[int] = DELTA_HOP,
        max_instances_per_source: Optional[int] = 300,
        max_combinations_per_source: int = 10000,
        structuring_threshold: Optional[float] = 9500.0,
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
        self.max_combinations_per_source = max_combinations_per_source
        self.structuring_threshold = structuring_threshold

    def match(
        self,
        df_primary: pl.DataFrame,
        df_extended: pl.DataFrame,
        index: TemporalIndex,
        window: WindowSpec,
        pattern: MotifPattern,
        output_buffer: MatcherOutputBuffer,
    ) -> Dict[str, Any]:

        if pattern.matcher_type != "split_merge":
            raise ValueError(
                f"SplitMergeMatcher expects matcher_type='split_merge'. "
                f"Got {pattern.matcher_type}"
            )

        start_time = time.time()
        num_instances_emitted = 0
        num_membership_rows_emitted = 0

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

        n_branches = len(pattern.edges) // 2

        for source, out_edges in index.out_edges.items():
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

            source_candidates_to_rank = []
            seen_source_keys = set()
            source_combos_count = 0

            for split_combo in combinations(split_candidates, n_branches):
                if source_combos_count >= self.max_combinations_per_source:
                    break

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

                # Build merge candidates for each intermediate
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
                        source_combos_count += 1
                        if source_combos_count > self.max_combinations_per_source:
                            break

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

                        edges_for_validation = split_edges + merge_edges_sorted

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
                            
                            if self.structuring_threshold is not None:
                                split_max = max(e.amount for e in split_edges)
                                if split_max > self.structuring_threshold:
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

                        # Score the combination
                        score = score_instance(edges_for_validation, index, pattern)

                        ck = make_canonical_key(
                            pattern.name,
                            [e.edge_id for e in edges_for_validation],
                            ordered=True,
                        )
                        if ck in seen_source_keys:
                            continue
                        seen_source_keys.add(ck)
                        source_candidates_to_rank.append((score, edges_for_validation, anchor))

                    if source_combos_count > self.max_combinations_per_source:
                        break
                if source_combos_count > self.max_combinations_per_source:
                    break

            # Rank and keep only top K per source
            source_candidates_to_rank.sort(key=lambda x: x[0], reverse=True)
            top_k_candidates = source_candidates_to_rank[:self.max_instances_per_source]

            num_rejected_source_cap += len(source_candidates_to_rank) - len(top_k_candidates)

            # Emit the ranked top K
            for rank_idx, (score, edges_val, anchor) in enumerate(top_k_candidates):
                if num_instances_emitted >= self.max_instances_per_window:
                    break

                try:
                    motif_row = make_motif_instance_row(
                        window_id=window.window_id,
                        pattern=pattern,
                        edges=edges_val,
                        anchor_edge_id=anchor.edge_id,
                        index=index,
                        candidate_rank=rank_idx,
                    )
                    member_rows = make_edge_motif_membership_rows(
                        motif_row=motif_row,
                        pattern=pattern,
                        edges=edges_val,
                    )
                    output_buffer.add(motif_row, member_rows)
                    num_instances_emitted += 1
                    num_membership_rows_emitted += len(member_rows)
                except Exception:
                    continue

            if num_instances_emitted >= self.max_instances_per_window:
                break

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

            "num_instances": int(num_instances_emitted),
            "num_membership_rows": int(num_membership_rows_emitted),

            "num_rejected_duration": int(num_rejected_duration),
            "num_rejected_nodes": int(num_rejected_nodes),
            "num_rejected_anchor": int(num_rejected_anchor),
            "num_rejected_amount": int(num_rejected_amount),
            "num_rejected_split_phase_delta": int(num_rejected_split_phase_delta),
            "num_rejected_merge_phase_delta": int(num_rejected_merge_phase_delta),
            "num_rejected_phase_gap": int(num_rejected_phase_gap),
            "num_rejected_source_cap": int(num_rejected_source_cap),

            "hit_max_instances_per_window": int(
                num_instances_emitted >= self.max_instances_per_window
            ),
            "elapsed_seconds": float(elapsed),
        }

        return stats


class CenterInOutMatcher:
    """
    Matcher for center-in-out motifs.
    Implements a score-then-rank emission strategy per center node to prevent supernode domination.
    """

    def __init__(
        self,
        max_in_candidates: int = 12,
        max_out_candidates: int = 6,
        max_instances_per_window: int = 20_000,
        amount_min: Optional[float] = None,
        use_amount_ratio: bool = True,
        candidate_policy: str = "hybrid",
        incoming_phase_delta: Optional[int] = DELTA_HOP,
        outgoing_phase_delta: Optional[int] = DELTA_HOP,
        center_handoff_delta: Optional[int] = DELTA_HOP,
        max_instances_per_center: Optional[int] = 300,
        max_combinations_per_center: int = 10000,
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
        self.max_combinations_per_center = max_combinations_per_center

    def match(
        self,
        df_primary: pl.DataFrame,
        df_extended: pl.DataFrame,
        index: TemporalIndex,
        window: WindowSpec,
        pattern: MotifPattern,
        output_buffer: MatcherOutputBuffer,
    ) -> Dict[str, Any]:

        if pattern.matcher_type != "center_in_out":
            raise ValueError(
                f"CenterInOutMatcher expects matcher_type='center_in_out'. "
                f"Got {pattern.matcher_type}"
            )

        start_time = time.time()
        num_instances_emitted = 0
        num_membership_rows_emitted = 0

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

        n_in = sum(1 for e in pattern.edges if e.dst == "center")
        n_out = sum(1 for e in pattern.edges if e.src == "center")

        candidate_centers = set(index.in_edges.keys()) & set(index.out_edges.keys())

        for center in candidate_centers:
            # Fast degree check before candidate selection.
            if len(index.in_edges.get(center, [])) < n_in:
                continue

            if len(index.out_edges.get(center, [])) < n_out:
                continue

            num_centers_scanned += 1

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

            center_candidates_to_rank = []
            seen_center_keys = set()
            center_combos_count = 0

            for in_combo in combinations(incoming, n_in):
                in_edges = sorted(
                    list(in_combo),
                    key=lambda e: (e.step, e.edge_id),
                )

                num_in_combinations_checked += 1

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

                # Group-aware outgoing query:
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
                    center_combos_count += 1
                    if center_combos_count > self.max_combinations_per_center:
                        break

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

                    # Score the combination
                    score = score_instance(edges, index, pattern)

                    ck = make_canonical_key(
                        pattern.name,
                        [e.edge_id for e in edges],
                        ordered=True,
                    )
                    if ck in seen_center_keys:
                        continue
                    seen_center_keys.add(ck)
                    center_candidates_to_rank.append((score, edges, anchor))

                if center_combos_count > self.max_combinations_per_center:
                    break

            # Rank and keep only top K per center
            center_candidates_to_rank.sort(key=lambda x: x[0], reverse=True)
            top_k_candidates = center_candidates_to_rank[:self.max_instances_per_center]

            num_rejected_center_cap += len(center_candidates_to_rank) - len(top_k_candidates)

            # Emit the ranked top K
            for rank_idx, (score, edges_val, anchor) in enumerate(top_k_candidates):
                if num_instances_emitted >= self.max_instances_per_window:
                    break

                try:
                    motif_row = make_motif_instance_row(
                        window_id=window.window_id,
                        pattern=pattern,
                        edges=edges_val,
                        anchor_edge_id=anchor.edge_id,
                        index=index,
                        candidate_rank=rank_idx,
                    )
                    member_rows = make_edge_motif_membership_rows(
                        motif_row=motif_row,
                        pattern=pattern,
                        edges=edges_val,
                    )
                    output_buffer.add(motif_row, member_rows)
                    num_instances_emitted += 1
                    num_membership_rows_emitted += len(member_rows)
                except Exception:
                    continue

            if num_instances_emitted >= self.max_instances_per_window:
                break

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

            "num_instances": int(num_instances_emitted),
            "num_membership_rows": int(num_membership_rows_emitted),

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
                num_instances_emitted >= self.max_instances_per_window
            ),
            "elapsed_seconds": float(elapsed),
        }

        return stats


# Matcher registration moved to Cell 12 in code_matcher_2.md
print("Cell 10 completed.")

