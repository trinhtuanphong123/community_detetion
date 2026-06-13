# ============================================================
# Cell 9: FanMatcher (generic, direction parameterized, top-K ranking)
# ============================================================

from itertools import combinations
import time
from typing import List, Dict, Tuple, Any, Optional
import polars as pl

if "MATCHER_REGISTRY" not in globals():
    MATCHER_REGISTRY = {}

def is_edge_in_primary_window(edge: EdgeRecord, window: WindowSpec) -> bool:
    return window.primary_start <= edge.step <= window.primary_end


def get_anchor_edge(edges: List[EdgeRecord]) -> EdgeRecord:
    return min(edges, key=lambda e: (e.step, e.edge_id))


def passes_amount_min(edges: List[EdgeRecord], amount_min: Optional[float]) -> bool:
    if amount_min is None:
        return True
    return all(e.amount >= amount_min for e in edges)


def passes_consecutive_delta_hop_sorted(
    edges_sorted_by_step: List[EdgeRecord],
    delta_hop: Optional[int],
) -> bool:
    if delta_hop is None or len(edges_sorted_by_step) <= 1:
        return True
    steps = [e.step for e in edges_sorted_by_step]
    return all(steps[i+1] - steps[i] <= delta_hop for i in range(len(steps)-1))


def sliding_temporal_groups(
    edges:        List[EdgeRecord],
    max_duration: int,
    n:            int,
) -> List[List[EdgeRecord]]:
    edges_sorted = sorted(edges, key=lambda e: (e.step, e.edge_id))
    groups = []
    total  = len(edges_sorted)

    for i in range(total):
        group = [edges_sorted[i]]
        for j in range(i + 1, total):
            if edges_sorted[j].step - edges_sorted[i].step <= max_duration:
                group.append(edges_sorted[j])
            else:
                break
        if len(group) >= n:
            groups.append(group)

    return groups


def passes_amount_coherence(
    edges:     List[EdgeRecord],
    max_ratio: float = 3.0,
) -> bool:
    amounts = [e.amount for e in edges if e.amount > 0]
    if len(amounts) < 2:
        return True
    return max(amounts) / min(amounts) <= max_ratio


class FanMatcher:
    """
    Generic Fan Matcher parameterized by direction ('in' or 'out').
    Implements a score-then-rank emission strategy to prevent supernode domination.
    """

    def __init__(
        self,
        direction: str,
        cap_schedule: Dict[int, int],
        max_instances_per_window: int = MAX_INSTANCES_PER_WINDOW,
        max_instances_per_center: int = 100,
        max_combinations_per_center: int = 10000,
        amount_min: Optional[float] = AMOUNT_MIN,
        amount_coherence_ratio: float = 3.0,
        delta_hop: Optional[int] = DELTA_HOP,
        lookback_delta: Optional[int] = None,
    ):
        self.direction = direction  # "in" or "out"
        self.cap_schedule = cap_schedule
        self.max_instances_per_window = max_instances_per_window
        self.max_instances_per_center = max_instances_per_center
        self.max_combinations_per_center = max_combinations_per_center
        self.amount_min = amount_min
        self.amount_coherence_ratio = amount_coherence_ratio
        self.delta_hop = delta_hop
        self.lookback_delta = lookback_delta

        if direction == "in":
            self.center_role = "dst"
            self.branch_role = "src"
        elif direction == "out":
            self.center_role = "src"
            self.branch_role = "dst"
        else:
            raise ValueError(f"Invalid direction: {direction}")

    def _get_cap(self, n_branches: int) -> int:
        return self.cap_schedule.get(n_branches, 10)

    def _source_has_prior_incoming(
        self,
        index: TemporalIndex,
        src: int,
        t_anchor: int,
    ) -> bool:
        if self.lookback_delta is None:
            return True
        t_min = t_anchor - self.lookback_delta
        t_max = t_anchor
        return index.has_any_incoming(dst=src, t_min=t_min, t_max=t_max)

    def _passes_fan_structure(self, edges: List[EdgeRecord]) -> bool:
        if not edges:
            return False
        if len({e.edge_id for e in edges}) != len(edges):
            return False
        
        # All edges must have the same center node
        centers = {getattr(e, self.center_role) for e in edges}
        if len(centers) != 1:
            return False
        center_node = next(iter(centers))
        
        # All branch nodes must be distinct
        branches = [getattr(e, self.branch_role) for e in edges]
        if len(branches) != len(set(branches)):
            return False
            
        # No branch node can be the center node
        return not any(b == center_node for b in branches)

    def match(
        self,
        df_primary:   pl.DataFrame,
        df_extended:  pl.DataFrame,
        index:        TemporalIndex,
        window:       WindowSpec,
        pattern:      MotifPattern,
        write_output: bool = False,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, Any]]:

        if pattern.matcher_type != f"fan_{self.direction}":
            raise ValueError(f"FanMatcher with direction={self.direction} requires fan_{self.direction}, got {pattern.matcher_type}")

        n_branches = len(pattern.edges)
        if n_branches < 3:
            raise ValueError(f"fan pattern needs >= 3 edges, got {n_branches}")

        start_time = time.time()
        motif_rows:      List[Dict] = []
        emitted_edges:   List[List[EdgeRecord]] = []

        n_centers_scanned     = 0
        n_centers_skipped_deg = 0
        n_centers_capped      = 0
        n_groups_checked  = 0
        n_combos_checked  = 0
        n_rej_duration    = 0
        n_rej_delta_hop   = 0
        n_rej_structure   = 0
        n_rej_coherence   = 0
        n_rej_anchor      = 0
        n_rej_lookback    = 0
        n_rej_center_cap  = 0

        pattern_edges_sorted = sorted(pattern.edges, key=lambda x: x.order)
        cap = self._get_cap(n_branches)
        
        seen_canonical_keys = set()
        scanned_center_degrees = []

        edge_dict = index.in_edges if self.center_role == "dst" else index.out_edges

        for center_node, incident_edges in edge_dict.items():

            if len(incident_edges) < n_branches:
                n_centers_skipped_deg += 1
                continue

            n_centers_scanned += 1
            scanned_center_degrees.append(len(incident_edges))

            # Amount floor filter
            candidates = list(incident_edges)
            if self.amount_min is not None:
                candidates = [e for e in candidates if e.amount >= self.amount_min]

            if len(candidates) < n_branches:
                continue

            # Sliding temporal groups
            groups = sliding_temporal_groups(candidates, pattern.max_duration, n_branches)
            n_groups_checked += len(groups)

            center_candidates_by_ck = {}
            center_capped = False
            center_combos_count = 0

            for group in groups:
                if center_combos_count >= self.max_combinations_per_center:
                    break

                if len(group) > cap:
                    group = cap_edges(group, max_candidates=cap, policy="hybrid")
                    if not center_capped:
                        n_centers_capped += 1
                        center_capped = True

                for edge_combo in combinations(group, n_branches):
                    center_combos_count += 1
                    if center_combos_count > self.max_combinations_per_center:
                        break

                    edges = sorted(list(edge_combo), key=lambda e: (e.step, e.edge_id))

                    # 1. Duration
                    if not is_within_total_duration(edges, pattern.max_duration):
                        n_rej_duration += 1
                        continue

                    # 2. Delta hop
                    if not passes_consecutive_delta_hop_sorted(edges, self.delta_hop):
                        n_rej_delta_hop += 1
                        continue

                    # 3. Structural validity
                    if not self._passes_fan_structure(edges):
                        n_rej_structure += 1
                        continue

                    # 4. Amount coherence
                    if not passes_amount_coherence(edges, self.amount_coherence_ratio):
                        n_rej_coherence += 1
                        continue

                    # 5. Anchor in primary window
                    anchor = get_anchor_edge(edges)
                    if not is_edge_in_primary_window(anchor, window):
                        n_rej_anchor += 1
                        continue

                    # 6. Lookback check (only for fan_out when lookback_delta is provided)
                    if self.direction == "out" and self.lookback_delta is not None:
                        if not self._source_has_prior_incoming(index, center_node, anchor.step):
                            n_rej_lookback += 1
                            continue

                    # 7. Deduplicate via canonical key (ordered=False consistently for fan-in/fan-out)
                    ck = make_canonical_key(
                        pattern.name,
                        [e.edge_id for e in edges],
                        ordered=False,
                    )
                    if ck in seen_canonical_keys:
                        continue

                    # Prep node map
                    node_map = {self.center_role: int(center_node)}
                    for i in range(n_branches):
                        node_map[f"{self.branch_role}_{i+1}"] = int(getattr(edges[i], self.branch_role))

                    role_map = {
                        p_e.role: int(r_e.edge_id)
                        for p_e, r_e in zip(pattern_edges_sorted, edges)
                    }

                    # Score the combination
                    score = score_instance(edges, index, pattern)

                    if ck not in center_candidates_by_ck or score > center_candidates_by_ck[ck][0]:
                        center_candidates_by_ck[ck] = (score, edges, node_map, role_map, anchor, ck)

            center_candidates_to_rank = list(center_candidates_by_ck.values())
            # Rank and keep only top K per center
            center_candidates_to_rank.sort(key=lambda x: x[0], reverse=True)
            top_k_candidates = center_candidates_to_rank[:self.max_instances_per_center]
            
            n_rej_center_cap += len(center_candidates_to_rank) - len(top_k_candidates)

            # Emit the ranked top K
            for rank_idx, (score, edges, node_map, role_map, anchor, ck) in enumerate(top_k_candidates):
                if len(motif_rows) >= self.max_instances_per_window:
                    break

                seen_canonical_keys.add(ck)

                try:
                    motif_row = make_motif_instance_row(
                        window_id=window.window_id,
                        pattern=pattern,
                        edges=edges,
                        node_map=node_map,
                        role_map=role_map,
                        anchor_edge_id=anchor.edge_id,
                        validate=False,
                        index=index,
                        candidate_rank=rank_idx,
                    )
                    motif_rows.append(motif_row)
                    emitted_edges.append(edges)
                except Exception:
                    continue

            n_combos_checked += center_combos_count
            if len(motif_rows) >= self.max_instances_per_window:
                break

        motif_df      = motif_instance_rows_to_polars(motif_rows)
        membership_df = make_membership_df_from_motif_rows(motif_rows, pattern, emitted_edges)
        elapsed       = time.time() - start_time

        max_deg = int(max(scanned_center_degrees)) if scanned_center_degrees else 0
        mean_deg = float(sum(scanned_center_degrees) / len(scanned_center_degrees)) if scanned_center_degrees else 0.0

        stats = {
            "window_id":                    int(window.window_id),
            "motif_type":                   pattern.name,
            "matcher_type":                 pattern.matcher_type,
            "n_branches":                   int(n_branches),
            "primary_num_edges":            int(df_primary.height),
            "extended_num_edges":           int(df_extended.height),
            f"n_centers_{self.direction}_scanned":  int(n_centers_scanned),
            f"n_centers_{self.direction}_skipped":  int(n_centers_skipped_deg),
            f"n_centers_{self.direction}_capped":   int(n_centers_capped),
            f"max_center_{self.direction}_degree":  max_deg,
            f"mean_center_{self.direction}_degree": mean_deg,
            "n_groups_checked":             int(n_groups_checked),
            "n_combos_checked":             int(n_combos_checked),
            "num_instances":                int(motif_df.height),
            "num_membership_rows":          int(membership_df.height),
            "n_rej_duration":               int(n_rej_duration),
            "n_rej_delta_hop":              int(n_rej_delta_hop),
            "n_rej_structure":              int(n_rej_structure),
            "n_rej_coherence":              int(n_rej_coherence),
            "n_rej_anchor":                 int(n_rej_anchor),
            "n_rej_lookback":               int(n_rej_lookback),
            "n_rej_center_cap":             int(n_rej_center_cap),
            "hit_max_instances_per_window": int(len(motif_rows) >= self.max_instances_per_window),
            "elapsed_seconds":              float(elapsed),
        }

        if write_output:
            mp, mep = write_motif_outputs(
                motif_rows, membership_df, window.window_id, pattern.name
            )
            stats["motif_path"]      = mp
            stats["membership_path"] = mep

        return motif_df, membership_df, stats


# Matcher registration moved to Cell 12 in code_matcher_2.md

# Compatibility aliases
fanin_matchers = {
    p.name: FanMatcher(
        direction="in",
        cap_schedule=FAN_IN_CAP_SCHEDULE,
        max_instances_per_window=MAX_INSTANCES_PER_WINDOW,
        max_instances_per_center=MAX_FAN_INSTANCES_PER_CENTER,
        amount_min=AMOUNT_MIN,
        amount_coherence_ratio=3.0,
        delta_hop=DELTA_HOP,
    )
    for p in fan_in_patterns
}
fanin_matcher = fanin_matchers[fan_in_4.name]

fanout_matchers = {
    p.name: FanMatcher(
        direction="out",
        cap_schedule=FAN_OUT_CAP_SCHEDULE,
        max_instances_per_window=MAX_INSTANCES_PER_WINDOW,
        max_instances_per_center=MAX_FAN_INSTANCES_PER_CENTER,
        amount_min=AMOUNT_MIN,
        amount_coherence_ratio=3.0,
        delta_hop=DELTA_HOP,
        lookback_delta=None,
    )
    for p in fan_out_patterns
}


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
        emitted_edges = []

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
                        source_candidates_to_rank.append((score, edges_for_validation, node_map, role_map, anchor))

                    if source_combos_count > self.max_combinations_per_source:
                        break
                if source_combos_count > self.max_combinations_per_source:
                    break

            # Rank and keep only top K per source
            source_candidates_to_rank.sort(key=lambda x: x[0], reverse=True)
            top_k_candidates = source_candidates_to_rank[:self.max_instances_per_source]
            
            num_rejected_source_cap += len(source_candidates_to_rank) - len(top_k_candidates)

            # Emit the ranked top K
            for rank_idx, (score, edges_val, node_map, role_map, anchor) in enumerate(top_k_candidates):
                if len(motif_rows) >= self.max_instances_per_window:
                    break

                try:
                    motif_row = make_motif_instance_row(
                        window_id=window.window_id,
                        pattern=pattern,
                        edges=edges_val,
                        node_map=node_map,
                        role_map=role_map,
                        anchor_edge_id=anchor.edge_id,
                        validate=False,
                        index=index,
                        candidate_rank=rank_idx,
                    )
                    motif_rows.append(motif_row)
                    emitted_edges.append(edges_val)
                except Exception:
                    continue

            if len(motif_rows) >= self.max_instances_per_window:
                break

        motif_df = motif_instance_rows_to_polars(motif_rows)
        membership_df = make_membership_df_from_motif_rows(motif_rows, pattern, emitted_edges)

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
                membership_rows=membership_df,
                window_id=window.window_id,
                motif_type=pattern.name,
            )

            stats["motif_path"] = motif_path
            stats["membership_path"] = membership_path

        return motif_df, membership_df, stats


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
        write_output: bool = False,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, Any]]:

        if pattern.matcher_type != "center_in_out":
            raise ValueError(
                f"CenterInOutMatcher expects matcher_type='center_in_out'. "
                f"Got {pattern.matcher_type}"
            )

        start_time = time.time()

        motif_rows = []
        emitted_edges = []

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
                    center_candidates_to_rank.append((score, edges, node_map, role_map, anchor, flow_features))

                if center_combos_count > self.max_combinations_per_center:
                    break

            # Rank and keep only top K per center
            center_candidates_to_rank.sort(key=lambda x: x[0], reverse=True)
            top_k_candidates = center_candidates_to_rank[:self.max_instances_per_center]
            
            num_rejected_center_cap += len(center_candidates_to_rank) - len(top_k_candidates)

            # Emit the ranked top K
            for rank_idx, (score, edges_val, node_map, role_map, anchor, flow_features) in enumerate(top_k_candidates):
                if len(motif_rows) >= self.max_instances_per_window:
                    break

                try:
                    motif_row = make_motif_instance_row(
                        window_id=window.window_id,
                        pattern=pattern,
                        edges=edges_val,
                        node_map=node_map,
                        role_map=role_map,
                        anchor_edge_id=anchor.edge_id,
                        validate=False,
                        index=index,
                        candidate_rank=rank_idx,
                    )
                    # Add flow features into motif row.
                    motif_row.update(flow_features)
                    motif_rows.append(motif_row)
                    emitted_edges.append(edges_val)
                except Exception:
                    continue

            if len(motif_rows) >= self.max_instances_per_window:
                break

        motif_df = motif_instance_rows_to_polars(motif_rows)
        membership_df = make_membership_df_from_motif_rows(motif_rows, pattern, emitted_edges)

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
                membership_rows=membership_df,
                window_id=window.window_id,
                motif_type=pattern.name,
            )

            stats["motif_path"] = motif_path
            stats["membership_path"] = membership_path

        return motif_df, membership_df, stats


# Matcher registration moved to Cell 12 in code_matcher_2.md
print("Cell 10 completed.")


# ============================================================
# Cell 11: Cycle matcher, diagnostic only
# ============================================================

BIDIRECTIONAL_CYCLE_THRESHOLD = 999

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
        emitted_edges: List[List[EdgeRecord]],
        window: WindowSpec,
        pattern: MotifPattern,
        edges: List[EdgeRecord],
        anchor_edge: EdgeRecord,
        k: int,
        validate: bool = True,
    ) -> bool:
        try:
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

            motif_row["amount_consistency"] = compute_amount_consistency(edges)

            motif_rows.append(motif_row)
            emitted_edges.append(edges)

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

        use_bidirectional = (k >= BIDIRECTIONAL_CYCLE_THRESHOLD)

        start_time = time.time()

        motif_rows = []
        emitted_edges = []

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

                path_nodes = [int(edges[0].src)]
                for e in edges[:-1]:
                    path_nodes.append(int(e.dst))

                if not has_distinct_nodes(path_nodes):
                    num_rejected_nodes += 1
                    continue

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
                    emitted_edges=emitted_edges,
                    window=window,
                    pattern=pattern,
                    edges=edges,
                    anchor_edge=anchor_edge,
                    k=k,
                    validate=True,
                )

        # ============================================================
        # Branch 2: DFS search for short cycles
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
                                motif_rows=motif_rows,
                                emitted_edges=emitted_edges,
                                window=window,
                                pattern=pattern,
                                edges=edges,
                                anchor_edge=anchor_edge,
                                k=k,
                                validate=True,
                            )

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
                        if len(motif_rows) >= self.max_instances_per_window:
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

                if len(motif_rows) >= self.max_instances_per_window:
                    break

        motif_df = motif_instance_rows_to_polars(motif_rows)
        membership_df = make_membership_df_from_motif_rows(motif_rows, pattern, emitted_edges)

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
                membership_rows=membership_df,
                window_id=window.window_id,
                motif_type=pattern.name,
            )

            stats["motif_path"] = motif_path
            stats["membership_path"] = membership_path

        return motif_df, membership_df, stats


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


# ============================================================
# Cell 12: Matcher registry
# ============================================================

# Control whether cycle motifs are enabled in diagnostic run
ENABLE_CYCLES = False

# Consolidated central matcher registration
MATCHER_REGISTRY = {
    "fan_in": FanMatcher(
        direction="in",
        cap_schedule=FAN_IN_CAP_SCHEDULE,
        max_instances_per_window=MAX_FAN_INSTANCES_PER_WINDOW,
        max_instances_per_center=MAX_FAN_INSTANCES_PER_CENTER,
        amount_min=AMOUNT_MIN,
        amount_coherence_ratio=3.0,
        delta_hop=DELTA_HOP,
    ),
    "fan_out": FanMatcher(
        direction="out",
        cap_schedule=FAN_OUT_CAP_SCHEDULE,
        max_instances_per_window=MAX_FAN_INSTANCES_PER_WINDOW,
        max_instances_per_center=MAX_FAN_INSTANCES_PER_CENTER,
        amount_min=AMOUNT_MIN,
        amount_coherence_ratio=3.0,
        delta_hop=DELTA_HOP,
        lookback_delta=None,
    ),
    "center_in_out": CenterInOutMatcher(
        max_in_candidates=12,
        max_out_candidates=6,
        max_instances_per_window=20_000,
        amount_min=None,
        use_amount_ratio=True,
        candidate_policy="hybrid",
        incoming_phase_delta=DELTA_HOP,
        outgoing_phase_delta=DELTA_HOP,
        center_handoff_delta=DELTA_HOP,
        max_instances_per_center=300,
    ),
    "split_merge": SplitMergeMatcher(
        max_out_candidates=10,
        max_merge_candidates_per_intermediate=2,
        max_instances_per_window=20_000,
        amount_min=None,
        use_amount_ratio=True,
        candidate_policy="hybrid",
        split_phase_delta=DELTA_HOP,
        merge_phase_delta=DELTA_HOP,
        split_to_merge_delta=DELTA_HOP,
        max_instances_per_source=300,
    ),
}

if ENABLE_CYCLES:
    MATCHER_REGISTRY["path_cycle"] = CycleKMatcher(
        max_branching=4,
        branching_schedule=CYCLE_MAX_BRANCHING,
        max_instances_per_window=20_000,
        amount_min=None,
        use_amount_ratio=False,
        candidate_policy="hybrid",
        delta_hop=DELTA_HOP,
        max_anchors_per_window=None,
    )

print("Matcher registry initialized.")
print("MATCHER_REGISTRY keys:", list(MATCHER_REGISTRY.keys()))
print("Cell 12 completed.")





