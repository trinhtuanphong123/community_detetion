# ============================================================
# Cell 9: FanMatcher (generic, direction parameterized, top-K ranking)
# ============================================================

from itertools import combinations
import time
import gc
from typing import List, Dict, Tuple, Any, Optional, Iterator
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


def iter_sliding_temporal_groups(
    edges:        List[EdgeRecord],
    max_duration: int,
    n:            int,
) -> Iterator[List[EdgeRecord]]:
    edges_sorted = sorted(edges, key=lambda e: (e.step, e.edge_id))
    total  = len(edges_sorted)

    for i in range(total):
        group = []
        start_step = edges_sorted[i].step
        for j in range(i, total):
            if edges_sorted[j].step - start_step <= max_duration:
                group.append(edges_sorted[j])
            else:
                break
        if len(group) >= n:
            yield group


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
        df_primary: pl.DataFrame,
        df_extended: pl.DataFrame,
        index: TemporalIndex,
        window: WindowSpec,
        pattern: MotifPattern,
        output_buffer: MatcherOutputBuffer,
    ) -> Dict[str, Any]:

        if pattern.matcher_type != f"fan_{self.direction}":
            raise ValueError(f"FanMatcher with direction={self.direction} requires fan_{self.direction}, got {pattern.matcher_type}")

        n_branches = len(pattern.edges)
        if n_branches < 3:
            raise ValueError(f"fan pattern needs >= 3 edges, got {n_branches}")

        start_time = time.time()
        num_instances_emitted = 0
        num_membership_rows_emitted = 0


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
        n_centers_degree_sum = 0
        n_centers_degree_max = 0

        edge_dict = index.in_edges if self.center_role == "dst" else index.out_edges

        for center_node, incident_edges in edge_dict.items():

            if len(incident_edges) < n_branches:
                n_centers_skipped_deg += 1
                continue

            n_centers_scanned += 1
            deg = len(incident_edges)
            n_centers_degree_sum += deg
            if deg > n_centers_degree_max:
                n_centers_degree_max = deg

            # Amount floor filter
            candidates = list(incident_edges)
            if self.amount_min is not None:
                candidates = [e for e in candidates if e.amount >= self.amount_min]

            if len(candidates) < n_branches:
                continue

            # Sliding temporal groups
            center_candidates_by_ck = {}
            center_capped = False
            center_combos_count = 0

            for group in iter_sliding_temporal_groups(candidates, pattern.max_duration, n_branches):
                n_groups_checked += 1
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

                    # 7. Deduplicate via fast key tuple
                    fast_key = (pattern.name, tuple(sorted(e.edge_id for e in edges)))
                    if fast_key in seen_canonical_keys:
                        continue

                    # Score the combination
                    score = score_instance(edges, index, pattern)

                    if fast_key not in center_candidates_by_ck or score > center_candidates_by_ck[fast_key][0]:
                        center_candidates_by_ck[fast_key] = (score, edges, anchor, fast_key)

                        if len(center_candidates_by_ck) > self.max_instances_per_center * 5:
                            kept = sorted(
                                center_candidates_by_ck.values(),
                                key=lambda x: x[0],
                                reverse=True,
                            )[:self.max_instances_per_center]
                            center_candidates_by_ck = {
                                item[3]: item
                                for item in kept
                            }
                            gc.collect()

            center_candidates_to_rank = list(center_candidates_by_ck.values())
            # Rank and keep only top K per center
            center_candidates_to_rank.sort(key=lambda x: x[0], reverse=True)
            top_k_candidates = center_candidates_to_rank[:self.max_instances_per_center]

            n_rej_center_cap += len(center_candidates_to_rank) - len(top_k_candidates)

            # Emit the ranked top K
            for rank_idx, (score, edges, anchor, fast_key) in enumerate(top_k_candidates):
                if num_instances_emitted >= self.max_instances_per_window:
                    break

                seen_canonical_keys.add(fast_key)

                try:
                    motif_row = make_motif_instance_row(
                        window_id=window.window_id,
                        pattern=pattern,
                        edges=edges,
                        anchor_edge_id=anchor.edge_id,
                        index=index,
                        candidate_rank=rank_idx,
                    )
                    member_rows = make_edge_motif_membership_rows(
                        motif_row=motif_row,
                        pattern=pattern,
                        edges=edges,
                    )
                    output_buffer.add(motif_row, member_rows)
                    num_instances_emitted += 1
                    num_membership_rows_emitted += len(member_rows)
                except Exception:
                    continue

            n_combos_checked += center_combos_count
            if num_instances_emitted >= self.max_instances_per_window:
                break

        elapsed       = time.time() - start_time

        max_deg = int(n_centers_degree_max)
        mean_deg = float(n_centers_degree_sum / n_centers_scanned) if n_centers_scanned > 0 else 0.0

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
            "num_instances":                int(num_instances_emitted),
            "num_membership_rows":          int(num_membership_rows_emitted),
            "n_rej_duration":               int(n_rej_duration),
            "n_rej_delta_hop":              int(n_rej_delta_hop),
            "n_rej_structure":              int(n_rej_structure),
            "n_rej_coherence":              int(n_rej_coherence),
            "n_rej_anchor":                 int(n_rej_anchor),
            "n_rej_lookback":               int(n_rej_lookback),
            "n_rej_center_cap":             int(n_rej_center_cap),
            "hit_max_instances_per_window": int(num_instances_emitted >= self.max_instances_per_window),
            "elapsed_seconds":              float(elapsed),
        }

        return stats


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
        lookback_delta=MAX_MOTIF_DURATION,
    )
    for p in fan_out_patterns
}
