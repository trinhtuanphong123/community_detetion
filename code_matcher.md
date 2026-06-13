# ============================================================
# Cell 9: FanInMatcher (sliding group + coherence)
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


def passes_distinct_nodes_for_fanin(edges: List[EdgeRecord]) -> bool:
    """
    All edges must share the same dst.
    All src nodes must be distinct.
    No src may equal dst.
    All edge_ids must be distinct.
    Works for any n >= 3.
    """
    if not edges:
        return False
    if len({e.edge_id for e in edges}) != len(edges):
        return False
    dst_values = {e.dst for e in edges}
    if len(dst_values) != 1:
        return False
    src_values = [e.src for e in edges]
    if len(src_values) != len(set(src_values)):
        return False
    center_dst = next(iter(dst_values))
    return not any(s == center_dst for s in src_values)


def passes_fanin_duration(edges: List[EdgeRecord], max_duration: int) -> bool:
    steps = [e.step for e in edges]
    return (max(steps) - min(steps)) <= max_duration


def passes_consecutive_delta_hop_sorted(
    edges_sorted_by_step: List[EdgeRecord],
    delta_hop: Optional[int],
) -> bool:
    """
    Expects edges already sorted by (step, edge_id).
    Checks step[i+1] - step[i] <= delta_hop for all consecutive pairs.
    """
    if delta_hop is None or len(edges_sorted_by_step) <= 1:
        return True
    steps = [e.step for e in edges_sorted_by_step]
    return all(steps[i+1] - steps[i] <= delta_hop for i in range(len(steps)-1))


def canonicalize_fanin_edges(edges: List[EdgeRecord]) -> List[EdgeRecord]:
    return sorted(edges, key=lambda e: (e.step, e.edge_id))



def sliding_temporal_groups(
    edges:        List[EdgeRecord],
    max_duration: int,
    n:            int,
) -> List[List[EdgeRecord]]:
    """
    Chia danh sách edges (đã sort theo step) thành các cửa sổ thời gian
    compact: với mỗi edge i làm anchor, lấy tất cả edges j >= i mà
    edges[j].step - edges[i].step <= max_duration.

    Chỉ trả về group có >= n edges. Các group trùng nhau được chấp nhận
    vì combinations() bên trong sẽ dedup qua canonical_key.

    Phức tạp: O(n * window_size). Tốt hơn earliest-only vì không bỏ sót
    burst xảy ra ở cuối window.
    """
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
    """
    Kiểm tra amount coherence: max(amount) / min(amount) <= max_ratio.

    Nếu một edge trong nhóm có amount lớn hơn 10x edge khác,
    đây thường không phải là coordinated transfer.

    max_ratio=10.0 là ngưỡng mềm cho exploration.
    Production nên thử 5.0 hoặc 3.0 rồi so sánh SAR enrichment.
    """
    amounts = [e.amount for e in edges if e.amount > 0]
    if len(amounts) < 2:
        return True
    return max(amounts) / min(amounts) <= max_ratio


def select_edges_by_hybrid_policy(
    edges: List[EdgeRecord],
    max_candidates: int,
    early_ratio: float = 0.5,
) -> List[EdgeRecord]:
    """
    Hybrid candidate selection:
        keep part earliest edges and part largest-amount edges.
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


class FanInMatcher:
    """
    Generalized fan-in matcher. Supports n = 3..7 incoming branches.

    Thay đổi so với version cũ:
    - Candidate selection dùng sliding_temporal_groups thay vì earliest-only.
    - Thêm amount coherence check.
    - Thêm max_instances_per_dst để kiểm soát supernode domination.
    """

    def __init__(
        self,
        cap_schedule:             Dict[int, int]  = FAN_IN_CAP_SCHEDULE,
        max_instances_per_window: int             = MAX_INSTANCES_PER_WINDOW,
        max_instances_per_dst:    int             = 500,
        amount_min:               Optional[float] = AMOUNT_MIN,
        amount_coherence_ratio:   float           = 3.0,
        delta_hop:                Optional[int]   = DELTA_HOP,
    ):
        self.cap_schedule             = cap_schedule
        self.max_instances_per_window = max_instances_per_window
        self.max_instances_per_dst    = max_instances_per_dst
        self.amount_min               = amount_min
        self.amount_coherence_ratio   = amount_coherence_ratio
        self.delta_hop                = delta_hop

    def _get_cap(self, n_branches: int) -> int:
        return self.cap_schedule.get(n_branches, 10)

    def match(
        self,
        df_primary:   pl.DataFrame,
        df_extended:  pl.DataFrame,
        index:        TemporalIndex,
        window:       WindowSpec,
        pattern:      MotifPattern,
        write_output: bool = False,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, Any]]:

        if pattern.matcher_type != "fan_in":
            raise ValueError(f"FanInMatcher requires fan_in, got {pattern.matcher_type}")

        n_branches = len(pattern.edges)
        if n_branches < 3:
            raise ValueError(f"fan_in needs >= 3 edges, got {n_branches}")

        start_time = time.time()
        motif_rows:      List[Dict] = []
        membership_rows: List[Dict] = []

        n_dst_scanned     = 0
        n_dst_skipped_deg = 0
        n_dst_capped      = 0
        n_groups_checked  = 0
        n_combos_checked  = 0
        n_rej_duration    = 0
        n_rej_delta_hop   = 0
        n_rej_structure   = 0
        n_rej_coherence   = 0
        n_rej_anchor      = 0
        n_rej_dst_cap     = 0

        pattern_edges_sorted = sorted(pattern.edges, key=lambda x: x.order)
        cap = self._get_cap(n_branches)
        
        seen_canonical_keys = set()
        scanned_dst_degrees = []

        for dst_node, incoming_edges in index.in_edges.items():

            if len(incoming_edges) < n_branches:
                n_dst_skipped_deg += 1
                continue

            n_dst_scanned += 1
            scanned_dst_degrees.append(len(incoming_edges))

            # Amount floor filter trước khi group
            candidates = list(incoming_edges)
            if self.amount_min is not None:
                candidates = [e for e in candidates if e.amount >= self.amount_min]

            if len(candidates) < n_branches:
                continue

            # Sliding temporal groups — thay thế earliest-only
            groups = sliding_temporal_groups(candidates, pattern.max_duration, n_branches)
            n_groups_checked += len(groups)

            dst_instance_count = 0
            dst_cap_reached = False
            dst_capped = False

            for group in groups:
                if len(motif_rows) >= self.max_instances_per_window:
                    break
                if dst_cap_reached:
                    break

                # Item 1: sliding trước, cap sau trong từng group bằng policy hybrid
                if len(group) > cap:
                    group = cap_edges(group, max_candidates=cap, policy="hybrid")
                    if not dst_capped:
                        n_dst_capped += 1
                        dst_capped = True

                for edge_combo in combinations(group, n_branches):
                    n_combos_checked += 1
                    edges = canonicalize_fanin_edges(list(edge_combo))

                    # 1. Duration
                    if not passes_fanin_duration(edges, pattern.max_duration):
                        n_rej_duration += 1
                        continue

                    # 2. Delta hop
                    if not passes_consecutive_delta_hop_sorted(edges, self.delta_hop):
                        n_rej_delta_hop += 1
                        continue

                    # 3. Structural validity
                    if not passes_distinct_nodes_for_fanin(edges):
                        n_rej_structure += 1
                        continue

                    # 4. Amount coherence (NEW)
                    if not passes_amount_coherence(edges, self.amount_coherence_ratio):
                        n_rej_coherence += 1
                        continue

                    # 5. Anchor in primary window
                    anchor = get_anchor_edge(edges)
                    if not is_edge_in_primary_window(anchor, window):
                        n_rej_anchor += 1
                        continue

                    # Item 2: canonical key dedup (chống duplicate do sliding groups chồng lấn)
                    ck = make_canonical_key(
                        pattern.name,
                        [e.edge_id for e in edges],
                        ordered=True,
                    )
                    if ck in seen_canonical_keys:
                        continue
                    seen_canonical_keys.add(ck)

                    node_map = {f"src_{i+1}": int(edges[i].src) for i in range(n_branches)}
                    node_map["dst"] = int(edges[0].dst)

                    role_map = {
                        p_e.role: int(r_e.edge_id)
                        for p_e, r_e in zip(pattern_edges_sorted, edges)
                    }

                    try:
                        motif_row = make_motif_instance_row(
                            window_id=window.window_id, pattern=pattern,
                            edges=edges, node_map=node_map, role_map=role_map,
                            anchor_edge_id=anchor.edge_id, validate=False,
                        )
                        # Item 5 & 6: Lưu các feature trực tiếp vào motif_row
                        amounts = [e.amount for e in edges if e.amount > 0]
                        motif_row["amount_min"] = float(min(amounts)) if amounts else 0.0
                        motif_row["amount_max"] = float(max(amounts)) if amounts else 0.0
                        motif_row["amount_sum"] = float(sum(amounts))
                        motif_row["amount_coherence"] = float(min(amounts) / max(amounts)) if amounts else 1.0
                        motif_row["time_span"] = int(max(e.step for e in edges) - min(e.step for e in edges))
                        motif_row["dst_in_degree_window"] = int(len(incoming_edges))
                    except Exception:
                        continue

                    motif_rows.append(motif_row)
                    membership_rows.extend(
                        make_edge_motif_membership_rows(motif_row, pattern, edges)
                    )
                    dst_instance_count += 1

                    # Item 3: kiểm tra max_instances_per_dst ngay sau khi emit
                    if dst_instance_count >= self.max_instances_per_dst:
                        n_rej_dst_cap += 1
                        dst_cap_reached = True
                        break

                    if len(motif_rows) >= self.max_instances_per_window:
                        break

            if len(motif_rows) >= self.max_instances_per_window:
                break

        motif_df      = motif_instance_rows_to_polars(motif_rows)
        membership_df = membership_rows_to_polars(membership_rows)
        elapsed       = time.time() - start_time

        # Item 6: Thống kê max & mean degree ở cấp window
        max_deg = int(max(scanned_dst_degrees)) if scanned_dst_degrees else 0
        mean_deg = float(sum(scanned_dst_degrees) / len(scanned_dst_degrees)) if scanned_dst_degrees else 0.0

        stats = {
            "window_id":                    int(window.window_id),
            "motif_type":                   pattern.name,
            "matcher_type":                 pattern.matcher_type,
            "n_branches":                   int(n_branches),
            "primary_num_edges":            int(df_primary.height),
            "extended_num_edges":           int(df_extended.height),
            "n_dst_scanned":                int(n_dst_scanned),
            "n_dst_skipped_low_degree":     int(n_dst_skipped_deg),
            "n_dst_capped":                 int(n_dst_capped),
            "max_dst_in_degree_scanned":    max_deg,
            "mean_dst_in_degree_scanned":   mean_deg,
            "n_groups_checked":             int(n_groups_checked),
            "n_combos_checked":             int(n_combos_checked),
            "num_instances":                int(motif_df.height),
            "num_membership_rows":          int(membership_df.height),
            "n_rej_duration":               int(n_rej_duration),
            "n_rej_delta_hop":              int(n_rej_delta_hop),
            "n_rej_structure":              int(n_rej_structure),
            "n_rej_coherence":              int(n_rej_coherence),        # NEW
            "n_rej_anchor":                 int(n_rej_anchor),
            "n_rej_dst_cap":                int(n_rej_dst_cap),          # NEW
            "hit_max_instances_per_window": int(len(motif_rows) >= self.max_instances_per_window),
            "elapsed_seconds":              float(elapsed),
        }

        if write_output:
            mp, mep = write_motif_outputs(
                motif_rows, membership_rows, window.window_id, pattern.name
            )
            stats["motif_path"]      = mp
            stats["membership_path"] = mep

        return motif_df, membership_df, stats


# Build one matcher instance per fan-in size and store in a dict.
# The pipeline in Cell 11 will dispatch by pattern.name.
fanin_matchers = {
    p.name: FanInMatcher(
        cap_schedule             = FAN_IN_CAP_SCHEDULE,
        max_instances_per_window = MAX_INSTANCES_PER_WINDOW,
        max_instances_per_dst    = 500,
        amount_coherence_ratio  = 3.0,
        amount_min               = AMOUNT_MIN,
        delta_hop                = DELTA_HOP,
    )
    for p in fan_in_patterns
}

# Backward-compatible alias used in Cell 11 registry.
fanin_matcher = fanin_matchers[fan_in_4.name]

MATCHER_REGISTRY["fan_in"] = FanInMatcher(
    cap_schedule             = FAN_IN_CAP_SCHEDULE,
    max_instances_per_window = MAX_INSTANCES_PER_WINDOW,
    max_instances_per_dst    = 500,
    amount_min               = AMOUNT_MIN,
    amount_coherence_ratio   = 3.0,
    delta_hop                = DELTA_HOP,
)

print("FanInMatcher (sliding group + coherence) registered.")
print("MATCHER_REGISTRY keys:", list(MATCHER_REGISTRY.keys()))
print("Cell 9 completed.")


# ============================================================
# Cell 10: FanOutMatcher (generalized, variable n)
# ============================================================

def passes_distinct_nodes_for_fanout(edges: List[EdgeRecord]) -> bool:
    if not edges:
        return False
    if len({e.edge_id for e in edges}) != len(edges):
        return False
    src_values = {e.src for e in edges}
    if len(src_values) != 1:
        return False
    dst_values = [e.dst for e in edges]
    if len(dst_values) != len(set(dst_values)):
        return False
    center_src = next(iter(src_values))
    return not any(d == center_src for d in dst_values)


def passes_fanout_duration(edges: List[EdgeRecord], max_duration: int) -> bool:
    steps = [e.step for e in edges]
    return (max(steps) - min(steps)) <= max_duration


def canonicalize_fanout_edges(edges: List[EdgeRecord]) -> List[EdgeRecord]:
    return sorted(edges, key=lambda e: (e.step, e.edge_id))


class FanOutMatcher:
    """
    Generalized fan-out matcher. Supports n = 3..7 outgoing branches.
    Mirrors FanInMatcher exactly, operating on out_edges instead of in_edges.
    """

    def __init__(
        self,
        cap_schedule:             Dict[int, int]  = FAN_OUT_CAP_SCHEDULE,
        max_instances_per_window: int             = MAX_INSTANCES_PER_WINDOW,
        max_instances_per_src:    int             = 500,
        amount_min:               Optional[float] = AMOUNT_MIN,
        amount_coherence_ratio:   float           = 3.0,
        delta_hop:                Optional[int]   = DELTA_HOP,
        lookback_delta:           Optional[int]   = None,
    ):
        self.cap_schedule             = cap_schedule
        self.max_instances_per_window = max_instances_per_window
        self.max_instances_per_src    = max_instances_per_src
        self.amount_min               = amount_min
        self.amount_coherence_ratio   = amount_coherence_ratio
        self.delta_hop                = delta_hop
        self.lookback_delta           = lookback_delta

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

    def match(
        self,
        df_primary:   pl.DataFrame,
        df_extended:  pl.DataFrame,
        index:        TemporalIndex,
        window:       WindowSpec,
        pattern:      MotifPattern,
        write_output: bool = False,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, Any]]:

        if pattern.matcher_type != "fan_out":
            raise ValueError(
                f"FanOutMatcher requires matcher_type='fan_out', got {pattern.matcher_type}"
            )

        n_branches = len(pattern.edges)
        if n_branches < 3:
            raise ValueError(f"fan_out pattern must have >= 3 edges, got {n_branches}")

        start_time = time.time()
        motif_rows:      List[Dict] = []
        membership_rows: List[Dict] = []

        n_src_scanned     = 0
        n_src_skipped_deg = 0
        n_src_capped      = 0
        n_groups_checked  = 0
        n_combos_checked  = 0
        n_rej_duration    = 0
        n_rej_delta_hop   = 0
        n_rej_structure   = 0
        n_rej_coherence   = 0
        n_rej_anchor      = 0
        n_rej_lookback    = 0
        n_rej_src_cap     = 0

        pattern_edges_sorted = sorted(pattern.edges, key=lambda x: x.order)
        cap = self._get_cap(n_branches)

        seen_canonical_keys = set()
        scanned_src_degrees = []

        for src_node, outgoing_edges in index.out_edges.items():

            if len(outgoing_edges) < n_branches:
                n_src_skipped_deg += 1
                continue

            n_src_scanned += 1
            scanned_src_degrees.append(len(outgoing_edges))

            # Amount floor filter trước khi group
            candidates = list(outgoing_edges)
            if self.amount_min is not None:
                candidates = [e for e in candidates if e.amount >= self.amount_min]

            if len(candidates) < n_branches:
                continue

            # Sliding temporal groups
            groups = sliding_temporal_groups(candidates, pattern.max_duration, n_branches)
            n_groups_checked += len(groups)

            src_instance_count = 0
            src_cap_reached = False
            src_capped = False

            for group in groups:
                if len(motif_rows) >= self.max_instances_per_window:
                    break
                if src_cap_reached:
                    break

                # Sliding trước, cap sau trong từng group
                if len(group) > cap:
                    group = cap_edges(group, max_candidates=cap, policy="hybrid")
                    if not src_capped:
                        n_src_capped += 1
                        src_capped = True

                for edge_combo in combinations(group, n_branches):
                    n_combos_checked += 1
                    edges = canonicalize_fanout_edges(list(edge_combo))

                    # 1. Duration
                    if not passes_fanout_duration(edges, pattern.max_duration):
                        n_rej_duration += 1
                        continue

                    # 2. Delta hop
                    if not passes_consecutive_delta_hop_sorted(edges, self.delta_hop):
                        n_rej_delta_hop += 1
                        continue

                    # 3. Structural validity
                    if not passes_distinct_nodes_for_fanout(edges):
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

                    # Lookback check using _source_has_prior_incoming
                    if not self._source_has_prior_incoming(index, src_node, anchor.step):
                        n_rej_lookback += 1
                        continue

                    # Canonical key dedup
                    ck = make_canonical_key(
                        pattern.name,
                        [e.edge_id for e in edges],
                        ordered=True,
                    )
                    if ck in seen_canonical_keys:
                        continue
                    seen_canonical_keys.add(ck)

                    node_map = {"src": int(edges[0].src)}
                    node_map.update({
                        f"dst_{i+1}": int(edges[i].dst) for i in range(n_branches)
                    })

                    role_map = {
                        p_edge.role: int(real_edge.edge_id)
                        for p_edge, real_edge in zip(pattern_edges_sorted, edges)
                    }

                    try:
                        motif_row = make_motif_instance_row(
                            window_id      = window.window_id,
                            pattern        = pattern,
                            edges          = edges,
                            node_map       = node_map,
                            role_map       = role_map,
                            anchor_edge_id = anchor.edge_id,
                            validate       = False,
                        )
                        # Lưu các feature trực tiếp vào motif_row
                        amounts = [e.amount for e in edges if e.amount > 0]
                        motif_row["amount_min"] = float(min(amounts)) if amounts else 0.0
                        motif_row["amount_max"] = float(max(amounts)) if amounts else 0.0
                        motif_row["amount_sum"] = float(sum(amounts))
                        motif_row["amount_coherence"] = float(min(amounts) / max(amounts)) if amounts else 1.0
                        motif_row["time_span"] = int(max(e.step for e in edges) - min(e.step for e in edges))
                        motif_row["src_out_degree_window"] = int(len(outgoing_edges))
                    except Exception:
                        continue

                    motif_rows.append(motif_row)
                    membership_rows.extend(
                        make_edge_motif_membership_rows(motif_row, pattern, edges)
                    )
                    src_instance_count += 1

                    # Check max_instances_per_src sau khi emit
                    if src_instance_count >= self.max_instances_per_src:
                        n_rej_src_cap += 1
                        src_cap_reached = True
                        break

                    if len(motif_rows) >= self.max_instances_per_window:
                        break

            if len(motif_rows) >= self.max_instances_per_window:
                break

        motif_df      = motif_instance_rows_to_polars(motif_rows)
        membership_df = membership_rows_to_polars(membership_rows)
        elapsed       = time.time() - start_time

        # Thống kê max & mean degree ở cấp window
        max_deg = int(max(scanned_src_degrees)) if scanned_src_degrees else 0
        mean_deg = float(sum(scanned_src_degrees) / len(scanned_src_degrees)) if scanned_src_degrees else 0.0

        stats = {
            "window_id":                    int(window.window_id),
            "motif_type":                   pattern.name,
            "matcher_type":                 pattern.matcher_type,
            "n_branches":                   int(n_branches),
            "primary_num_edges":            int(df_primary.height),
            "extended_num_edges":           int(df_extended.height),
            "n_src_scanned":                int(n_src_scanned),
            "n_src_skipped_low_degree":     int(n_src_skipped_deg),
            "n_src_capped":                 int(n_src_capped),
            "max_src_out_degree_scanned":   max_deg,
            "mean_src_out_degree_scanned":  mean_deg,
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
            "n_rej_src_cap":                int(n_rej_src_cap),
            "hit_max_instances_per_window": int(len(motif_rows) >= self.max_instances_per_window),
            "elapsed_seconds":              float(elapsed),
        }

        if write_output:
            mp, mep = write_motif_outputs(
                motif_rows, membership_rows, window.window_id, pattern.name
            )
            stats["motif_path"]      = mp
            stats["membership_path"] = mep

        return motif_df, membership_df, stats


# Build one matcher instance per fan-out size and store in a dict.
# The pipeline in Cell 11 will dispatch by pattern.name.
fanout_matchers = {
    p.name: FanOutMatcher(
        cap_schedule             = FAN_OUT_CAP_SCHEDULE,
        max_instances_per_window = MAX_INSTANCES_PER_WINDOW,
        max_instances_per_src    = 500,
        amount_coherence_ratio   = 3.0,
        amount_min               = AMOUNT_MIN,
        delta_hop                = DELTA_HOP,
        lookback_delta           = None,
    )
    for p in fan_out_patterns
}

MATCHER_REGISTRY["fan_out"] = FanOutMatcher(
    cap_schedule             = FAN_OUT_CAP_SCHEDULE,
    max_instances_per_window = MAX_INSTANCES_PER_WINDOW,
    max_instances_per_src    = 500,
    amount_min               = AMOUNT_MIN,
    amount_coherence_ratio   = 3.0,
    delta_hop                = DELTA_HOP,
    lookback_delta           = None,
)

print("FanOutMatcher (generalized) ready.")
print("MATCHER_REGISTRY keys:", list(MATCHER_REGISTRY.keys()))
print("Cell 10 completed.")



# ============================================================
# Cell 11: Run Fan-in and Fan-out over Temporal Windows
# ============================================================

def get_pattern_by_name(patterns: List[MotifPattern], pattern_name: str) -> MotifPattern:
    """
    Retrieve a MotifPattern by name.
    """
    matches = [p for p in patterns if p.name == pattern_name]

    if len(matches) == 0:
        raise ValueError(f"Pattern not found: {pattern_name}")

    if len(matches) > 1:
        raise ValueError(f"Duplicated pattern name: {pattern_name}")

    return matches[0]


def get_output_paths_for_window_pattern(
    window_id: int,
    motif_type: str,
    motif_instance_dir: str = MOTIF_INSTANCE_DIR,
    membership_dir: str = MEMBERSHIP_DIR,
) -> Tuple[str, str]:
    """
    Return expected output shard paths for one window-pattern pair.
    """
    motif_path = f"{motif_instance_dir}/window_{int(window_id):06d}_{motif_type}.parquet"
    membership_path = f"{membership_dir}/window_{int(window_id):06d}_{motif_type}.parquet"

    return motif_path, membership_path


def output_shards_exist(window_id: int, motif_type: str) -> bool:
    """
    Check whether both motif and membership shards already exist.
    """
    motif_path, membership_path = get_output_paths_for_window_pattern(
        window_id=window_id,
        motif_type=motif_type,
    )

    return os.path.exists(motif_path) and os.path.exists(membership_path)


def write_stats_log(stats_rows: List[Dict[str, Any]], stats_path: str) -> pl.DataFrame:
    """
    Write stats rows to parquet and return the stats dataframe.
    """
    if len(stats_rows) == 0:
        stats_df = pl.DataFrame()
    else:
        stats_df = pl.DataFrame(stats_rows)

    stats_df.write_parquet(stats_path)

    return stats_df


def run_matchers_over_windows(
    df_edges: pl.DataFrame,
    windows: List[WindowSpec],
    patterns_to_run: List[MotifPattern],
    matcher_registry: Dict[str, Any],
    max_windows: Optional[int] = None,
    skip_existing: bool = True,
    write_empty_outputs: bool = True,
) -> pl.DataFrame:
    """
    Run selected matchers over temporal windows.

    For each window:
        1. Slice primary and extended edges.
        2. Build TemporalIndex on extended edges.
        3. Run each selected pattern matcher.
        4. Write motif_instances and edge_motif_membership parquet shards.
        5. Record runtime and output statistics.

    Parameters:
        max_windows:
            If not None, only run first max_windows windows for testing.

        skip_existing:
            If True, skip a window-pattern if both output shards already exist.

        write_empty_outputs:
            If True, write empty parquet shards even when no instances are found.
            This makes resume and auditing easier.
    """

    run_start_time = time.time()

    stats_rows = []

    if max_windows is not None:
        windows_to_run = windows[:max_windows]
    else:
        windows_to_run = windows

    print("Starting motif mining run.")
    print("Number of windows to run:", len(windows_to_run))
    print("Patterns:", [p.name for p in patterns_to_run])
    print("skip_existing:", skip_existing)
    print("write_empty_outputs:", write_empty_outputs)

    for idx, window in enumerate(windows_to_run):
        window_start_time = time.time()

        print(
            f"\n[Window {idx + 1}/{len(windows_to_run)}] "
            f"window_id={window.window_id}, "
            f"primary=[{window.primary_start}, {window.primary_end}], "
            f"extended=[{window.extended_start}, {window.extended_end}]"
        )

        df_primary, df_extended = slice_window_edges(df_edges, window)

        window_summary = build_window_candidate_summary(df_extended)

        if df_primary.height == 0:
            print("  Primary window is empty. Skipping matcher execution.")

            for pattern in patterns_to_run:
                stats_rows.append({
                    "window_id": int(window.window_id),
                    "motif_type": pattern.name,
                    "matcher_type": pattern.matcher_type,
                    "status": "skipped_empty_primary",
                    "primary_start": int(window.primary_start),
                    "primary_end": int(window.primary_end),
                    "extended_start": int(window.extended_start),
                    "extended_end": int(window.extended_end),
                    "primary_num_edges": int(df_primary.height),
                    "extended_num_edges": int(df_extended.height),
                    "num_instances": 0,
                    "num_membership_rows": 0,
                    "elapsed_seconds": 0.0,
                })

            continue

        if df_extended.height == 0:
            print("  Extended window is empty. Skipping matcher execution.")

            for pattern in patterns_to_run:
                stats_rows.append({
                    "window_id": int(window.window_id),
                    "motif_type": pattern.name,
                    "matcher_type": pattern.matcher_type,
                    "status": "skipped_empty_extended",
                    "primary_start": int(window.primary_start),
                    "primary_end": int(window.primary_end),
                    "extended_start": int(window.extended_start),
                    "extended_end": int(window.extended_end),
                    "primary_num_edges": int(df_primary.height),
                    "extended_num_edges": int(df_extended.height),
                    "num_instances": 0,
                    "num_membership_rows": 0,
                    "elapsed_seconds": 0.0,
                })

            continue

        index_start_time = time.time()
        temporal_index = build_temporal_index_from_polars(df_extended)
        index_elapsed = time.time() - index_start_time

        print("  df_primary edges:", df_primary.height)
        print("  df_extended edges:", df_extended.height)
        print("  index stats:", temporal_index.stats())
        print("  index build time:", round(index_elapsed, 3), "seconds")

        for pattern in patterns_to_run:
            pattern_start_time = time.time()

            if pattern.matcher_type not in matcher_registry:
                raise ValueError(
                    f"No matcher registered for matcher_type={pattern.matcher_type}. "
                    f"Available registry keys: {list(matcher_registry.keys())}"
                )

            motif_path, membership_path = get_output_paths_for_window_pattern(
                window_id=window.window_id,
                motif_type=pattern.name,
            )

            if skip_existing and output_shards_exist(window.window_id, pattern.name):
                print(f"  [{pattern.name}] output exists. Skipping.")

                stats_rows.append({
                    "window_id": int(window.window_id),
                    "motif_type": pattern.name,
                    "matcher_type": pattern.matcher_type,
                    "status": "skipped_existing",
                    "primary_start": int(window.primary_start),
                    "primary_end": int(window.primary_end),
                    "extended_start": int(window.extended_start),
                    "extended_end": int(window.extended_end),
                    "primary_num_edges": int(df_primary.height),
                    "extended_num_edges": int(df_extended.height),
                    "index_build_seconds": float(index_elapsed),
                    "num_instances": None,
                    "num_membership_rows": None,
                    "elapsed_seconds": 0.0,
                    "motif_path": motif_path,
                    "membership_path": membership_path,
                })

                continue

            matcher = matcher_registry[pattern.matcher_type]

            motif_df, membership_df, stats = matcher.match(
                df_primary=df_primary,
                df_extended=df_extended,
                index=temporal_index,
                window=window,
                pattern=pattern,
                write_output=False,
            )

            if write_empty_outputs or motif_df.height > 0:
                motif_df.write_parquet(motif_path)
                membership_df.write_parquet(membership_path)

            pattern_elapsed = time.time() - pattern_start_time

            stats["status"] = "completed"
            stats["index_build_seconds"] = float(index_elapsed)
            stats["total_pattern_seconds"] = float(pattern_elapsed)
            stats["motif_path"] = motif_path
            stats["membership_path"] = membership_path

            stats_rows.append(stats)

            print(
                f"  [{pattern.name}] "
                f"instances={motif_df.height}, "
                f"membership_rows={membership_df.height}, "
                f"time={round(pattern_elapsed, 3)}s"
            )

            # Explicit cleanup per pattern.
            del motif_df
            del membership_df
            gc.collect()

        window_elapsed = time.time() - window_start_time
        print("  Window elapsed:", round(window_elapsed, 3), "seconds")

        # Explicit cleanup per window.
        del temporal_index
        del df_primary
        del df_extended
        gc.collect()

    total_elapsed = time.time() - run_start_time

    print("\nMotif mining run completed.")
    print("Total elapsed:", round(total_elapsed, 3), "seconds")

    stats_path = f"{LOG_DIR}/fanin_fanout_run_stats.parquet"
    stats_df = write_stats_log(stats_rows, stats_path)

    print("Stats log saved to:")
    print(stats_path)

    return stats_df


# ------------------------------------------------------------
# Select patterns for this run
# ------------------------------------------------------------

PATTERNS_TO_RUN_CELL11 = fan_in_patterns + fan_out_patterns

# Ensure matchers are registered.
required_matcher_types = sorted(set(p.matcher_type for p in PATTERNS_TO_RUN_CELL11))
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

print("Cell 11 setup ready.")
print("Patterns to run:", [p.name for p in PATTERNS_TO_RUN_CELL11])
print("Matcher registry:", list(MATCHER_REGISTRY.keys()))


# ------------------------------------------------------------
# Run mode
# ------------------------------------------------------------
# For first run, keep this small to validate the pipeline.
# After checking outputs, set MAX_WINDOWS_TO_RUN = None to run all windows.

MAX_WINDOWS_TO_RUN = 2

SKIP_EXISTING_OUTPUTS = False
WRITE_EMPTY_OUTPUTS = True

run_stats_df = run_matchers_over_windows(
    df_edges=df_edges,
    windows=windows,
    patterns_to_run=PATTERNS_TO_RUN_CELL11,
    matcher_registry=MATCHER_REGISTRY,
    max_windows=MAX_WINDOWS_TO_RUN,
    skip_existing=SKIP_EXISTING_OUTPUTS,
    write_empty_outputs=WRITE_EMPTY_OUTPUTS,
)

print("\nRun stats preview:")
display(run_stats_df)


# ------------------------------------------------------------
# Quick aggregate summary
# ------------------------------------------------------------

if run_stats_df.height > 0 and "status" in run_stats_df.columns:
    completed_stats = run_stats_df.filter(pl.col("status") == "completed")

    if completed_stats.height > 0:
        aggregate_summary = (
            completed_stats
            .group_by("motif_type")
            .agg([
                pl.col("num_instances").sum().alias("total_instances"),
                pl.col("num_membership_rows").sum().alias("total_membership_rows"),
                pl.col("elapsed_seconds").sum().alias("total_match_seconds"),
                pl.col("window_id").n_unique().alias("num_windows_completed"),
            ])
            .sort("motif_type")
        )

        print("\nAggregate summary:")
        display(aggregate_summary)
    else:
        print("\nNo completed matcher rows found in run_stats_df.")
else:
    print("\nrun_stats_df is empty or missing status column.")

print("\nCell 11 completed.")



