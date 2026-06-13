# ============================================================
# Cell 9: FanInMatcher (generalized, variable n)
# ============================================================

from itertools import combinations
import time
from typing import List, Dict, Tuple, Any, Optional
import polars as pl

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

MATCHER_REGISTRY["fan_in"] = FanInMatcher(
    cap_schedule             = FAN_IN_CAP_SCHEDULE,
    max_instances_per_window = MAX_INSTANCES_PER_WINDOW,
    max_instances_per_dst    = 500,
    amount_min               = AMOUNT_MIN,
    amount_coherence_ratio   =3.0,
    delta_hop                = DELTA_HOP,
)

print("FanInMatcher (sliding group + coherence) registered.")
print("Cell 9 completed.")