

# ============================================================
# Cell 7b: Common validation, scoring, and candidate selection helpers
# ============================================================

import json
import math
from typing import List, Dict, Tuple, Any, Optional

def has_unique_edge_ids(edges: List[EdgeRecord]) -> bool:
    edge_ids = [e.edge_id for e in edges]
    return len(edge_ids) == len(set(edge_ids))


def has_distinct_nodes(node_values: List[int]) -> bool:
    return len(node_values) == len(set(node_values))


def is_within_total_duration(edges: List[EdgeRecord], max_duration: int) -> bool:
    if not edges:
        return True
    steps = [e.step for e in edges]
    return (max(steps) - min(steps)) <= max_duration


def passes_consecutive_step_gap(edges: List[EdgeRecord], max_gap: Optional[int]) -> bool:
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
    if max_gap is None:
        return True
    if len(earlier_edges) == 0 or len(later_edges) == 0:
        return False
    earlier_end = max(int(e.step) for e in earlier_edges)
    later_start = min(int(e.step) for e in later_edges)
    return 0 < (later_start - earlier_end) <= max_gap


def amount_consistency(edges: List[EdgeRecord]) -> float:
    amounts = [float(e.amount) for e in edges if float(e.amount) > 0]
    if not amounts:
        return 0.0
    max_amount = max(amounts)
    if max_amount <= 0:
        return 0.0
    return float(min(amounts) / max_amount)

# Alias for backward compatibility
compute_amount_consistency = amount_consistency


def flow_ratio(in_edges: List[EdgeRecord], out_edges: List[EdgeRecord]) -> float:
    in_sum = float(sum(e.amount for e in in_edges))
    out_sum = float(sum(e.amount for e in out_edges))
    return out_sum / in_sum if in_sum > 0 else 0.0


def make_canonical_key(
    motif_type: str,
    edge_ids: List[int],
    ordered: bool = True,
) -> str:
    if ordered:
        canonical_edge_ids = [int(eid) for eid in edge_ids]
    else:
        canonical_edge_ids = sorted(int(eid) for eid in edge_ids)

    key_obj = {
        "motif_type": motif_type,
        "edge_ids": canonical_edge_ids,
    }
    return json.dumps(key_obj, sort_keys=True)


# ------------------------------------------------------------
# SUSPICIOUSNESS SCORING FUNCTIONS
# ------------------------------------------------------------

def score_temporal_compactness(edges: List[EdgeRecord], max_duration: int) -> float:
    if len(edges) <= 1:
        return 1.0
    if max_duration <= 0:
        return 0.0
    steps = [e.step for e in edges]
    duration = max(steps) - min(steps)
    return max(0.0, min(1.0, 1.0 - (duration / max_duration)))


def score_amount_consistency(edges: List[EdgeRecord]) -> float:
    return amount_consistency(edges)


def score_amount_scale(edges: List[EdgeRecord], amount_quantiles: Dict[float, float]) -> float:
    if not edges:
        return 0.0
    mean_amount = sum(e.amount for e in edges) / len(edges)

    q50 = amount_quantiles.get(0.5, 0.0)
    q90 = amount_quantiles.get(0.9, 0.0)
    q99 = amount_quantiles.get(0.99, 0.0)

    if mean_amount >= q99:
        return 1.0
    elif mean_amount >= q90:
        return 0.8 + 0.2 * (mean_amount - q90) / (q99 - q90) if q99 > q90 else 0.8
    elif mean_amount >= q50:
        return 0.5 + 0.3 * (mean_amount - q50) / (q90 - q50) if q90 > q50 else 0.5
    else:
        return 0.5 * mean_amount / q50 if q50 > 0 else 0.0


def score_degree_penalty(
    edges: List[EdgeRecord],
    node_in_degree: Dict[int, int],
    node_out_degree: Dict[int, int]
) -> float:
    nodes = set()
    for e in edges:
        nodes.add(e.src)
        nodes.add(e.dst)

    max_deg = 0
    for n in nodes:
        in_d = node_in_degree.get(n, 0)
        out_d = node_out_degree.get(n, 0)
        deg = in_d + out_d
        if deg > max_deg:
            max_deg = deg

    if max_deg <= 10:
        return 1.0
    elif max_deg <= 100:
        return 1.0 - 0.5 * (max_deg - 10) / 90
    else:
        return max(0.1, 0.5 * 100 / max_deg)


def score_flow_conservation(in_edges: List[EdgeRecord], out_edges: List[EdgeRecord]) -> float:
    ratio = flow_ratio(in_edges, out_edges)
    if ratio <= 0:
        return 0.0
    return math.exp(-((ratio - 1.0) ** 2) / 0.5)


def score_instance(edges: List[EdgeRecord], index: TemporalIndex, pattern: MotifPattern) -> float:
    if pattern.matcher_type == "split_merge":
        n = len(edges) // 2
        split_edges = edges[:n]
        merge_edges = edges[n:]

        split_comp = score_temporal_compactness(split_edges, pattern.max_duration)
        merge_comp = score_temporal_compactness(merge_edges, pattern.max_duration)

        in_end = max(e.step for e in split_edges)
        out_start = min(e.step for e in merge_edges)
        gap = max(0, out_start - in_end)
        gap_score = math.exp(-gap / 3.0)

        flow_score = score_flow_conservation(split_edges, merge_edges)
        consistency = score_amount_consistency(edges)
        penalty = score_degree_penalty(edges, index.node_in_degree, index.node_out_degree)

        overall_score = (split_comp * 0.15) + (merge_comp * 0.15) + (gap_score * 0.15) + (flow_score * 0.25) + (consistency * 0.20) + (penalty * 0.10)
        return float(overall_score)

    elif pattern.matcher_type == "center_in_out":
        n_in = sum(1 for e in pattern.edges if e.dst == "center")
        in_edges = edges[:n_in]
        out_edges = edges[n_in:]

        in_comp = score_temporal_compactness(in_edges, pattern.max_duration)
        out_comp = score_temporal_compactness(out_edges, pattern.max_duration)

        in_end = max(e.step for e in in_edges)
        out_start = min(e.step for e in out_edges)
        gap = max(0, out_start - in_end)
        gap_score = math.exp(-gap / 3.0)

        flow_score = score_flow_conservation(in_edges, out_edges)
        consistency = score_amount_consistency(edges)
        penalty = score_degree_penalty(edges, index.node_in_degree, index.node_out_degree)

        overall_score = (in_comp * 0.15) + (out_comp * 0.15) + (gap_score * 0.15) + (flow_score * 0.25) + (consistency * 0.20) + (penalty * 0.10)
        return float(overall_score)

    else:
        compactness = score_temporal_compactness(edges, pattern.max_duration)
        consistency = score_amount_consistency(edges)
        scale = score_amount_scale(edges, index.amount_quantiles)
        penalty = score_degree_penalty(edges, index.node_in_degree, index.node_out_degree)

        overall_score = (compactness * 0.25) + (consistency * 0.25) + (scale * 0.25) + (penalty * 0.25)
        return float(overall_score)


# ------------------------------------------------------------
# RANKING-BASED CANDIDATE SELECTION
# ------------------------------------------------------------

def cap_edges(
    edges: List[EdgeRecord],
    max_candidates: int,
    policy: str = "hybrid",
    anchor_edge: Optional[EdgeRecord] = None,
    amount_quantiles: Optional[Dict[float, float]] = None,
) -> List[EdgeRecord]:
    """
    Candidate cap helper implementing multiple policies.
    Policies:
        - earliest: earliest by step
        - top_amount: highest transaction amount
        - temporally_dense: closest in time to anchor_edge
        - amount_coherent: closest in amount to anchor_edge (or index median)
        - hybrid: a combination of earliest, top_amount, temporally_dense, and amount_coherent

    Important: is_sar is never used in candidate selection.
    """
    if max_candidates is None or len(edges) <= max_candidates:
        return sorted(edges, key=lambda e: (e.step, e.edge_id))

    if policy == "earliest":
        selected = sorted(edges, key=lambda e: (e.step, e.edge_id))[:max_candidates]
        return sorted(selected, key=lambda e: (e.step, e.edge_id))

    if policy == "top_amount":
        selected = sorted(edges, key=lambda e: (-e.amount, e.step, e.edge_id))[:max_candidates]
        return sorted(selected, key=lambda e: (e.step, e.edge_id))

    if policy == "temporally_dense" and anchor_edge is not None:
        selected = sorted(edges, key=lambda e: (abs(e.step - anchor_edge.step), e.step, e.edge_id))[:max_candidates]
        return sorted(selected, key=lambda e: (e.step, e.edge_id))

    if policy == "amount_coherent" and anchor_edge is not None:
        selected = sorted(edges, key=lambda e: (abs(e.amount - anchor_edge.amount), e.step, e.edge_id))[:max_candidates]
        return sorted(selected, key=lambda e: (e.step, e.edge_id))

    if anchor_edge is None:
        k_earliest = max_candidates // 2
        k_amount = max_candidates - k_earliest
        earliest = sorted(edges, key=lambda e: (e.step, e.edge_id))[:k_earliest]
        top_amount = sorted(edges, key=lambda e: (-e.amount, e.step, e.edge_id))[:k_amount]
        selected_dict = {e.edge_id: e for e in earliest + top_amount}
        if len(selected_dict) < max_candidates:
            remaining = sorted(edges, key=lambda e: (e.step, e.edge_id))
            for e in remaining:
                if e.edge_id not in selected_dict:
                    selected_dict[e.edge_id] = e
                    if len(selected_dict) >= max_candidates:
                        break
        return sorted(selected_dict.values(), key=lambda e: (e.step, e.edge_id))[:max_candidates]

    # Fallback to hybrid combination when anchor_edge is not None
    k_earliest = max(1, max_candidates // 4)
    k_amount = max(1, max_candidates // 4)
    k_dense = max(1, max_candidates // 4)
    k_coherent = max(1, max_candidates - (k_earliest + k_amount + k_dense))

    earliest = sorted(edges, key=lambda e: (e.step, e.edge_id))[:k_earliest]
    top_amount = sorted(edges, key=lambda e: (-e.amount, e.step, e.edge_id))[:k_amount]
    dense = sorted(edges, key=lambda e: (abs(e.step - anchor_edge.step), e.step, e.edge_id))[:k_dense]
    coherent = sorted(edges, key=lambda e: (abs(e.amount - anchor_edge.amount), e.step, e.edge_id))[:k_coherent]

    selected_dict = {}
    for e in earliest + top_amount + dense + coherent:
        selected_dict[e.edge_id] = e

    if len(selected_dict) < max_candidates:
        remaining = sorted(edges, key=lambda e: (e.step, e.edge_id))
        for e in remaining:
            if e.edge_id not in selected_dict:
                selected_dict[e.edge_id] = e
                if len(selected_dict) >= max_candidates:
                    break

    return sorted(selected_dict.values(), key=lambda e: (e.step, e.edge_id))[:max_candidates]


def select_edges_by_hybrid_policy(
    edges: List[EdgeRecord],
    max_candidates: int,
    early_ratio: float = 0.5,
) -> List[EdgeRecord]:
    return cap_edges(edges, max_candidates, "hybrid")


print("Cell 7b: Common validation, scoring, and candidate selection helpers ready.")


