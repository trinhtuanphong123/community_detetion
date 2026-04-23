"""
second_order.py — Step 4.2: 2nd Order (Line) Graph Construction.

Aggregates temporal edges into 2nd-order relay edges.

Output schema (per graph_schema.md):
    src_2nd, dst_2nd, count, total_amount_src,
    total_amount_dst, avg_time_gap, n_alert, _max_node_id
"""

from __future__ import annotations

try:
    import cudf as pd_lib
except ImportError:
    import pandas as pd_lib


def build_second_order_edges(
    temporal_edges: "pd_lib.DataFrame",
) -> "pd_lib.DataFrame":
    """Aggregate temporal edges into 2nd-order relay edges.

    Groups by (src_1, dst_2) — the relay pair's origin and final
    destination — and computes aggregate statistics.

    Parameters
    ----------
    temporal_edges : DataFrame
        Output from build_temporal_edges().

    Returns
    -------
    DataFrame with columns:
        src_2nd, dst_2nd, count, total_amount_src,
        total_amount_dst, avg_time_gap, n_alert, _max_node_id
    """
    if len(temporal_edges) == 0:
        return pd_lib.DataFrame({
            "src_2nd": pd_lib.Series(dtype="int64"),
            "dst_2nd": pd_lib.Series(dtype="int64"),
            "count": pd_lib.Series(dtype="int64"),
            "total_amount_src": pd_lib.Series(dtype="float64"),
            "total_amount_dst": pd_lib.Series(dtype="float64"),
            "avg_time_gap": pd_lib.Series(dtype="float64"),
            "n_alert": pd_lib.Series(dtype="int64"),
            "_max_node_id": pd_lib.Series(dtype="int64"),
        })

    te = temporal_edges.copy()
    te["time_gap"] = te["step_2"] - te["step_1"]
    te["n_alert_per_pair"] = te["alert_1"] + te["alert_2"]

    grouped = te.groupby(["src_1", "dst_2"], as_index=False).agg({
        "amount_1": "sum",
        "amount_2": "sum",
        "time_gap": "mean",
        "n_alert_per_pair": "sum",
        "src_2": "count",  # count of relay edges
    })

    grouped.columns = [
        "src_2nd", "dst_2nd",
        "total_amount_src", "total_amount_dst",
        "avg_time_gap", "n_alert", "count",
    ]

    # Compute _max_node_id for downstream canonical pair construction
    max_src = grouped["src_2nd"].max()
    max_dst = grouped["dst_2nd"].max()
    if hasattr(max_src, "item"):
        max_src = max_src.item()
    if hasattr(max_dst, "item"):
        max_dst = max_dst.item()
    grouped["_max_node_id"] = max(int(max_src), int(max_dst))

    return grouped
