"""
temporal.py — Step 4.1: Temporal Edge Construction.

Builds temporal relay edges by self-joining transactions where
dst_1 == src_2 and the time gap is within DELTA_W steps.

Output schema (per graph_schema.md):
    src_1, dst_1, step_1, amount_1, alert_1,
    src_2, dst_2, step_2, amount_2, alert_2
"""

from __future__ import annotations

try:
    import cudf as pd_lib
except ImportError:
    import pandas as pd_lib


def build_temporal_edges(
    df: "pd_lib.DataFrame",
    delta_w: int = 5,
    amount_col: str = "amount",
    step_col: str = "step",
    src_col: str = "src_node",
    dst_col: str = "dst_node",
    alert_col: str = "is_sar",
) -> "pd_lib.DataFrame":
    """Build temporal relay edges via self-join on relay node.

    A temporal edge (tx_A -> tx_B) exists when:
      - tx_A.dst_node == tx_B.src_node   (relay intermediary)
      - 0 < tx_B.step - tx_A.step <= delta_w   (temporal proximity)

    Parameters
    ----------
    df : DataFrame
        Window of transactions with canonical columns.
    delta_w : int
        Maximum step gap for temporal edge (default: 5).

    Returns
    -------
    DataFrame with columns:
        src_1, dst_1, step_1, amount_1, alert_1,
        src_2, dst_2, step_2, amount_2, alert_2
    """
    # Prepare left (tx_1) and right (tx_2) sides
    left = df[[src_col, dst_col, step_col, amount_col, alert_col]].copy()
    left.columns = ["src_1", "dst_1", "step_1", "amount_1", "alert_1"]

    right = df[[src_col, dst_col, step_col, amount_col, alert_col]].copy()
    right.columns = ["src_2", "dst_2", "step_2", "amount_2", "alert_2"]

    # Join on relay: dst_1 == src_2
    merged = left.merge(right, left_on="dst_1", right_on="src_2", how="inner")

    # Temporal constraint: 0 < step_2 - step_1 <= delta_w
    time_gap = merged["step_2"] - merged["step_1"]
    mask = (time_gap > 0) & (time_gap <= delta_w)
    temporal_edges = merged[mask].reset_index(drop=True)

    return temporal_edges
