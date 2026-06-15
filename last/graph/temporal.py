"""
temporal.py — Temporal edge construction for AML motif mining.

Two outputs are produced from a window DataFrame:

1. build_temporal_edges()
   Self-join on relay node (dst_1 == src_2) within delta_w steps.
   Used by motif mining (fan-in, fan-out, relay, cycle seed detection).

2. build_snapshot_edges()
   Aggregate all raw transactions in a window into a directed weighted
   edge table (one row per unique src→dst pair).
   Used by community detection as the input to sparse adjacency construction.

Guarantees:
- Time      : step_2 > step_1 always enforced; output sorted by step_1.
- Direction : src_node → dst_node preserved throughout; never symmetrized.
- Memory    : no full DataFrame copy; column rename done on lightweight slices.
"""

from __future__ import annotations

try:
    import cudf as pd_lib
    _GPU = True
except ImportError:
    import pandas as pd_lib  # type: ignore
    _GPU = False


# ---------------------------------------------------------------------------
# Temporal relay edges (motif mining input)
# ---------------------------------------------------------------------------

def build_temporal_edges(
    df: "pd_lib.DataFrame",
    delta_w: int = 5,
    src_col: str = "src_node",
    dst_col: str = "dst_node",
    step_col: str = "step",
    amount_col: str = "amount",
    alert_col: str = "is_sar",
) -> "pd_lib.DataFrame":
    """
    Build temporal relay edges via self-join on an intermediary node.

    A temporal edge (tx_A → tx_B) is formed when:
      - tx_A.dst_node == tx_B.src_node   (relay: money passes through a node)
      - 0 < tx_B.step - tx_A.step ≤ delta_w  (temporal proximity)

    This captures the core relay structure needed for motif mining:
    fan-in, fan-out, relay chain, and cycle seeds all use this join.

    Parameters
    ----------
    df : DataFrame
        Single time-window of transactions.  Must have src_col, dst_col,
        step_col, amount_col.  alert_col is optional; filled with 0 if absent.
    delta_w : int
        Maximum step gap allowed between two linked transactions.
    src_col, dst_col, step_col, amount_col, alert_col : str
        Column names in df.

    Returns
    -------
    DataFrame with columns:
        src_1, dst_1, step_1, amount_1, alert_1,
        src_2, dst_2, step_2, amount_2, alert_2
    Sorted by step_1 ascending (temporal order preserved).

    Notes
    -----
    - No full copy of df is made; only the required columns are selected.
    - The join can be large for dense windows; caller should use bounded
      windows (iter_windows) to keep memory controlled.
    """
    # Guard: alert column may be absent in unlabelled data
    has_alert = alert_col in df.columns
    cols = [src_col, dst_col, step_col, amount_col]
    if has_alert:
        cols.append(alert_col)

    # Lightweight slice — no copy, just a column selection
    base = df[cols]

    # Left side: outgoing transaction (tx_A)
    left = base.rename(columns={
        src_col:    "src_1",
        dst_col:    "dst_1",
        step_col:   "step_1",
        amount_col: "amount_1",
        **({alert_col: "alert_1"} if has_alert else {}),
    })

    # Right side: incoming transaction (tx_B)
    right = base.rename(columns={
        src_col:    "src_2",
        dst_col:    "dst_2",
        step_col:   "step_2",
        amount_col: "amount_2",
        **({alert_col: "alert_2"} if has_alert else {}),
    })

    # Self-join: relay node is dst_1 == src_2
    merged = left.merge(right, left_on="dst_1", right_on="src_2", how="inner")

    # Temporal constraint: strict forward order within delta_w steps
    time_gap = merged["step_2"] - merged["step_1"]
    mask = (time_gap > 0) & (time_gap <= delta_w)
    temporal_edges = merged[mask].reset_index(drop=True)

    # Fill missing alert columns with 0 if unlabelled
    if not has_alert:
        temporal_edges["alert_1"] = 0
        temporal_edges["alert_2"] = 0

    # Sort by step_1 to preserve temporal order for downstream motif search
    temporal_edges = temporal_edges.sort_values("step_1").reset_index(drop=True)

    return temporal_edges


# ---------------------------------------------------------------------------
# Snapshot edge table (community detection input)
# ---------------------------------------------------------------------------

def build_snapshot_edges(
    df: "pd_lib.DataFrame",
    src_col: str = "src_node",
    dst_col: str = "dst_node",
    amount_col: str = "amount",
    step_col: str = "step",
) -> "pd_lib.DataFrame":
    """
    Aggregate raw transactions into a directed weighted edge table.

    Groups all transactions in the window by (src_node, dst_node) and sums
    amounts.  This is the input required by build_snapshot_graph() and by
    community detection algorithms.

    Per graph_schema.md §6.2:
        W^(t)_{uv} = sum of all amounts from u to v within the window.

    Parameters
    ----------
    df : DataFrame
        Single time-window of transactions.  Must have src_col, dst_col,
        amount_col.  step_col is used to record the window's step range.
    src_col, dst_col, amount_col, step_col : str
        Column names in df.

    Returns
    -------
    DataFrame with columns:
        src_node, dst_node, weight, tx_count, step_min, step_max

    Notes
    -----
    - Direction is preserved: (u, v) and (v, u) remain separate rows.
    - No symmetrization is applied.
    - Self-loops should have been removed upstream by load_transactions().
    """
    agg = df.groupby([src_col, dst_col], as_index=False).agg(
        weight=(amount_col, "sum"),
        tx_count=(amount_col, "count"),
        step_min=(step_col, "min"),
        step_max=(step_col, "max"),
    )

    # Normalize column names to canonical form
    agg = agg.rename(columns={src_col: "src_node", dst_col: "dst_node"})

    return agg.reset_index(drop=True)
