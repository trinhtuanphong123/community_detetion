"""
second_order.py — Second-order (line graph) construction and sparse adjacency.

Two outputs are produced:

1. build_second_order_edges()
   Aggregates temporal relay pairs (src_1 → relay → dst_2) into a
   second-order edge table.  Captures multi-hop flow intensity and timing.
   Used by community detection to represent structural relay behavior.

2. build_snapshot_graph()
   Converts a snapshot edge table (from build_snapshot_edges) into a
   directed, weighted sparse adjacency matrix A^(t).
   Used as the core input for modularity-based community detection.

Guarantees:
- Time      : avg_time_gap preserves temporal signal; step range recorded.
- Direction : src_2nd → dst_2nd is the u→w relay direction (not reversed).
- Memory    : no full DataFrame copy; scipy.sparse CSR used (not dense matrix).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

try:
    import cudf as pd_lib
    _GPU = True
except ImportError:
    import pandas as pd_lib  # type: ignore
    _GPU = False


# ---------------------------------------------------------------------------
# Second-order edge table (multi-hop relay aggregation)
# ---------------------------------------------------------------------------

def build_second_order_edges(
    temporal_edges: "pd_lib.DataFrame",
) -> "pd_lib.DataFrame":
    """
    Aggregate temporal relay edges into second-order (u → w) relay edges.

    Groups by (src_1, dst_2) — the origin and final destination of a relay
    chain — and computes aggregate statistics over all relay hops between them.

    Parameters
    ----------
    temporal_edges : DataFrame
        Output of build_temporal_edges().  Expected columns:
        src_1, dst_1, step_1, amount_1, alert_1,
        src_2, dst_2, step_2, amount_2, alert_2.

    Returns
    -------
    DataFrame with columns:
        src_2nd     : origin node of the relay chain
        dst_2nd     : final destination of the relay chain
        count       : number of relay hops between this pair
        weight_src  : total amount sent by src (sum of amount_1)
        weight_dst  : total amount received by dst (sum of amount_2)
        avg_gap     : mean step gap across relay hops
        n_alert     : total alert flags across all hops (alert_1 + alert_2)

    Notes
    -----
    - (src_2nd, dst_2nd) preserves direction: u → w, not w → u.
    - Self-relays (src_1 == dst_2) are excluded to avoid trivial loops.
    - Returns an empty DataFrame with correct schema if input is empty.
    """
    _EMPTY_SCHEMA = {
        "src_2nd":    "int64",
        "dst_2nd":    "int64",
        "count":      "int64",
        "weight_src": "float32",
        "weight_dst": "float32",
        "avg_gap":    "float32",
        "n_alert":    "int64",
    }

    if len(temporal_edges) == 0:
        return pd_lib.DataFrame(
            {col: pd_lib.Series(dtype=dtype) for col, dtype in _EMPTY_SCHEMA.items()}
        )

    te = temporal_edges

    # Exclude self-relays: src_1 == dst_2 would be a trivial loop
    te = te[te["src_1"] != te["dst_2"]]

    if len(te) == 0:
        return pd_lib.DataFrame(
            {col: pd_lib.Series(dtype=dtype) for col, dtype in _EMPTY_SCHEMA.items()}
        )

    # Compute time gap and combined alert in-place (no copy)
    te = te.copy()  # single copy here to safely assign new columns
    te["_gap"] = te["step_2"] - te["step_1"]
    te["_n_alert"] = te["alert_1"] + te["alert_2"]

    grouped = te.groupby(["src_1", "dst_2"], as_index=False).agg(
        count=("_gap", "count"),
        weight_src=("amount_1", "sum"),
        weight_dst=("amount_2", "sum"),
        avg_gap=("_gap", "mean"),
        n_alert=("_n_alert", "sum"),
    )

    grouped = grouped.rename(columns={"src_1": "src_2nd", "dst_2": "dst_2nd"})

    # Cast to memory-efficient types
    grouped["weight_src"] = grouped["weight_src"].astype("float32")
    grouped["weight_dst"] = grouped["weight_dst"].astype("float32")
    grouped["avg_gap"] = grouped["avg_gap"].astype("float32")

    return grouped.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sparse directed adjacency matrix (community detection input)
# ---------------------------------------------------------------------------

def build_snapshot_graph(
    snapshot_edges: "pd_lib.DataFrame",
    n_nodes: int | None = None,
    src_col: str = "src_node",
    dst_col: str = "dst_node",
    weight_col: str = "weight",
) -> tuple[csr_matrix, int]:
    """
    Build a directed, weighted sparse adjacency matrix from snapshot edges.

    Per community_spec.md §7.1:
        A^(t)[i, j] = total amount transferred from node i to node j.

    Parameters
    ----------
    snapshot_edges : DataFrame
        Output of build_snapshot_edges().  Must have src_col, dst_col,
        weight_col as integer node IDs (already encoded by NodeEncoder).
    n_nodes : int, optional
        Size of the square matrix.  If None, inferred as max(node_id) + 1.
        Pass explicitly when the encoder's n_nodes is known, to ensure
        consistent matrix shape across windows.
    src_col, dst_col, weight_col : str
        Column names in snapshot_edges.

    Returns
    -------
    (A, n_nodes) where:
        A       : scipy.sparse.csr_matrix of shape (n_nodes, n_nodes).
                  A[u, v] = total weight from u to v.  Not symmetrized.
        n_nodes : int, the dimension used (useful when n_nodes was inferred).

    Notes
    -----
    - Uses CSR format: efficient for row (out-degree) operations.
    - Direction is preserved: A[u,v] ≠ A[v,u] in general.
    - No dense matrix is created; memory scales with number of edges, not nodes².
    - If snapshot_edges is empty, returns a zero matrix of shape (n_nodes, n_nodes).
    """
    if len(snapshot_edges) == 0:
        dim = n_nodes if n_nodes is not None else 0
        return csr_matrix((dim, dim), dtype=np.float32), dim

    # Extract arrays — go through numpy for both GPU and CPU DataFrames
    def _to_np(series):
        if hasattr(series, "to_pandas"):
            return series.to_pandas().to_numpy()
        return series.to_numpy()

    row = _to_np(snapshot_edges[src_col]).astype(np.int64)
    col = _to_np(snapshot_edges[dst_col]).astype(np.int64)
    data = _to_np(snapshot_edges[weight_col]).astype(np.float32)

    if n_nodes is None:
        n_nodes = int(max(row.max(), col.max())) + 1

    A = csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes), dtype=np.float32)
    return A, n_nodes
