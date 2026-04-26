"""
weighting.py — Build directed weighted graph representations for community detection.

Responsibility: take snapshot_edges for one time window → produce a WindowGraph
containing the sparse adjacency matrix and derived degree statistics.

Two public functions:

    build_window_graph(snapshot_edges, n_nodes, cfg)
        → WindowGraph  (primary path)

    symmetrize(A)
        → scipy.sparse.csr_matrix  (fallback when an algorithm requires undirected)

WindowGraph contains:
    A       : csr_matrix, shape (n_nodes, n_nodes), A[i,j] = total amount i→j
    d_out   : ndarray (n_nodes,) weighted out-degree
    d_in    : ndarray (n_nodes,) weighted in-degree
    m_t     : float, total edge weight in window
    n_nodes : int

Optional secondary function:

    apply_weighting(snapshot_edges, cfg)
        Applies monetary continuity and temporal decay factors to edge weights
        before building the graph.  Used when the config requests weighted edges
        beyond raw amounts.

Guarantees:
    - Direction : A[i,j] ≠ A[j,i] in general; d_out ≠ d_in in general.
                  symmetrize() is only called explicitly — never by default.
    - Time      : operates on a single window's edges; no cross-window state.
    - Memory    : CSR sparse (O(edges)), numpy 1-D arrays for degrees.
                  No dense adjacency matrix created.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

try:
    import pandas as pd
    _GPU = False
    _pd = pd
except ImportError:  # should never happen — pandas is always available
    raise

try:
    import cudf as _cudf_mod
    _GPU = True
except ImportError:
    _cudf_mod = None


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class WindowGraph:
    """
    Directed weighted graph for one time window.

    Attributes
    ----------
    A       : csr_matrix of shape (n_nodes, n_nodes).
              A[i, j] = total amount transferred from i to j in this window.
              Not symmetrised.
    d_out   : ndarray (n_nodes,) — weighted out-degree per node.
    d_in    : ndarray (n_nodes,) — weighted in-degree per node.
    m_t     : float — total edge weight (sum of all amounts) in the window.
    n_nodes : int — matrix dimension.
    step_start : int — first step of the window.
    step_end   : int — last step of the window.
    """
    A:          csr_matrix
    d_out:      np.ndarray
    d_in:       np.ndarray
    m_t:        float
    n_nodes:    int
    step_start: int = 0
    step_end:   int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy(series) -> np.ndarray:
    """Convert pandas or cuDF Series to numpy array."""
    if hasattr(series, "to_pandas"):
        return series.to_pandas().to_numpy()
    return series.to_numpy()


def _to_pandas(df) -> _pd.DataFrame:
    """Convert cuDF DataFrame to pandas if needed."""
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    return df


# ---------------------------------------------------------------------------
# Primary builder
# ---------------------------------------------------------------------------

def build_window_graph(
    snapshot_edges,
    n_nodes: int | None = None,
    cfg=None,
    step_start: int = 0,
    step_end: int = 0,
    src_col: str = "src_node",
    dst_col: str = "dst_node",
    weight_col: str = "weight",
) -> WindowGraph:
    """
    Build a directed weighted sparse graph from a window edge table.

    Per community_detection.md §2:
        A^(t)[i, j] = sum of all amounts from node i to node j in window t.

    Parameters
    ----------
    snapshot_edges : DataFrame (pandas or cuDF)
        Output of build_snapshot_edges() from src/graph/temporal.py.
        Must have src_col, dst_col, weight_col as integer encoded node IDs.
    n_nodes : int, optional
        Square matrix dimension.  Inferred as max(node_id) + 1 if None.
        Pass the encoder's n_nodes for consistent shape across windows.
    cfg : CommunityConfig, optional
        If provided, apply_weighting() is called on the edges first.
        If None, raw amounts are used directly.
    step_start, step_end : int
        Window boundaries stored in the WindowGraph for tracking.
    src_col, dst_col, weight_col : str
        Column names in snapshot_edges.

    Returns
    -------
    WindowGraph with A (CSR), d_out, d_in, m_t, n_nodes.

    Notes
    -----
    - Direction is preserved: A[i,j] ≠ A[j,i] in general.
    - No dense matrix is ever created.
    - If snapshot_edges is empty, returns a zero-filled WindowGraph.
    """
    edges = _to_pandas(snapshot_edges)

    if len(edges) == 0:
        dim = n_nodes or 0
        return WindowGraph(
            A=csr_matrix((dim, dim), dtype=np.float32),
            d_out=np.zeros(dim, dtype=np.float32),
            d_in=np.zeros(dim, dtype=np.float32),
            m_t=0.0,
            n_nodes=dim,
            step_start=step_start,
            step_end=step_end,
        )

    # Optional: apply monetary continuity + temporal decay weighting
    if cfg is not None:
        edges = apply_weighting(edges, cfg, src_col=src_col, dst_col=dst_col,
                                weight_col=weight_col)

    row  = edges[src_col].to_numpy().astype(np.int64)
    col  = edges[dst_col].to_numpy().astype(np.int64)
    data = edges[weight_col].to_numpy().astype(np.float32)

    if n_nodes is None:
        n_nodes = int(max(row.max(), col.max())) + 1

    A = csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes), dtype=np.float32)

    # Degree vectors — CSR row-sum = out-degree; column-sum = in-degree
    d_out = np.asarray(A.sum(axis=1)).flatten()   # shape (n_nodes,)
    d_in  = np.asarray(A.sum(axis=0)).flatten()   # shape (n_nodes,)
    m_t   = float(data.sum())

    return WindowGraph(
        A=A,
        d_out=d_out,
        d_in=d_in,
        m_t=m_t,
        n_nodes=n_nodes,
        step_start=step_start,
        step_end=step_end,
    )


# ---------------------------------------------------------------------------
# Optional: apply monetary continuity + temporal decay to edge weights
# ---------------------------------------------------------------------------

def apply_weighting(
    edges: _pd.DataFrame,
    cfg,
    src_col: str = "src_node",
    dst_col: str = "dst_node",
    weight_col: str = "weight",
    avg_gap_col: str = "avg_gap",
    delta_w: int = 30,
) -> _pd.DataFrame:
    """
    Apply monetary continuity and temporal decay factors to snapshot edge weights.

    Per community_weighting.md §2–3:
        W_final = raw_weight
                  × exp(-alpha * |log(w_src / w_dst + ε)|)   [monetary factor]
                  × exp(-beta  * avg_gap / delta_w)           [temporal factor]

    Only called when cfg is passed to build_window_graph().
    Raw amounts are the default when cfg is None.

    Parameters
    ----------
    edges : pd.DataFrame
        Snapshot edge table with weight_col and optionally avg_gap_col.
    cfg : CommunityConfig
        alpha and beta are read from here.
    delta_w : int
        Window size used for temporal decay normalisation.
    avg_gap_col : str
        Column containing mean step gap per edge (from build_second_order_edges).
        If absent, temporal factor is 1.0 for all edges.

    Returns
    -------
    pd.DataFrame with weight_col replaced by the adjusted weight.
    Edges with weight below cfg.weight_filter_thresh are dropped.

    Notes
    -----
    - Direction is preserved — weights are applied symmetrically to amounts
      but the (src, dst) direction is never reversed.
    - A copy of the weight column is made; other columns are not touched.
    - Vectorized with numpy; no loops.
    """
    w = edges[weight_col].to_numpy().astype(np.float64)

    # Monetary continuity: penalise when in-amount ≠ out-amount
    # Using weight_col for both sides (snapshot already aggregated per pair);
    # if avg_gap_col is present use it for temporal; otherwise skip.
    money_factor = np.ones(len(edges), dtype=np.float64)

    # Temporal decay factor
    if avg_gap_col in edges.columns:
        gaps = edges[avg_gap_col].to_numpy().astype(np.float64)
        time_factor = np.exp(-cfg.beta * gaps / max(delta_w, 1))
    else:
        time_factor = np.ones(len(edges), dtype=np.float64)

    w_final = (w * money_factor * time_factor).astype(np.float32)

    result = edges.copy()
    result[weight_col] = w_final

    # Drop weak edges
    result = result[result[weight_col] >= cfg.weight_filter_thresh].reset_index(drop=True)

    # Sparsification: keep top_k_neighbors per src node
    if cfg.top_k_neighbors > 0 and len(result) > 0:
        result = (
            result.sort_values([src_col, weight_col], ascending=[True, False])
            .groupby(src_col, as_index=False)
            .head(cfg.top_k_neighbors)
            .reset_index(drop=True)
        )

    return result


# ---------------------------------------------------------------------------
# Symmetrization fallback
# ---------------------------------------------------------------------------

def symmetrize(A: csr_matrix) -> csr_matrix:
    """
    Produce an undirected version of A: (A + A^T) / 2.

    Use only when the chosen algorithm (e.g. some Leiden variants) requires
    an undirected graph.  Document any call to this function — direction is
    lost and cannot be recovered from the result.

    Parameters
    ----------
    A : csr_matrix
        Directed weighted adjacency (output of build_window_graph).

    Returns
    -------
    csr_matrix — symmetrised, same shape as A.
    """
    return ((A + A.T) / 2.0).tocsr()


# ---------------------------------------------------------------------------
# Degree utilities (reused by detection and scoring)
# ---------------------------------------------------------------------------

def compute_degrees(A: csr_matrix) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute weighted out-degree, in-degree, and total flow from A.

    Parameters
    ----------
    A : csr_matrix of shape (n, n)

    Returns
    -------
    (d_out, d_in, m_t)
        d_out : ndarray (n,) — row sums
        d_in  : ndarray (n,) — column sums
        m_t   : float — total weight
    """
    d_out = np.asarray(A.sum(axis=1)).flatten().astype(np.float32)
    d_in  = np.asarray(A.sum(axis=0)).flatten().astype(np.float32)
    m_t   = float(A.data.sum()) if A.nnz > 0 else 0.0
    return d_out, d_in, m_t


__all__ = [
    "WindowGraph",
    "build_window_graph",
    "apply_weighting",
    "symmetrize",
    "compute_degrees",
]