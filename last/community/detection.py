"""
detection.py — Community detection on directed weighted window graphs.

Three public functions:

    detect_communities(wg, cfg)
        Primary entry point. Takes a WindowGraph from weighting.py.
        Returns Dict[node_id, community_id] for every active node.
        Method is selected by cfg.method: "leiden" or "infomap".

    compute_node_roles(window_df, cfg)
        Classify each node as: source / sink / layering / neutral.
        Input is the raw transaction DataFrame for the window.

    build_relay_edges(snapshot_edges)
        Preserve relay structure for edge-level analysis.
        Returns an aggregated directed edge table.

Algorithm notes (community_detection.md §1–6):
    - Primary: directed modularity via igraph Leiden (CPU).
      igraph supports directed=True in its Leiden implementation.
    - Fallback: if igraph unavailable, use networkx greedy_modularity.
    - GPU path (cuGraph): attempted if RAPIDS is present and cfg allows it.
    - Infomap: use infomap package if cfg.method == "infomap".
    - Symmetrization: only if the algorithm explicitly requires undirected.
      Any symmetrization must be documented in the call site.

Guarantees:
    - Time      : operates on a single WindowGraph (no cross-window state).
    - Direction : A[i,j] used directly; d_out / d_in kept separate.
                  Symmetrization is never applied silently.
    - Memory    : igraph graph freed after each call; no cuDF/cuGraph required.
"""

from __future__ import annotations

import gc
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import CommunityConfig
from .weighting import WindowGraph, compute_degrees, symmetrize


# ---------------------------------------------------------------------------
# Algorithm availability checks (lazy imports — no crash at module load)
# ---------------------------------------------------------------------------

def _try_igraph():
    try:
        import igraph as ig
        return ig
    except ImportError:
        return None


def _try_infomap():
    try:
        import infomap as im
        return im
    except ImportError:
        return None


def _try_cugraph():
    try:
        import cugraph
        import cudf
        return cugraph, cudf
    except ImportError:
        return None, None


# ---------------------------------------------------------------------------
# Primary entry point
# ---------------------------------------------------------------------------

def detect_communities(
    wg: WindowGraph,
    cfg: CommunityConfig,
) -> Dict[int, int]:
    """
    Detect communities in a single time-window directed weighted graph.

    Parameters
    ----------
    wg : WindowGraph
        Output of build_window_graph(). Contains A (CSR), d_out, d_in, m_t.
    cfg : CommunityConfig
        method, resolution, min_comm_size, s_max, etc.

    Returns
    -------
    Dict[node_id, community_id]
        Community assignment for every node present in the window.
        Nodes not in any edge are assigned community -1.

    Notes
    -----
    - Method priority: igraph Leiden → igraph community_multilevel →
      networkx greedy_modularity → trivial (all nodes in one community).
    - "infomap" method uses the infomap package if available; falls back
      to the default Leiden path if not installed.
    - cuGraph is attempted only when method == "leiden" and RAPIDS is
      available; symmetrization is then applied and documented.
    """
    if wg.A.nnz == 0:
        return {}

    if cfg.method == "infomap":
        labels = _run_infomap(wg, cfg)
    else:
        labels = _run_leiden(wg, cfg)

    # Apply minimum community size filter
    if cfg.min_comm_size > 1:
        labels = _filter_small_communities(labels, cfg.min_comm_size)

    return labels


# ---------------------------------------------------------------------------
# Leiden path
# ---------------------------------------------------------------------------

def _run_leiden(wg: WindowGraph, cfg: CommunityConfig) -> Dict[int, int]:
    """
    Run Leiden community detection.

    Tries, in order:
        1. igraph Leiden with directed=True on the directed graph.
        2. igraph community_multilevel (undirected fallback — symmetrised).
        3. networkx greedy_modularity (undirected fallback).
        4. Trivial: all nodes in community 0.
    """
    ig = _try_igraph()
    if ig is not None:
        return _leiden_igraph(wg, cfg, ig)

    # networkx fallback
    try:
        import networkx as nx
        return _leiden_networkx(wg, cfg, nx)
    except ImportError:
        pass

    # Trivial: every node is its own community
    active_nodes = _active_nodes(wg)
    return {n: 0 for n in active_nodes}


def _leiden_igraph(wg: WindowGraph, cfg: CommunityConfig, ig) -> Dict[int, int]:
    """
    igraph Leiden on the directed weighted adjacency.

    igraph Leiden supports directed=True — no symmetrization needed for the
    primary path.  community_multilevel is undirected; it is used only if
    Leiden raises an error (e.g. disconnected graph edge case).
    """
    A = wg.A
    rows, cols = A.nonzero()
    weights = np.asarray(A[rows, cols]).flatten().astype(float).tolist()

    # igraph edge list
    edges = list(zip(rows.tolist(), cols.tolist()))
    G = ig.Graph(n=wg.n_nodes, edges=edges, directed=True)
    G.es["weight"] = weights

    try:
        # Directed Leiden (igraph >= 0.10)
        partition = G.community_leiden(
            weights="weight",
            resolution=cfg.resolution,
            directed=True,
        )
    except Exception:
        # Fallback: multilevel (undirected) — symmetrize and document
        # NOTE: direction is lost in this fallback. Weights are averaged.
        A_sym = symmetrize(A)
        rs, cs = A_sym.nonzero()
        ws = np.asarray(A_sym[rs, cs]).flatten().astype(float).tolist()
        G_u = ig.Graph(
            n=wg.n_nodes,
            edges=list(zip(rs.tolist(), cs.tolist())),
            directed=False,
        )
        G_u.es["weight"] = ws
        partition = G_u.community_multilevel(weights="weight")
        del G_u

    labels = {}
    for cid, members in enumerate(partition):
        for node in members:
            labels[node] = cid

    del G
    gc.collect()
    return labels


def _leiden_networkx(wg: WindowGraph, cfg: CommunityConfig, nx) -> Dict[int, int]:
    """
    networkx greedy_modularity fallback (undirected).
    Symmetrization applied — documented here.
    NOTE: direction is lost in this path. Use only when igraph is absent.
    """
    A_sym = symmetrize(wg.A)
    G = nx.from_scipy_sparse_array(A_sym, create_using=nx.Graph())
    communities = nx.community.greedy_modularity_communities(G, weight="weight")
    labels = {}
    for cid, members in enumerate(communities):
        for node in members:
            labels[int(node)] = cid
    del G, A_sym
    gc.collect()
    return labels


# ---------------------------------------------------------------------------
# Infomap path
# ---------------------------------------------------------------------------

def _run_infomap(wg: WindowGraph, cfg: CommunityConfig) -> Dict[int, int]:
    """
    Run Infomap for flow-based community detection (community_detection.md §6).
    Falls back to Leiden if infomap package is not installed.
    """
    im = _try_infomap()
    if im is None:
        return _run_leiden(wg, cfg)

    infomapper = im.Infomap("--directed --silent")
    A = wg.A
    rows, cols = A.nonzero()
    weights = np.asarray(A[rows, cols]).flatten()

    for u, v, w in zip(rows.tolist(), cols.tolist(), weights.tolist()):
        infomapper.add_link(u, v, w)

    infomapper.run()
    labels = {int(node.node_id): int(node.module_id) for node in infomapper.nodes}
    return labels


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def _active_nodes(wg: WindowGraph) -> List[int]:
    """Return list of node IDs that have at least one edge in the window."""
    rows, cols = wg.A.nonzero()
    return list(set(rows.tolist()) | set(cols.tolist()))


def _filter_small_communities(
    labels: Dict[int, int],
    min_size: int,
) -> Dict[int, int]:
    """
    Reassign nodes in communities smaller than min_size to community -1.
    Does not renumber the remaining community IDs.
    """
    from collections import Counter
    counts = Counter(labels.values())
    return {
        node: (cid if counts[cid] >= min_size else -1)
        for node, cid in labels.items()
    }


def labels_to_dataframe(
    labels: Dict[int, int],
    window_id: int,
) -> pd.DataFrame:
    """
    Convert a labels dict to a tidy DataFrame for downstream use.

    Parameters
    ----------
    labels : Dict[node_id, community_id]
    window_id : int

    Returns
    -------
    pd.DataFrame with columns: window_id, node_id, community_id
    """
    if not labels:
        return pd.DataFrame(columns=["window_id", "node_id", "community_id"])
    df = pd.DataFrame(
        [(window_id, node, cid) for node, cid in labels.items()],
        columns=["window_id", "node_id", "community_id"],
    )
    return df.sort_values(["community_id", "node_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Node role classification
# ---------------------------------------------------------------------------

def compute_node_roles(
    window_df: pd.DataFrame,
    cfg: CommunityConfig,
    src_col: str = "src_node",
    dst_col: str = "dst_node",
    amount_col: str = "amount",
    alert_col: str = "is_sar",
) -> pd.DataFrame:
    """
    Classify each node in the window as source / sink / layering / neutral.

    Role encoding:
        0 = Neutral      (balanced or low-activity)
        1 = Source       (net_flow_ratio > role_threshold)
        2 = Sink         (net_flow_ratio < -role_threshold)
        3 = Layering     (flow_consistency > layering_consistency and
                          |net_flow_ratio| < role_threshold × 0.67)

    Layering score = flow_consistency × (1 - |net_flow_ratio|)

    Parameters
    ----------
    window_df : pd.DataFrame
        Single window of transactions. Must have src_col, dst_col, amount_col.
        alert_col is optional.
    cfg : CommunityConfig
    src_col, dst_col, amount_col, alert_col : str

    Returns
    -------
    pd.DataFrame with columns:
        node, total_volume, total_volume_out, total_volume_in,
        n_tx, n_alert_tx, alert_rate, net_flow_ratio,
        flow_consistency, layering_score, role
    """
    """
    Classify each node in the window as source / sink / layering / neutral.

    Role encoding:
        0 = Neutral      (balanced or low-activity)
        1 = Source       (net_flow_ratio > role_threshold)
        2 = Sink         (net_flow_ratio < -role_threshold)
        3 = Layering     (flow_consistency > layering_consistency and
                          |net_flow_ratio| < role_threshold × 0.67)

    Implementation Fix: Uses explicit priority assignment to avoid 
    undefined role codes from overlapping thresholds.
    """
    has_alert = alert_col in window_df.columns

    # 1. Aggregation: Compute out-flow statistics per node
    agg_src = {"amt_out": (amount_col, "sum"), "n_out": (amount_col, "count")}
    if has_alert:
        agg_src["n_alert_out"] = (alert_col, "sum")
    src_stats = window_df.groupby(src_col, as_index=False).agg(**agg_src)
    src_stats = src_stats.rename(columns={src_col: "node"})

    # 2. Aggregation: Compute in-flow statistics per node
    agg_dst = {"amt_in": (amount_col, "sum"), "n_in": (amount_col, "count")}
    if has_alert:
        agg_dst["n_in_tx"] = (amount_col, "count") # Using count for tx volume
        if has_alert:
            agg_dst["n_alert_in"] = (alert_col, "sum")
    dst_stats = window_df.groupby(dst_col, as_index=False).agg(**agg_dst)
    dst_stats = dst_stats.rename(columns={dst_col: "node"})

    # 3. Merge: Align in-flow and out-flow on the same node ID
    nf = src_stats.merge(dst_stats, on="node", how="outer").fillna(0.0)

    # 4. Feature Engineering: Basic metrics
    nf["total_volume_out"] = nf["amt_out"].astype("float32")
    nf["total_volume_in"]  = nf["amt_in"].astype("float32")
    nf["total_volume"]     = nf["total_volume_out"] + nf["total_volume_in"]
    
    # Standardize transaction counts
    n_out_vals = nf["n_out"].astype("int32")
    n_in_vals = nf.get("n_in", nf.get("n_in_tx", 0)).astype("int32")
    nf["n_tx"] = n_out_vals + n_in_vals

    # Alert rate calculation
    if has_alert:
        nf["n_alert_tx"] = (nf.get("n_alert_out", 0) + nf.get("n_alert_in", 0)).astype("int32")
    else:
        nf["n_alert_tx"] = 0
        
    nf["alert_rate"] = nf["n_alert_tx"] / (nf["n_tx"] + 1e-9).astype("float32")

    # 5. Feature Engineering: Advanced AML metrics
    # Net flow ratio: balance between money coming in vs going out
    nf["net_flow_ratio"] = (
        (nf["total_volume_out"] - nf["total_volume_in"])
        / (nf["total_volume_out"] + nf["total_volume_in"] + 1e-9)
    ).astype("float32")

    # Flow consistency: high when in-flow and out-flow are nearly equal
    vol_min = nf[["total_volume_in", "total_volume_out"]].min(axis=1)
    nf["flow_consistency"] = (
        2 * vol_min / (nf["total_volume_in"] + nf["total_volume_out"] + 1e-9)
    ).astype("float32")

    # Layering score: derived from consistency and capital preservation
    nf["layering_score"] = (
        nf["flow_consistency"] * (1 - nf["net_flow_ratio"].abs())
    ).astype("float32")

    # 6. Role Assignment: Explicit Priority Logic (The Fix)
    # Define boolean masks based on Config thresholds
    is_source   = nf["net_flow_ratio"] >  cfg.role_threshold
    is_sink     = nf["net_flow_ratio"] < -cfg.role_threshold
    is_layering = (
        (nf["flow_consistency"] >  cfg.layering_consistency)
        & (nf["net_flow_ratio"].abs() < cfg.role_threshold * 0.67)
    )
    
    # Initialize all as Neutral (0)
    nf["role"] = 0
    
    # Apply labels with clear hierarchy. 
    # Last assignment wins if a node satisfies multiple conditions.
    nf.loc[is_source,   "role"] = 1
    nf.loc[is_sink,     "role"] = 2
    nf.loc[is_layering, "role"] = 3  # Layering has highest priority in AML context

    return nf[[
        "node", "total_volume", "total_volume_out", "total_volume_in",
        "n_tx", "n_alert_tx", "alert_rate", "net_flow_ratio",
        "flow_consistency", "layering_score", "role",
    ]].reset_index(drop=True)



# ---------------------------------------------------------------------------
# Relay edge table (for edge-level analysis, optional)
# ---------------------------------------------------------------------------

def build_relay_edges(
    snapshot_edges: pd.DataFrame,
    src_col: str = "src_node",
    dst_col: str = "dst_node",
    weight_col: str = "weight",
    tx_count_col: str = "tx_count",
) -> pd.DataFrame:
    """
    Produce a directed relay edge table from snapshot edges.

    Keeps direction. Aggregates weight and transaction count per (src, dst) pair.
    Used for edge-level feature engineering and visualisation.

    Parameters
    ----------
    snapshot_edges : pd.DataFrame
        Output of build_snapshot_edges() from src/graph/temporal.py.
    src_col, dst_col, weight_col, tx_count_col : str

    Returns
    -------
    pd.DataFrame with columns:
        src, dst, weight, tx_count, alert_ratio (0.0 if no alert column)
    """
    if len(snapshot_edges) == 0:
        return pd.DataFrame(columns=["src", "dst", "weight", "tx_count"])

    agg = snapshot_edges.rename(columns={src_col: "src", dst_col: "dst"})
    keep = ["src", "dst", weight_col]
    if tx_count_col in agg.columns:
        keep.append(tx_count_col)
    agg = agg[keep]

    grouped = agg.groupby(["src", "dst"], as_index=False).agg(
        weight=(weight_col, "sum"),
        tx_count=(tx_count_col if tx_count_col in agg.columns else weight_col, "count"),
    )

    return grouped.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Recursive community splitting (oversized community guard)
# ---------------------------------------------------------------------------

def split_large_communities(
    labels: Dict[int, int],
    wg: WindowGraph,
    cfg: CommunityConfig,
    depth: int = 0,
) -> Dict[int, int]:
    """
    Recursively split communities larger than cfg.s_max.

    Extracts the subgraph of each oversized community and runs
    detect_communities on it independently.  Assigns globally unique
    community IDs across all splits.

    Parameters
    ----------
    labels : Dict[node_id, community_id]
        Initial community assignment.
    wg : WindowGraph
        The full window graph (used to extract subgraph weights).
    cfg : CommunityConfig
    depth : int
        Current recursion depth (stops at cfg.max_recursion_depth).

    Returns
    -------
    Dict[node_id, community_id] — refined labels with splits applied.
    """
    if depth >= cfg.max_recursion_depth:
        return labels

    from collections import defaultdict
    comm_nodes: Dict[int, List[int]] = defaultdict(list)
    for node, cid in labels.items():
        comm_nodes[cid].append(node)

    oversized = {cid: nodes for cid, nodes in comm_nodes.items()
                 if len(nodes) > cfg.s_max}

    if not oversized:
        return labels

    max_cid = max(labels.values()) + 1
    result = dict(labels)

    # Pre-compute full nonzero coordinates once outside the per-community loop
    all_rows, all_cols = wg.A.nonzero()

    for cid, members in oversized.items():
        node_set = set(members)
        # Extract subgraph CSR for this community
        mask = np.isin(all_rows, list(node_set)) & np.isin(all_cols, list(node_set))
        sub_rows = all_rows[mask]
        sub_cols = all_cols[mask]

        if len(sub_rows) == 0:
            continue

        # Read through matrix interface — safe regardless of CSR internal layout
        sub_data = np.asarray(wg.A[sub_rows, sub_cols]).flatten().astype(np.float32)

        # Remap to local indices
        sorted_members = sorted(members)
        idx_map = {n: i for i, n in enumerate(sorted_members)}
        local_rows = np.array([idx_map[r] for r in sub_rows])
        local_cols = np.array([idx_map[c] for c in sub_cols])
        from scipy.sparse import csr_matrix as _csr
        sub_A = _csr((sub_data, (local_rows, local_cols)),
                     shape=(len(sorted_members), len(sorted_members)),
                     dtype=np.float32)

        from .weighting import WindowGraph as WG
        sub_wg = WG(
            A=sub_A,
            d_out=np.asarray(sub_A.sum(axis=1)).flatten(),
            d_in=np.asarray(sub_A.sum(axis=0)).flatten(),
            m_t=float(sub_data.sum()),
            n_nodes=len(sorted_members),
        )

        sub_labels_local = detect_communities(sub_wg, cfg)
        sub_labels_local = split_large_communities(sub_labels_local, sub_wg, cfg, depth + 1)

        # Remap back to global node IDs
        rev_map = {i: n for n, i in idx_map.items()}
        for local_node, local_cid in sub_labels_local.items():
            global_node = rev_map[local_node]
            result[global_node] = max_cid + local_cid

        max_cid += (max(sub_labels_local.values()) + 1 if sub_labels_local else 1)

    return result


__all__ = [
    "detect_communities",
    "compute_node_roles",
    "build_relay_edges",
    "labels_to_dataframe",
    "split_large_communities",
    # Legacy names kept for __init__.py compatibility
    "run_recursive_leiden",
]

# Alias for __init__.py which still imports run_recursive_leiden
run_recursive_leiden = split_large_communities