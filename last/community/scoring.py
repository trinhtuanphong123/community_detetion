"""
scoring.py — AML-aware community suspicion scoring.

Two public functions:

    extract_community_features(window_df, snap_edges, global_labels, wg, node_roles)
        Compute AML-relevant features for each community in one window.
        Called once per window by pipeline.py — never accumulates all windows.

    score_communities(feature_df, cfg, persistence_map)
        Apply the weighted suspicion formula to the feature table.
        Returns the same DataFrame with added score columns.

Score formula (community_scoring.md §2):
    S(C) = w_internal_flow  * InternalFlow(C)
         + w_reciprocity    * Reciprocity(C)
         + w_persistence    * Persistence(C)
         + w_motif          * MotifEnrichment(C)
         - w_external_noise * ExternalNoise(C)

Feature definitions:
    internal_flow    : total_internal_weight / total_weight (in window)
    reciprocity      : bidirectional_volume / total_internal_volume
    persistence      : n_windows_seen / max_possible_windows (0–1, from tracker)
    motif_enrichment : motif_count_inside / community_size (0 if no motif table)
    external_noise   : total_external_weight / total_weight (1 - internal_flow)

Guarantees:
    - Time      : per-window; no cross-window state inside this module.
    - Direction : A[i,j] used directly for internal/external flow; no symmetrization.
    - Memory    : all operations are vectorized pandas groupby; no cuDF/cupy.
"""

from __future__ import annotations

import gc
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import CommunityConfig
from .weighting import WindowGraph, compute_degrees


# ---------------------------------------------------------------------------
# Per-window feature extraction
# ---------------------------------------------------------------------------

def extract_community_features(
    window_df: pd.DataFrame,
    global_labels: Dict[int, int],
    wg: WindowGraph,
    node_roles: Optional[pd.DataFrame] = None,
    motif_counts: Optional[Dict[int, float]] = None,
    window_id: int = 0,
    step_start: int = 0,
    step_end: int = 0,
    src_col: str = "src_node",
    dst_col: str = "dst_node",
    amount_col: str = "amount",
    alert_col: str = "is_sar",
) -> pd.DataFrame:
    """
    Extract AML community features for one time window.

    Parameters
    ----------
    window_df : pd.DataFrame
        Raw transaction rows for this window. Provides amount and alert info.
    global_labels : Dict[node_id, global_cid]
        Output of match_communities_jaccard() for this window.
    wg : WindowGraph
        Directed weighted adjacency for this window (from build_window_graph).
    node_roles : pd.DataFrame, optional
        Output of compute_node_roles(). Must have columns: node, role,
        flow_consistency, layering_score.
    motif_counts : Dict[global_cid, float], optional
        Motif instance count per community (from motif module integration).
        If None, motif_enrichment is 0 for all communities.
    window_id, step_start, step_end : int
        Window metadata stored in the output.
    src_col, dst_col, amount_col, alert_col : str
        Column names in window_df.

    Returns
    -------
    pd.DataFrame with one row per active community. Columns:
        window_id, step_start, step_end, global_cid,
        size, internal_flow, external_flow, external_noise,
        reciprocity, internal_recirc,
        sink_concentration, source_concentration,
        n_sources, n_sinks, n_layering, avg_layering_score,
        total_volume, n_internal_edges,
        alert_ratio, motif_enrichment, vol_density, edge_density,
        [persistence is added later by score_communities]
    """
    # Filter out noise nodes (-1 community)
    active = {n: c for n, c in global_labels.items() if c != -1}
    if not active:
        return pd.DataFrame()

    node_to_cid = pd.Series(active, name="global_cid")
    node_to_cid.index.name = "node"
    node_to_cid = node_to_cid.reset_index()

    # ── Map edges to communities ─────────────────────────────────────────────
    has_alert = alert_col in window_df.columns
    agg_dict = {
        "weight": (amount_col, "sum"),
        "n_tx": (amount_col, "count"),
    }
    if has_alert:
        agg_dict["n_alert"] = (alert_col, "sum")

    edges = (
        window_df.groupby([src_col, dst_col], as_index=False)
        .agg(**agg_dict)
        .rename(columns={src_col: "src", dst_col: "dst"})
    )

    edges = edges.merge(
        node_to_cid.rename(columns={"node": "src", "global_cid": "src_cid"}),
        on="src", how="inner",
    )
    edges = edges.merge(
        node_to_cid.rename(columns={"node": "dst", "global_cid": "dst_cid"}),
        on="dst", how="inner",
    )

    total_weight = float(edges["weight"].sum()) if len(edges) > 0 else 1.0

    internal = edges[edges["src_cid"] == edges["dst_cid"]].copy()
    internal = internal.rename(columns={"src_cid": "global_cid"}).drop(columns=["dst_cid"])

    external = edges[edges["src_cid"] != edges["dst_cid"]].copy()

    # ── Basic internal features ───────────────────────────────────────────────
    if len(internal) == 0:
        return pd.DataFrame()

    if has_alert:
        internal["is_alert_edge"] = (internal["n_alert"] > 0).astype("float32")
        comm_basic = internal.groupby("global_cid", as_index=False).agg(
            total_volume      =("weight",        "sum"),
            n_internal_edges  =("weight",        "count"),
            alert_ratio       =("is_alert_edge", "mean"),
        )
    else:
        comm_basic = internal.groupby("global_cid", as_index=False).agg(
            total_volume      =("weight",   "sum"),
            n_internal_edges  =("weight",   "count"),
        )
        comm_basic["alert_ratio"] = 0.0

    # ── External flow ─────────────────────────────────────────────────────────
    ext_agg = external.groupby("src_cid", as_index=False).agg(
        ext_volume=("weight", "sum"),
    ).rename(columns={"src_cid": "global_cid"})
    comm_basic = comm_basic.merge(ext_agg, on="global_cid", how="left")
    comm_basic["ext_volume"] = comm_basic["ext_volume"].fillna(0.0)
    comm_basic["internal_flow"]   = (comm_basic["total_volume"] / (total_weight + 1e-9)).astype("float32")
    comm_basic["external_flow"]   = (comm_basic["ext_volume"]   / (total_weight + 1e-9)).astype("float32")
    comm_basic["external_noise"]  = comm_basic["external_flow"]

    # ── Reciprocity (bidirectional volume ratio) ───────────────────────────────
    reverse = internal[["src", "dst", "global_cid", "weight"]].rename(
        columns={"src": "dst_r", "dst": "src_r", "weight": "rev_w"}
    )
    recirc = internal[["src", "dst", "global_cid", "weight"]].merge(
        reverse,
        left_on=["src", "dst", "global_cid"],
        right_on=["src_r", "dst_r", "global_cid"],
        how="inner",
    )
    if len(recirc) > 0:
        recirc_agg = recirc.groupby("global_cid", as_index=False).agg(
            recirc_volume=("weight", "sum")
        )
    else:
        recirc_agg = pd.DataFrame({"global_cid": pd.Series(dtype="int64"),
                                   "recirc_volume": pd.Series(dtype="float32")})

    comm_basic = comm_basic.merge(recirc_agg, on="global_cid", how="left")
    comm_basic["recirc_volume"] = comm_basic["recirc_volume"].fillna(0.0)
    comm_basic["reciprocity"] = (
        comm_basic["recirc_volume"] / (comm_basic["total_volume"] + 1e-9)
    ).astype("float32")
    comm_basic["internal_recirc"] = comm_basic["reciprocity"]
    comm_basic = comm_basic.drop(columns=["recirc_volume", "ext_volume"])

    # ── Sink / source concentration ───────────────────────────────────────────
    node_in = internal.groupby(["global_cid", "dst"], as_index=False).agg(in_vol=("weight", "sum"))
    max_in  = node_in.groupby("global_cid", as_index=False).agg(max_in=("in_vol", "max"))
    sum_in  = node_in.groupby("global_cid", as_index=False).agg(sum_in=("in_vol", "sum"))
    sink_c  = max_in.merge(sum_in, on="global_cid")
    sink_c["sink_concentration"] = (sink_c["max_in"] / (sink_c["sum_in"] + 1e-9)).astype("float32")
    comm_basic = comm_basic.merge(sink_c[["global_cid", "sink_concentration"]], on="global_cid", how="left")

    node_out  = internal.groupby(["global_cid", "src"], as_index=False).agg(out_vol=("weight", "sum"))
    max_out   = node_out.groupby("global_cid", as_index=False).agg(max_out=("out_vol", "max"))
    sum_out   = node_out.groupby("global_cid", as_index=False).agg(sum_out=("out_vol", "sum"))
    source_c  = max_out.merge(sum_out, on="global_cid")
    source_c["source_concentration"] = (source_c["max_out"] / (source_c["sum_out"] + 1e-9)).astype("float32")
    comm_basic = comm_basic.merge(source_c[["global_cid", "source_concentration"]], on="global_cid", how="left")

    # ── Community size ────────────────────────────────────────────────────────
    sizes = node_to_cid.groupby("global_cid", as_index=False).agg(size=("node", "count"))
    comm_basic = comm_basic.merge(sizes, on="global_cid", how="left")

    # ── Node roles ────────────────────────────────────────────────────────────
    if node_roles is not None and len(node_roles) > 0:
        nr = node_roles[["node", "role", "flow_consistency", "layering_score"]].merge(
            node_to_cid, on="node", how="inner"
        )
        nr["is_source"]   = (nr["role"] == 1).astype("int32")
        nr["is_sink"]     = (nr["role"] == 2).astype("int32")
        nr["is_layering"] = (nr["role"] == 3).astype("int32")
        role_agg = nr.groupby("global_cid", as_index=False).agg(
            n_sources          =("is_source",       "sum"),
            n_sinks            =("is_sink",          "sum"),
            n_layering         =("is_layering",      "sum"),
            avg_layering_score =("layering_score",   "mean"),
        )
        comm_basic = comm_basic.merge(role_agg, on="global_cid", how="left")
    else:
        for col in ["n_sources", "n_sinks", "n_layering", "avg_layering_score"]:
            comm_basic[col] = 0

    # ── Motif enrichment ──────────────────────────────────────────────────────
    if motif_counts:
        # Pre-build size lookup to avoid O(n²) per-row DataFrame filter
        # and IndexError when a community is in motif_counts but not comm_basic.
        size_dict = comm_basic.set_index("global_cid")["size"].to_dict()
        comm_basic["motif_enrichment"] = comm_basic["global_cid"].map(
            lambda gcid: motif_counts.get(gcid, 0.0) / max(size_dict.get(gcid, 1), 1)
        ).astype("float32")
    else:
        comm_basic["motif_enrichment"] = 0.0

    # ── Size-normalised features ───────────────────────────────────────────────
    comm_basic["vol_density"]  = (comm_basic["total_volume"] / (comm_basic["size"] + 1e-9)).astype("float32")
    comm_basic["edge_density"] = (
        comm_basic["n_internal_edges"].astype("float32")
        / (comm_basic["size"] * (comm_basic["size"] - 1) + 1e-9)
    ).astype("float32")

    # ── Metadata ──────────────────────────────────────────────────────────────
    comm_basic["window_id"]  = window_id
    comm_basic["step_start"] = step_start
    comm_basic["step_end"]   = step_end
    comm_basic["persistence"] = 0.0   # filled by score_communities from persistence_map

    # Fill remaining NaN
    for col in comm_basic.select_dtypes(include=[np.number]).columns:
        comm_basic[col] = comm_basic[col].fillna(0.0)

    # Filter by min community size
    return comm_basic[comm_basic["size"] >= 1].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Suspicion scoring
# ---------------------------------------------------------------------------

def score_communities(
    feature_df: pd.DataFrame,
    cfg: CommunityConfig,
    persistence_map: Optional[Dict[int, int]] = None,
    max_windows: int = 1,
    n_alert: int = 0,
    n_rows: int = 1,
) -> pd.DataFrame:
    """
    Compute suspicion scores for all communities in feature_df.

    Formula (community_scoring.md §2):
        S(C) = w_internal_flow  * internal_flow
             + w_reciprocity    * reciprocity
             + w_persistence    * persistence_norm
             + w_motif          * motif_enrichment_norm
             - w_external_noise * external_noise

    Then apply a log size-penalty to avoid large communities dominating.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Output of extract_community_features() — one or many windows concatenated.
    cfg : CommunityConfig
        Score weights and threshold.
    persistence_map : Dict[global_cid, int], optional
        {global_cid: n_windows_seen}. If None, persistence = 0 for all.
    max_windows : int
        Total number of windows processed — used to normalise persistence to [0, 1].
    n_alert : int
        Total SAR-labelled transactions in the dataset (for AER reporting).
    n_rows : int
        Total transactions in the dataset (for AER reporting).

    Returns
    -------
    pd.DataFrame — same as input, with added columns:
        persistence, persistence_norm, motif_enrichment_norm,
        suspicion_score, is_suspicious,
        flag_internal_flow, flag_reciprocity, flag_persistence, flag_motif
    """
    if len(feature_df) == 0:
        return feature_df

    df = feature_df.copy()

    # ── Persistence normalisation ─────────────────────────────────────────────
    if persistence_map:
        df["persistence"] = df["global_cid"].map(
            lambda gcid: persistence_map.get(gcid, 0)
        ).astype("float32")
    # persistence_norm ∈ [0, 1]
    df["persistence_norm"] = (df["persistence"] / max(max_windows, 1)).clip(0.0, 1.0).astype("float32")

    # ── Motif enrichment normalisation ────────────────────────────────────────
    max_motif = float(df["motif_enrichment"].max()) if df["motif_enrichment"].max() > 0 else 1.0
    df["motif_enrichment_norm"] = (df["motif_enrichment"] / max_motif).clip(0.0, 1.0).astype("float32")

    # ── Weighted suspicion score ───────────────────────────────────────────────
    df["suspicion_score"] = (
        cfg.w_internal_flow  * df["internal_flow"].clip(0.0, 1.0)
        + cfg.w_reciprocity  * df["reciprocity"].clip(0.0, 1.0)
        + cfg.w_persistence  * df["persistence_norm"]
        + cfg.w_motif        * df["motif_enrichment_norm"]
        - cfg.w_external_noise * df["external_noise"].clip(0.0, 1.0)
    ).astype("float32")

    # Size-penalty: log penalty to avoid large communities dominating.
    # gamma is configurable via cfg.size_penalty_gamma (default 0.05).
    gamma = cfg.size_penalty_gamma
    size_penalty = gamma * np.log1p(df["size"].astype("float64")).astype("float32")
    df["suspicion_score"] = (df["suspicion_score"] - size_penalty).clip(lower=0.0).astype("float32")

    # ── Binary flag ───────────────────────────────────────────────────────────
    df["is_suspicious"] = (df["suspicion_score"] >= cfg.global_susp_threshold).astype("int8")

    # ── Explainability flags ───────────────────────────────────────────────────
    df["flag_internal_flow"] = (df["internal_flow"]  >= 0.5).astype("int8")
    df["flag_reciprocity"]   = (df["reciprocity"]    >= 0.3).astype("int8")
    df["flag_persistence"]   = (df["persistence_norm"] >= 0.3).astype("int8")
    df["flag_motif"]         = (df["motif_enrichment"] >= 1.0).astype("int8")

    # ── Summary ────────────────────────────────────────────────────────────────
    total  = len(df)
    n_susp = int(df["is_suspicious"].sum())
    global_sar_rate = n_alert / max(n_rows, 1)
    susp_sar_rate   = float(df[df["is_suspicious"] == 1]["alert_ratio"].mean()) if n_susp > 0 else 0.0
    aer = susp_sar_rate / max(global_sar_rate, 1e-9)

    print(f"  Communities: {total:,} | Suspicious: {n_susp:,} ({n_susp / max(total, 1) * 100:.1f}%)")
    print(f"  Mean score: {float(df['suspicion_score'].mean()):.4f}")
    if n_alert > 0:
        print(f"  AER: {aer:.2f}×  (global={global_sar_rate * 100:.2f}%, susp={susp_sar_rate * 100:.2f}%)")

    unknown_unknowns = df[
        (df["is_suspicious"] == 1) & (df["alert_ratio"] < cfg.alert_thresh)
    ]
    if len(unknown_unknowns) > 0:
        print(f"  Unknown-unknowns (suspicious, alert_ratio < {cfg.alert_thresh}): {len(unknown_unknowns)}")

    return df.sort_values("suspicion_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Shortlist helper
# ---------------------------------------------------------------------------

def get_shortlist(
    scored_df: pd.DataFrame,
    cfg: CommunityConfig,
) -> pd.DataFrame:
    """
    Return the top-K suspicious communities for investigation.

    Parameters
    ----------
    scored_df : pd.DataFrame
        Output of score_communities().
    cfg : CommunityConfig
        top_k_export and global_susp_threshold.

    Returns
    -------
    pd.DataFrame — top-K rows from scored_df, filtered to is_suspicious == 1.
    """
    suspicious = scored_df[scored_df["is_suspicious"] == 1]
    return suspicious.head(cfg.top_k_export).reset_index(drop=True)


__all__ = [
    "extract_community_features",
    "score_communities",
    "get_shortlist",
]