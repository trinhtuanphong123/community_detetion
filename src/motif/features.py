"""
features.py — Motif feature extraction for ML downstream use.

Two primary feature tables:
    1. Entity-level (per node, per motif type)
       Output: build_entity_motif_features()  → long format
               build_entity_feature_wide()     → wide format (one row per node)

    2. Window-level (per time bucket, per motif type)
       Output: build_window_motif_features()

Export:
    save_features() — writes parquet to a configurable path (Google Drive).

Feature columns produced (motif_spec §7 / guide §8):
    count             — raw instance count
    avg_amount        — mean amount across all edges in matched instances
    avg_ratio         — mean amount preservation ratio across hops
    ratio_std         — std of amount ratio (higher = more irregular)
    avg_lag           — mean step gap between consecutive hops
    max_lag           — worst-case lag (flags delayed relay)
    avg_n_alert_edges — mean flagged edges per instance
    zscore            — significance vs. null model (from compute_null_zscore)
    freq_by_degree    — count / node degree (passed in by caller)
    freq_by_volume    — count / total transaction volume in window

Guarantees:
    - Time   : window_start assigned from min(steps) of each instance.
    - Direction: node role (src/dst) preserved in entity table; not collapsed.
    - Memory : row-building is O(instances × nodes_per_instance), not O(all transactions).
               groupby aggregation is vectorized. No unnecessary intermediate copies.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Entity-level features
# ---------------------------------------------------------------------------

def build_entity_motif_features(
    motif_instances: List[dict],
    zscore_table: Optional[Dict[str, dict]] = None,
    node_degree: Optional[Dict[int, int]] = None,
    total_volume: float = 0.0,
) -> pd.DataFrame:
    """
    Aggregate motif instances into a feature table per (node, motif_type).

    For each node participating in at least one motif instance, computes:
        count             — number of instances the node appears in
        avg_amount        — mean of per-instance mean amounts
        avg_ratio         — mean amount preservation ratio across hops
        ratio_std         — std of ratio sequence (instability signal)
        avg_lag           — mean hop lag
        max_lag           — maximum hop lag seen
        avg_n_alert_edges — mean SAR-flagged edges per instance
        zscore            — motif significance (from compute_null_zscore)
        freq_by_degree    — count / node out-degree (0 if degree unknown)
        freq_by_volume    — count / total_volume (0 if volume == 0)

    Parameters
    ----------
    motif_instances : list[dict]
        Combined output of run_all_matchers() or filter_motifs().
    zscore_table : dict, optional
        Output of compute_null_zscore(). Keys are motif_type strings.
    node_degree : dict, optional
        {node_id: degree} mapping. Used to compute freq_by_degree.
        If None, freq_by_degree is set to 0 for all rows.
    total_volume : float
        Total transaction amount in the current window.
        Used to compute freq_by_volume. Set to 0 to skip.

    Returns
    -------
    pd.DataFrame with columns:
        node, motif_type, count, avg_amount, avg_ratio, ratio_std,
        avg_lag, max_lag, avg_n_alert_edges, zscore,
        freq_by_degree, freq_by_volume
    One row per (node, motif_type). Sorted by node, motif_type.
    """
    _EMPTY_COLS = [
        "node", "motif_type", "count",
        "avg_amount", "avg_ratio", "ratio_std",
        "avg_lag", "max_lag", "avg_n_alert_edges",
        "zscore", "freq_by_degree", "freq_by_volume",
    ]

    if not motif_instances:
        return pd.DataFrame(columns=_EMPTY_COLS)

    rows = []
    for inst in motif_instances:
        mt      = inst["motif_type"]
        amounts = inst.get("amounts", [])
        lags    = inst.get("lags", [])
        ratios  = inst.get("ratios", [])
        n_alert = inst.get("n_alert", 0)
        zs      = zscore_table[mt]["zscore"] if (zscore_table and mt in zscore_table) else np.nan

        avg_amt   = float(np.mean(amounts))   if amounts else 0.0
        avg_ratio = float(np.mean(ratios))    if ratios  else np.nan
        r_std     = float(np.std(ratios))     if ratios  else np.nan
        avg_lag   = float(np.mean(lags))      if lags    else 0.0
        max_lag   = float(max(lags))          if lags    else 0.0

        for node in set(inst.get("nodes", [])):
            rows.append({
                "node":             int(node),
                "motif_type":       mt,
                "avg_amount":       avg_amt,
                "avg_ratio":        avg_ratio,
                "ratio_std":        r_std,
                "avg_lag":          avg_lag,
                "max_lag":          max_lag,
                "n_alert_edges":    n_alert,
                "zscore":           zs,
            })

    if not rows:
        return pd.DataFrame(columns=_EMPTY_COLS)

    df = pd.DataFrame(rows)

    agg = (
        df.groupby(["node", "motif_type"], as_index=False)
        .agg(
            count               =("avg_amount",     "count"),
            avg_amount          =("avg_amount",      "mean"),
            avg_ratio           =("avg_ratio",       "mean"),
            ratio_std           =("ratio_std",       "mean"),
            avg_lag             =("avg_lag",         "mean"),
            max_lag             =("max_lag",         "max"),
            avg_n_alert_edges   =("n_alert_edges",   "mean"),
            zscore              =("zscore",          "first"),
        )
    )

    # Normalised frequency by node degree
    if node_degree:
        agg["freq_by_degree"] = agg.apply(
            lambda r: r["count"] / node_degree.get(int(r["node"]), 1),
            axis=1,
        )
    else:
        agg["freq_by_degree"] = 0.0

    # Normalised frequency by window transaction volume
    if total_volume > 0:
        agg["freq_by_volume"] = agg["count"] / total_volume
    else:
        agg["freq_by_volume"] = 0.0

    return agg.sort_values(["node", "motif_type"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Wide-format pivot (one row per node, all motif types as columns)
# ---------------------------------------------------------------------------

def build_entity_feature_wide(entity_features: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot entity_motif_features from long → wide format.

    Long:  [node, motif_type, count, avg_amount, ...]
    Wide:  [node, fanin_count, fanin_avg_amount, ..., cycle3_count, ...]

    Use when a single feature vector per node is needed for model input.
    Missing (node, motif_type) combinations are filled with 0.

    Parameters
    ----------
    entity_features : pd.DataFrame
        Output of build_entity_motif_features().

    Returns
    -------
    pd.DataFrame: one row per node.
    Column naming: {motif_type}_{metric}  e.g. fanin_count, relay4_avg_lag.
    """
    if entity_features.empty:
        return pd.DataFrame()

    metric_cols = [
        "count", "avg_amount", "avg_ratio", "ratio_std",
        "avg_lag", "max_lag", "avg_n_alert_edges", "zscore",
        "freq_by_degree", "freq_by_volume",
    ]
    available = [c for c in metric_cols if c in entity_features.columns]

    wide = entity_features.pivot_table(
        index="node",
        columns="motif_type",
        values=available,
        aggfunc="first",
    )

    # Flatten multi-level columns: (metric, motif_type) → "motif_type_metric"
    wide.columns = [f"{mt}_{metric}" for metric, mt in wide.columns]
    wide = wide.fillna(0).reset_index()

    return wide


# ---------------------------------------------------------------------------
# Window-level features
# ---------------------------------------------------------------------------

def build_window_motif_features(
    motif_instances: List[dict],
    window_size: int = 7,
) -> pd.DataFrame:
    """
    Aggregate motif instances into a feature table per (window_start, motif_type).

    For each time bucket, computes:
        count             — number of matched instances
        total_amount      — total transaction amount across all edges
        avg_lag           — mean hop lag across all instances
        n_alert_edges     — total SAR-flagged edges
        suspicious_ratio  — fraction of instances with at least one alert edge

    Parameters
    ----------
    motif_instances : list[dict]
        Output from run_all_matchers() or filter_motifs().
    window_size : int
        Number of steps per bucket. Default 7 = one week if step == 1 day.

    Returns
    -------
    pd.DataFrame with columns:
        window_start, motif_type, count, total_amount,
        avg_lag, n_alert_edges, suspicious_ratio
    Sorted by window_start, motif_type.
    """
    _EMPTY_COLS = [
        "window_start", "motif_type", "count",
        "total_amount", "avg_lag", "n_alert_edges", "suspicious_ratio",
    ]

    if not motif_instances:
        return pd.DataFrame(columns=_EMPTY_COLS)

    rows = []
    for inst in motif_instances:
        steps   = inst.get("steps", [])
        amounts = inst.get("amounts", [])
        lags    = inst.get("lags", [])
        n_alert = inst.get("n_alert", 0)

        if not steps:
            continue

        # Assign to bucket by first event's step — time-preserving
        window_start = (min(steps) // window_size) * window_size

        rows.append({
            "window_start": window_start,
            "motif_type":   inst["motif_type"],
            "total_amount": float(sum(amounts)) if amounts else 0.0,
            "avg_lag":      float(np.mean(lags)) if lags else 0.0,
            "n_alert_edges": n_alert,
            "has_alert":    int(n_alert > 0),
        })

    if not rows:
        return pd.DataFrame(columns=_EMPTY_COLS)

    df = pd.DataFrame(rows)

    agg = (
        df.groupby(["window_start", "motif_type"], as_index=False)
        .agg(
            count            =("total_amount",  "count"),
            total_amount     =("total_amount",  "sum"),
            avg_lag          =("avg_lag",        "mean"),
            n_alert_edges    =("n_alert_edges",  "sum"),
            suspicious_ratio =("has_alert",      "mean"),
        )
    )

    return agg.sort_values(["window_start", "motif_type"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Export to parquet / CSV (configurable path for Google Drive)
# ---------------------------------------------------------------------------

# Default export directory — override in Colab:
#   import src.motif.features as mf
#   mf.DEFAULT_EXPORT_DIR = "/content/drive/MyDrive/aml_outputs"
DEFAULT_EXPORT_DIR: str = "outputs/motif_features"


def save_features(
    df: pd.DataFrame,
    filename: str,
    export_dir: Optional[str] = None,
    fmt: str = "parquet",
) -> str:
    """
    Save a feature DataFrame to disk (parquet or CSV).

    Parameters
    ----------
    df : pd.DataFrame
        Any feature table produced by this module.
    filename : str
        Base filename without extension, e.g. "entity_features_w30".
    export_dir : str, optional
        Target directory.  Defaults to DEFAULT_EXPORT_DIR.
        Set to "/content/drive/MyDrive/<your_path>" in Colab.
    fmt : str
        "parquet" (default, smaller) or "csv".

    Returns
    -------
    str : full path of the saved file.

    Notes
    -----
    - Parquet is preferred for downstream pandas / spark reads.
    - CSV is a fallback for manual inspection or Drive sharing.
    - The directory is created if it does not exist.
    """
    out_dir = Path(export_dir or DEFAULT_EXPORT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    ext  = "parquet" if fmt == "parquet" else "csv"
    path = out_dir / f"{filename}.{ext}"

    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

    print(f"  Saved {len(df):,} rows → {path}")
    return str(path)


__all__ = [
    "build_entity_motif_features",
    "build_entity_feature_wide",
    "build_window_motif_features",
    "save_features",
    "DEFAULT_EXPORT_DIR",
]