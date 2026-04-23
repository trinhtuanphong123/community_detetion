# src/motif/features.py
# Aggregate motif instances thành numeric feature table cho ML.
#
# Hai loại feature:
#   1. entity-level: feature per node (dùng cho node scoring)
#   2. window-level: feature per (step_window) (dùng cho time-series)
#
# Output là pandas DataFrame sẵn sàng cho XGBoost/LightGBM.

import numpy as np
import pandas as pd


def build_entity_motif_features(
    motif_instances: list[dict],
    zscore_table: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """
    Aggregate motif instances thành feature table theo entity (node).

    Với mỗi node tham gia ít nhất một motif instance, tính:
        - count per motif type
        - avg_amount per motif type
        - avg_lag per motif type (avg khoảng cách bước)
        - n_alert_edges: số cạnh SAR trong motif instances
        - zscore per motif type (nếu cung cấp zscore_table)

    Args:
        motif_instances: list[dict] — output gộp từ các matchers
        zscore_table:    dict từ compute_null_zscore() — optional

    Returns:
        pandas DataFrame với columns:
            [node, motif_type, count, avg_amount, avg_lag,
             avg_n_alert_edges, zscore]
        Một row per (node, motif_type).
    """
    if not motif_instances:
        return pd.DataFrame(columns=[
            "node", "motif_type", "count",
            "avg_amount", "avg_lag", "avg_n_alert_edges", "zscore",
        ])

    rows = []
    for inst in motif_instances:
        mt      = inst["motif_type"]
        amounts = inst.get("amounts", [])
        lags    = inst.get("lags", [])
        n_alert = inst.get("n_alert", 0)

        avg_amt = float(np.mean(amounts)) if amounts else 0.0
        avg_lag = float(np.mean(lags))    if lags    else 0.0
        zs      = zscore_table[mt]["zscore"] if zscore_table and mt in zscore_table else np.nan

        for node in set(inst.get("nodes", [])):
            rows.append({
                "node":             int(node),
                "motif_type":       mt,
                "avg_amount":       avg_amt,
                "avg_lag":          avg_lag,
                "n_alert_edges":    n_alert,
                "zscore":           zs,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    agg = (
        df.groupby(["node", "motif_type"], as_index=False)
        .agg(
            count           =("avg_amount",    "count"),
            avg_amount      =("avg_amount",    "mean"),
            avg_lag         =("avg_lag",        "mean"),
            avg_n_alert_edges=("n_alert_edges", "mean"),
            zscore          =("zscore",         "first"),
        )
    )

    return agg.sort_values(["node", "motif_type"]).reset_index(drop=True)


def build_entity_feature_wide(entity_features: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot entity_motif_features từ long → wide format.

    Long:  [node, motif_type, count, avg_amount, ...]
    Wide:  [node, fanin_count, fanin_avg_amount, ..., cycle3_count, ...]

    Dùng khi cần một vector feature duy nhất per node cho model input.

    Args:
        entity_features: output của build_entity_motif_features()

    Returns:
        pandas DataFrame: một row per node, nhiều cột feature.
        Giá trị NaN (node không có motif type đó) được fill bằng 0.
    """
    if entity_features.empty:
        return pd.DataFrame()

    metric_cols = ["count", "avg_amount", "avg_lag", "avg_n_alert_edges"]
    available   = [c for c in metric_cols if c in entity_features.columns]

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


def build_window_motif_features(
    motif_instances: list[dict],
    window_size: int = 7,
) -> pd.DataFrame:
    """
    Aggregate motif instances thành feature table theo time window.

    Với mỗi window (step_bucket), tính:
        - count per motif type
        - total_amount per motif type
        - avg_lag per motif type
        - n_alert_edges tổng
        - suspicious_ratio: tỷ lệ instance có ít nhất 1 alert edge

    Args:
        motif_instances: list[dict]
        window_size:     số steps mỗi bucket (default 7 = tuần)

    Returns:
        pandas DataFrame: [window_start, motif_type, count,
                           total_amount, avg_lag, n_alert_edges,
                           suspicious_ratio]
    """
    if not motif_instances:
        return pd.DataFrame(columns=[
            "window_start", "motif_type", "count",
            "total_amount", "avg_lag", "n_alert_edges", "suspicious_ratio",
        ])

    rows = []
    for inst in motif_instances:
        steps   = inst.get("steps", [])
        amounts = inst.get("amounts", [])
        lags    = inst.get("lags", [])
        n_alert = inst.get("n_alert", 0)

        if not steps:
            continue

        window_start = (min(steps) // window_size) * window_size
        total_amt    = float(sum(amounts)) if amounts else 0.0
        avg_lag      = float(np.mean(lags)) if lags else 0.0

        rows.append({
            "window_start":   window_start,
            "motif_type":     inst["motif_type"],
            "total_amount":   total_amt,
            "avg_lag":        avg_lag,
            "n_alert_edges":  n_alert,
            "has_alert":      int(n_alert > 0),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    agg = (
        df.groupby(["window_start", "motif_type"], as_index=False)
        .agg(
            count            =("total_amount",   "count"),
            total_amount     =("total_amount",   "sum"),
            avg_lag          =("avg_lag",         "mean"),
            n_alert_edges    =("n_alert_edges",   "sum"),
            suspicious_ratio =("has_alert",       "mean"),
        )
    )

    return agg.sort_values(["window_start", "motif_type"]).reset_index(drop=True)


__all__ = [
    "build_entity_motif_features",
    "build_entity_feature_wide",
    "build_window_motif_features",
]