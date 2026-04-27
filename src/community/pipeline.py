"""
pipeline.py — End-to-end community detection orchestrator.

Execution order (community_pipeline.md §1):
    1. For each time window:
       a. Build directed weighted WindowGraph (from weighting.py)
       b. Run community detection (detection.py)
       c. Split oversized communities (detection.py)
       d. Track communities across windows (tracking.py)
       e. Extract community features (scoring.py)
       f. Release large objects immediately
    2. After all windows:
       a. Score all accumulated feature rows
       b. Build community assignment table
       c. Build suspicious community shortlist

Output tables:
    assignment_df : node-level table  [window_id, step_start, step_end,
                                       node, global_cid, community_size,
                                       overlap_score]
    feature_df    : community-level table with all scoring columns
    shortlist_df  : top-K suspicious communities

Memory rules (pipeline.md §4):
    - Window graph (WindowGraph) is built, used, and released per window.
    - No snapshot dict list accumulates over time.
    - Tracking buffer is capped at cfg.tracking_memory frames.
    - Feature rows are appended to a list; concatenated once at the end.

Guarantees:
    - Time      : windows processed in step_start order.
    - Direction : WindowGraph.A is directed; never symmetrised here.
    - Memory    : large objects deleted + gc.collect() after each window.
"""

from __future__ import annotations

import gc
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import CommunityConfig
from .weighting import build_window_graph, WindowGraph
from .detection import (
    detect_communities,
    split_large_communities,
    compute_node_roles,
    labels_to_dataframe,
)
from .tracking import (
    match_communities_jaccard,
    build_tracking_record,
    update_buffer,
)
from .scoring import extract_community_features, score_communities, get_shortlist


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def run_community_pipeline(
    window_iter,
    cfg: CommunityConfig,
    n_nodes: int,
    motif_counts_by_window: Optional[Dict[int, Dict[int, float]]] = None,
    n_alert: int = 0,
    n_rows: int = 1,
    src_col: str = "src_node",
    dst_col: str = "dst_node",
    amount_col: str = "amount",
    alert_col: str = "is_sar",
    weight_col: str = "amount",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the end-to-end community detection pipeline.

    Parameters
    ----------
    window_iter : iterable of (step_start, step_end, window_df)
        Produces one window at a time. Matches iter_windows() from loader.py.
        window_df must have src_col, dst_col, amount_col.
    cfg : CommunityConfig
        All detection, tracking, and scoring hyperparameters.
    n_nodes : int
        Total number of encoded nodes (from NodeEncoder.n_nodes).
        Keeps adjacency matrix shape consistent across windows.
    motif_counts_by_window : Dict[window_id, Dict[global_cid, float]], optional
        Motif enrichment counts per community per window.
        Produced by motif module integration (features.py). If None, motif
        enrichment is 0 in the suspicion score.
    n_alert : int
        Total SAR-labelled rows in the full dataset (for AER reporting).
    n_rows : int
        Total rows in the full dataset (for AER reporting).
    src_col, dst_col, amount_col, alert_col, weight_col : str
        Column names in window DataFrames.
    verbose : bool
        Print per-window progress.

    Returns
    -------
    assignment_df : pd.DataFrame
        Node-level tracking table — one row per (window, node).
        Columns: window_id, step_start, step_end, node, global_cid,
                 community_size, overlap_score.

    feature_df : pd.DataFrame
        Community-level feature + score table — one row per (window, community).
        All fields from extract_community_features() + score_communities().

    shortlist_df : pd.DataFrame
        Top-K suspicious communities, filtered from feature_df.
    """
    global_cid_counter  = 0
    tracking_buffer:  List[pd.DataFrame] = []
    persistence_map:  Dict[int, int]     = {}   # global_cid → n_windows_seen
    assignment_rows:  List[pd.DataFrame] = []
    feature_rows:     List[pd.DataFrame] = []
    window_id         = 0
    total_windows     = 0

    for step_start, step_end, window_df in window_iter:
        total_windows += 1

        if len(window_df) < 2:   # skip windows with too few transactions
            window_id += 1
            continue

        if verbose:
            print(f"  Window {window_id}: steps [{step_start}, {step_end}] "
                  f"({len(window_df):,} rows)")

        # Guard: fail clearly if the weight column is missing
        # (build_window_graph would silently produce an all-zero adjacency).
        if weight_col not in window_df.columns:
            raise ValueError(
                f"weight_col='{weight_col}' not found in window_df. "
                f"Available columns: {list(window_df.columns)}"
            )

        # ── 1. Build directed weighted graph ──────────────────────────────────
        wg = build_window_graph(
            window_df,
            n_nodes=n_nodes,
            cfg=None,           # raw amount weights (no decay unless cfg passed)
            step_start=step_start,
            step_end=step_end,
            src_col=src_col,
            dst_col=dst_col,
            weight_col=weight_col,
        )

        if wg.m_t == 0:
            window_id += 1
            continue

        # ── 2. Community detection ─────────────────────────────────────────────
        labels = detect_communities(wg, cfg)
        if not labels:
            del wg
            gc.collect()
            window_id += 1
            continue

        # ── 3. Split oversized communities ────────────────────────────────────
        if cfg.s_max > 0:
            labels = split_large_communities(labels, wg, cfg)

        # ── 4. Cross-window tracking ──────────────────────────────────────────
        global_labels, global_cid_counter = match_communities_jaccard(
            labels, tracking_buffer, cfg, global_cid_counter
        )

        # ── 5. Build tracking record ──────────────────────────────────────────
        record = build_tracking_record(
            global_labels, window_id, step_start, step_end,
            prev_buffer=tracking_buffer,
        )
        tracking_buffer = update_buffer(tracking_buffer, record, cfg)

        # Update persistence counters
        for gcid in set(global_labels.values()):
            if gcid != -1:
                persistence_map[gcid] = persistence_map.get(gcid, 0) + 1

        assignment_rows.append(record)

        # ── 6. Node roles ─────────────────────────────────────────────────────
        node_roles = compute_node_roles(
            window_df, cfg,
            src_col=src_col, dst_col=dst_col,
            amount_col=amount_col, alert_col=alert_col,
        )

        # ── 7. Extract community features ────────────────────────────────────
        motif_counts = (
            motif_counts_by_window.get(window_id) if motif_counts_by_window else None
        )
        feat = extract_community_features(
            window_df=window_df,
            global_labels=global_labels,
            wg=wg,
            node_roles=node_roles,
            motif_counts=motif_counts,
            window_id=window_id,
            step_start=step_start,
            step_end=step_end,
            src_col=src_col,
            dst_col=dst_col,
            amount_col=amount_col,
            alert_col=alert_col,
        )

        if len(feat) > 0:
            feature_rows.append(feat)

        # ── 8. Release large objects ──────────────────────────────────────────
        del wg, labels, global_labels, node_roles, feat, record
        gc.collect()
        window_id += 1

    if verbose:
        print(f"\n  Processed {total_windows} windows, {global_cid_counter} global communities created.")

    # ── Assemble assignment table ─────────────────────────────────────────────
    if assignment_rows:
        assignment_df = pd.concat(assignment_rows, ignore_index=True)
    else:
        assignment_df = pd.DataFrame(columns=[
            "window_id", "step_start", "step_end", "node",
            "global_cid", "community_size", "overlap_score",
        ])
    del assignment_rows
    gc.collect()

    # ── Score all features ────────────────────────────────────────────────────
    if feature_rows:
        feature_df = pd.concat(feature_rows, ignore_index=True)
        del feature_rows
        gc.collect()

        feature_df = score_communities(
            feature_df,
            cfg,
            persistence_map=persistence_map,
            max_windows=max(total_windows, 1),
            n_alert=n_alert,
            n_rows=n_rows,
        )
    else:
        feature_df = pd.DataFrame()
        del feature_rows

    # ── Shortlist ─────────────────────────────────────────────────────────────
    if len(feature_df) > 0:
        shortlist_df = get_shortlist(feature_df, cfg)
    else:
        shortlist_df = pd.DataFrame()

    if verbose:
        print(f"  Shortlist: {len(shortlist_df)} suspicious communities (top-{cfg.top_k_export})")

    return assignment_df, feature_df, shortlist_df


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------

def save_pipeline_outputs(
    assignment_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    shortlist_df: pd.DataFrame,
    export_dir: str = "outputs/community",
    fmt: str = "parquet",
) -> None:
    """
    Save the three pipeline output tables to disk.

    Parameters
    ----------
    assignment_df, feature_df, shortlist_df : pd.DataFrame
    export_dir : str
        Target directory (set to "/content/drive/MyDrive/..." in Colab).
    fmt : str
        "parquet" (default) or "csv".

    Notes
    -----
    Directory is created if it does not exist.
    Parquet is preferred for downstream model consumption.
    """
    from pathlib import Path
    out = Path(export_dir)
    out.mkdir(parents=True, exist_ok=True)

    ext = "parquet" if fmt == "parquet" else "csv"
    tables = {
        "community_assignments": assignment_df,
        "community_features":    feature_df,
        "community_shortlist":   shortlist_df,
    }
    for name, df in tables.items():
        if len(df) == 0:
            continue
        path = out / f"{name}.{ext}"
        if fmt == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        print(f"  Saved {len(df):,} rows → {path}")


__all__ = [
    "run_community_pipeline",
    "save_pipeline_outputs",
]