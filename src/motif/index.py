"""
index.py — Event-level search indexes for motif matching.

Builds lightweight Python dict indexes from a window of transactions.
Matchers use these indexes instead of scanning the full DataFrame repeatedly.

Indexes produced by build_event_indexes():
    out_index  : {src_node -> [event_dict, ...]} sorted by step ascending
    in_index   : {dst_node -> [event_dict, ...]} sorted by step ascending
    step_index : {step     -> [event_dict, ...]}

Helper:
    edges_after_step(out_index, node, step) — forward-only edge lookup.
    filter_window(event_df, step_start, step_end) — slice a time window.

Guarantees:
- Time      : all buckets are sorted by step; edges_after_step enforces
              forward-only search (no backward traversal).
- Direction : out_index keys on src; in_index keys on dst; never merged.
- Memory    : plain Python dicts after indexing (no DataFrame held);
              gc.collect() called after build.
"""

from __future__ import annotations

import gc
from bisect import bisect_right
from collections import defaultdict
from typing import Dict, List

import pandas as pd
import os
# ---------------------------------------------------------------------------
# Type alias for an event record stored in the index
# ---------------------------------------------------------------------------

# Each event is a plain dict for zero-overhead lookup in matcher loops.
# Keys: event_id, step, src, dst, amount, is_sar
EventDict = Dict[str, object]


# ---------------------------------------------------------------------------
# Main index builder
# ---------------------------------------------------------------------------

def build_event_indexes(
    event_df: pd.DataFrame,
    src_col: str = "src_node",
    dst_col: str = "dst_node",
    step_col: str = "step",
    amount_col: str = "amount",
    alert_col: str = "is_sar",
) -> tuple[dict, dict, dict]:
    """
    Build three search indexes from a window of transactions.

    Parameters
    ----------
    event_df : pd.DataFrame
        A single time window of transactions.  Must be sorted by step
        (loader.py and iter_windows guarantee this).
        Supported column name variants:
          - canonical: src_node, dst_node (from loader.py)
          - legacy:    nameOrig, nameDest  (raw AMLGentex)
          - short:     src, dst           (old pipeline)
    src_col, dst_col, step_col, amount_col, alert_col : str
        Column name overrides.

    Returns
    -------
    out_index  : {src_node: [EventDict, ...]}  sorted by step ascending
    in_index   : {dst_node: [EventDict, ...]}  sorted by step ascending
    step_index : {step:     [EventDict, ...]}

    Notes
    -----
    - event_id is auto-assigned as the row index if absent (motif_spec §3.1).
    - itertuples() is used intentionally: the single-pass loop is O(n) and
      avoids creating three separate groupby-materialised DataFrames.
    - All three dicts are populated in one pass to minimise memory pressure.
    - Buckets are already in step order because event_df is pre-sorted.
    """
    # cuDF → pandas: motif index works on plain Python dicts after this point;
    # converting once here avoids per-row GPU tensor overhead in itertuples.

    if hasattr(event_df, "to_pandas"):
        event_df = event_df.to_pandas()

    # --- FIX WARNING 4: Defensive Integrity Guard ---
    # Validate monotonicity after pandas conversion but before indexing.
    if os.getenv("MOTIF_DEBUG") == "1":
        if not event_df[step_col].is_monotonic_increasing:
            raise ValueError(
                f"CRITICAL DATA ERROR: event_df must be sorted by '{step_col}' before indexing. "
                "Unsorted data will cause binary search (bisect) to return silent failures."
            )

    df = _normalize_columns(event_df, src_col, dst_col, step_col, amount_col, alert_col)
    df = _ensure_event_id(df)
    _validate_required(df)

    has_alert = "is_sar" in df.columns
    out_index:  dict = defaultdict(list)
    in_index:   dict = defaultdict(list)
    step_index: dict = defaultdict(list)

    # Single O(n) pass — itertuples permitted per coding conventions when
    # building an index that cannot be expressed as a vectorized operation.
    for row in df.itertuples(index=False):
        e: EventDict = {
            "event_id": int(row.event_id),
            "step":     int(row.step),
            "src":      int(row.src_node),
            "dst":      int(row.dst_node),
            "amount":   float(row.amount),
            "is_sar":   int(row.is_sar) if has_alert else 0,
        }
        out_index[e["src"]].append(e)
        in_index[e["dst"]].append(e)
        step_index[e["step"]].append(e)

    # Freeze to regular dicts: prevents accidental key creation on miss
    out_index  = dict(out_index)
    in_index   = dict(in_index)
    step_index = dict(step_index)
    # At the end of build_event_indexes, after populating out_index:
    out_steps = {node: [e["step"] for e in edges]
             for node, edges in out_index.items()}
    gc.collect()
    return out_index, in_index, step_index, out_steps

    

   
def build_event_indexes(
    event_df: pd.DataFrame,
    src_col: str = "src_node",
    dst_col: str = "dst_node",
    step_col: str = "step",
    amount_col: str = "amount",
    alert_col: str = "is_sar",
) -> tuple[dict, dict, dict, dict]:
    """
    Build search indexes and pre-computed step arrays for optimized motif mining.

    This function performs a single O(n) pass to build graph adjacency lists and 
    pre-calculates step arrays to enable O(log n) binary search in matchers.

    Returns
    -------
    out_index  : {node: [EventDict, ...]}
    in_index   : {node: [EventDict, ...]}
    step_index : {step: [EventDict, ...]}
    out_steps  : {node: [int, ...]} - Pre-computed step lists for fast bisect.
    """
    
    # Move to CPU/Pandas immediately to avoid GPU-to-CPU overhead during iteration
    if hasattr(event_df, "to_pandas"):
        event_df = event_df.to_pandas()

    # --- FIX WARNING 4: Defensive Integrity Guard ---
    # Validate monotonicity after pandas conversion but before indexing.
    if os.getenv("MOTIF_DEBUG") == "1":
        if not event_df[step_col].is_monotonic_increasing:
            raise ValueError(
                f"CRITICAL DATA ERROR: event_df must be sorted by '{step_col}' before indexing. "
                "Unsorted data will cause binary search (bisect) to return silent failures."
            )

    df = _normalize_columns(event_df, src_col, dst_col, step_col, amount_col, alert_col)
    df = _ensure_event_id(df)
    _validate_required(df)

    has_alert = "is_sar" in df.columns
    out_index:  dict = defaultdict(list)
    in_index:   dict = defaultdict(list)
    step_index: dict = defaultdict(list)

    # Single O(n) pass to build core adjacency structures
    for row in df.itertuples(index=False):
        e: EventDict = {
            "event_id": int(row.event_id),
            "step":     int(row.step),
            "src":      int(row.src_node),
            "dst":      int(row.dst_node),
            "amount":   float(row.amount),
            "is_sar":   int(row.is_sar) if has_alert else 0,
        }
        out_index[e["src"]].append(e)
        in_index[e["dst"]].append(e)
        step_index[e["step"]].append(e)

    # Freeze to regular dicts to avoid accidental key creation/memory leaks
    out_index  = dict(out_index)
    in_index   = dict(in_index)
    step_index = dict(step_index)


# ---------------------------------------------------------------------------
# Forward-only lookup helper
# ---------------------------------------------------------------------------


def edges_after_step(
    out_index: dict,
    node: int,
    step: int,
    out_steps: dict,
) -> List[EventDict]:
    """
    Return all outgoing edges from `node` with step > `step`.

    Used by matchers to expand a relay chain forward in time only.
    Binary search on the pre-sorted bucket avoids a linear scan.

    Parameters
    ----------
    out_index : dict
        Output of build_event_indexes().
    node : int
        Source node ID.
    step : int
        All returned edges have step strictly greater than this value.

    Returns
    -------
    List of EventDict, empty if node has no outgoing edges after step.
    """
    bucket = out_index.get(node)
    if not bucket:
        return []
    idx = bisect_right(out_steps[node], step)
    return bucket[idx:]



# ---------------------------------------------------------------------------
# Window slice helper
# ---------------------------------------------------------------------------

def filter_window(
    event_df: pd.DataFrame,
    step_start: int,
    step_end: int,
) -> pd.DataFrame:
    """
    Slice event_df to [step_start, step_end] (inclusive).

    Use before build_event_indexes() when working from a full loaded
    DataFrame rather than through iter_windows().

    Parameters
    ----------
    event_df : pd.DataFrame
        Must have a `step` column.
    step_start, step_end : int
        Inclusive step bounds.

    Returns
    -------
    Filtered DataFrame with reset index.  Direction and step order preserved.
    """
    mask = (event_df["step"] >= step_start) & (event_df["step"] <= step_end)
    return event_df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_columns(
    df: pd.DataFrame,
    src_col: str,
    dst_col: str,
    step_col: str,
    amount_col: str,
    alert_col: str,
) -> pd.DataFrame:
    """
    Rename columns to canonical names (src_node, dst_node, step, amount, is_sar).
    Supports three naming variants without copying the full DataFrame.
    """
    rename: dict = {}

    # Caller-specified override names
    if src_col != "src_node" and src_col in df.columns:
        rename[src_col] = "src_node"
    if dst_col != "dst_node" and dst_col in df.columns:
        rename[dst_col] = "dst_node"
    if step_col != "step" and step_col in df.columns:
        rename[step_col] = "step"
    if amount_col != "amount" and amount_col in df.columns:
        rename[amount_col] = "amount"
    if alert_col != "is_sar" and alert_col in df.columns:
        rename[alert_col] = "is_sar"

    # Legacy / raw AMLGentex column names
    cols = set(df.columns)
    if "src_node" not in cols and "nameOrig" in cols:
        rename["nameOrig"] = "src_node"
    if "dst_node" not in cols and "nameDest" in cols:
        rename["nameDest"] = "dst_node"
    # Short-form names (old pipeline)
    if "src_node" not in cols and "src" in cols:
        rename["src"] = "src_node"
    if "dst_node" not in cols and "dst" in cols:
        rename["dst"] = "dst_node"
    # is_sar / Is Laundering / is_laundering aliases
    if "is_sar" not in cols:
        for alias in ("is_laundering", "Is Laundering", "isSAR"):
            if alias in cols:
                rename[alias] = "is_sar"
                break

    if rename:
        df = df.rename(columns=rename)
    return df


def _ensure_event_id(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-assign event_id from row index if the column is absent."""
    if "event_id" not in df.columns:
        df = df.copy()
        df["event_id"] = range(len(df))
    return df


def _validate_required(df: pd.DataFrame) -> None:
    """Raise ValueError if mandatory columns are missing."""
    required = {"event_id", "step", "src_node", "dst_node", "amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"event_df is missing required columns: {sorted(missing)}. "
            f"Available columns: {sorted(df.columns)}"
        )


__all__ = [
    "build_event_indexes",
    "edges_after_step",
    "filter_window",
]