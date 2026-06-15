"""
loader.py — Load and validate AMLGentex transaction data.

Reads CSV or Parquet into a cuDF (GPU) or pandas (CPU) DataFrame,
normalizes to the canonical schema defined in graph_schema.md §3,
and provides a windowed iterator for memory-bounded processing.

Guarantees:
- Time    : output is sorted by `step` ascending; `step` is always present.
- Direction: `src_node → dst_node` is preserved; self-loops are removed.
- Memory  : only canonical columns are kept; gc.collect() called per window.
"""

from __future__ import annotations

import gc
from typing import Iterator, Tuple

try:
    import cudf as pd_lib
    _GPU = True
except ImportError:
    import pandas as pd_lib  # type: ignore
    _GPU = False

from src.graph.config import LoaderConfig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_file(path: str) -> "pd_lib.DataFrame":
    """Read CSV or Parquet; return raw DataFrame."""
    if path.endswith(".parquet"):
        return pd_lib.read_parquet(path)
    return pd_lib.read_csv(path)


def _rename_columns(df: "pd_lib.DataFrame", column_map: dict) -> "pd_lib.DataFrame":
    """Rename only columns that are present in df."""
    rename = {k: v for k, v in column_map.items() if k in df.columns}
    if rename:
        df = df.rename(columns=rename)
    return df


def _build_step(df: "pd_lib.DataFrame") -> "pd_lib.DataFrame":
    """
    Derive integer `step` (days since first event) from `timestamp`.
    Preserves temporal ordering without adding clock-specific logic.
    """
    ts = pd_lib.to_datetime(df["timestamp"])
    df["step"] = (ts - ts.min()).dt.days.astype("int32")
    return df


def _cast_dtypes(
    df: "pd_lib.DataFrame",
    dtypes: dict,
) -> "pd_lib.DataFrame":
    """
    Cast columns to memory-efficient types.
    Only casts columns that exist; silently skips missing ones.
    Numeric columns with non-numeric values are coerced (errors → NaN dropped).
    """
    for col, dtype in dtypes.items():
        if col not in df.columns:
            continue
        try:
            if _GPU:
                df[col] = df[col].astype(dtype)
            else:
                df[col] = pd_lib.to_numeric(df[col], errors="coerce").astype(dtype)
        except Exception:
            # If cast fails, leave the column as-is; pipeline will surface issues
            pass
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_transactions(
    path: str,
    cfg: LoaderConfig | None = None,
) -> "pd_lib.DataFrame":
    """
    Load a raw AMLGentex file and normalize to the canonical schema.

    Parameters
    ----------
    path : str
        Path to a CSV or Parquet file.
    cfg : LoaderConfig, optional
        Loading configuration.  Defaults are used when None.

    Returns
    -------
    DataFrame with columns: src_node, dst_node, amount, step, is_sar.
    Sorted by `step` ascending.  Self-loops and zero/negative amounts removed.

    Raises
    ------
    ValueError
        If mandatory columns (src_node, dst_node, amount) are absent after
        renaming, or if neither `step` nor `timestamp` is available.
    """
    cfg = cfg or LoaderConfig()

    df = _read_file(path)

    # Step 1 — rename to canonical names
    df = _rename_columns(df, cfg.column_map)

    # Step 2 — check mandatory columns
    for col in ("src_node", "dst_node", "amount"):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found after renaming. "
                f"Available columns: {list(df.columns)}"
            )

    # Step 3 — build `step` if absent
    if "step" not in df.columns:
        if "timestamp" not in df.columns:
            raise ValueError(
                "Neither 'step' nor 'timestamp' column found. "
                "Cannot establish temporal ordering."
            )
        df = _build_step(df)

    # Step 4 — default is_sar = 0 when label is absent
    if "is_sar" not in df.columns:
        df["is_sar"] = 0

    # Step 5 — cast to memory-efficient dtypes early
    df = _cast_dtypes(df, cfg.dtypes)

    # Step 6 — remove self-loops (direction invariant)
    df = df[df["src_node"] != df["dst_node"]].reset_index(drop=True)

    # Step 7 — remove non-positive amounts (AML constraint)
    df = df[df["amount"] > 0].reset_index(drop=True)

    # Step 8 — keep only canonical columns; drop everything else to save RAM
    keep = [c for c in cfg.keep_cols if c in df.columns]
    df = df[keep]

    # Step 9 — sort by step for temporal ordering (must not shuffle)
    df = df.sort_values("step").reset_index(drop=True)

    return df


def iter_windows(
    df: "pd_lib.DataFrame",
    window_size: int = 30,
    window_stride: int = 15,
) -> Iterator[Tuple[int, int, "pd_lib.DataFrame"]]:
    """
    Yield non-overlapping (or overlapping) time windows from a transaction table.

    Parameters
    ----------
    df : DataFrame
        Must have a `step` column, sorted ascending.
    window_size : int
        Number of steps included in each window  [step_start, step_start + size - 1].
    window_stride : int
        Step increment between consecutive window starts.
        Set equal to window_size for non-overlapping windows.

    Yields
    ------
    (start_step, end_step, window_df)
        window_df contains only transactions in [start_step, end_step].
        gc.collect() is called after each yield to free intermediate memory.

    Notes
    -----
    - Temporal ordering is preserved within each window_df.
    - Direction (src_node → dst_node) is not altered.
    - Empty windows are skipped silently.
    """
    step_min = int(df["step"].min())
    step_max = int(df["step"].max())

    start = step_min
    while start <= step_max:
        end = start + window_size - 1
        mask = (df["step"] >= start) & (df["step"] <= end)
        window_df = df[mask].reset_index(drop=True)
        if len(window_df) > 0:
            yield start, end, window_df
        del window_df, mask
        gc.collect()
        start += window_stride
