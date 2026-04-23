"""
loader.py — Load and validate AMLGentex transaction data.

Reads CSV/parquet into a cuDF (or pandas fallback) DataFrame,
validates schema per graph_schema.md, and provides windowed iteration.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Iterator, Tuple

try:
    import cudf as pd_lib
    _GPU = True
except ImportError:
    import pandas as pd_lib
    _GPU = False


@dataclass
class LoaderConfig:
    """Configuration for data loading."""
    # Required columns in raw data (mapped to canonical names)
    column_map: dict = field(default_factory=lambda: {
        "From Bank": "src_bank",
        "To Bank": "dst_bank",
        "Account": "src_node",
        "Account.1": "dst_node",
        "Amount Received": "amount",
        "Timestamp": "timestamp",
        "Is Laundering": "is_sar",
    })
    # Columns to keep after mapping
    keep_cols: list = field(default_factory=lambda: [
        "src_node", "dst_node", "amount", "timestamp", "is_sar", "step",
    ])
    window_size: int = 30     # steps per window
    window_stride: int = 15   # stride between windows (overlap = size - stride)


def load_transactions(
    path: str,
    cfg: LoaderConfig | None = None,
) -> "pd_lib.DataFrame":
    """Load raw transaction file and normalize to canonical schema.

    Parameters
    ----------
    path : str
        Path to CSV or parquet file.
    cfg : LoaderConfig, optional
        Loading configuration. Uses defaults if None.

    Returns
    -------
    DataFrame with canonical columns: src_node, dst_node, amount, step, is_sar.
    Sorted by step ascending.
    """
    cfg = cfg or LoaderConfig()

    # Read file
    if path.endswith(".parquet"):
        df = pd_lib.read_parquet(path)
    else:
        df = pd_lib.read_csv(path)

    # Rename columns that exist in the mapping
    rename = {k: v for k, v in cfg.column_map.items() if k in df.columns}
    if rename:
        df = df.rename(columns=rename)

    # Ensure required columns
    for col in ["src_node", "dst_node", "amount"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Build step from timestamp if not already present
    if "step" not in df.columns:
        if "timestamp" in df.columns:
            ts = pd_lib.to_datetime(df["timestamp"])
            df["step"] = (ts - ts.min()).dt.days
        else:
            raise ValueError("Need either 'step' or 'timestamp' column")

    # Default is_sar to 0 if absent
    if "is_sar" not in df.columns:
        df["is_sar"] = 0

    # Filter self-loops
    df = df[df["src_node"] != df["dst_node"]].reset_index(drop=True)

    # Filter non-positive amounts
    df = df[df["amount"] > 0].reset_index(drop=True)

    # Keep only canonical columns that exist
    keep = [c for c in cfg.keep_cols if c in df.columns]
    df = df[keep]

    # Sort by step for temporal ordering
    df = df.sort_values("step").reset_index(drop=True)

    return df


def iter_windows(
    df: "pd_lib.DataFrame",
    window_size: int = 30,
    window_stride: int = 15,
) -> Iterator[Tuple[int, int, "pd_lib.DataFrame"]]:
    """Yield (window_start, window_end, window_df) for each time window.

    Parameters
    ----------
    df : DataFrame
        Must have a 'step' column, sorted ascending.
    window_size : int
        Number of steps per window.
    window_stride : int
        Step increment between windows.

    Yields
    ------
    (start_step, end_step, window_df) tuples.
    """
    if _GPU:
        step_min = int(df["step"].min())
        step_max = int(df["step"].max())
    else:
        step_min = int(df["step"].min())
        step_max = int(df["step"].max())

    start = step_min
    while start <= step_max:
        end = start + window_size - 1
        mask = (df["step"] >= start) & (df["step"] <= end)
        window_df = df[mask].reset_index(drop=True)
        if len(window_df) > 0:
            yield start, end, window_df
        start += window_stride
        gc.collect()
