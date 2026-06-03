# ============================================================
# CELL 0 — Environment setup
#
# HƯỚNG DẪN:
# 1. Chạy Cell này DUY NHẤT một lần.
# 2. Nếu thông báo "RESTART" hiện ra, chọn Runtime > Restart runtime.
# 3. Sau khi restart, chạy các Cell từ 1 đến 6 theo thứ tự.
# ============================================================

import os
import sys
import subprocess

# 1. Kiểm tra và cài đặt RAPIDS cuDF cho T4 GPU
try:
    import cudf
    print("✅ cuDF đã sẵn sàng, bỏ qua bước cài đặt.")
except ImportError:
    print("⏳ Đang cài đặt cuDF (RAPIDS)... Việc này có thể mất 1-2 phút.")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--quiet",
            "cudf-cu12",
            "--extra-index-url", "https://pypi.nvidia.com"
        ])
        print("✅ Cài đặt cuDF thành công!")
        print("⚠️ VUI LÒNG RESTART RUNTIME (Runtime > Restart runtime) ngay bây giờ.")
        print("Sau đó chạy lại từ Cell 1.")
    except Exception as e:
        print(f"❌ Lỗi khi cài đặt cuDF: {e}")

# 2. Kết nối Google Drive
print("\n--- Kiểm tra Google Drive ---")
DRIVE_PATH = "/content/drive"
if not os.path.exists(DRIVE_PATH):
    try:
        from google.colab import drive
        drive.mount(DRIVE_PATH)
        print("✅ Google Drive đã được kết nối.")
    except Exception as e:
        print(f"❌ Không thể kết nối Google Drive: {e}")
else:
    print("ℹ️ Google Drive đã được kết nối từ trước.")



from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class GraphConfig:
    """
    Hyperparameters for temporal graph construction.

    window_sizes : standard analysis windows (days/steps), per graph_schema §6.3
    window_stride : step increment between rolling windows (overlap = size - stride)
    delta_w : max step gap allowed between two linked temporal events
    min_amount : filter out transactions at or below this value
    """

    # Rolling-window sizes used by community detection (§6.3)
    window_sizes: List[int] = field(default_factory=lambda: [7, 14, 30])

    # Default window used by the loader iterator
    window_size: int = 7
    window_stride: int = 7

    # Max step gap for a temporal edge (motif / second-order links)
    delta_w: int = 5

    # Transactions with amount <= this are discarded
    min_amount: float = 0.0


@dataclass
class LoaderConfig:
    """
    Configuration for loading and normalizing AMLGentex raw data.

    column_map : rename raw columns → canonical names
    keep_cols  : columns to retain after normalization (drop everything else)
    dtypes     : cast canonical columns to memory-efficient types
    window_size, window_stride : forwarded to iter_windows default
    """

    # AMLGentex column names → canonical schema (graph_schema §3.2)
    column_map: dict = field(default_factory=lambda: {
        # # AMLSim / IBM-style headers (CSV variant)
        # "Timestamp":          "timestamp",
        # "From Bank":          "src_bank",
        # "Account":            "src_node",
        # "To Bank":            "dst_bank",
        # "Account.1":          "dst_node",
        # "Amount Received":    "amount",
        # "Receiving Currency": "currency",
        # "Is Laundering":      "is_sar",


        # AMLGentex / PaySim-style headers (alternative CSV variant)
        "step":               "step",
        "nameOrig":           "src_node",
        "nameDest":           "dst_node",
        "amount":             "amount",
        "type":               "type",
        "isSAR":              "is_sar",
        "oldbalanceOrig":     "old_bal_src",
        "newbalanceOrig":     "new_bal_src",
        "oldbalanceDest":     "old_bal_dst",
        "newbalanceDest":     "new_bal_dst",
    })

    # Columns to keep in the normalized DataFrame (§3.3)
    keep_cols: List[str] = field(default_factory=lambda: [
        "event_id", "src_node", "dst_node", "amount", "step", "type_code", "is_sar",
    ])

    # Memory-efficient dtypes (§3.2 recommended types)
    dtypes: dict = field(default_factory=lambda: {
        "event_id": "int64",
        "step":     "int32",
        "amount":   "float32",
        "type_code":"int8",
        "is_sar":   "int8",
        "src_node": "int64",
        "dst_node": "int64",
    })

    window_size: int = 30
    window_stride: int = 15


from __future__ import annotations

import gc
import os
from typing import Iterator, Tuple

try:
    import cudf as pd_lib
    _GPU = True
except ImportError:
    import pandas as pd_lib  # type: ignore
    _GPU = False


# Colab / Google Drive — set path to your AML transaction file (CSV or Parquet).
# Mount Drive first if needed: from google.colab import drive; drive.mount('/content/drive')

AML_DATA_PATH = "/content/drive/MyDrive/AML/dataset/tx_log.csv"

# Internal helpers

def _read_file(path: str) -> "pd_lib.DataFrame":
    """Read CSV or Parquet; return raw DataFrame. Path is typically a mounted Drive path on Colab."""
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Data file not found: {path!r}. "
            "Mount Google Drive if needed and set AML_DATA_PATH or pass a valid path to load_transactions."
        )
    if path.endswith(".parquet"):
        return pd_lib.read_parquet(path)
    if _GPU:
        return pd_lib.read_csv(path)
    return pd_lib.read_csv(path, encoding="utf-8", low_memory=False)


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


# Public API

def load_transactions(
    path: str | None = None,
    cfg: LoaderConfig | None = None,
) -> "pd_lib.DataFrame":
    """
    Load a raw AMLGentex file and normalize to the canonical schema
    """
    cfg = cfg or LoaderConfig()
    resolved_path = path if path is not None else AML_DATA_PATH

    df = _read_file(resolved_path)

    # Step 1 — rename to canonical names
    df = _rename_columns(df, cfg.column_map)

    # Optional: hot-code transaction type into a compact numeric feature.
    # Keeps strings out of the canonical table and avoids affecting graph shards.
    if "type" in df.columns and "type_code" not in df.columns:
        _type_map = {
            "TRANSFER": 0,
            "INITALBALANCE": 1,  # dataset spelling
            "CASH": 2,
        }
        if hasattr(df["type"], "to_pandas"):
            _codes = df["type"].to_pandas().map(_type_map).fillna(-1).astype("int8")
            df["type_code"] = pd_lib.Series(_codes.values, dtype="int8")
        else:
            df["type_code"] = df["type"].map(_type_map).fillna(-1).astype("int8")

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

    # Step 9 — preserve raw row order for stable ties
    df["_raw_row_id"] = pd_lib.Series(range(len(df)), dtype="int64")

    # Step 10 — stable deterministic sort, then add unique event_id
    df = df.sort_values(
        ["step", "src_node", "dst_node", "amount", "_raw_row_id"]
    ).reset_index(drop=True)
    df["event_id"] = pd_lib.Series(range(len(df)), dtype="int64")
    df = df.drop(columns=["_raw_row_id"])

    # Reorder to canonical output schema
    out_cols = [c for c in cfg.keep_cols if c in df.columns]
    df = df[out_cols]

    return df


def iter_windows(
    df: "pd_lib.DataFrame",
    window_size: int = 7,
    window_stride: int = 7,
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


from __future__ import annotations

import numpy as np

try:
    import cudf as pd_lib
    _GPU = True
except ImportError:
    import pandas as pd_lib  # type: ignore
    _GPU = False


class NodeEncoder:
    """
    Maps raw node labels → contiguous integer IDs.

    The mapping is incremental: calling fit() multiple times (e.g., once
    per time window) extends the mapping without reassigning existing IDs.
    This guarantees stable encoding across the whole pipeline.

    Usage
    -----
    >>> enc = NodeEncoder()
    >>> df = enc.fit_transform(df, src_col="src_node", dst_col="dst_node")
    >>> enc.encode_column(series)   # encode a single Series
    >>> enc.decode(encoded_ids)     # reverse lookup
    >>> enc.n_nodes                 # total unique nodes seen so far
    """

    def __init__(self) -> None:
        self._label_to_id: dict = {}
        self._id_to_label: dict = {}
        self._next_id: int = 0

    # Fit

    def fit(
        self,
        df: "pd_lib.DataFrame",
        src_col: str = "src_node",
        dst_col: str = "dst_node",
    ) -> "NodeEncoder":
        """
        Learn mapping from all unique nodes in df[src_col] and df[dst_col].

        Incremental: new nodes are appended; existing IDs are never changed.
        Safe to call multiple times (once per window).

        """
        src_vals, dst_vals = _unique_values(df, src_col, dst_col)
        all_nodes = np.union1d(src_vals, dst_vals)
        self._register(all_nodes)
        return self


    # Transform

    def transform(
        self,
        df: "pd_lib.DataFrame",
        src_col: str = "src_node",
        dst_col: str = "dst_node",
    ) -> "pd_lib.DataFrame":
        """
        Replace raw node labels with encoded integer IDs.

        Modifies only src_col and dst_col in-place (no full DataFrame copy).
        Direction (src → dst) is preserved.
        """
        df[src_col] = encode_series(df[src_col], self._label_to_id)
        df[dst_col] = encode_series(df[dst_col], self._label_to_id)
        return df

    def fit_transform(
        self,
        df: "pd_lib.DataFrame",
        src_col: str = "src_node",
        dst_col: str = "dst_node",
    ) -> "pd_lib.DataFrame":
        """Fit on df then transform it. Convenience wrapper."""
        self.fit(df, src_col, dst_col)
        return self.transform(df, src_col, dst_col)

    # Single-column helper (used by temporal.py, second_order.py)


    def encode_column(self, series: "pd_lib.Series") -> "pd_lib.Series":
        """
        Encode a single Series of raw node IDs.

        Any unseen node is registered on the fly (incremental fit).
        Useful when later modules pass node lists that may contain new nodes.

        Returns
        -------
        Series of int64 encoded IDs, same index as input.
        """
        raw = _to_numpy(series)
        # Deterministic: register unseen nodes in sorted unique order.
        for node in np.unique(raw):
            if node not in self._label_to_id:
                self._label_to_id[node] = int(self._next_id)
                self._id_to_label[int(self._next_id)] = node
                self._next_id += 1
        return encode_series(series, self._label_to_id)


    # Decode

    def decode(self, encoded_ids) -> list:
        """
        Map encoded integer IDs back to original raw labels.

        Parameters
        ----------
        encoded_ids : int, list, or ndarray
        """
        if np.isscalar(encoded_ids):
            return self._id_to_label.get(int(encoded_ids), encoded_ids)
        return [self._id_to_label.get(int(i), i) for i in encoded_ids]


    # Properties / dunder helpers

    @property
    def n_nodes(self) -> int:
        """Total number of unique nodes registered so far."""
        return self._next_id

    def __len__(self) -> int:
        return self._next_id

    def __contains__(self, raw_id) -> bool:
        """True if raw_id has been registered."""
        return raw_id in self._label_to_id

    # Internal

    def _register(self, nodes: np.ndarray) -> None:
        """Add new nodes to the mapping (deterministic, avoids key-array conversions)."""
        for node in np.unique(nodes):
            if node not in self._label_to_id:
                self._label_to_id[node] = int(self._next_id)
                self._id_to_label[int(self._next_id)] = node
                self._next_id += 1


# ---------------------------------------------------------------------------
# Module-level helpers (reusable by temporal.py / second_order.py)
# ---------------------------------------------------------------------------

def _to_numpy(series: "pd_lib.Series") -> np.ndarray:
    """Return numpy array from a pandas or cuDF Series."""
    if hasattr(series, "to_pandas"):
        return series.to_pandas().to_numpy()
    return series.to_numpy()


def _unique_values(
    df: "pd_lib.DataFrame",
    src_col: str,
    dst_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract unique values from two columns as numpy arrays."""
    return _to_numpy(df[src_col]), _to_numpy(df[dst_col])


def encode_series(
    series: "pd_lib.Series",
    mapping: dict,
) -> "pd_lib.Series":
    """
    Map a Series of raw IDs through a label→id dict.

    Works for both pandas and cuDF by going through pandas .map(),
    then wrapping back into the correct Series type.

    """
    if _GPU and hasattr(series, "to_pandas"):
        mapped = series.to_pandas().map(mapping)
        return pd_lib.Series(mapped.values, dtype="int64")
    return series.map(mapping).astype("int64")




from __future__ import annotations

try:
    import cudf as pd_lib
    _GPU = True
except ImportError:
    import pandas as pd_lib  # type: ignore
    _GPU = False


# ---------------------------------------------------------------------------
# Temporal relay edges (motif mining input)
# ---------------------------------------------------------------------------

def build_temporal_edges(
    df,
    delta_w=5,
    max_fan=100,   # NEW: pruning
    return_trim_stats: bool = False,
):
    base = df[["event_id", "src_node", "dst_node", "step", "amount"]]

    left = base.rename(columns={
        "event_id": "event_id_1",
        "src_node": "src_1",
        "dst_node": "dst_1",
        "step": "step_1",
        "amount": "amount_1",
    })

    right = base.rename(columns={
        "event_id": "event_id_2",
        "src_node": "src_2",
        "dst_node": "dst_2",
        "step": "step_2",
        "amount": "amount_2",
    })

    # --- CRITICAL: prune hubs BEFORE join ---
    left = left.sort_values(["dst_1", "step_1", "event_id_1"])
    right = right.sort_values(["src_2", "step_2", "event_id_2"])

    left_rows_before = len(left)
    right_rows_before = len(right)
    left = left.groupby("dst_1").head(max_fan)
    right = right.groupby("src_2").head(max_fan)
    left_rows_after = len(left)
    right_rows_after = len(right)

    merged = left.merge(right, left_on="dst_1", right_on="src_2", how="inner")

    gap = merged["step_2"] - merged["step_1"]
    mask = (gap > 0) & (gap <= delta_w)

    te = merged.loc[mask].copy()
    te["_gap"] = gap[mask]

    te = te.reset_index(drop=True)
    if not return_trim_stats:
        return te

    trim_stats = {
        "max_fan_used": int(max_fan),
        "left_rows_before": int(left_rows_before),
        "left_rows_after": int(left_rows_after),
        "right_rows_before": int(right_rows_before),
        "right_rows_after": int(right_rows_after),
    }
    return te, trim_stats


def build_temporal_edges_debug(
    df,
    delta_w=5,
    max_fan=100,
    return_trim_stats: bool = False,
):
    """
    Debug-only temporal edges that preserve label-derived alert columns.
    Do NOT use these shards for model features.
    """
    base = df[["event_id", "src_node", "dst_node", "step", "amount", "is_sar"]]

    left = base.rename(columns={
        "event_id": "event_id_1",
        "src_node": "src_1",
        "dst_node": "dst_1",
        "step": "step_1",
        "amount": "amount_1",
        "is_sar": "alert_1",
    })

    right = base.rename(columns={
        "event_id": "event_id_2",
        "src_node": "src_2",
        "dst_node": "dst_2",
        "step": "step_2",
        "amount": "amount_2",
        "is_sar": "alert_2",
    })

    left = left.sort_values(["dst_1", "step_1", "event_id_1"])
    right = right.sort_values(["src_2", "step_2", "event_id_2"])

    left_rows_before = len(left)
    right_rows_before = len(right)
    left = left.groupby("dst_1").head(max_fan)
    right = right.groupby("src_2").head(max_fan)
    left_rows_after = len(left)
    right_rows_after = len(right)

    merged = left.merge(right, left_on="dst_1", right_on="src_2", how="inner")

    gap = merged["step_2"] - merged["step_1"]
    mask = (gap > 0) & (gap <= delta_w)

    te = merged.loc[mask].copy()
    te["_gap"] = gap[mask]

    te = te.reset_index(drop=True)
    if not return_trim_stats:
        return te

    trim_stats = {
        "max_fan_used": int(max_fan),
        "left_rows_before": int(left_rows_before),
        "left_rows_after": int(left_rows_after),
        "right_rows_before": int(right_rows_before),
        "right_rows_after": int(right_rows_after),
    }
    return te, trim_stats

def build_snapshot_edges(
    df: "pd_lib.DataFrame",
    src_col: str = "src_node",
    dst_col: str = "dst_node",
    amount_col: str = "amount",
    step_col: str = "step",
) -> "pd_lib.DataFrame":
    """
    Aggregate raw transactions into a directed weighted edge table.

    Groups all transactions in the window by (src_node, dst_node) and sums
    amounts.  This is the input required by build_snapshot_graph() and by
    community detection algorithms.

    Per graph_schema.md §6.2:
        W^(t)_{uv} = sum of all amounts from u to v within the window.

    Parameters
    ----------
    df : DataFrame
        Single time-window of transactions.  Must have src_col, dst_col,
        amount_col.  step_col is used to record the window's step range.
    src_col, dst_col, amount_col, step_col : str
        Column names in df.

    Returns
    -------
    DataFrame with columns:
        src_node, dst_node, weight, tx_count, step_min, step_max

    Notes
    -----
    - Direction is preserved: (u, v) and (v, u) remain separate rows.
    - No symmetrization is applied.
    - Self-loops should have been removed upstream by load_transactions().
    """
    agg = df.groupby([src_col, dst_col], as_index=False).agg(
        weight=(amount_col, "sum"),
        tx_count=(amount_col, "count"),
        step_min=(step_col, "min"),
        step_max=(step_col, "max"),
    )

    # Normalize column names to canonical form
    agg = agg.rename(columns={src_col: "src_node", dst_col: "dst_node"})

    return agg.reset_index(drop=True)
# Snapshot edge table (community detection input)
#
# NOTE: a full, typed `build_second_order_edges()` is defined below.
# Keep this legacy helper name distinct to avoid accidental overrides.

def build_second_order_edges_legacy(te):
    if len(te) == 0:
        return pd.DataFrame(
            columns=["src_2nd", "dst_2nd", "count", "weight_src", "weight_dst", "avg_gap"]
        )

    te = te[te["src_1"] != te["dst_2"]].copy()

    grouped = te.groupby(["src_1", "dst_2"], as_index=False).agg(
        count=("_gap", "count"),
        weight_src=("amount_1", "sum"),
        weight_dst=("amount_2", "sum"),
        avg_gap=("_gap", "mean"),
    )

    grouped = grouped.rename(columns={"src_1": "src_2nd", "dst_2": "dst_2nd"})
    return grouped





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

"""
import numpy as np
from scipy.sparse import csr_matrix
from __future__ import annotations

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
        src_1, dst_1, step_1, amount_1,
        src_2, dst_2, step_2, amount_2, _gap.

    Returns
    -------
    DataFrame with columns:
        src_2nd     : origin node of the relay chain
        dst_2nd     : final destination of the relay chain
        count       : number of relay hops between this pair
        weight_src  : total amount sent by src (sum of amount_1)
        weight_dst  : total amount received by dst (sum of amount_2)
        avg_gap     : mean step gap across relay hops

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

    # Compute time gap in-place (no copy)
    te = te.copy()  # single copy here to safely assign a new column
    te["_gap"] = te["step_2"] - te["step_1"]



    grouped = te.groupby(["src_1", "dst_2"], as_index=False).agg(
        count=("_gap", "count"),
        weight_src=("amount_1", "sum"),
        weight_dst=("amount_2", "sum"),
        avg_gap=("_gap", "mean"),
    )

    grouped = grouped.rename(columns={"src_1": "src_2nd", "dst_2": "dst_2nd"})

    # Cast to memory-efficient types
    grouped["weight_src"] = grouped["weight_src"].astype("float32")
    grouped["weight_dst"] = grouped["weight_dst"].astype("float32")
    grouped["avg_gap"] = grouped["avg_gap"].astype("float32")

    return grouped.reset_index(drop=True)


def build_second_order_edges_debug(
    temporal_edges_debug: "pd_lib.DataFrame",
) -> "pd_lib.DataFrame":
    """
    Debug-only second-order edges that preserve label-derived aggregates (n_alert).
    Do NOT use these shards for model features.
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

    if len(temporal_edges_debug) == 0:
        return pd_lib.DataFrame(
            {col: pd_lib.Series(dtype=dtype) for col, dtype in _EMPTY_SCHEMA.items()}
        )

    te = temporal_edges_debug
    te = te[te["src_1"] != te["dst_2"]]

    if len(te) == 0:
        return pd_lib.DataFrame(
            {col: pd_lib.Series(dtype=dtype) for col, dtype in _EMPTY_SCHEMA.items()}
        )

    te = te.copy()
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



# ============================================================
# CELL 6 — Graph Pipeline: stream all windows and save artifacts
# ============================================================

import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as _pd
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# Drive guard
# ---------------------------------------------------------------------------

if not os.path.isdir("/content/drive/MyDrive"):
    raise RuntimeError(
        "Google Drive is not mounted. "
        "Run Cell 0 first and mount /content/drive."
    )


# ---------------------------------------------------------------------------
# USER CONFIG
# ---------------------------------------------------------------------------

AML_DATA_PATH = "/content/drive/MyDrive/AML/dataset/tx_log.csv"
OUTPUT_DIR = Path("/content/drive/MyDrive/AML/outputs")

WINDOW_SIZE = 7
WINDOW_STRIDE = 7
DELTA_W = 5
WRITE_DEBUG_SHARDS = True

TEMPORAL_DIR = OUTPUT_DIR / "temporal_edges"
TEMPORAL_DEBUG_DIR = OUTPUT_DIR / "temporal_edges_debug"
SECOND_ORDER_DIR = OUTPUT_DIR / "second_order_edges"
SECOND_ORDER_DEBUG_DIR = OUTPUT_DIR / "second_order_edges_debug"
SNAPSHOT_DIR = OUTPUT_DIR / "snapshot_edges"
META_PATH = OUTPUT_DIR / "windows_meta.parquet"

for d in [
    OUTPUT_DIR,
    TEMPORAL_DIR,
    TEMPORAL_DEBUG_DIR,
    SECOND_ORDER_DIR,
    SECOND_ORDER_DEBUG_DIR,
    SNAPSHOT_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _to_np(s):
    if hasattr(s, "to_pandas"):
        return s.to_pandas().to_numpy()
    return s.to_numpy()


FORBIDDEN_FEATURE_LABEL_COLS = {"is_sar", "alert_1", "alert_2", "n_alert", "_n_alert"}


def assert_no_label_columns(df, artifact_name: str) -> None:
    leaked = FORBIDDEN_FEATURE_LABEL_COLS & set(df.columns)
    if leaked:
        raise ValueError(
            f"{artifact_name} contains label columns (forbidden in feature shards): {sorted(leaked)}"
        )


# ---------------------------------------------------------------------------
# Section 1 — Load and validate transactions
# ---------------------------------------------------------------------------

print("[1/7] Loading transactions...")
tx_df = load_transactions(AML_DATA_PATH)

print(f"      Rows: {len(tx_df):,}")
print(f"      Columns: {list(tx_df.columns)}")

_required_cols = {"event_id", "src_node", "dst_node", "amount", "step", "is_sar"}
missing = _required_cols - set(tx_df.columns)
if missing:
    raise ValueError(
        f"Missing required columns after load_transactions(): {sorted(missing)}. "
        f"Got: {sorted(set(tx_df.columns))}"
    )

assert str(tx_df["event_id"].dtype) == "int64", f"event_id dtype: {tx_df['event_id'].dtype}"
assert tx_df["event_id"].is_unique, "event_id must be unique per transaction."
assert str(tx_df["step"].dtype) == "int32", f"step dtype: {tx_df['step'].dtype}"
assert str(tx_df["amount"].dtype) == "float32", f"amount dtype: {tx_df['amount'].dtype}"
assert str(tx_df["is_sar"].dtype) == "int8", f"is_sar dtype: {tx_df['is_sar'].dtype}"
if "type_code" in tx_df.columns:
    assert str(tx_df["type_code"].dtype) == "int8", f"type_code dtype: {tx_df['type_code'].dtype}"

print("      Schema OK.")


# ---------------------------------------------------------------------------
# Section 2 — Global node encoding
# ---------------------------------------------------------------------------

print("\n[2/7] Encoding nodes globally...")

_raw_src = _to_np(tx_df["src_node"]).copy()
_raw_dst = _to_np(tx_df["dst_node"]).copy()
unique_raw = np.unique(np.concatenate([_raw_src, _raw_dst]))

encoder = NodeEncoder()
encoder.fit_transform(tx_df)

assert encoder.n_nodes == len(unique_raw), (
    f"Expected {len(unique_raw)} nodes, got {encoder.n_nodes}"
)
print(f"      Unique nodes: {encoder.n_nodes:,}")

# Round-trip sanity check
_enc = encoder.encode_column(_pd.Series(_raw_src))
_dec = encoder.decode(_to_np(_enc).tolist())
assert list(_dec) == _raw_src.tolist(), "Round-trip encode->decode failed."
print("      Round-trip encode->decode: OK")

del _raw_src, _raw_dst, _enc, _dec, unique_raw
gc.collect()


# ---------------------------------------------------------------------------
# Section 3 — Save global artifacts needed by downstream notebooks
# ---------------------------------------------------------------------------

print("\n[3/7] Saving global transaction artifacts...")

tx_path = OUTPUT_DIR / "transactions.parquet"
tx_df.to_parquet(tx_path, index=False)

node_map_df = _pd.DataFrame(
    [{"raw_id": k, "node_id": v} for k, v in encoder._label_to_id.items()]
)
node_map_path = OUTPUT_DIR / "node_map.parquet"
node_map_df.to_parquet(node_map_path, index=False)

print(f"      transactions.parquet -> {tx_path}")
print(f"      node_map.parquet     -> {node_map_path}")

del node_map_df
gc.collect()


# ---------------------------------------------------------------------------
# Section 4 — Stream all windows and build graph artifacts
# ---------------------------------------------------------------------------

print(f"\n[4/7] Streaming windows (size={WINDOW_SIZE}, stride={WINDOW_STRIDE})...")

window_stats = []
n_windows = 0

for window_id, (step_start, step_end, window_df) in enumerate(
    iter_windows(tx_df, window_size=WINDOW_SIZE, window_stride=WINDOW_STRIDE)
):
    n_windows += 1
    n_tx = len(window_df)

    if n_tx == 0:
        continue

    print(
        f"      Window {window_id:03d} [{step_start}, {step_end}] "
        f"rows={n_tx:,}",
        end="\r",
    )

    # --------------------------------------------------------
    # A. Temporal relay edges (exact motif substrate)
    # --------------------------------------------------------
    temporal_edges, temporal_trim_stats = build_temporal_edges(
        window_df, delta_w=DELTA_W, return_trim_stats=True
    )
    if WRITE_DEBUG_SHARDS:
        temporal_edges_debug, temporal_trim_stats_debug = build_temporal_edges_debug(
            window_df, delta_w=DELTA_W, return_trim_stats=True
        )
    else:
        temporal_edges_debug = None
        temporal_trim_stats_debug = {}

    # Validate temporal edges if non-empty
    if len(temporal_edges) > 0:
        expected_te = {
            "event_id_1", "src_1", "dst_1", "step_1", "amount_1",
            "event_id_2", "src_2", "dst_2", "step_2", "amount_2",
            "_gap",
        }
        missing_te = expected_te - set(temporal_edges.columns)
        if missing_te:
            raise ValueError(
                f"Temporal edges missing columns: {sorted(missing_te)}"
            )

        gaps = _to_np(temporal_edges["step_2"] - temporal_edges["step_1"])
        assert (gaps > 0).all(), "Temporal ordering violated in temporal_edges."
        assert (gaps <= DELTA_W).all(), f"Temporal gap exceeds DELTA_W={DELTA_W}."
        assert (_to_np(temporal_edges["_gap"]) == gaps).all(), "Mismatch in stored _gap."

    # Validate debug temporal edges if non-empty
    if WRITE_DEBUG_SHARDS and temporal_edges_debug is not None and len(temporal_edges_debug) > 0:
        expected_te_dbg = {
            "event_id_1", "src_1", "dst_1", "step_1", "amount_1", "alert_1",
            "event_id_2", "src_2", "dst_2", "step_2", "amount_2", "alert_2",
            "_gap",
        }
        missing_te_dbg = expected_te_dbg - set(temporal_edges_debug.columns)
        if missing_te_dbg:
            raise ValueError(
                f"Debug temporal edges missing columns: {sorted(missing_te_dbg)}"
            )

    # --------------------------------------------------------
    # B. Snapshot edges (community/global graph use)
    # --------------------------------------------------------
    snapshot_edges = build_snapshot_edges(window_df)

    if len(snapshot_edges) > 0:
        expected_se = {
            "src_node", "dst_node", "weight", "tx_count", "step_min", "step_max"
        }
        missing_se = expected_se - set(snapshot_edges.columns)
        if missing_se:
            raise ValueError(
                f"Snapshot edges missing columns: {sorted(missing_se)}"
            )

        assert (
            _to_np(snapshot_edges["src_node"]) != _to_np(snapshot_edges["dst_node"])
        ).all(), "Self-loops found in snapshot_edges."

        assert (_to_np(snapshot_edges["weight"]) > 0).all(), (
            "Non-positive weights found in snapshot_edges."
        )

    # --------------------------------------------------------
    # C. Second-order edges (feature engineering only)
    # --------------------------------------------------------
    second_order_edges = build_second_order_edges(temporal_edges)
    second_order_edges_debug = (
        build_second_order_edges_debug(temporal_edges_debug)
        if WRITE_DEBUG_SHARDS and temporal_edges_debug is not None
        else None
    )

    if len(second_order_edges) > 0:
        expected_so = {
            "src_2nd", "dst_2nd", "count",
            "weight_src", "weight_dst", "avg_gap",
        }
        missing_so = expected_so - set(second_order_edges.columns)
        if missing_so:
            raise ValueError(
                f"Second-order edges missing columns: {sorted(missing_so)}"
            )

        assert (
            _to_np(second_order_edges["src_2nd"]) != _to_np(second_order_edges["dst_2nd"])
        ).all(), "Self-relays found in second_order_edges."

    if WRITE_DEBUG_SHARDS and second_order_edges_debug is not None and len(second_order_edges_debug) > 0:
        expected_so_dbg = {
            "src_2nd", "dst_2nd", "count",
            "weight_src", "weight_dst", "avg_gap", "n_alert",
        }
        missing_so_dbg = expected_so_dbg - set(second_order_edges_debug.columns)
        if missing_so_dbg:
            raise ValueError(
                f"Debug second-order edges missing columns: {sorted(missing_so_dbg)}"
            )

    # --------------------------------------------------------
    # D. Snapshot adjacency check (do not save matrix per window)
    # --------------------------------------------------------
    A, n_dim = build_snapshot_graph(snapshot_edges, n_nodes=encoder.n_nodes)

    assert isinstance(A, csr_matrix)
    assert A.shape == (encoder.n_nodes, encoder.n_nodes)
    assert n_dim == encoder.n_nodes

    # --------------------------------------------------------
    # E. Save shards
    # --------------------------------------------------------
    shard_name = f"w_{step_start}_{step_end}.parquet"

    temporal_path = TEMPORAL_DIR / shard_name
    temporal_debug_path = TEMPORAL_DEBUG_DIR / shard_name
    second_order_path = SECOND_ORDER_DIR / shard_name
    second_order_debug_path = SECOND_ORDER_DEBUG_DIR / shard_name
    snapshot_path = SNAPSHOT_DIR / shard_name

    assert_no_label_columns(temporal_edges, "temporal_edges (feature)")
    temporal_edges.to_parquet(temporal_path, index=False)
    if WRITE_DEBUG_SHARDS and temporal_edges_debug is not None:
        temporal_edges_debug.to_parquet(temporal_debug_path, index=False)
    assert_no_label_columns(second_order_edges, "second_order_edges (feature)")
    second_order_edges.to_parquet(second_order_path, index=False)
    if WRITE_DEBUG_SHARDS and second_order_edges_debug is not None:
        second_order_edges_debug.to_parquet(second_order_debug_path, index=False)
    snapshot_edges.to_parquet(snapshot_path, index=False)

    # --------------------------------------------------------
    # F. Window metadata for notebook 2 candidate selection
    # --------------------------------------------------------
    # Window-local diagnostics (do not change snapshot/community logic)
    if len(snapshot_edges) > 0:
        _src = _to_np(snapshot_edges["src_node"])
        _dst = _to_np(snapshot_edges["dst_node"])
        n_nodes_window = int(np.unique(np.concatenate([_src, _dst])).shape[0])
    else:
        n_nodes_window = 0

    adj_sparsity_window = (
        float(A.nnz / (n_nodes_window ** 2)) if n_nodes_window > 0 else 0.0
    )

    window_stats.append({
        "window": int(window_id),
        "start": int(step_start),
        "end": int(step_end),
        "window_key": f"w_{int(step_start)}_{int(step_end)}",
        "n_tx": int(n_tx),
        "n_temporal": int(len(temporal_edges)),
        "n_second": int(len(second_order_edges)),
        "n_snapshot": int(len(snapshot_edges)),
        "n_nodes_window": int(n_nodes_window),
        "adj_nnz": int(A.nnz),
        "adj_sparsity": float(A.nnz / (encoder.n_nodes ** 2)) if encoder.n_nodes > 0 else 0.0,
        "adj_sparsity_window": float(adj_sparsity_window),
        # Hub trimming (max_fan pruning) diagnostics
        "max_fan_used": int(temporal_trim_stats.get("max_fan_used", 0)),
        "trim_left_rows_before": int(temporal_trim_stats.get("left_rows_before", 0)),
        "trim_left_rows_after": int(temporal_trim_stats.get("left_rows_after", 0)),
        "trim_right_rows_before": int(temporal_trim_stats.get("right_rows_before", 0)),
        "trim_right_rows_after": int(temporal_trim_stats.get("right_rows_after", 0)),
        "debug_trim_left_rows_before": int(temporal_trim_stats_debug.get("left_rows_before", 0))
        if WRITE_DEBUG_SHARDS
        else 0,
        "debug_trim_left_rows_after": int(temporal_trim_stats_debug.get("left_rows_after", 0))
        if WRITE_DEBUG_SHARDS
        else 0,
        "debug_trim_right_rows_before": int(temporal_trim_stats_debug.get("right_rows_before", 0))
        if WRITE_DEBUG_SHARDS
        else 0,
        "debug_trim_right_rows_after": int(temporal_trim_stats_debug.get("right_rows_after", 0))
        if WRITE_DEBUG_SHARDS
        else 0,
    })

    del (
        window_df,
        temporal_edges,
        temporal_edges_debug,
        snapshot_edges,
        second_order_edges,
        second_order_edges_debug,
        A,
    )
    gc.collect()

print(f"\n      Total non-empty windows processed: {len(window_stats):,}")


# ---------------------------------------------------------------------------
# Section 5 — Save metadata
# ---------------------------------------------------------------------------

print("\n[5/7] Saving windows metadata...")

windows_meta_df = _pd.DataFrame(window_stats).sort_values(
    ["start", "end"]
).reset_index(drop=True)

# Chronological split (window-level): train/val/test = 70/15/15
if len(windows_meta_df) > 0:
    n = len(windows_meta_df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    split = _pd.Series(["test"] * n, index=windows_meta_df.index, dtype="object")
    split.iloc[:n_train] = "train"
    split.iloc[n_train:n_train + n_val] = "val"
    windows_meta_df["split"] = split

windows_meta_df.to_parquet(META_PATH, index=False)

print(f"      windows_meta.parquet -> {META_PATH}")
print(f"      shape: {windows_meta_df.shape}")

if len(windows_meta_df) > 0:
    print("\n      Top windows by temporal relay count:")
    print(
        windows_meta_df.sort_values("n_temporal", ascending=False)
        .head(10)
        .to_string(index=False)
    )


# ---------------------------------------------------------------------------
# Section 6 — Save light pipeline config / manifest
# ---------------------------------------------------------------------------

print("\n[6/7] Saving graph manifest...")

manifest = {
    "artifact_schema_version": 2,
    "feature_shards_labeled": False,
    "debug_shards_written": True,
    "write_debug_shards": bool(WRITE_DEBUG_SHARDS),
    "forbidden_feature_label_cols": sorted(FORBIDDEN_FEATURE_LABEL_COLS),
    "aml_data_path": AML_DATA_PATH,
    "output_dir": str(OUTPUT_DIR),
    "window_size": int(WINDOW_SIZE),
    "window_stride": int(WINDOW_STRIDE),
    "delta_w": int(DELTA_W),
    "n_nodes": int(encoder.n_nodes),
    "n_windows": int(len(window_stats)),
    "artifacts": {
        "transactions": str(tx_path),
        "node_map": str(node_map_path),
        "temporal_edges_dir": str(TEMPORAL_DIR),
        "temporal_edges_debug_dir": str(TEMPORAL_DEBUG_DIR),
        "second_order_edges_dir": str(SECOND_ORDER_DIR),
        "second_order_edges_debug_dir": str(SECOND_ORDER_DEBUG_DIR),
        "snapshot_edges_dir": str(SNAPSHOT_DIR),
        "windows_meta": str(META_PATH),
    },
}

manifest_path = OUTPUT_DIR / "graph_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"      graph_manifest.json -> {manifest_path}")


# ---------------------------------------------------------------------------
# Section 7 — Final summary
# ---------------------------------------------------------------------------

print("\n[7/7] Done.")

for path in [
    tx_path,
    node_map_path,
    META_PATH,
    manifest_path,
]:
    print(f"   {path} ({os.path.getsize(path)/1024:.1f} KB)")

print("\nShard directories:")
for d in [TEMPORAL_DIR, SECOND_ORDER_DIR, SNAPSHOT_DIR]:
    n_files = len(list(d.glob("*.parquet")))
    print(f"   {d}  files={n_files:,}")

del tx_df, windows_meta_df, window_stats
gc.collect()
print("\nAll graph artifacts saved successfully.")






