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





cell 1:

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class GraphConfig:
    """
    Hyperparameters for temporal graph construction.

    window_sizes : standard analysis windows (days/steps), per graph_schema
    window_stride : step increment between rolling windows (overlap = size - stride)
    delta_w : max step gap allowed between two linked temporal events
    min_amount : filter out transactions at or below this value
    """

    # Rolling-window sizes used by community detection
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
        "src_node", "dst_node", "amount", "step", "is_sar",
    ])

    # Memory-efficient dtypes (§3.2 recommended types)
    dtypes: dict = field(default_factory=lambda: {
        "step":     "int32",
        "amount":   "float32",
        "is_sar":   "int8",
        "src_node": "int64",
        "dst_node": "int64",
    })

    window_size: int = 30
    window_stride: int = 15



cell 2: 


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


# Colab / Google Drive — set path to  AML transaction file (CSV or Parquet).
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


cell 3:



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
        new_nodes = np.setdiff1d(raw, np.array(list(self._label_to_id.keys())))
        if len(new_nodes) > 0:
            self._register(new_nodes)
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
        """Add new nodes to the mapping. Vectorized; no Python loop."""
        # Filter to nodes not yet registered
        known = np.array(list(self._label_to_id.keys())) if self._label_to_id else np.array([])
        if len(known) > 0:
            new_nodes = np.setdiff1d(nodes, known)
        else:
            new_nodes = nodes

        if len(new_nodes) == 0:
            return

        # Assign contiguous IDs starting from _next_id
        new_ids = np.arange(self._next_id, self._next_id + len(new_nodes))
        for node, nid in zip(new_nodes.tolist(), new_ids.tolist()):
            self._label_to_id[node] = int(nid)
            self._id_to_label[int(nid)] = node

        self._next_id += len(new_nodes)


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



cell 4: 



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
    df: "pd_lib.DataFrame",
    delta_w: int = 5,
    src_col: str = "src_node",
    dst_col: str = "dst_node",
    step_col: str = "step",
    amount_col: str = "amount",
    alert_col: str = "is_sar",
) -> "pd_lib.DataFrame":
    """
    Build temporal relay edges via self-join on an intermediary node.

    A temporal edge (tx_A → tx_B) is formed when:
      - tx_A.dst_node == tx_B.src_node   (relay: money passes through a node)
      - 0 < tx_B.step - tx_A.step ≤ delta_w  (temporal proximity)

    This captures the core relay structure needed for motif mining:
    fan-in, fan-out, relay chain, and cycle seeds all use this join.

    Parameters
    ----------
    df : DataFrame
        Single time-window of transactions.  Must have src_col, dst_col,
        step_col, amount_col.  alert_col is optional; filled with 0 if absent.
    delta_w : int
        Maximum step gap allowed between two linked transactions.
    src_col, dst_col, step_col, amount_col, alert_col : str
        Column names in df.

    Returns
    -------
    DataFrame with columns:
        src_1, dst_1, step_1, amount_1, alert_1,
        src_2, dst_2, step_2, amount_2, alert_2
    Sorted by step_1 ascending (temporal order preserved).

    Notes
    -----
    - No full copy of df is made; only the required columns are selected.
    - The join can be large for dense windows; caller should use bounded
      windows (iter_windows) to keep memory controlled.
    """
    # Guard: alert column may be absent in unlabelled data
    has_alert = alert_col in df.columns
    cols = [src_col, dst_col, step_col, amount_col]
    if has_alert:
        cols.append(alert_col)

    # Lightweight slice — no copy, just a column selection
    base = df[cols]

    # Left side: outgoing transaction (tx_A)
    left = base.rename(columns={
        src_col:    "src_1",
        dst_col:    "dst_1",
        step_col:   "step_1",
        amount_col: "amount_1",
        **({alert_col: "alert_1"} if has_alert else {}),
    })

    # Right side: incoming transaction (tx_B)
    right = base.rename(columns={
        src_col:    "src_2",
        dst_col:    "dst_2",
        step_col:   "step_2",
        amount_col: "amount_2",
        **({alert_col: "alert_2"} if has_alert else {}),
    })

    # Self-join: relay node is dst_1 == src_2
    merged = left.merge(right, left_on="dst_1", right_on="src_2", how="inner")

    # Temporal constraint: strict forward order within delta_w steps
    time_gap = merged["step_2"] - merged["step_1"]
    mask = (time_gap > 0) & (time_gap <= delta_w)
    temporal_edges = merged[mask].reset_index(drop=True)

    # Fill missing alert columns with 0 if unlabelled
    if not has_alert:
        temporal_edges["alert_1"] = 0
        temporal_edges["alert_2"] = 0

    # Sort by step_1 to preserve temporal order for downstream motif search
    temporal_edges = temporal_edges.sort_values("step_1").reset_index(drop=True)

    return temporal_edges


# Snapshot edge table (community detection input)

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



cell 5:

"""
second_order — Second-order (line graph) construction and sparse adjacency.

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
        src_1, dst_1, step_1, amount_1, alert_1,
        src_2, dst_2, step_2, amount_2, alert_2.

    Returns
    -------
    DataFrame with columns:
        src_2nd     : origin node of the relay chain
        dst_2nd     : final destination of the relay chain
        count       : number of relay hops between this pair
        weight_src  : total amount sent by src (sum of amount_1)
        weight_dst  : total amount received by dst (sum of amount_2)
        avg_gap     : mean step gap across relay hops
        n_alert     : total alert flags across all hops (alert_1 + alert_2)

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
        "n_alert":    "int64",
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

    # Compute time gap and combined alert in-place (no copy)
    te = te.copy()  # single copy here to safely assign new columns
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

    # Cast to memory-efficient types
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


cell 6: 

# ============================================================
# CELL 6 — Graph Pipeline: run end-to-end and save artifacts
# ============================================================

import os
import gc
import numpy as np
from scipy.sparse import csr_matrix

# -- Drive guard -------------------------------------------------------------
if not os.path.isdir("/content/drive/MyDrive"):
    raise RuntimeError(
        "Google Drive is not mounted. "
        "Run Cell 0 first: from google.colab import drive; "
        "drive.mount('/content/drive')"
    )

# -- USER CONFIG -------------------------------------------------------------
AML_DATA_PATH = "/content/drive/MyDrive/AML/dataset/tx_log.csv"
OUTPUT_DIR    = "/content/drive/MyDrive/AML/outputs"
WINDOW_SIZE   = 30   # must match GraphConfig.window_size
WINDOW_STRIDE = 15   # must match GraphConfig.window_stride
DELTA_W       = 5    # must match GraphConfig.delta_w
# ---------------------------------------------------------------------------

import pandas as _pd  # local alias: needed only for encode_column + node_map_df


def _to_np(s):
    """Coerce pandas or cuDF Series to numpy (null-safe, routes through pandas)."""
    if hasattr(s, "to_pandas"):
        return s.to_pandas().to_numpy()
    return s.to_numpy()


# -- Section 1: Load transactions --------------------------------------------
print("[1/6] Loading transactions...")
tx_df = load_transactions(AML_DATA_PATH)
print(f"      Rows: {len(tx_df):,}  |  Columns: {list(tx_df.columns)}")

_expected_cols = {"src_node", "dst_node", "amount", "step", "is_sar"}
assert _expected_cols == set(tx_df.columns), (
    f"Unexpected columns: {set(tx_df.columns) - _expected_cols}"
)
assert str(tx_df["step"].dtype)   == "int32",   f"step dtype: {tx_df['step'].dtype}"
assert str(tx_df["amount"].dtype) == "float32", f"amount dtype: {tx_df['amount'].dtype}"
assert str(tx_df["is_sar"].dtype) == "int8",    f"is_sar dtype: {tx_df['is_sar'].dtype}"
print("      Schema OK.")

# -- Section 2: Node encoding (global fit on the full dataset) ---------------
# Encode before windowing so node IDs are stable across all windows.
print("\n[2/6] NodeEncoder (global fit)...")
_raw_src   = _to_np(tx_df["src_node"]).copy()
_raw_dst   = _to_np(tx_df["dst_node"]).copy()
unique_raw = np.unique(np.concatenate([_raw_src, _raw_dst]))

encoder = NodeEncoder()
encoder.fit_transform(tx_df)   # tx_df now holds encoded int64 IDs in-place
# tx_df["step"] / tx_df["amount"] / tx_df["is_sar"] are unchanged

assert encoder.n_nodes == len(unique_raw), (
    f"Expected {len(unique_raw)} nodes, got {encoder.n_nodes}"
)
print(f"      Unique nodes: {encoder.n_nodes:,}")

# Round-trip: encode raw IDs -> decode back -> must match originals
_enc = encoder.encode_column(_pd.Series(_raw_src))
_dec = encoder.decode(_to_np(_enc).tolist())
assert list(_dec) == _raw_src.tolist(), "Round-trip encode->decode failed for src_node"
print("      Round-trip encode->decode: OK")

del _raw_src, _raw_dst, _enc, _dec
gc.collect()

# -- Section 3: Stream windows; preview first 3; find first non-empty --------
# Single pass: no list() materialisation, no double iteration.
print(f"\n[3/6] iter_windows(window_size={WINDOW_SIZE}, stride={WINDOW_STRIDE})...")

_n_windows    = 0
_window_df    = None
_window_start = None
_window_end   = None

for _s, _e, _w in iter_windows(tx_df, window_size=WINDOW_SIZE, window_stride=WINDOW_STRIDE):
    _n_windows += 1
    if _n_windows <= 3:
        print(f"      [{_s}, {_e}]  rows={len(_w)}")
    if _window_df is None:
        _snap_probe = build_snapshot_edges(_w)
        if len(_snap_probe) > 0:
            _window_df    = _w
            _window_start = _s
            _window_end   = _e
        del _snap_probe

print(f"      Total windows: {_n_windows}")
assert _n_windows >= 1, "iter_windows produced no windows -- check data range."

if _window_df is None:
    raise RuntimeError(
        "No window produced non-empty snapshot edges. "
        "Try a larger WINDOW_SIZE or check that AML_DATA_PATH is correct."
    )
print(f"      First non-empty window: [{_window_start}, {_window_end}] "
      f"({len(_window_df):,} rows)")

# -- Section 4: Temporal edges -----------------------------------------------
print("\n[4/6] build_temporal_edges...")
temporal_edges = build_temporal_edges(_window_df, delta_w=DELTA_W)

_expected_te = {
    "src_1", "dst_1", "step_1", "amount_1", "alert_1",
    "src_2", "dst_2", "step_2", "amount_2", "alert_2",
}
assert _expected_te <= set(temporal_edges.columns)
if len(temporal_edges) > 0:
    _gaps = _to_np(temporal_edges["step_2"] - temporal_edges["step_1"])
    assert (_gaps > 0).all(),        "Temporal ordering violated: step_2 must be > step_1"
    assert (_gaps <= DELTA_W).all(), f"Gap exceeds DELTA_W={DELTA_W}"
print(f"      Temporal edges shape: {temporal_edges.shape}")

# -- Section 5: Snapshot edges -----------------------------------------------
print("\n[5/6] build_snapshot_edges...")
snapshot_edges = build_snapshot_edges(_window_df)

_expected_se = {"src_node", "dst_node", "weight", "tx_count", "step_min", "step_max"}
assert _expected_se == set(snapshot_edges.columns)
assert len(snapshot_edges) == 0 or bool(
    (_to_np(snapshot_edges["src_node"]) != _to_np(snapshot_edges["dst_node"])).all()
), "Self-loops found in snapshot edges"
assert len(snapshot_edges) == 0 or bool(
    (_to_np(snapshot_edges["weight"]) > 0).all()
), "Non-positive weights found in snapshot edges"
print(f"      Snapshot edges shape: {snapshot_edges.shape}")

# -- Section 6: Second-order edges + sparse adjacency -----------------------
print("\n[6/6] build_second_order_edges + build_snapshot_graph...")
second_order_edges = build_second_order_edges(temporal_edges)

_expected_so = {"src_2nd", "dst_2nd", "count", "weight_src", "weight_dst", "avg_gap", "n_alert"}
assert _expected_so <= set(second_order_edges.columns)
if len(second_order_edges) > 0:
    assert bool(
        (_to_np(second_order_edges["src_2nd"]) != _to_np(second_order_edges["dst_2nd"])).all()
    ), "Self-relay found in second-order edges"
print(f"      2nd-order edges shape: {second_order_edges.shape}")

A, n_dim = build_snapshot_graph(snapshot_edges, n_nodes=encoder.n_nodes)
assert isinstance(A, csr_matrix)
assert A.shape == (encoder.n_nodes, encoder.n_nodes)
assert n_dim == encoder.n_nodes
assert A.nnz > 0, (
    "Adjacency matrix is all-zero. "
    "Check that snapshot_edges has non-zero weights and node IDs are encoded integers."
)
_sparsity = A.nnz / (encoder.n_nodes ** 2)
print(f"      CSR shape: {A.shape}  nnz: {A.nnz}  sparsity: {_sparsity:.6f}")

print("\nAll validation checks passed.")

# -- Save artifacts ----------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ← ADD THIS LINE HERE
tx_df.to_parquet(os.path.join(OUTPUT_DIR, 'transactions.parquet'), index=False)

# node_map: serialize the encoder label->id mapping as a parquet-safe DataFrame
node_map_df = _pd.DataFrame(
    [{"raw_id": k, "node_id": v} for k, v in encoder._label_to_id.items()]
)
node_map_df.to_parquet(os.path.join(OUTPUT_DIR, "node_map.parquet"), index=False)

temporal_edges.to_parquet(
    os.path.join(OUTPUT_DIR, "temporal_edges.parquet"), index=False
)
second_order_edges.to_parquet(
    os.path.join(OUTPUT_DIR, "second_order_edges.parquet"), index=False
)

print("\nArtifacts saved:")
for _fname in ("transactions.parquet", "node_map.parquet",   # ← add transactions here
               "temporal_edges.parquet", "second_order_edges.parquet"):
    _path = os.path.join(OUTPUT_DIR, _fname)
    _size = os.path.getsize(_path) / 1024
    print(f"   {_path}  ({_size:.1f} KB)")
del A, node_map_df, _window_df
gc.collect()