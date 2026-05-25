
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