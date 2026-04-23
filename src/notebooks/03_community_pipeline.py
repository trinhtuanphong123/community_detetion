# notebooks/03_community_pipeline.py
# Community detection pipeline — dán từng block vào Colab cell.
# Chạy sau 00_setup.py, 01_graph_pipeline.py (đã có temporal + 2nd order edges).

# ============================================================
# CELL 1 — Imports
# ============================================================
"""
from src.graph.config import Config
from src.community.config import CommunityConfig
from src.community.weighting import compute_weights
from src.community.pipeline import run_community_pipeline

graph_cfg   = Config()
comm_cfg    = CommunityConfig()
print("Configs OK")
"""

# ============================================================
# CELL 2 — Load graph artifacts từ checkpoint (nếu đã lưu)
# ============================================================
"""
import cudf

temporal_edges     = cudf.read_parquet(f"{graph_cfg.OUTPUT_DIR}/temporal_edges.parquet")
second_order_edges = cudf.read_parquet(f"{graph_cfg.OUTPUT_DIR}/second_order_edges.parquet")

print(f"Temporal edges  : {len(temporal_edges):,}")
print(f"2nd order edges : {len(second_order_edges):,}")
"""

# ============================================================
# CELL 3 — Bước 4.3: Tính trọng số
# ============================================================
"""
weighted_temporal = compute_weights(
    second_order_edges=second_order_edges,
    temporal_edges=temporal_edges,
    cfg=comm_cfg,
    delta_w=graph_cfg.DELTA_W,
)
print(f"Weighted temporal edges: {len(weighted_temporal):,}")
"""

# ============================================================
# CELL 4 — Bước 4.4 + 4.5: Community pipeline đầy đủ
# ============================================================
"""
# df là cuDF DataFrame đã encode từ 01_graph_pipeline
# n_alert và n_rows được in ra ở bước load_raw

snapshots, comm_df, n_global_cids = run_community_pipeline(
    df=df,
    weighted_temporal_edges=weighted_temporal,
    cfg=comm_cfg,
    n_alert=n_alert,   # từ load_raw summary
    n_rows=n_rows,     # từ load_raw summary
)

print(f"Snapshots: {len(snapshots)}")
print(f"Global community IDs: {n_global_cids}")
print(f"Community records: {len(comm_df):,}")
"""

# ============================================================
# CELL 5 — Xem top suspicious communities
# ============================================================
"""
top = (
    comm_df[comm_df["is_suspicious"] == 1]
    .sort_values("suspicion_score", ascending=False)
    .head(comm_cfg.TOP_K_EXPORT)
)
print(f"Suspicious communities: {int(comm_df['is_suspicious'].sum())}")
print(top[["global_cid", "size", "suspicion_score", "alert_ratio",
           "n_layering", "step_start", "step_end"]].to_pandas())
"""

# ============================================================
# CELL 6 — Lưu kết quả
# ============================================================
"""
import os
os.makedirs(graph_cfg.OUTPUT_DIR, exist_ok=True)

comm_df.to_parquet(f"{graph_cfg.OUTPUT_DIR}/community_scores.parquet")
print(f"✅ Đã lưu: {graph_cfg.OUTPUT_DIR}/community_scores.parquet")

# Community assignment per node (dùng để join với motif features)
node_assignments = cudf.concat(
    [snap["partition_df"] for snap in snapshots], ignore_index=True
)
node_assignments.to_parquet(f"{graph_cfg.OUTPUT_DIR}/node_community_assignments.parquet")
print(f"✅ Đã lưu: {graph_cfg.OUTPUT_DIR}/node_community_assignments.parquet")
"""