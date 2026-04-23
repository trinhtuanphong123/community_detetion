# notebooks/01_graph_pipeline.py
# Pipeline đầy đủ cho src/graph/ — chạy tuần tự trong Colab.
# Mỗi block là một cell riêng trong notebook.

# ============================================================
# CELL 1 — Setup (sau khi đã chạy 00_setup.py)
# ============================================================
"""
from src.graph.config import Config
from src.graph.loader import load_raw, validate_schema
from src.graph.encoder import build_node_map, encode_nodes
from src.graph.temporal import create_temporal_graph
from src.graph.second_order import create_second_order_graph

cfg = Config()
"""

# ============================================================
# CELL 2 — Load raw data
# ============================================================
"""
df_raw = load_raw(cfg)
# df_raw columns: [step, src, dst, amount, is_laundering, ...]
"""

# ============================================================
# CELL 3 — Node encoding
# ============================================================
"""
node_map = build_node_map(df_raw)
df = encode_nodes(df_raw, node_map)

# Sau bước này:
# df["src"], df["dst"] là int32
# node_map: ['node_name', 'node_id'] dùng để decode về sau

del df_raw  # giải phóng bản gốc
import gc; gc.collect()
"""

# ============================================================
# CELL 4 — Temporal graph (bước 4.1)
# ============================================================
"""
temporal_edges = create_temporal_graph(df, cfg)

# temporal_edges columns:
#   src_1, dst_1, step_1, amount_1, alert_1,
#   src_2, dst_2, step_2, amount_2, alert_2
print(f"Temporal edges: {len(temporal_edges):,}")
print(temporal_edges.head(3).to_pandas())
"""

# ============================================================
# CELL 5 — 2nd Order graph (bước 4.2)
# ============================================================
"""
second_order_edges = create_second_order_graph(temporal_edges, cfg)

# second_order_edges columns:
#   src_2nd, dst_2nd, count, total_amount_src,
#   total_amount_dst, avg_time_gap, n_alert, _max_node_id
print(f"2nd order edges: {len(second_order_edges):,}")
print(second_order_edges.head(3).to_pandas())
"""

# ============================================================
# CELL 6 — Checkpoint: lưu kết quả để dùng ở community module
# ============================================================
"""
import os
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

temporal_edges.to_parquet(f"{cfg.OUTPUT_DIR}/temporal_edges.parquet")
second_order_edges.to_parquet(f"{cfg.OUTPUT_DIR}/second_order_edges.parquet")
node_map.to_parquet(f"{cfg.OUTPUT_DIR}/node_map.parquet")

print("✅ Graph artifacts đã lưu:")
print(f"  {cfg.OUTPUT_DIR}/temporal_edges.parquet")
print(f"  {cfg.OUTPUT_DIR}/second_order_edges.parquet")
print(f"  {cfg.OUTPUT_DIR}/node_map.parquet")
"""