# notebooks/02_motif_pipeline.py
# Pipeline motif mining — dán từng block vào Colab cell.
# Chạy sau khi 00_setup.py và 01_graph_pipeline.py đã xong.

# ============================================================
# CELL 1 — Imports
# ============================================================
"""
import pandas as pd
import numpy as np

from src.motif.config import MotifConfig
from src.motif.index import build_event_indexes, filter_events_by_window
from src.motif.matchers import (
    find_fanin, find_fanout, find_cycle3, find_relay4, find_split_merge
)
from src.motif.scoring import count_support, compute_null_zscore
from src.motif.features import (
    build_entity_motif_features,
    build_entity_feature_wide,
    build_window_motif_features,
)

cfg_motif = MotifConfig()
print("MotifConfig OK — DELTA:", cfg_motif.DELTA, "| RHO:", cfg_motif.RHO_MIN, "–", cfg_motif.RHO_MAX)
"""

# ============================================================
# CELL 2 — Chuẩn bị event_df từ df đã encode (GPU → CPU)
# ============================================================
"""
# df là cuDF DataFrame từ 01_graph_pipeline.py
# Chuyển sang pandas vì motif matchers dùng Python iteration

event_df = df[["step", "src", "dst", "amount", "is_laundering"]].to_pandas()
event_df = event_df.sort_values(["step"]).reset_index(drop=True)
event_df["event_id"] = np.arange(len(event_df), dtype=np.int64)

print(f"event_df: {len(event_df):,} rows, steps {event_df['step'].min()}–{event_df['step'].max()}")
"""

# ============================================================
# CELL 3 — (Optional) Chạy trên một window nhỏ để kiểm tra
# ============================================================
"""
# Lọc 30 steps đầu để test nhanh
sample_df = filter_events_by_window(event_df, step_start=0, step_end=29)
out_idx_s, in_idx_s, _ = build_event_indexes(sample_df)

fanin_s  = find_fanin(in_idx_s, cfg_motif)
fanout_s = find_fanout(out_idx_s, cfg_motif)
cycle3_s = find_cycle3(out_idx_s, cfg_motif)
relay4_s = find_relay4(out_idx_s, cfg_motif)
sm_s     = find_split_merge(out_idx_s, in_idx_s, cfg_motif)

all_s = fanin_s + fanout_s + cycle3_s + relay4_s + sm_s
print("Sample motif counts:", count_support(all_s))
"""

# ============================================================
# CELL 4 — Chạy trên toàn bộ dataset
# ============================================================
"""
out_index, in_index, step_index = build_event_indexes(event_df)

fanin   = find_fanin(in_index, cfg_motif)
fanout  = find_fanout(out_index, cfg_motif)
cycle3  = find_cycle3(out_index, cfg_motif)
relay4  = find_relay4(out_index, cfg_motif)
sm      = find_split_merge(out_index, in_index, cfg_motif)

all_motifs = fanin + fanout + cycle3 + relay4 + sm

support = count_support(all_motifs)
print("Motif support:", support)
"""

# ============================================================
# CELL 5 — Z-score (null model)
# ============================================================
"""
# Chạy lâu hơn (~N_PERMUTATIONS lần search)
# Giảm N_PERMUTATIONS nếu chậm: cfg_motif.N_PERMUTATIONS = 10

zscore_table = compute_null_zscore(
    observed_counts=support,
    event_df=event_df,
    cfg=cfg_motif,
    verbose=True,
)
for mt, v in zscore_table.items():
    sig = "✅ significant" if v["zscore"] >= cfg_motif.Z_MIN else "—"
    print(f"  {mt:15s}: obs={v['observed']:5d}, z={v['zscore']:.2f}  {sig}")
"""

# ============================================================
# CELL 6 — Feature tables
# ============================================================
"""
# Entity-level features (long format)
entity_feat = build_entity_motif_features(all_motifs, zscore_table)
print(f"Entity features: {len(entity_feat):,} rows")
print(entity_feat.head(10))

# Entity-level features (wide format — dùng cho model input)
entity_wide = build_entity_feature_wide(entity_feat)
print(f"Entity wide: {entity_wide.shape}")
print(entity_wide.head(5))

# Window-level features
window_feat = build_window_motif_features(all_motifs, window_size=7)
print(f"Window features: {len(window_feat):,} rows")
print(window_feat.head(10))
"""

# ============================================================
# CELL 7 — Lưu kết quả
# ============================================================
"""
import os
from src.graph.config import Config
cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

entity_wide.to_parquet(f"{cfg.OUTPUT_DIR}/motif_entity_features.parquet", index=False)
window_feat.to_parquet(f"{cfg.OUTPUT_DIR}/motif_window_features.parquet", index=False)

print("✅ Motif features đã lưu:")
print(f"  {cfg.OUTPUT_DIR}/motif_entity_features.parquet")
print(f"  {cfg.OUTPUT_DIR}/motif_window_features.parquet")
"""