

# ============================================================
# Cell 19: Export Final Edge-Level Motif Participation Table
# ============================================================

from pathlib import Path
import os
import polars as pl

# ------------------------------------------------------------
# 1. Chọn run cần tổng hợp
# ------------------------------------------------------------
# Nếu bạn vừa chạy production thì để "production".
# Nếu chỉ mới chạy Cell 14 validation thì đổi thành "validation".
RUN_ID_FOR_EDGE_EXPORT = "production"

EDGE_OUTPUT_DIR = FEATURE_DIR
os.makedirs(EDGE_OUTPUT_DIR, exist_ok=True)

MEMBERSHIP_ROOT = f"{MEMBERSHIP_DIR}/{RUN_ID_FOR_EDGE_EXPORT}"

FINAL_EDGE_OUTPUT_PATH = (
    f"{EDGE_OUTPUT_DIR}/final_edge_motif_participation_{RUN_ID_FOR_EDGE_EXPORT}.parquet"
)

print("Running final edge-level export...")
print("RUN_ID:", RUN_ID_FOR_EDGE_EXPORT)
print("MEMBERSHIP_ROOT:", MEMBERSHIP_ROOT)
print("FINAL_EDGE_OUTPUT_PATH:", FINAL_EDGE_OUTPUT_PATH)


# ------------------------------------------------------------
# 2. Kiểm tra membership parquet files
# ------------------------------------------------------------
membership_files = sorted(Path(MEMBERSHIP_ROOT).glob("**/*.parquet"))
membership_files = [str(p) for p in membership_files if p.stat().st_size > 0]

print("Number of membership parquet files:", len(membership_files))

if len(membership_files) == 0:
    raise FileNotFoundError(
        f"No membership parquet files found under: {MEMBERSHIP_ROOT}. "
        "Check whether Cell 14/15 has produced edge_motif_membership outputs."
    )


# ------------------------------------------------------------
# 3. Đọc membership bằng lazy scan và tổng hợp theo edge_id
# ------------------------------------------------------------
membership_lf = pl.scan_parquet(membership_files)

if 'df_motif_deduped' in globals():
    deduped_ids_for_edge = df_motif_deduped["motif_instance_id"].to_list()
    membership_lf = membership_lf.filter(pl.col("motif_instance_id").is_in(deduped_ids_for_edge))

edge_motif_summary = (
    membership_lf
    .group_by("edge_id")
    .agg([
        # Số motif instance mà edge tham gia
        pl.col("motif_instance_id")
          .n_unique()
          .alias("num_motif_instances"),

        # Danh sách instance mà edge tham gia
        pl.col("motif_instance_id")
          .unique()
          .alias("motif_instance_ids"),

        # Số loại motif mà edge tham gia
        pl.col("motif_type")
          .n_unique()
          .alias("num_motif_types"),

        # Danh sách loại motif
        pl.col("motif_type")
          .unique()
          .alias("motif_types_list"),

        # Matcher types
        pl.col("matcher_type")
          .unique()
          .alias("matcher_types_list"),

        # Role của edge trong các motif
        pl.col("role_in_motif")
          .unique()
          .alias("roles_in_motif"),

        # Điểm motif tổng hợp
        pl.col("instance_score")
          .sum()
          .alias("motif_score_sum"),

        pl.col("instance_score")
          .max()
          .alias("motif_score_max"),

        pl.col("instance_score")
          .mean()
          .alias("motif_score_mean"),

        # Count theo matcher family
        (pl.col("matcher_type") == "fan_in")
          .sum()
          .alias("fan_in_count"),

        (pl.col("matcher_type") == "fan_out")
          .sum()
          .alias("fan_out_count"),

        (pl.col("matcher_type") == "split_merge")
          .sum()
          .alias("split_merge_count"),

        (pl.col("matcher_type") == "center_in_out")
          .sum()
          .alias("center_in_out_count"),

        (pl.col("matcher_type") == "path_cycle")
          .sum()
          .alias("cycle_count"),
    ])
    .with_columns([
        (
            pl.col("num_motif_instances").log1p() * 0.40
            + pl.col("num_motif_types") * 0.30
            + pl.col("motif_score_max") * 0.30
        ).alias("edge_motif_score")
    ])
    .collect(streaming=True)
)

print("edge_motif_summary shape:", edge_motif_summary.shape)


# ------------------------------------------------------------
# 4. Join lại với df_edges để có toàn bộ thông tin cạnh gốc
# ------------------------------------------------------------
edge_base_cols = [
    c for c in [
        "edge_id",
        "src",
        "dst",
        "step",
        "amount",
        "is_sar",
    ]
    if c in df_edges.columns
]

final_edge_table = (
    df_edges
    .select(edge_base_cols)
    .join(edge_motif_summary, on="edge_id", how="left")
)


# ------------------------------------------------------------
# 5. Fill null cho các edge không tham gia motif nào
# ------------------------------------------------------------
numeric_fill_zero_cols = [
    "num_motif_instances",
    "num_motif_types",
    "motif_score_sum",
    "motif_score_max",
    "motif_score_mean",
    "fan_in_count",
    "fan_out_count",
    "split_merge_count",
    "center_in_out_count",
    "cycle_count",
    "edge_motif_score",
]

list_fill_empty_cols = [
    "motif_instance_ids",
    "motif_types_list",
    "matcher_types_list",
    "roles_in_motif",
]

for col in numeric_fill_zero_cols:
    if col in final_edge_table.columns:
        final_edge_table = final_edge_table.with_columns(
            pl.col(col).fill_null(0)
        )

for col in list_fill_empty_cols:
    if col in final_edge_table.columns:
        final_edge_table = final_edge_table.with_columns(
            pl.when(pl.col(col).is_null())
            .then(pl.lit([], dtype=final_edge_table.schema[col]))
            .otherwise(pl.col(col))
            .alias(col)
        )

final_edge_table = final_edge_table.with_columns([
    (pl.col("num_motif_instances") > 0).cast(pl.Int8).alias("edge_in_any_motif")
])


# ------------------------------------------------------------
# 6. Ghi ra 1 file parquet duy nhất
# ------------------------------------------------------------
final_edge_table.write_parquet(FINAL_EDGE_OUTPUT_PATH)

print("\nFinal edge-level motif participation file saved:")
print(FINAL_EDGE_OUTPUT_PATH)

print("\nFinal edge table shape:", final_edge_table.shape)

print("\nPreview: edges with highest motif participation")
display(
    final_edge_table
    .sort("num_motif_instances", descending=True)
    .head(30)
)

print("\nCell 19 completed.")