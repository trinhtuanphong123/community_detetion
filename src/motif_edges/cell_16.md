
# ============================================================
# Cell 16: Post-run deduplication
# ============================================================

from pathlib import Path

def merge_and_deduplicate_motif_shards(
    motif_instance_dir: str,
    output_path:        str,
) -> pl.DataFrame:
    """
    Merge all per-window motif instance shards and deduplicate by canonical_key.
    Rule: keep the instance with the highest instance_score; if there's a tie,
    keep the one from the earliest window_id.
    """
    paths = sorted(Path(motif_instance_dir).glob("**/*.parquet"))
    if not paths:
        print("No motif instance shards found.")
        return pl.DataFrame()

    # LazyFrame: no data is loaded until collect().
    df_lazy = pl.scan_parquet([str(p) for p in paths])

    # Deduplicate with lazy sort + unique so Polars can stream with bounded RAM.
    df_deduped = (
        df_lazy
        .sort(["canonical_key", "instance_score", "window_id"], descending=[False, True, False])
        .unique(subset=["canonical_key"], keep="first")
        .collect(streaming=True)
    )

    df_deduped.write_parquet(output_path)
    print(f"Deduplicated: {df_deduped.height} instances")
    return df_deduped


def build_edge_participation_summary(
    membership_dir: str,
    df_edges:       pl.DataFrame,
    df_motifs_deduped: pl.DataFrame = None,
) -> pl.DataFrame:
    """
    For each edge_id, aggregate all motif memberships into summary statistics.
    Uses lazy scanning of parquet shards to keep memory usage low.
    """
    paths = sorted(Path(membership_dir).glob("**/*.parquet"))
    valid_paths = [str(p) for p in paths if p.stat().st_size > 0]
    if not valid_paths:
        print("No membership shards found.")
        return df_edges.with_columns([
            pl.lit(0).alias("num_motif_instances"),
            pl.lit(0).alias("num_motif_types"),
            pl.lit([], dtype=pl.List(pl.String)).alias("motif_types_list"),
            pl.lit(0.0).alias("motif_score_sum"),
            pl.lit(0.0).alias("motif_score_max"),
            pl.lit(0.0).alias("motif_score_mean"),
            pl.lit(0).alias("fan_in_count"),
            pl.lit(0).alias("fan_out_count"),
            pl.lit(0).alias("split_merge_count"),
            pl.lit(0).alias("center_in_out_count"),
            pl.lit(0.0).alias("edge_motif_score"),
        ])
    lf = pl.scan_parquet(valid_paths)
    if df_motifs_deduped is not None:
        deduped_ids = df_motifs_deduped["motif_instance_id"].to_list()
        lf = lf.filter(pl.col("motif_instance_id").is_in(deduped_ids))
        
    summary = (
        lf
        .group_by("edge_id")
        .agg([
            pl.col("motif_instance_id").n_unique().alias("num_motif_instances"),
            pl.col("motif_type").n_unique().alias("num_motif_types"),
            pl.col("motif_type").unique().alias("motif_types_list"),
            pl.col("instance_score").sum().alias("motif_score_sum"),
            pl.col("instance_score").max().alias("motif_score_max"),
            pl.col("instance_score").mean().alias("motif_score_mean"),
            (pl.col("matcher_type") == "fan_in").sum().alias("fan_in_count"),
            (pl.col("matcher_type") == "fan_out").sum().alias("fan_out_count"),
            (pl.col("matcher_type") == "split_merge").sum().alias("split_merge_count"),
            (pl.col("matcher_type") == "center_in_out").sum().alias("center_in_out_count"),
        ])
        .with_columns([
            (
                pl.col("num_motif_instances").log1p() * 0.4
                + pl.col("num_motif_types") * 0.3
                + pl.col("motif_score_max") * 0.3
            ).alias("edge_motif_score")
        ])
        .collect(streaming=True)
    )
    result = df_edges.join(summary, on="edge_id", how="left")
    result = result.with_columns([
        pl.col("num_motif_instances").fill_null(0),
        pl.col("num_motif_types").fill_null(0),
        pl.col("motif_score_sum").fill_null(0.0),
        pl.col("motif_score_max").fill_null(0.0),
        pl.col("motif_score_mean").fill_null(0.0),
        pl.col("fan_in_count").fill_null(0),
        pl.col("fan_out_count").fill_null(0),
        pl.col("split_merge_count").fill_null(0),
        pl.col("center_in_out_count").fill_null(0),
        pl.col("edge_motif_score").fill_null(0.0),
    ])
    return result

# Select the run namespace to aggregate (e.g., "production", "validation")
RUN_ID = "production"
CURRENT_RUN_MOTIF_DIR = f"{MOTIF_INSTANCE_DIR}/{RUN_ID}"
CURRENT_RUN_MEMBERSHIP_DIR = f"{MEMBERSHIP_DIR}/{RUN_ID}"

AUTO_RUN_AGGREGATION = True
DEDUPED_MOTIF_PATH = f"{OUTPUT_DIR}/all_motif_instances_deduped.parquet"

if AUTO_RUN_AGGREGATION and 'df_edges' in globals():
    print(f"Running post-run deduplication on '{RUN_ID}' outputs...")
    df_motif_deduped = merge_and_deduplicate_motif_shards(
        motif_instance_dir=CURRENT_RUN_MOTIF_DIR,
        output_path=DEDUPED_MOTIF_PATH
    )
    edge_participation = build_edge_participation_summary(
        membership_dir=CURRENT_RUN_MEMBERSHIP_DIR,
        df_edges=df_edges,
        df_motifs_deduped=df_motif_deduped
    )
else:
    if 'df_edges' not in globals():
        print("[NOTE] df_edges not found in globals. Skipping deduplication execution.")
    else:
        print(f"[NOTE] AUTO_RUN_AGGREGATION is False. Skipping deduplication for run '{RUN_ID}'.")

print("Cell 16 completed.")



