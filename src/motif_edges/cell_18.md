

# ============================================================
# Cell 18: Final summary tables
# ============================================================

def generate_summary_reports(
    run_stats_df: Optional[pl.DataFrame] = None,
    eval_dir: str = FEATURE_DIR,
):
    """
    Reads output metrics and prints human-readable summary tables.
    """
    print("=" * 60)
    print("                   FINAL MOTIF MINING SUMMARY REPORT")
    print("=" * 60)

    family_eval_path = f"{eval_dir}/motif_family_eval.parquet"

    if not os.path.exists(family_eval_path):
        print("Evaluation files not found. Please run aggregation and evaluation cells first.")
        return

    family_df = pl.read_parquet(family_eval_path)

    # 1. Instance count & SAR enrichment by motif type
    print("\n### 1. MOTIF PERFORMANCE AND ENRICHMENT SUMMARY")
    print("-" * 115)
    print(f"{'Motif Type':<25} | {'Instances':<10} | {'Prim SAR Rate':<13} | {'Prim Enrichment':<15} | {'Prim Capture':<12} | {'Ext Enrichment':<14}")
    print("-" * 115)
    for row in family_df.iter_rows(named=True):
        print(f"{row['motif_type']:<25} | {row['num_instances']:<10} | {row['primary_edge_sar_rate']:<13.4%} | {row['primary_anchor_enrichment']:<15.2f}x | {row['primary_sar_capture_rate']:<12.2%} | {row['extended_edge_enrichment']:<14.2f}x")

    # 2. Runtime by motif type
    if run_stats_df is not None and run_stats_df.height > 0:
        print("\n### 2. RUNTIME BY MOTIF TYPE")
        print("-" * 65)
        print(f"{'Motif Type':<25} | {'Total Match Time':<18} | {'Avg Window Time':<18}")
        print("-" * 65)

        completed_stats = run_stats_df.filter(pl.col("status") == "completed")
        if completed_stats.height > 0:
            runtime_df = (
                completed_stats
                .group_by("motif_type")
                .agg([
                    pl.col("elapsed_seconds").sum().alias("total_time"),
                    pl.col("elapsed_seconds").mean().alias("avg_time"),
                ])
                .sort("total_time", descending=True)
            )
            for row in runtime_df.iter_rows(named=True):
                print(f"{row['motif_type']:<25} | {row['total_time']:<18.2f}s | {row['avg_time']:<18.2f}s")
        else:
            print("No completed pattern runs found in stats.")

        # 3. Windows where caps were hit
        print("\n### 3. WINDOW CAPS SUMMARY")
        print("-" * 60)
        capped_df = run_stats_df.filter(
            (pl.col("status") == "completed") &
            (pl.col("hit_max_instances_per_window") > 0)
        )
        if capped_df.height > 0:
            print(f"{'Window ID':<10} | {'Motif Type':<25} | {'Status':<15}")
            print("-" * 60)
            for row in capped_df.iter_rows(named=True):
                print(f"{row['window_id']:<10} | {row['motif_type']:<25} | {'CAP HIT':<15}")
        else:
            print("No windows hit the maximum instance limit cap.")
    else:
        print("\n[NOTE] Run stats dataframe not provided; skipping runtime & cap tables.")

    # 4. Decision Rule analysis (Top families to keep / remove)
    print("\n### 4. RECOMMENDATIONS AND MOTIF KEEP/REMOVE DECISIONS")
    print("-" * 60)

    keep_df = family_df.filter(pl.col("primary_anchor_enrichment") >= 1.5)
    remove_df = family_df.filter(pl.col("primary_anchor_enrichment") < 1.0)
    monitor_df = family_df.filter((pl.col("primary_anchor_enrichment") >= 1.0) & (pl.col("primary_anchor_enrichment") < 1.5))

    print("  🟢 TOP MOTIF FAMILIES TO KEEP (Primary Enrichment >= 1.5):")
    if keep_df.height > 0:
        for row in keep_df.iter_rows(named=True):
            priority = "PRIORITIZE" if row['primary_anchor_enrichment'] > 2.0 else "USEFUL"
            print(f"    - {row['motif_type']:<20} (Primary Enrichment: {row['primary_anchor_enrichment']:.2f}x) -> DECISION: {priority}")
    else:
        print("    None")

    print("\n  🟡 MARGINAL MOTIF FAMILIES (Primary Enrichment 1.0 - 1.5):")
    if monitor_df.height > 0:
        for row in monitor_df.iter_rows(named=True):
            print(f"    - {row['motif_type']:<20} (Primary Enrichment: {row['primary_anchor_enrichment']:.2f}x) -> DECISION: MONITOR/RE-TUNE")
    else:
        print("    None")

    print("\n  🔴 MOTIF FAMILIES TO REMOVE (Primary Enrichment < 1.0):")
    if remove_df.height > 0:
        for row in remove_df.iter_rows(named=True):
            print(f"    - {row['motif_type']:<20} (Primary Enrichment: {row['primary_anchor_enrichment']:.2f}x) -> DECISION: REMOVE")
    else:
        print("    None")

    print("=" * 60)

# Run report generation automatically
if 'all_run_stats_df' in globals():
    generate_summary_reports(run_stats_df=all_run_stats_df)
elif 'validation_stats_df' in globals():
    print("\n[NOTE] Using validation run stats for summary tables:")
    generate_summary_reports(run_stats_df=validation_stats_df)
else:
    generate_summary_reports(run_stats_df=None)

print("Cell 18 completed.")



