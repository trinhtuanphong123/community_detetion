


# ============================================================
# Cell 17: SAR evaluation and retrieval report
# ============================================================

def compute_structural_recall(
    edge_participation: pl.DataFrame,
) -> Dict[str, Any]:
    """
    Compute what fraction of SAR-labelled edges appear in at least one motif.
    """
    if "is_sar" not in edge_participation.columns:
        return {"error": "is_sar column not found"}
    sar_edges = edge_participation.filter(pl.col("is_sar") == 1)
    total_sar = sar_edges.height
    if total_sar == 0:
        return {"total_sar": 0, "recall": 0.0}
    sar_in_motif = sar_edges.filter(pl.col("num_motif_instances") > 0)
    covered_sar = sar_in_motif.height
    return {
        "total_sar": total_sar,
        "covered_sar": covered_sar,
        "recall": covered_sar / total_sar if total_sar > 0 else 0.0
    }

def evaluate_motif_performance(
    df_edges: pl.DataFrame,
    df_motifs: pl.DataFrame,
    df_memberships: pl.DataFrame,
    output_dir: str = FEATURE_DIR,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Computes rigorous SAR evaluation metrics on matched motifs, separating
    primary-scope and extended-scope metrics.
    Returns:
        motif_family_eval, window_eval, top_instances_by_score, top_instances_by_sar_ratio
    """
    if df_motifs.height == 0 or df_memberships.height == 0:
        print("No motif instances to evaluate.")
        empty = pl.DataFrame()
        return empty, empty, empty, empty

    # Find the windows list and create a dictionary
    windows_list = globals().get("windows", [])
    window_dict = {w.window_id: w for w in windows_list} if windows_list else {}

    # Find unique windows in this evaluation run
    run_window_ids = df_motifs["window_id"].unique().to_list()
    run_specs = [window_dict[w] for w in run_window_ids if w in window_dict]

    # Add window spec info to motifs
    if run_specs:
        specs_df = pl.DataFrame([
            {
                "window_id": int(ws.window_id),
                "primary_start": int(ws.primary_start),
                "primary_end": int(ws.primary_end),
                "extended_start": int(ws.extended_start),
                "extended_end": int(ws.extended_end),
            }
            for ws in run_specs
        ])
        df_motifs_eval = df_motifs.join(specs_df, on="window_id", how="left")
    else:
        df_motifs_eval = df_motifs.with_columns([
            pl.lit(-1).alias("primary_start"),
            pl.lit(999999).alias("primary_end"),
            pl.lit(-1).alias("extended_start"),
            pl.lit(999999).alias("extended_end"),
        ])

    # Join memberships with edges to get step and is_sar for each edge
    df_edge_info = df_edges.select(["edge_id", "step", "is_sar"])
    df_member_info = df_memberships.join(df_edge_info, on="edge_id", how="left").with_columns([
        pl.col("step").fill_null(-1),
        pl.col("is_sar").fill_null(0)
    ])

    # Join member info with motif specs
    df_member_info = df_member_info.join(
        df_motifs_eval.select(["motif_instance_id", "window_id", "primary_start", "primary_end", "extended_start", "extended_end"]),
        on="motif_instance_id",
        how="left"
    )

    # Label primary and extended membership edges
    df_member_info = df_member_info.with_columns([
        ((pl.col("step") >= pl.col("primary_start")) & (pl.col("step") <= pl.col("primary_end"))).alias("is_primary"),
        ((pl.col("step") >= pl.col("extended_start")) & (pl.col("step") <= pl.col("extended_end"))).alias("is_extended")
    ])

    # Compute instance-level SAR stats
    df_instance_sar = (
        df_member_info
        .group_by("motif_instance_id")
        .agg([
            pl.len().alias("k_edges"),
            pl.col("is_sar").sum().alias("sar_edges"),
        ])
        .with_columns(
            (pl.col("sar_edges") / pl.col("k_edges")).alias("instance_sar_ratio")
        )
    )
    df_motifs_eval = df_motifs_eval.join(df_instance_sar, on="motif_instance_id", how="left")

    # Run-level baselines
    if run_specs:
        conditions = []
        for ws in run_specs:
            conditions.append((pl.col("step") >= ws.primary_start) & (pl.col("step") <= ws.primary_end))
        df_primary_baseline = df_edges.filter(pl.any_horizontal(conditions))

        ext_conditions = []
        for ws in run_specs:
            ext_conditions.append((pl.col("step") >= ws.extended_start) & (pl.col("step") <= ws.extended_end))
        df_extended_baseline = df_edges.filter(pl.any_horizontal(ext_conditions))
    else:
        df_primary_baseline = df_edges
        df_extended_baseline = df_edges

    orig_primary_edges = df_primary_baseline.height
    orig_primary_sar = df_primary_baseline.filter(pl.col("is_sar") == 1).height
    orig_primary_sar_rate = orig_primary_sar / orig_primary_edges if orig_primary_edges > 0 else 0.0

    orig_extended_edges = df_extended_baseline.height
    orig_extended_sar = df_extended_baseline.filter(pl.col("is_sar") == 1).height
    orig_extended_sar_rate = orig_extended_sar / orig_extended_edges if orig_extended_edges > 0 else 0.0

    # 1. Evaluate by Motif Family (motif_type)
    motif_types = df_motifs_eval["motif_type"].unique().to_list()
    family_rows = []

    for mtype in motif_types:
        df_mtype_instances = df_motifs_eval.filter(pl.col("motif_type") == mtype)
        instance_ids = df_mtype_instances["motif_instance_id"].unique()

        df_mtype_members = df_member_info.filter(pl.col("motif_instance_id").is_in(instance_ids))

        # Primary-scope calculations
        df_mtype_members_prim = df_mtype_members.filter(pl.col("is_primary") == True)
        uniq_prim_edges = df_mtype_members_prim.unique(subset=["edge_id"])
        motif_uniq_edges_prim = uniq_prim_edges.height
        motif_uniq_sar_prim = uniq_prim_edges.filter(pl.col("is_sar") == 1).height
        motif_sar_rate_prim = motif_uniq_sar_prim / motif_uniq_edges_prim if motif_uniq_edges_prim > 0 else 0.0
        capture_rate_prim = motif_uniq_sar_prim / orig_primary_sar if orig_primary_sar > 0 else 0.0
        enrichment_prim = motif_sar_rate_prim / orig_primary_sar_rate if orig_primary_sar_rate > 0 else 1.0

        # Extended-scope calculations
        uniq_ext_edges = df_mtype_members.unique(subset=["edge_id"])
        motif_uniq_edges_ext = uniq_ext_edges.height
        motif_uniq_sar_ext = uniq_ext_edges.filter(pl.col("is_sar") == 1).height
        motif_sar_rate_ext = motif_uniq_sar_ext / motif_uniq_edges_ext if motif_uniq_edges_ext > 0 else 0.0
        capture_rate_ext = motif_uniq_sar_ext / orig_extended_sar if orig_extended_sar > 0 else 0.0
        enrichment_ext = motif_sar_rate_ext / orig_extended_sar_rate if orig_extended_sar_rate > 0 else 1.0

        # Instance level metrics
        mean_ratio = df_mtype_instances["instance_sar_ratio"].mean()
        median_ratio = df_mtype_instances["instance_sar_ratio"].median()
        any_sar = df_mtype_instances.filter(pl.col("sar_edges") > 0).height
        all_sar = df_mtype_instances.filter(pl.col("sar_edges") == pl.col("k_edges")).height

        family_rows.append({
            "motif_type": mtype,
            "original_primary_edges": orig_primary_edges,
            "original_primary_sar_edges": orig_primary_sar,
            "original_primary_sar_rate": orig_primary_sar_rate,
            "motif_unique_primary_edges": motif_uniq_edges_prim,
            "motif_unique_primary_sar_edges": motif_uniq_sar_prim,
            "primary_edge_sar_rate": motif_sar_rate_prim,
            "primary_sar_capture_rate": capture_rate_prim,
            "primary_anchor_enrichment": enrichment_prim,

            "original_extended_edges": orig_extended_edges,
            "original_extended_sar_edges": orig_extended_sar,
            "original_extended_sar_rate": orig_extended_sar_rate,
            "motif_unique_extended_edges": motif_uniq_edges_ext,
            "motif_unique_extended_sar_edges": motif_uniq_sar_ext,
            "extended_edge_sar_rate": motif_sar_rate_ext,
            "extended_sar_capture_rate": capture_rate_ext,
            "extended_edge_enrichment": enrichment_ext,

            "mean_instance_sar_ratio": mean_ratio,
            "median_instance_sar_ratio": median_ratio,
            "instances_with_any_sar": any_sar,
            "instances_with_all_sar": all_sar,
            "num_instances": df_mtype_instances.height,
        })

    motif_family_eval = pl.DataFrame(family_rows).sort("primary_anchor_enrichment", descending=True)

    # 2. Evaluate by Window (window_id)
    window_ids = df_motifs_eval["window_id"].unique().to_list()
    window_rows = []

    for w_id in window_ids:
        df_w_instances = df_motifs_eval.filter(pl.col("window_id") == w_id)
        w_instance_ids = df_w_instances["motif_instance_id"].unique()

        # Determine the baseline for this specific window
        if w_id in window_dict:
            ws = window_dict[w_id]
            df_w_edges_prim = df_edges.filter((pl.col("step") >= ws.primary_start) & (pl.col("step") <= ws.primary_end))
            df_w_edges_ext = df_edges.filter((pl.col("step") >= ws.extended_start) & (pl.col("step") <= ws.extended_end))
        else:
            df_w_edges_prim = df_edges
            df_w_edges_ext = df_edges

        w_orig_total_prim = df_w_edges_prim.height
        w_orig_sar_prim = df_w_edges_prim.filter(pl.col("is_sar") == 1).height
        w_orig_sar_rate_prim = w_orig_sar_prim / w_orig_total_prim if w_orig_total_prim > 0 else 0.0

        w_orig_total_ext = df_w_edges_ext.height
        w_orig_sar_ext = df_w_edges_ext.filter(pl.col("is_sar") == 1).height
        w_orig_sar_rate_ext = w_orig_sar_ext / w_orig_total_ext if w_orig_total_ext > 0 else 0.0

        # Unique edges in this window's motifs
        df_w_members = df_member_info.filter(pl.col("motif_instance_id").is_in(w_instance_ids))

        # Primary-scope calculations
        df_w_members_prim = df_w_members.filter(pl.col("is_primary") == True)
        uniq_w_prim = df_w_members_prim.unique(subset=["edge_id"])
        w_motif_uniq_edges_prim = uniq_w_prim.height
        w_motif_uniq_sar_prim = uniq_w_prim.filter(pl.col("is_sar") == 1).height
        w_motif_sar_rate_prim = w_motif_uniq_sar_prim / w_motif_uniq_edges_prim if w_motif_uniq_edges_prim > 0 else 0.0
        w_capture_prim = w_motif_uniq_sar_prim / w_orig_sar_prim if w_orig_sar_prim > 0 else 0.0
        w_enrichment_prim = w_motif_sar_rate_prim / w_orig_sar_rate_prim if w_orig_sar_rate_prim > 0 else 1.0

        # Extended-scope calculations
        uniq_w_ext = df_w_members.unique(subset=["edge_id"])
        w_motif_uniq_edges_ext = uniq_w_ext.height
        w_motif_uniq_sar_ext = uniq_w_ext.filter(pl.col("is_sar") == 1).height
        w_motif_sar_rate_ext = w_motif_uniq_sar_ext / w_motif_uniq_edges_ext if w_motif_uniq_edges_ext > 0 else 0.0
        w_capture_ext = w_motif_uniq_sar_ext / w_orig_sar_ext if w_orig_sar_ext > 0 else 0.0
        w_enrichment_ext = w_motif_sar_rate_ext / w_orig_sar_rate_ext if w_orig_sar_rate_ext > 0 else 1.0

        mean_ratio = df_w_instances["instance_sar_ratio"].mean()
        median_ratio = df_w_instances["instance_sar_ratio"].median()
        any_sar = df_w_instances.filter(pl.col("sar_edges") > 0).height
        all_sar = df_w_instances.filter(pl.col("sar_edges") == pl.col("k_edges")).height

        window_rows.append({
            "window_id": w_id,
            "original_primary_edges": w_orig_total_prim,
            "original_primary_sar_edges": w_orig_sar_prim,
            "original_primary_sar_rate": w_orig_sar_rate_prim,
            "motif_unique_primary_edges": w_motif_uniq_edges_prim,
            "motif_unique_primary_sar_edges": w_motif_uniq_sar_prim,
            "primary_edge_sar_rate": w_motif_sar_rate_prim,
            "primary_sar_capture_rate": w_capture_prim,
            "primary_anchor_enrichment": w_enrichment_prim,

            "original_extended_edges": w_orig_total_ext,
            "original_extended_sar_edges": w_orig_sar_ext,
            "original_extended_sar_rate": w_orig_sar_rate_ext,
            "motif_unique_extended_edges": w_motif_uniq_edges_ext,
            "motif_unique_extended_sar_edges": w_motif_uniq_sar_ext,
            "extended_edge_sar_rate": w_motif_sar_rate_ext,
            "extended_sar_capture_rate": w_capture_ext,
            "extended_edge_enrichment": w_enrichment_ext,

            "mean_instance_sar_ratio": mean_ratio,
            "median_instance_sar_ratio": median_ratio,
            "instances_with_any_sar": any_sar,
            "instances_with_all_sar": all_sar,
            "num_instances": df_w_instances.height,
        })

    window_eval = pl.DataFrame(window_rows).sort("window_id")

    # 3. Top instances by score
    top_instances_by_score = df_motifs_eval.sort("instance_score", descending=True).head(500)

    # 4. Top instances by SAR ratio
    top_instances_by_sar_ratio = df_motifs_eval.sort("instance_sar_ratio", descending=True).head(500)

    # Write files
    os.makedirs(output_dir, exist_ok=True)
    motif_family_eval.write_parquet(f"{output_dir}/motif_family_eval.parquet")
    window_eval.write_parquet(f"{output_dir}/window_eval.parquet")
    top_instances_by_score.write_parquet(f"{output_dir}/top_instances_by_score.parquet")
    top_instances_by_sar_ratio.write_parquet(f"{output_dir}/top_instances_by_sar_ratio.parquet")

    print("SAR evaluation files saved successfully.")
    return motif_family_eval, window_eval, top_instances_by_score, top_instances_by_sar_ratio

AUTO_RUN_EVALUATION = True

# Run evaluation automatically if data is available
if AUTO_RUN_EVALUATION and 'df_motif_deduped' in globals() and 'df_edges' in globals():
    # RUN_ID and CURRENT_RUN_MEMBERSHIP_DIR are inherited from Cell 16 globals
    print(f"Running evaluation on '{RUN_ID}' outputs...")
    deduped_instance_ids = df_motif_deduped["motif_instance_id"].to_list()
    df_deduped_members = (
        pl.scan_parquet(CURRENT_RUN_MEMBERSHIP_DIR + "/**/*.parquet")
        .filter(pl.col("motif_instance_id").is_in(deduped_instance_ids))
        .collect(streaming=True)
    )
    motif_family_eval, window_eval, top_instances_by_score, top_instances_by_sar_ratio = evaluate_motif_performance(
        df_edges=df_edges,
        df_motifs=df_motif_deduped,
        df_memberships=df_deduped_members
    )

    # Run structural recall
    recall_stats = compute_structural_recall(edge_participation)
    print("\nOverall Structural Recall on background SAR:")
    for k, v in recall_stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4%}")
        else:
            print(f"  {k}: {v}")
else:
    if 'df_motif_deduped' not in globals() or 'df_edges' not in globals():
        print("[NOTE] df_motif_deduped or df_edges not found in globals. Skipping evaluation execution.")
    else:
        print(f"[NOTE] AUTO_RUN_EVALUATION is False. Skipping evaluation for run '{RUN_ID}'.")

print("Cell 17 completed.")



