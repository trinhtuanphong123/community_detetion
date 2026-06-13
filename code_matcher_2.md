# ============================================================
# Cell 13: Runner and orchestration
# ============================================================

import os
import gc
import time
import traceback
import polars as pl
from typing import List, Dict, Any, Optional, Tuple

def run_matchers_over_windows(
    df_edges: pl.DataFrame,
    windows: List[WindowSpec],
    patterns_to_run: Optional[List[MotifPattern]] = None,
    matcher_registry: Optional[Dict[str, Any]] = None,
    run_mode: str = "production",
    max_windows: Optional[int] = None,
    skip_existing: bool = True,
    write_empty_outputs: bool = True,
    pattern_group: Optional[str] = None,
) -> pl.DataFrame:
    """
    Orchestrates the running of motif matchers over temporal windows.
    Supports debug, validation, and production modes, failure isolation,
    and structured output directory partitioning.
    """
    run_start_time = time.time()
    run_mode = run_mode.lower()
    if matcher_registry is None:
        matcher_registry = globals().get("MATCHER_REGISTRY", {})

    if patterns_to_run is None:
        if run_mode in ["debug", "validation"]:
            patterns_to_run = globals().get("VALIDATION_PATTERNS", [])
        else:
            patterns_to_run = globals().get("PATTERNS_TO_RUN_ALL", [])

    if max_windows is None:
        if run_mode == "debug":
            max_windows = 2
        elif run_mode == "validation":
            max_windows = 1

    if pattern_group is not None:
        pattern_group = pattern_group.lower()
        if pattern_group == "fan":
            patterns_to_run = [p for p in patterns_to_run if p.matcher_type in ["fan_in", "fan_out"]]
        elif pattern_group == "flow":
            patterns_to_run = [p for p in patterns_to_run if p.matcher_type in ["split_merge", "center_in_out"]]
        elif pattern_group == "cycle":
            patterns_to_run = [p for p in patterns_to_run if p.matcher_type == "path_cycle"]
        else:
            patterns_to_run = [
                p for p in patterns_to_run 
                if pattern_group in p.name.lower() or pattern_group in p.matcher_type.lower()
            ]

    windows_to_run = windows[:max_windows] if max_windows is not None else windows

    print("=" * 60)
    print(f"Starting motif mining runner (Mode: {run_mode.upper()})")
    print(f"Windows to run: {len(windows_to_run)}")
    print(f"Patterns ({len(patterns_to_run)}): {[p.name for p in patterns_to_run]}")
    print(f"Skip existing: {skip_existing}, Write empty outputs: {write_empty_outputs}")
    print("=" * 60)

    stats_rows = []

    for idx, window in enumerate(windows_to_run):
        window_start_time = time.time()
        w_id = int(window.window_id)

        print(f"\n[Window {idx + 1}/{len(windows_to_run)}] window_id={w_id}")
        
        # Setup paths
        motif_dir = f"{MOTIF_INSTANCE_DIR}/{run_mode}/window_id={w_id:06d}"
        membership_dir = f"{MEMBERSHIP_DIR}/{run_mode}/window_id={w_id:06d}"
        log_dir = f"{LOG_DIR}/{run_mode}/window_id={w_id:06d}"

        os.makedirs(motif_dir, exist_ok=True)
        os.makedirs(membership_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        df_primary, df_extended = slice_window_edges(df_edges, window)

        if df_primary.height == 0:
            print("  Primary window is empty. Skipping window.")
            for pattern in patterns_to_run:
                stats_rows.append({
                    "window_id": w_id,
                    "motif_type": pattern.name,
                    "matcher_type": pattern.matcher_type,
                    "status": "skipped_empty_primary",
                    "elapsed_seconds": 0.0,
                })
            continue

        if df_extended.height == 0:
            print("  Extended window is empty. Skipping window.")
            for pattern in patterns_to_run:
                stats_rows.append({
                    "window_id": w_id,
                    "motif_type": pattern.name,
                    "matcher_type": pattern.matcher_type,
                    "status": "skipped_empty_extended",
                    "elapsed_seconds": 0.0,
                })
            continue

        index_start_time = time.time()
        if 'temporal_index' in locals():
            del temporal_index
            gc.collect()

        temporal_index = build_temporal_index_from_polars(df_extended)
        index_elapsed = time.time() - index_start_time

        print(f"  Extended edges: {df_extended.height}, Primary edges: {df_primary.height}")
        print(f"  Temporal index build time: {round(index_elapsed, 3)}s")

        for pattern in patterns_to_run:
            pattern_start_time = time.time()
            
            motif_path = f"{motif_dir}/{pattern.name}.parquet"
            membership_path = f"{membership_dir}/{pattern.name}.parquet"

            # Check if output already exists
            if skip_existing and os.path.exists(motif_path) and os.path.exists(membership_path):
                print(f"  [{pattern.name}] outputs already exist. Skipping.")
                stats_rows.append({
                    "window_id": w_id,
                    "motif_type": pattern.name,
                    "matcher_type": pattern.matcher_type,
                    "status": "skipped_existing",
                    "elapsed_seconds": 0.0,
                })
                continue

            if pattern.matcher_type not in matcher_registry:
                err_msg = f"No matcher registered for type '{pattern.matcher_type}'"
                print(f"  [ERROR] [{pattern.name}]: {err_msg}")
                with open(f"{log_dir}/{pattern.name}_error.log", "w", encoding="utf-8") as f:
                    f.write(err_msg)
                stats_rows.append({
                    "window_id": w_id,
                    "motif_type": pattern.name,
                    "matcher_type": pattern.matcher_type,
                    "status": "failed",
                    "error_message": err_msg,
                    "elapsed_seconds": 0.0,
                })
                continue

            # Failure isolation wrapper
            try:
                matcher = matcher_registry[pattern.matcher_type]
                motif_df, membership_df, stats = matcher.match(
                    df_primary=df_primary,
                    df_extended=df_extended,
                    index=temporal_index,
                    window=window,
                    pattern=pattern,
                    write_output=False,
                )

                if write_empty_outputs or motif_df.height > 0:
                    motif_df.write_parquet(motif_path)
                    membership_df.write_parquet(membership_path)

                pattern_elapsed = time.time() - pattern_start_time
                stats["status"] = "completed"
                stats["index_build_seconds"] = float(index_elapsed)
                stats["total_pattern_seconds"] = float(pattern_elapsed)
                stats["motif_path"] = motif_path
                stats["membership_path"] = membership_path
                stats_rows.append(stats)

                print(
                    f"  [{pattern.name}] "
                    f"instances={motif_df.height}, "
                    f"membership_rows={membership_df.height}, "
                    f"time={round(pattern_elapsed, 3)}s"
                )

                # Explicit memory cleanup
                del motif_df
                del membership_df
                gc.collect()

            except Exception as e:
                tb = traceback.format_exc()
                print(f"  [ERROR] [{pattern.name}] Failed: {e}")
                
                # Write detailed error log
                with open(f"{log_dir}/{pattern.name}_error.log", "w", encoding="utf-8") as f:
                    f.write(f"Exception: {str(e)}\n\nTraceback:\n{tb}")

                stats_rows.append({
                    "window_id": w_id,
                    "motif_type": pattern.name,
                    "matcher_type": pattern.matcher_type,
                    "status": "failed",
                    "error_message": str(e),
                    "elapsed_seconds": float(time.time() - pattern_start_time),
                })

        window_elapsed = time.time() - window_start_time
        print(f"  Window {w_id} completed in {round(window_elapsed, 3)}s")

        if 'temporal_index' in locals():
            del temporal_index
            gc.collect()

    # Aggregate and return stats
    if len(stats_rows) == 0:
        run_stats_df = pl.DataFrame()
    else:
        run_stats_df = pl.DataFrame(stats_rows)

    # Save aggregated run stats to the LOG_DIR folder
    os.makedirs(LOG_DIR, exist_ok=True)
    stats_path = f"{LOG_DIR}/run_stats_{run_mode}.parquet"
    run_stats_df.write_parquet(stats_path)
    print(f"\nRun stats saved to {stats_path}")

    print(f"\nMotif mining runner finished in {round(time.time() - run_start_time, 3)}s")
    return run_stats_df

print("Cell 13 completed.")


# ============================================================
# Cell 14: Validation run
# ============================================================

import os
import polars as pl
from pathlib import Path

# Run exactly 1 window using validation mode
print("Running validation pipeline...")

# Select validation patterns (exactly: fan_in_4, fan_out_4, center_inout_3in_2out, split_merge_6, cycle_5 if ENABLE_CYCLES)
VALIDATION_PATTERNS = [
    fan_in_patterns[0],       # fan_in_4
    fan_out_patterns[0],      # fan_out_4
    center_inout_patterns[0],  # center_inout_3in_2out
    split_merge_patterns[0]   # split_merge_6
]

if globals().get("ENABLE_CYCLES", False):
    VALIDATION_PATTERNS.append(cycle_patterns[0])  # cycle_5

# Run matchers over validation settings (run_mode="validation" targets 1 window by default)
if 'df_edges' in globals() and 'windows' in globals():
    validation_stats_df = run_matchers_over_windows(
        df_edges=df_edges,
        windows=windows,
        patterns_to_run=VALIDATION_PATTERNS,
        matcher_registry=MATCHER_REGISTRY,
        run_mode="validation",
        skip_existing=False,
        write_empty_outputs=True
    )

    # Perform validation checks
    print("\nRunning post-run validation checks...")

    VALIDATION_MAX_INSTANCES_PER_PATTERN = {
        "fan_in": 100_000,
        "fan_out": 100_000,
        "split_merge": 20_000,
        "center_in_out": 20_000,
        "path_cycle": 20_000,
    }

    # 1. No matcher crash
    if validation_stats_df.height > 0 and "status" in validation_stats_df.columns:
        failed_runs = validation_stats_df.filter(pl.col("status") == "failed")
        if failed_runs.height > 0:
            print("[FAIL] One or more matchers crashed during execution:")
            for row in failed_runs.iter_rows(named=True):
                print(f"  Pattern: {row['motif_type']}, Error: {row.get('error_message', 'Unknown error')}")
            raise ValueError("Validation failed: Matcher crashed")
        else:
            print("[PASS] No matcher crashes detected.")
    else:
        raise ValueError("Validation failed: Empty stats returned from run.")

    # 2. Schema correctness, canonical_key uniqueness, membership rows count consistency
    motif_dir_path = Path(MOTIF_INSTANCE_DIR) / "validation" / "window_id=000000"
    membership_dir_path = Path(MEMBERSHIP_DIR) / "validation" / "window_id=000000"

    if motif_dir_path.exists() and membership_dir_path.exists():
        for pattern in VALIDATION_PATTERNS:
            motif_file = motif_dir_path / f"{pattern.name}.parquet"
            membership_file = membership_dir_path / f"{pattern.name}.parquet"
            
            if motif_file.exists() and membership_file.exists():
                df_motifs = pl.read_parquet(motif_file)
                df_members = pl.read_parquet(membership_file)
                
                num_instances = df_motifs.height
                num_members = df_members.height
                k_edges = len(pattern.edges)
                
                print(f"\nChecking pattern '{pattern.name}' (instances={num_instances}, expected_edges_per_instance={k_edges}):")
                
                # Check schema correctness
                expected_motif_cols = {
                    "motif_instance_id", "window_id", "motif_type", "anchor_edge_id",
                    "canonical_key", "instance_score", "candidate_rank",
                    "time_compactness", "amount_consistency", "amount_sum_log",
                    "degree_penalty", "flow_ratio", "flow_gap", "center_degree"
                }
                missing_motif_cols = expected_motif_cols - set(df_motifs.columns)
                if missing_motif_cols:
                    raise ValueError(f"Validation failed: motif instances schema missing columns: {missing_motif_cols}")
                
                expected_member_cols = {"motif_instance_id", "edge_id", "role_in_motif"}
                missing_member_cols = expected_member_cols - set(df_members.columns)
                if missing_member_cols:
                    raise ValueError(f"Validation failed: membership schema missing columns: {missing_member_cols}")
                    
                print(f"  [PASS] Schemas for motifs and memberships are correct.")
                
                # Check canonical_key uniqueness
                if "canonical_key" in df_motifs.columns:
                    num_unique_keys = df_motifs["canonical_key"].n_unique()
                    if num_unique_keys != num_instances:
                        raise ValueError(f"Validation failed: duplicate canonical_keys found in motifs ({num_unique_keys} unique vs {num_instances} total)")
                    print(f"  [PASS] Canonical keys are unique.")
                
                # Check membership rows match num_instances * num_edges
                expected_members_count = num_instances * k_edges
                if num_members != expected_members_count:
                    raise ValueError(f"Validation failed: membership rows mismatch for {pattern.name}. Got {num_members}, expected {expected_members_count}")
                print(f"  [PASS] Membership rows count matches num_instances * num_edges.")
                
                # Check if any pattern hits max_instances_per_window
                pattern_stats = validation_stats_df.filter(pl.col("motif_type") == pattern.name)
                if pattern_stats.height > 0 and "hit_max_instances_per_window" in pattern_stats.columns:
                    hit_cap = pattern_stats["hit_max_instances_per_window"][0]
                    if hit_cap == 1:
                        raise ValueError(f"Validation failed: Pattern {pattern.name} hit max_instances_per_window cap!")
                    else:
                        print(f"  [PASS] Pattern {pattern.name} did not hit max_instances_per_window cap.")
                else:
                    print(f"  [PASS] Pattern {pattern.name} did not hit max_instances_per_window cap.")
                        
                # Check num_instances is reasonable
                max_allowed = VALIDATION_MAX_INSTANCES_PER_PATTERN.get(pattern.matcher_type, 1_000_000)
                if num_instances < 0 or num_instances > max_allowed:
                    raise ValueError(f"Validation failed: unreasonable instance count ({num_instances}) for pattern {pattern.name} (max allowed for {pattern.matcher_type} is {max_allowed})")
                print(f"  [PASS] Instance count ({num_instances}) is reasonable (<= {max_allowed}).")
    else:
        raise ValueError("[FAIL] Could not locate output directories for window 0.")

    print("\n[SUCCESS] Validation run checks passed!")
else:
    print("[NOTE] df_edges or windows not found in globals. Skipping validation run execution.")

print("Cell 14 completed.")


# ============================================================
# Cell 15: Full Production Run
# ============================================================
# To make debugging and runtime control easier, motif families can be run separately
# using the RUN_GROUP control.

# Choose which group to run: "all", "fan", "flow", "cycle"
RUN_GROUP = "flow" 

# Full production set consists of Core and Exploration patterns
PATTERNS_TO_RUN_ALL = CORE_PATTERNS + EXPLORATION_PATTERNS

if globals().get("ENABLE_CYCLES", False):
    PATTERNS_TO_RUN_ALL = PATTERNS_TO_RUN_ALL + DIAGNOSTIC_PATTERNS

# Filter patterns based on RUN_GROUP
if RUN_GROUP == "fan":
    patterns_to_run = [p for p in PATTERNS_TO_RUN_ALL if p.matcher_type in ["fan_in", "fan_out"]]
elif RUN_GROUP == "flow":
    patterns_to_run = [p for p in PATTERNS_TO_RUN_ALL if p.matcher_type in ["split_merge", "center_in_out"]]
elif RUN_GROUP == "cycle":
    patterns_to_run = [p for p in PATTERNS_TO_RUN_ALL if p.matcher_type == "path_cycle"]
else:
    patterns_to_run = PATTERNS_TO_RUN_ALL

print(f"Running production run for group: {RUN_GROUP}")
print(f"Patterns to match: {[p.name for p in patterns_to_run]}")

MAX_WINDOWS_TO_RUN_ALL    = 2    # Set to small int for quick test
SKIP_EXISTING_OUTPUTS_ALL = True
# Write empty outputs to prevent missing parquet file errors in downstream evaluation cells
WRITE_EMPTY_OUTPUTS_ALL   = True
AUTO_RUN_PRODUCTION       = False

if AUTO_RUN_PRODUCTION and 'df_edges' in globals() and 'windows' in globals():
    all_run_stats_df = run_matchers_over_windows(
        df_edges=df_edges,
        windows=windows,
        patterns_to_run=patterns_to_run,
        matcher_registry=MATCHER_REGISTRY,
        run_mode="production",
        max_windows=MAX_WINDOWS_TO_RUN_ALL,
        skip_existing=SKIP_EXISTING_OUTPUTS_ALL,
        write_empty_outputs=WRITE_EMPTY_OUTPUTS_ALL
    )
    print("\nProduction run complete.")
else:
    print("[NOTE] AUTO_RUN_PRODUCTION is False or df_edges/windows not found. Skipping production run execution.")

print("Cell 15 completed.")


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
) -> pl.DataFrame:
    """
    For each edge_id, aggregate all motif memberships into summary statistics.

    Output columns:
        edge_id, num_motif_instances, num_motif_types,
        motif_types_list, roles_list,
        + all original df_edges columns (joined)
    """
    paths  = sorted(Path(membership_dir).glob("**/*.parquet"))
    shards = [pl.read_parquet(p) for p in paths if p.stat().st_size > 0]

    if not shards:
        print("No membership shards found.")
        return df_edges

    df_membership = pl.concat(shards)
    df_membership = df_membership.unique(subset=["edge_id", "motif_instance_id"])

    summary = (
        df_membership
        .group_by("edge_id")
        .agg([
            pl.col("motif_instance_id").n_unique().alias("num_motif_instances"),
            pl.col("motif_type").n_unique().alias("num_motif_types"),
            pl.col("motif_type").unique().alias("motif_types_list"),
            pl.col("role_in_motif").unique().alias("roles_list"),
        ])
    )

    result = df_edges.join(summary, on="edge_id", how="left")
    result = result.with_columns([
        pl.col("num_motif_instances").fill_null(0),
        pl.col("num_motif_types").fill_null(0),
    ])

    return result

# Select the run namespace to aggregate (e.g., "production", "validation")
RUN_ID = "production"
CURRENT_RUN_MOTIF_DIR = f"{MOTIF_INSTANCE_DIR}/{RUN_ID}"
CURRENT_RUN_MEMBERSHIP_DIR = f"{MEMBERSHIP_DIR}/{RUN_ID}"

AUTO_RUN_AGGREGATION = False
DEDUPED_MOTIF_PATH = f"{OUTPUT_DIR}/all_motif_instances_deduped.parquet"

if AUTO_RUN_AGGREGATION and 'df_edges' in globals():
    print(f"Running post-run deduplication on '{RUN_ID}' outputs...")
    df_motif_deduped = merge_and_deduplicate_motif_shards(
        motif_instance_dir=CURRENT_RUN_MOTIF_DIR,
        output_path=DEDUPED_MOTIF_PATH
    )
    edge_participation = build_edge_participation_summary(
        membership_dir=CURRENT_RUN_MEMBERSHIP_DIR,
        df_edges=df_edges
    )
else:
    if 'df_edges' not in globals():
        print("[NOTE] df_edges not found in globals. Skipping deduplication execution.")
    else:
        print(f"[NOTE] AUTO_RUN_AGGREGATION is False. Skipping deduplication for run '{RUN_ID}'.")

print("Cell 16 completed.")


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

AUTO_RUN_EVALUATION = False

# Run evaluation automatically if data is available
if AUTO_RUN_EVALUATION and 'df_motif_deduped' in globals() and 'df_edges' in globals():
    # RUN_ID and CURRENT_RUN_MEMBERSHIP_DIR are inherited from Cell 16 globals
    print(f"Running evaluation on '{RUN_ID}' outputs...")
    df_all_members = pl.read_parquet(CURRENT_RUN_MEMBERSHIP_DIR + "/**/*.parquet")
    df_deduped_members = df_all_members.filter(pl.col("motif_instance_id").is_in(df_motif_deduped["motif_instance_id"]))
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
