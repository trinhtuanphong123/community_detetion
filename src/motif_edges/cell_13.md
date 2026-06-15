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
            del df_primary
            del df_extended
            gc.collect()
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
            del df_primary
            del df_extended
            gc.collect()
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

            pattern_motif_dir = f"{motif_dir}/{pattern.name}"
            pattern_membership_dir = f"{membership_dir}/{pattern.name}"

            # Check if output already exists (shard directories containing any parquets)
            has_motifs = os.path.exists(pattern_motif_dir) and any(f.endswith('.parquet') for f in os.listdir(pattern_motif_dir)) if os.path.exists(pattern_motif_dir) else False
            has_memberships = os.path.exists(pattern_membership_dir) and any(f.endswith('.parquet') for f in os.listdir(pattern_membership_dir)) if os.path.exists(pattern_membership_dir) else False

            # Check if output already exists
            if skip_existing and has_motifs and has_memberships:
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
                output_buffer = MatcherOutputBuffer(
                    motif_dir=pattern_motif_dir,
                    membership_dir=pattern_membership_dir,
                    flush_every_instances=500,
                )
                stats = matcher.match(
                    df_primary=df_primary,
                    df_extended=df_extended,
                    index=temporal_index,
                    window=window,
                    pattern=pattern,
                    output_buffer=output_buffer,
                )
                output_buffer.close(write_empty_outputs=write_empty_outputs)
                pattern_elapsed = time.time() - pattern_start_time
                stats["status"] = "completed"
                stats["index_build_seconds"] = float(index_elapsed)
                stats["total_pattern_seconds"] = float(pattern_elapsed)
                stats["motif_path"] = pattern_motif_dir
                stats["membership_path"] = pattern_membership_dir
                stats_rows.append(stats)
                print(
                    f"  [{pattern.name}] "
                    f"instances={stats['num_instances']}, "
                    f"membership_rows={stats['num_membership_rows']}, "
                    f"time={round(pattern_elapsed, 3)}s"
                )

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
        del df_primary
        del df_extended
        gc.collect()

        try:
            import psutil
            ram_mb = psutil.Process(os.getpid()).memory_info().rss / 1e6
            print(f"  RAM after window {w_id}: {ram_mb:.0f} MB")
        except ImportError:
            pass

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
