

import os
import polars as pl
from pathlib import Path

# Run exactly 1 window using validation mode
print("Running validation pipeline...")

# Select validation patterns (exactly: fan_in_4, fan_out_4, center_inout_3in_2out, split_merge_6 if ENABLE_CYCLES)
VALIDATION_PATTERNS = [
    fan_in_patterns[0],       # fan_in_4
    fan_out_patterns[0],      # fan_out_4
    center_inout_patterns[0],  # center_inout_3in_2out
    split_merge_patterns[0]   # split_merge_6
]

if globals().get("ENABLE_CYCLES", False):
    VALIDATION_PATTERNS.append(cycle_patterns[0])  # cycle_5

VALIDATION_MAX_INSTANCES_PER_PATTERN = {
    "fan_in": 100_000,
    "fan_out": 100_000,
    "split_merge": 20_000,
    "center_in_out": 20_000,
    "path_cycle": 20_000,
}

# Create a temporary matcher registry for validation and adjust limits
validation_matcher_registry = {k: v for k, v in MATCHER_REGISTRY.items()}
for pattern_type in ["fan_in", "fan_out", "split_merge", "center_in_out", "path_cycle"]:
    if pattern_type in validation_matcher_registry:
        matcher_instance = validation_matcher_registry[pattern_type]
        if hasattr(matcher_instance, 'max_instances_per_window'):
            matcher_instance.max_instances_per_window = VALIDATION_MAX_INSTANCES_PER_PATTERN.get(pattern_type, matcher_instance.max_instances_per_window)

# Run matchers over validation settings (run_mode="validation" targets 1 window by default)
if 'df_edges' in globals() and 'windows' in globals():
    validation_stats_df = run_matchers_over_windows(
        df_edges=df_edges,
        windows=windows,
        patterns_to_run=VALIDATION_PATTERNS,
        matcher_registry=validation_matcher_registry, # Use the modified registry
        run_mode="validation",
        skip_existing=False,
        write_empty_outputs=True
    )

    # Perform validation checks
    print("\nRunning post-run validation checks...")


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
            motif_files = sorted((motif_dir_path / pattern.name).glob("*.parquet"))
            membership_files = sorted((membership_dir_path / pattern.name).glob("*.parquet"))
            df_motifs = pl.scan_parquet([str(p) for p in motif_files]).collect() if motif_files else pl.DataFrame()
            df_members = pl.scan_parquet([str(p) for p in membership_files]).collect() if membership_files else pl.DataFrame()

            if motif_files and membership_files:

                num_instances = df_motifs.height
                num_members = df_members.height
                k_edges = len(pattern.edges)

                print(f"\nChecking pattern '{pattern.name}' (instances={num_instances}, expected_edges_per_instance={k_edges}):")

                # Check schema correctness
                expected_motif_cols = {
                    "motif_instance_id", "window_id", "motif_type", "matcher_type", "canonical_key",
                    "anchor_edge_id", "edge_ids", "instance_score", "candidate_rank", "num_edges",
                    "start_step", "end_step", "duration"
                }
                missing_motif_cols = expected_motif_cols - set(df_motifs.columns)
                if missing_motif_cols:
                    raise ValueError(f"Validation failed: motif instances schema missing columns: {missing_motif_cols}")

                expected_member_cols = {
                    "edge_id", "motif_instance_id", "window_id", "motif_type", "matcher_type",
                    "instance_score", "role_in_motif"
                }
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

