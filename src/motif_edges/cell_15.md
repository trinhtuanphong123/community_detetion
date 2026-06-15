

# ============================================================
# Cell 15: Full Production Run
# ============================================================
# To make debugging and runtime control easier, motif families can be run separately
# using the RUN_GROUP control.

# Choose which group to run: "all", "fan", "flow", "cycle"
RUN_GROUP = "all"

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

MAX_WINDOWS_TO_RUN_ALL    = None    # Set to small int for quick test
SKIP_EXISTING_OUTPUTS_ALL = True
# Write empty outputs to prevent missing parquet file errors in downstream evaluation cells
WRITE_EMPTY_OUTPUTS_ALL   = True
AUTO_RUN_PRODUCTION       = True

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
