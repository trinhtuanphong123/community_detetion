

# ============================================================
# Cell 12: Matcher registry
# ============================================================

# Control whether cycle motifs are enabled in diagnostic run
ENABLE_CYCLES = False

# Consolidated central matcher registration
MATCHER_REGISTRY = {
    "fan_in": FanMatcher(
        direction="in",
        cap_schedule=FAN_IN_CAP_SCHEDULE,
        max_instances_per_window=MAX_FAN_INSTANCES_PER_WINDOW,
        max_instances_per_center=MAX_FAN_INSTANCES_PER_CENTER,
        amount_min=AMOUNT_MIN,
        amount_coherence_ratio=3.0,
        delta_hop=DELTA_HOP,
    ),
    "fan_out": FanMatcher(
        direction="out",
        cap_schedule=FAN_OUT_CAP_SCHEDULE,
        max_instances_per_window=MAX_FAN_INSTANCES_PER_WINDOW,
        max_instances_per_center=MAX_FAN_INSTANCES_PER_CENTER,
        amount_min=AMOUNT_MIN,
        amount_coherence_ratio=3.0,
        delta_hop=DELTA_HOP,
        lookback_delta=None,
    ),
    "center_in_out": CenterInOutMatcher(
        max_in_candidates=12,
        max_out_candidates=6,
        max_instances_per_window=20_000,
        amount_min=None,
        use_amount_ratio=True,
        candidate_policy="hybrid",
        incoming_phase_delta=DELTA_HOP,
        outgoing_phase_delta=DELTA_HOP,
        center_handoff_delta=DELTA_HOP,
        max_instances_per_center=300,
    ),
    "split_merge": SplitMergeMatcher(
        max_out_candidates=10,
        max_merge_candidates_per_intermediate=2,
        max_instances_per_window=20_000,
        amount_min=None,
        use_amount_ratio=True,
        candidate_policy="hybrid",
        split_phase_delta=DELTA_HOP,
        merge_phase_delta=DELTA_HOP,
        split_to_merge_delta=DELTA_HOP,
        max_instances_per_source=300,
    ),
}

if ENABLE_CYCLES:
    MATCHER_REGISTRY["path_cycle"] = CycleKMatcher(
        max_branching=4,
        branching_schedule=CYCLE_MAX_BRANCHING,
        max_instances_per_window=20_000,
        amount_min=None,
        use_amount_ratio=False,
        candidate_policy="hybrid",
        delta_hop=DELTA_HOP,
        max_anchors_per_window=None,
    )

print("Matcher registry initialized.")
print("MATCHER_REGISTRY keys:", list(MATCHER_REGISTRY.keys()))
print("Cell 12 completed.")





