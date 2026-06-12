# ============================================================
# Cell 9: FanInMatcher (generalized, variable n)
# ============================================================

from itertools import combinations


def is_edge_in_primary_window(edge: EdgeRecord, window: WindowSpec) -> bool:
    return window.primary_start <= edge.step <= window.primary_end


def get_anchor_edge(edges: List[EdgeRecord]) -> EdgeRecord:
    return min(edges, key=lambda e: (e.step, e.edge_id))


def passes_amount_min(edges: List[EdgeRecord], amount_min: Optional[float]) -> bool:
    if amount_min is None:
        return True
    return all(e.amount >= amount_min for e in edges)


def passes_distinct_nodes_for_fanin(edges: List[EdgeRecord]) -> bool:
    """
    All edges must share the same dst.
    All src nodes must be distinct.
    No src may equal dst.
    All edge_ids must be distinct.
    Works for any n >= 3.
    """
    if not edges:
        return False
    if len({e.edge_id for e in edges}) != len(edges):
        return False
    dst_values = {e.dst for e in edges}
    if len(dst_values) != 1:
        return False
    src_values = [e.src for e in edges]
    if len(src_values) != len(set(src_values)):
        return False
    center_dst = next(iter(dst_values))
    return not any(s == center_dst for s in src_values)


def passes_fanin_duration(edges: List[EdgeRecord], max_duration: int) -> bool:
    steps = [e.step for e in edges]
    return (max(steps) - min(steps)) <= max_duration


def passes_consecutive_delta_hop_sorted(
    edges_sorted_by_step: List[EdgeRecord],
    delta_hop: Optional[int],
) -> bool:
    """
    Expects edges already sorted by (step, edge_id).
    Checks step[i+1] - step[i] <= delta_hop for all consecutive pairs.
    """
    if delta_hop is None or len(edges_sorted_by_step) <= 1:
        return True
    steps = [e.step for e in edges_sorted_by_step]
    return all(steps[i+1] - steps[i] <= delta_hop for i in range(len(steps)-1))


def canonicalize_fanin_edges(edges: List[EdgeRecord]) -> List[EdgeRecord]:
    return sorted(edges, key=lambda e: (e.step, e.edge_id))


def prefilter_by_time_window(
    candidates: List[EdgeRecord],
    max_duration: int,
) -> List[EdgeRecord]:
    """
    Cheap O(n) pre-filter before combinations().

    After sorting by step, any edge whose step is more than
    max_duration ahead of the earliest edge's step can only
    appear in a combination that violates duration.
    Remove it from the candidate list entirely.

    This reduces combinations from C(n, k) to C(n', k)
    where n' = candidates within the valid time window.
    """
    if not candidates:
        return candidates
    candidates_sorted = sorted(candidates, key=lambda e: (e.step, e.edge_id))
    earliest_step = candidates_sorted[0].step
    return [e for e in candidates_sorted if e.step - earliest_step <= max_duration]


class FanInMatcher:
    """
    Generalized fan-in matcher. Supports n = 3..7 incoming branches.

    Pattern: n distinct sources each send one edge to one common destination.

    Changes from original:
    - n_branches read from pattern, not hard-coded.
    - Cap looked up from FAN_IN_CAP_SCHEDULE.
    - Time-window pre-filter applied before combinations().
    - Filter order: duration → delta_hop → distinct_nodes → amount → anchor.
    - validate=False passed to make_motif_instance_row.
    """

    def __init__(
        self,
        cap_schedule:             Dict[int, int] = FAN_IN_CAP_SCHEDULE,
        max_instances_per_window: int            = MAX_INSTANCES_PER_WINDOW,
        amount_min:               Optional[float]= AMOUNT_MIN,
        delta_hop:                Optional[int]  = DELTA_HOP,
    ):
        self.cap_schedule             = cap_schedule
        self.max_instances_per_window = max_instances_per_window
        self.amount_min               = amount_min
        self.delta_hop                = delta_hop

    def _get_cap(self, n_branches: int) -> int:
        return self.cap_schedule.get(n_branches, 10)

    def _select_candidates(
        self,
        edges:        List[EdgeRecord],
        n_branches:   int,
        max_duration: int,
    ) -> List[EdgeRecord]:
        cap = self._get_cap(n_branches)

        # Amount floor filter first — O(n), removes disqualified edges entirely.
        if self.amount_min is not None:
            edges = [e for e in edges if e.amount >= self.amount_min]

        # Time-window pre-filter.
        edges = prefilter_by_time_window(edges, max_duration)

        # Apply cap: keep earliest.
        if len(edges) > cap:
            edges = edges[:cap]   # already sorted by prefilter_by_time_window

        return edges

    def match(
        self,
        df_primary:   pl.DataFrame,
        df_extended:  pl.DataFrame,
        index:        TemporalIndex,
        window:       WindowSpec,
        pattern:      MotifPattern,
        write_output: bool = False,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, Any]]:

        if pattern.matcher_type != "fan_in":
            raise ValueError(
                f"FanInMatcher requires matcher_type='fan_in', got {pattern.matcher_type}"
            )

        n_branches = len(pattern.edges)
        if n_branches < 3:
            raise ValueError(f"fan_in pattern must have >= 3 edges, got {n_branches}")

        start_time = time.time()
        motif_rows:      List[Dict] = []
        membership_rows: List[Dict] = []

        n_dst_scanned      = 0
        n_dst_skipped_deg  = 0
        n_dst_capped       = 0
        n_combos_checked   = 0
        n_rej_duration     = 0
        n_rej_delta_hop    = 0
        n_rej_structure    = 0
        n_rej_amount       = 0
        n_rej_anchor       = 0

        pattern_edges_sorted = sorted(pattern.edges, key=lambda x: x.order)

        for dst_node, incoming_edges in index.in_edges.items():

            # Fast degree check before any other work.
            if len(incoming_edges) < n_branches:
                n_dst_skipped_deg += 1
                continue

            n_dst_scanned += 1

            candidates = self._select_candidates(
                list(incoming_edges), n_branches, pattern.max_duration
            )

            if len(incoming_edges) > len(candidates):
                n_dst_capped += 1

            if len(candidates) < n_branches:
                continue

            for edge_combo in combinations(candidates, n_branches):
                n_combos_checked += 1
                edges = canonicalize_fanin_edges(list(edge_combo))

                # 1. Duration — cheapest check (just max-min of steps)
                if not passes_fanin_duration(edges, pattern.max_duration):
                    n_rej_duration += 1
                    continue

                # 2. Delta hop — O(n) on sorted list
                if not passes_consecutive_delta_hop_sorted(edges, self.delta_hop):
                    n_rej_delta_hop += 1
                    continue

                # 3. Structural validity — distinct nodes, same dst
                if not passes_distinct_nodes_for_fanin(edges):
                    n_rej_structure += 1
                    continue

                # 4. Anchor in primary window — O(1)
                anchor = get_anchor_edge(edges)
                if not is_edge_in_primary_window(anchor, window):
                    n_rej_anchor += 1
                    continue

                # Build maps.
                node_map = {
                    f"src_{i+1}": int(edges[i].src) for i in range(n_branches)
                }
                node_map["dst"] = int(edges[0].dst)

                role_map = {
                    p_edge.role: int(real_edge.edge_id)
                    for p_edge, real_edge in zip(pattern_edges_sorted, edges)
                }

                try:
                    motif_row = make_motif_instance_row(
                        window_id      = window.window_id,
                        pattern        = pattern,
                        edges          = edges,
                        node_map       = node_map,
                        role_map       = role_map,
                        anchor_edge_id = anchor.edge_id,
                        validate       = False,
                    )
                except Exception:
                    continue

                membership = make_edge_motif_membership_rows(motif_row, pattern, edges)
                motif_rows.append(motif_row)
                membership_rows.extend(membership)

                if len(motif_rows) >= self.max_instances_per_window:
                    break

            if len(motif_rows) >= self.max_instances_per_window:
                break

        motif_df      = motif_instance_rows_to_polars(motif_rows)
        membership_df = membership_rows_to_polars(membership_rows)
        elapsed       = time.time() - start_time

        stats = {
            "window_id":                   int(window.window_id),
            "motif_type":                  pattern.name,
            "matcher_type":                pattern.matcher_type,
            "n_branches":                  int(n_branches),
            "primary_start":               int(window.primary_start),
            "primary_end":                 int(window.primary_end),
            "primary_num_edges":           int(df_primary.height),
            "extended_num_edges":          int(df_extended.height),
            "n_dst_scanned":               int(n_dst_scanned),
            "n_dst_skipped_low_degree":    int(n_dst_skipped_deg),
            "n_dst_capped":                int(n_dst_capped),
            "n_combos_checked":            int(n_combos_checked),
            "num_instances":               int(motif_df.height),
            "num_membership_rows":         int(membership_df.height),
            "n_rej_duration":              int(n_rej_duration),
            "n_rej_delta_hop":             int(n_rej_delta_hop),
            "n_rej_structure":             int(n_rej_structure),
            "n_rej_anchor":                int(n_rej_anchor),
            "hit_max_instances_per_window":int(len(motif_rows) >= self.max_instances_per_window),
            "elapsed_seconds":             float(elapsed),
        }

        if write_output:
            mp, mep = write_motif_outputs(
                motif_rows, membership_rows, window.window_id, pattern.name
            )
            stats["motif_path"]      = mp
            stats["membership_path"] = mep

        return motif_df, membership_df, stats


# Build one matcher instance per fan-in size and store in a dict.
# The pipeline in Cell 11 will dispatch by pattern.name.
fanin_matchers = {
    p.name: FanInMatcher(
        cap_schedule             = FAN_IN_CAP_SCHEDULE,
        max_instances_per_window = MAX_INSTANCES_PER_WINDOW,
        amount_min               = AMOUNT_MIN,
        delta_hop                = DELTA_HOP,
    )
    for p in fan_in_patterns
}

# Backward-compatible alias used in Cell 11 registry.
fanin_matcher = fanin_matchers[fan_in_4.name]

MATCHER_REGISTRY = {"fan_in": FanInMatcher(
    cap_schedule             = FAN_IN_CAP_SCHEDULE,
    max_instances_per_window = MAX_INSTANCES_PER_WINDOW,
    amount_min               = AMOUNT_MIN,
    delta_hop                = DELTA_HOP,
)}

print("FanInMatcher (generalized) ready.")
print("MATCHER_REGISTRY keys:", list(MATCHER_REGISTRY.keys()))
print("Cell 9 completed.")


# ============================================================
# Cell 10: FanOutMatcher (generalized, variable n)
# ============================================================

def passes_distinct_nodes_for_fanout(edges: List[EdgeRecord]) -> bool:
    if not edges:
        return False
    if len({e.edge_id for e in edges}) != len(edges):
        return False
    src_values = {e.src for e in edges}
    if len(src_values) != 1:
        return False
    dst_values = [e.dst for e in edges]
    if len(dst_values) != len(set(dst_values)):
        return False
    center_src = next(iter(src_values))
    return not any(d == center_src for d in dst_values)


def passes_fanout_duration(edges: List[EdgeRecord], max_duration: int) -> bool:
    steps = [e.step for e in edges]
    return (max(steps) - min(steps)) <= max_duration


def canonicalize_fanout_edges(edges: List[EdgeRecord]) -> List[EdgeRecord]:
    return sorted(edges, key=lambda e: (e.step, e.edge_id))


class FanOutMatcher:
    """
    Generalized fan-out matcher. Supports n = 3..7 outgoing branches.
    Mirrors FanInMatcher exactly, operating on out_edges instead of in_edges.
    """

    def __init__(
        self,
        cap_schedule:             Dict[int, int] = FAN_OUT_CAP_SCHEDULE,
        max_instances_per_window: int            = MAX_INSTANCES_PER_WINDOW,
        amount_min:               Optional[float]= AMOUNT_MIN,
        delta_hop:                Optional[int]  = DELTA_HOP,
    ):
        self.cap_schedule             = cap_schedule
        self.max_instances_per_window = max_instances_per_window
        self.amount_min               = amount_min
        self.delta_hop                = delta_hop

    def _get_cap(self, n_branches: int) -> int:
        return self.cap_schedule.get(n_branches, 10)

    def _select_candidates(
        self,
        edges:        List[EdgeRecord],
        n_branches:   int,
        max_duration: int,
    ) -> List[EdgeRecord]:
        cap = self._get_cap(n_branches)
        if self.amount_min is not None:
            edges = [e for e in edges if e.amount >= self.amount_min]
        edges = prefilter_by_time_window(edges, max_duration)
        if len(edges) > cap:
            edges = edges[:cap]
        return edges

    def match(
        self,
        df_primary:   pl.DataFrame,
        df_extended:  pl.DataFrame,
        index:        TemporalIndex,
        window:       WindowSpec,
        pattern:      MotifPattern,
        write_output: bool = False,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, Any]]:

        if pattern.matcher_type != "fan_out":
            raise ValueError(
                f"FanOutMatcher requires matcher_type='fan_out', got {pattern.matcher_type}"
            )

        n_branches = len(pattern.edges)
        if n_branches < 3:
            raise ValueError(f"fan_out pattern must have >= 3 edges, got {n_branches}")

        start_time = time.time()
        motif_rows:      List[Dict] = []
        membership_rows: List[Dict] = []

        n_src_scanned     = 0
        n_src_skipped_deg = 0
        n_src_capped      = 0
        n_combos_checked  = 0
        n_rej_duration    = 0
        n_rej_delta_hop   = 0
        n_rej_structure   = 0
        n_rej_anchor      = 0

        pattern_edges_sorted = sorted(pattern.edges, key=lambda x: x.order)

        for src_node, outgoing_edges in index.out_edges.items():

            if len(outgoing_edges) < n_branches:
                n_src_skipped_deg += 1
                continue

            n_src_scanned += 1

            candidates = self._select_candidates(
                list(outgoing_edges), n_branches, pattern.max_duration
            )

            if len(outgoing_edges) > len(candidates):
                n_src_capped += 1

            if len(candidates) < n_branches:
                continue

            for edge_combo in combinations(candidates, n_branches):
                n_combos_checked += 1
                edges = canonicalize_fanout_edges(list(edge_combo))

                if not passes_fanout_duration(edges, pattern.max_duration):
                    n_rej_duration += 1
                    continue

                if not passes_consecutive_delta_hop_sorted(edges, self.delta_hop):
                    n_rej_delta_hop += 1
                    continue

                if not passes_distinct_nodes_for_fanout(edges):
                    n_rej_structure += 1
                    continue

                anchor = get_anchor_edge(edges)
                if not is_edge_in_primary_window(anchor, window):
                    n_rej_anchor += 1
                    continue

                node_map = {"src": int(edges[0].src)}
                node_map.update({
                    f"dst_{i+1}": int(edges[i].dst) for i in range(n_branches)
                })

                role_map = {
                    p_edge.role: int(real_edge.edge_id)
                    for p_edge, real_edge in zip(pattern_edges_sorted, edges)
                }

                try:
                    motif_row = make_motif_instance_row(
                        window_id      = window.window_id,
                        pattern        = pattern,
                        edges          = edges,
                        node_map       = node_map,
                        role_map       = role_map,
                        anchor_edge_id = anchor.edge_id,
                        validate       = False,
                    )
                except Exception:
                    continue

                membership = make_edge_motif_membership_rows(motif_row, pattern, edges)
                motif_rows.append(motif_row)
                membership_rows.extend(membership)

                if len(motif_rows) >= self.max_instances_per_window:
                    break

            if len(motif_rows) >= self.max_instances_per_window:
                break

        motif_df      = motif_instance_rows_to_polars(motif_rows)
        membership_df = membership_rows_to_polars(membership_rows)
        elapsed       = time.time() - start_time

        stats = {
            "window_id":                    int(window.window_id),
            "motif_type":                   pattern.name,
            "matcher_type":                 pattern.matcher_type,
            "n_branches":                   int(n_branches),
            "primary_num_edges":            int(df_primary.height),
            "extended_num_edges":           int(df_extended.height),
            "n_src_scanned":                int(n_src_scanned),
            "n_src_skipped_low_degree":     int(n_src_skipped_deg),
            "n_src_capped":                 int(n_src_capped),
            "n_combos_checked":             int(n_combos_checked),
            "num_instances":                int(motif_df.height),
            "num_membership_rows":          int(membership_df.height),
            "n_rej_duration":               int(n_rej_duration),
            "n_rej_delta_hop":              int(n_rej_delta_hop),
            "n_rej_structure":              int(n_rej_structure),
            "n_rej_anchor":                 int(n_rej_anchor),
            "hit_max_instances_per_window": int(len(motif_rows) >= self.max_instances_per_window),
            "elapsed_seconds":              float(elapsed),
        }

        if write_output:
            mp, mep = write_motif_outputs(
                motif_rows, membership_rows, window.window_id, pattern.name
            )
            stats["motif_path"]      = mp
            stats["membership_path"] = mep

        return motif_df, membership_df, stats


MATCHER_REGISTRY["fan_out"] = FanOutMatcher(
    cap_schedule             = FAN_OUT_CAP_SCHEDULE,
    max_instances_per_window = MAX_INSTANCES_PER_WINDOW,
    amount_min               = AMOUNT_MIN,
    delta_hop                = DELTA_HOP,
)

print("FanOutMatcher (generalized) ready.")
print("MATCHER_REGISTRY keys:", list(MATCHER_REGISTRY.keys()))
print("Cell 10 completed.")



# ============================================================
# Cell 11: Run Fan-in and Fan-out over Temporal Windows
# ============================================================

def get_pattern_by_name(patterns: List[MotifPattern], pattern_name: str) -> MotifPattern:
    """
    Retrieve a MotifPattern by name.
    """
    matches = [p for p in patterns if p.name == pattern_name]

    if len(matches) == 0:
        raise ValueError(f"Pattern not found: {pattern_name}")

    if len(matches) > 1:
        raise ValueError(f"Duplicated pattern name: {pattern_name}")

    return matches[0]


def get_output_paths_for_window_pattern(
    window_id: int,
    motif_type: str,
    motif_instance_dir: str = MOTIF_INSTANCE_DIR,
    membership_dir: str = MEMBERSHIP_DIR,
) -> Tuple[str, str]:
    """
    Return expected output shard paths for one window-pattern pair.
    """
    motif_path = f"{motif_instance_dir}/window_{int(window_id):06d}_{motif_type}.parquet"
    membership_path = f"{membership_dir}/window_{int(window_id):06d}_{motif_type}.parquet"

    return motif_path, membership_path


def output_shards_exist(window_id: int, motif_type: str) -> bool:
    """
    Check whether both motif and membership shards already exist.
    """
    motif_path, membership_path = get_output_paths_for_window_pattern(
        window_id=window_id,
        motif_type=motif_type,
    )

    return os.path.exists(motif_path) and os.path.exists(membership_path)


def write_stats_log(stats_rows: List[Dict[str, Any]], stats_path: str) -> pl.DataFrame:
    """
    Write stats rows to parquet and return the stats dataframe.
    """
    if len(stats_rows) == 0:
        stats_df = pl.DataFrame()
    else:
        stats_df = pl.DataFrame(stats_rows)

    stats_df.write_parquet(stats_path)

    return stats_df


def run_matchers_over_windows(
    df_edges: pl.DataFrame,
    windows: List[WindowSpec],
    patterns_to_run: List[MotifPattern],
    matcher_registry: Dict[str, Any],
    max_windows: Optional[int] = None,
    skip_existing: bool = True,
    write_empty_outputs: bool = True,
) -> pl.DataFrame:
    """
    Run selected matchers over temporal windows.

    For each window:
        1. Slice primary and extended edges.
        2. Build TemporalIndex on extended edges.
        3. Run each selected pattern matcher.
        4. Write motif_instances and edge_motif_membership parquet shards.
        5. Record runtime and output statistics.

    Parameters:
        max_windows:
            If not None, only run first max_windows windows for testing.

        skip_existing:
            If True, skip a window-pattern if both output shards already exist.

        write_empty_outputs:
            If True, write empty parquet shards even when no instances are found.
            This makes resume and auditing easier.
    """

    run_start_time = time.time()

    stats_rows = []

    if max_windows is not None:
        windows_to_run = windows[:max_windows]
    else:
        windows_to_run = windows

    print("Starting motif mining run.")
    print("Number of windows to run:", len(windows_to_run))
    print("Patterns:", [p.name for p in patterns_to_run])
    print("skip_existing:", skip_existing)
    print("write_empty_outputs:", write_empty_outputs)

    for idx, window in enumerate(windows_to_run):
        window_start_time = time.time()

        print(
            f"\n[Window {idx + 1}/{len(windows_to_run)}] "
            f"window_id={window.window_id}, "
            f"primary=[{window.primary_start}, {window.primary_end}], "
            f"extended=[{window.extended_start}, {window.extended_end}]"
        )

        df_primary, df_extended = slice_window_edges(df_edges, window)

        window_summary = build_window_candidate_summary(df_extended)

        if df_primary.height == 0:
            print("  Primary window is empty. Skipping matcher execution.")

            for pattern in patterns_to_run:
                stats_rows.append({
                    "window_id": int(window.window_id),
                    "motif_type": pattern.name,
                    "matcher_type": pattern.matcher_type,
                    "status": "skipped_empty_primary",
                    "primary_start": int(window.primary_start),
                    "primary_end": int(window.primary_end),
                    "extended_start": int(window.extended_start),
                    "extended_end": int(window.extended_end),
                    "primary_num_edges": int(df_primary.height),
                    "extended_num_edges": int(df_extended.height),
                    "num_instances": 0,
                    "num_membership_rows": 0,
                    "elapsed_seconds": 0.0,
                })

            continue

        if df_extended.height == 0:
            print("  Extended window is empty. Skipping matcher execution.")

            for pattern in patterns_to_run:
                stats_rows.append({
                    "window_id": int(window.window_id),
                    "motif_type": pattern.name,
                    "matcher_type": pattern.matcher_type,
                    "status": "skipped_empty_extended",
                    "primary_start": int(window.primary_start),
                    "primary_end": int(window.primary_end),
                    "extended_start": int(window.extended_start),
                    "extended_end": int(window.extended_end),
                    "primary_num_edges": int(df_primary.height),
                    "extended_num_edges": int(df_extended.height),
                    "num_instances": 0,
                    "num_membership_rows": 0,
                    "elapsed_seconds": 0.0,
                })

            continue

        index_start_time = time.time()
        temporal_index = build_temporal_index_from_polars(df_extended)
        index_elapsed = time.time() - index_start_time

        print("  df_primary edges:", df_primary.height)
        print("  df_extended edges:", df_extended.height)
        print("  index stats:", temporal_index.stats())
        print("  index build time:", round(index_elapsed, 3), "seconds")

        for pattern in patterns_to_run:
            pattern_start_time = time.time()

            if pattern.matcher_type not in matcher_registry:
                raise ValueError(
                    f"No matcher registered for matcher_type={pattern.matcher_type}. "
                    f"Available registry keys: {list(matcher_registry.keys())}"
                )

            motif_path, membership_path = get_output_paths_for_window_pattern(
                window_id=window.window_id,
                motif_type=pattern.name,
            )

            if skip_existing and output_shards_exist(window.window_id, pattern.name):
                print(f"  [{pattern.name}] output exists. Skipping.")

                stats_rows.append({
                    "window_id": int(window.window_id),
                    "motif_type": pattern.name,
                    "matcher_type": pattern.matcher_type,
                    "status": "skipped_existing",
                    "primary_start": int(window.primary_start),
                    "primary_end": int(window.primary_end),
                    "extended_start": int(window.extended_start),
                    "extended_end": int(window.extended_end),
                    "primary_num_edges": int(df_primary.height),
                    "extended_num_edges": int(df_extended.height),
                    "index_build_seconds": float(index_elapsed),
                    "num_instances": None,
                    "num_membership_rows": None,
                    "elapsed_seconds": 0.0,
                    "motif_path": motif_path,
                    "membership_path": membership_path,
                })

                continue

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

            # Explicit cleanup per pattern.
            del motif_df
            del membership_df
            gc.collect()

        window_elapsed = time.time() - window_start_time
        print("  Window elapsed:", round(window_elapsed, 3), "seconds")

        # Explicit cleanup per window.
        del temporal_index
        del df_primary
        del df_extended
        gc.collect()

    total_elapsed = time.time() - run_start_time

    print("\nMotif mining run completed.")
    print("Total elapsed:", round(total_elapsed, 3), "seconds")

    stats_path = f"{LOG_DIR}/fanin_fanout_run_stats.parquet"
    stats_df = write_stats_log(stats_rows, stats_path)

    print("Stats log saved to:")
    print(stats_path)

    return stats_df


# ------------------------------------------------------------
# Select patterns for this run
# ------------------------------------------------------------

PATTERNS_TO_RUN_CELL11 = fan_in_patterns + fan_out_patterns

# Ensure matchers are registered.
required_matcher_types = sorted(set(p.matcher_type for p in PATTERNS_TO_RUN_CELL11))
missing_matchers = [
    matcher_type
    for matcher_type in required_matcher_types
    if matcher_type not in MATCHER_REGISTRY
]

if missing_matchers:
    raise ValueError(
        f"Missing matchers in MATCHER_REGISTRY: {missing_matchers}. "
        f"Available keys: {list(MATCHER_REGISTRY.keys())}"
    )

print("Cell 11 setup ready.")
print("Patterns to run:", [p.name for p in PATTERNS_TO_RUN_CELL11])
print("Matcher registry:", list(MATCHER_REGISTRY.keys()))


# ------------------------------------------------------------
# Run mode
# ------------------------------------------------------------
# For first run, keep this small to validate the pipeline.
# After checking outputs, set MAX_WINDOWS_TO_RUN = None to run all windows.

MAX_WINDOWS_TO_RUN = 2

SKIP_EXISTING_OUTPUTS = False
WRITE_EMPTY_OUTPUTS = True

run_stats_df = run_matchers_over_windows(
    df_edges=df_edges,
    windows=windows,
    patterns_to_run=PATTERNS_TO_RUN_CELL11,
    matcher_registry=MATCHER_REGISTRY,
    max_windows=MAX_WINDOWS_TO_RUN,
    skip_existing=SKIP_EXISTING_OUTPUTS,
    write_empty_outputs=WRITE_EMPTY_OUTPUTS,
)

print("\nRun stats preview:")
display(run_stats_df)


# ------------------------------------------------------------
# Quick aggregate summary
# ------------------------------------------------------------

if run_stats_df.height > 0 and "status" in run_stats_df.columns:
    completed_stats = run_stats_df.filter(pl.col("status") == "completed")

    if completed_stats.height > 0:
        aggregate_summary = (
            completed_stats
            .group_by("motif_type")
            .agg([
                pl.col("num_instances").sum().alias("total_instances"),
                pl.col("num_membership_rows").sum().alias("total_membership_rows"),
                pl.col("elapsed_seconds").sum().alias("total_match_seconds"),
                pl.col("window_id").n_unique().alias("num_windows_completed"),
            ])
            .sort("motif_type")
        )

        print("\nAggregate summary:")
        display(aggregate_summary)
    else:
        print("\nNo completed matcher rows found in run_stats_df.")
else:
    print("\nrun_stats_df is empty or missing status column.")

print("\nCell 11 completed.")



