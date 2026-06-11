# ============================================================
# Cell 9: FanInMatcher
# ============================================================

from itertools import combinations


def is_edge_in_primary_window(edge: EdgeRecord, window: WindowSpec) -> bool:
    """
    Check whether an edge belongs to the primary window.

    Only motif instances whose anchor edge is in the primary window
    should be emitted. This prevents duplicate output across windows.
    """
    return window.primary_start <= edge.step <= window.primary_end


def get_anchor_edge(edges: List[EdgeRecord]) -> EdgeRecord:
    """
    Define the canonical anchor edge of an instance.

    For unordered branch motifs like fan-in/fan-out, use the earliest edge.
    Tie-break by edge_id.
    """
    return min(edges, key=lambda e: (e.step, e.edge_id))


def passes_amount_min(edges: List[EdgeRecord], amount_min: Optional[float]) -> bool:
    """
    Optional absolute amount filter.
    """
    if amount_min is None:
        return True

    return all(e.amount >= amount_min for e in edges)


def passes_distinct_nodes_for_fanin(edges: List[EdgeRecord]) -> bool:
    """
    Fan-in structure:
        a -> d
        b -> d
        c -> d

    Conditions:
        all dst are the same
        all src are distinct
        src nodes are different from dst
        edge IDs are distinct
    """
    if len(edges) == 0:
        return False

    edge_ids = [e.edge_id for e in edges]
    if len(edge_ids) != len(set(edge_ids)):
        return False

    dst_values = [e.dst for e in edges]
    if len(set(dst_values)) != 1:
        return False

    src_values = [e.src for e in edges]
    if len(src_values) != len(set(src_values)):
        return False

    center_dst = dst_values[0]

    if any(src == center_dst for src in src_values):
        return False

    return True


def passes_fanin_duration(edges: List[EdgeRecord], max_duration: int) -> bool:
    """
    Fan-in motif duration constraint:
        max(step_i) - min(step_i) <= max_duration
    """
    steps = [e.step for e in edges]
    return (max(steps) - min(steps)) <= max_duration

def passes_consecutive_delta_hop(
    edges: List[EdgeRecord],
    delta_hop: Optional[int],
) -> bool:
    """
    Check whether consecutive edges are temporally close enough.

    After edges are sorted by step, edge_id:
        step[i + 1] - step[i] <= delta_hop

    If delta_hop is None, this constraint is disabled.
    """
    if delta_hop is None:
        return True

    if len(edges) <= 1:
        return True

    steps = [int(e.step) for e in edges]

    for i in range(len(steps) - 1):
        if steps[i + 1] - steps[i] > delta_hop:
            return False

    return True


def canonicalize_fanin_edges(edges: List[EdgeRecord]) -> List[EdgeRecord]:
    """
    Canonical ordering for fan-in branches.

    Since fan-in branches are unordered, we sort by step then edge_id
    before building output. This avoids duplicate representation.
    """
    return sorted(edges, key=lambda e: (e.step, e.edge_id))


class FanInMatcher:
    """
    Specialized matcher for fan_in_4.

    Pattern:
        a -> d
        b -> d
        c -> d

    Matching strategy:
        1. Use incoming index grouped by dst.
        2. For each dst, take candidate incoming edges.
        3. Apply degree/candidate cap.
        4. Generate combinations of size 3.
        5. Keep only combinations satisfying:
            - same dst
            - distinct sources
            - duration <= max_duration
            - optional amount_min
            - anchor edge is in primary window
        6. Emit motif_instances and edge_motif_membership rows.

    This matcher deliberately allows overlap between motif instances.
    It only prevents duplicate enumeration via combinations and canonical anchor.
    """

    def __init__(
        self,
        max_in_candidates: int = MAX_IN_CANDIDATES,
        max_instances_per_window: int = MAX_INSTANCES_PER_WINDOW,
        amount_min: Optional[float] = AMOUNT_MIN,
        delta_hop: Optional[int] = DELTA_HOP,
    ):
        self.max_in_candidates = max_in_candidates
        self.max_instances_per_window = max_instances_per_window
        self.amount_min = amount_min
        self.delta_hop = delta_hop

    def _select_candidate_edges_for_dst(
        self,
        edges: List[EdgeRecord],
        max_candidates: int,
    ) -> List[EdgeRecord]:
        """
        Candidate selection for one destination node.

        Current policy:
            sort by step, edge_id and keep earliest max_candidates.

        Later alternatives:
            top-K by amount
            top-K by local risk score
            hybrid earliest + high amount
        """
        edges_sorted = sorted(edges, key=lambda e: (e.step, e.edge_id))

        if max_candidates is not None and len(edges_sorted) > max_candidates:
            return edges_sorted[:max_candidates]

        return edges_sorted

    def match(
        self,
        df_primary: pl.DataFrame,
        df_extended: pl.DataFrame,
        index: TemporalIndex,
        window: WindowSpec,
        pattern: MotifPattern,
        write_output: bool = False,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, Any]]:
        """
        Run fan-in motif matching for one window.

        Returns:
            motif_df
            membership_df
            stats
        """

        if pattern.matcher_type != "fan_in":
            raise ValueError(
                f"FanInMatcher can only handle matcher_type='fan_in'. "
                f"Got: {pattern.matcher_type}"
            )

        if len(pattern.edges) != 3:
            raise ValueError(
                f"Current FanInMatcher expects exactly 3 incoming edges. "
                f"Pattern {pattern.name} has {len(pattern.edges)} edges."
            )

        start_time = time.time()

        motif_rows = []
        membership_rows = []

        num_dst_groups_scanned = 0
        num_dst_groups_capped = 0
        num_combinations_checked = 0
        num_combinations_rejected_duration = 0
        num_combinations_rejected_delta_hop = 0
        num_combinations_rejected_structure = 0
        num_combinations_rejected_amount = 0
        num_combinations_rejected_anchor = 0

        # Iterate over incoming groups from temporal index.
        # Each key is a destination node.
        for dst_node, incoming_edges in index.in_edges.items():
            num_dst_groups_scanned += 1

            candidates = self._select_candidate_edges_for_dst(
                incoming_edges,
                self.max_in_candidates,
            )

            if len(incoming_edges) > len(candidates):
                num_dst_groups_capped += 1

            if len(candidates) < 3:
                continue

            for edge_triplet in combinations(candidates, 3):
                num_combinations_checked += 1

                edges = canonicalize_fanin_edges(list(edge_triplet))

                if not passes_fanin_duration(edges, pattern.max_duration):
                    num_combinations_rejected_duration += 1
                    continue

                if not passes_consecutive_delta_hop(edges, self.delta_hop):
                    num_combinations_rejected_delta_hop += 1
                    continue

                if not passes_distinct_nodes_for_fanin(edges):
                    num_combinations_rejected_structure += 1
                    continue

                if not passes_amount_min(edges, self.amount_min):
                    num_combinations_rejected_amount += 1
                    continue

                anchor_edge = get_anchor_edge(edges)

                if not is_edge_in_primary_window(anchor_edge, window):
                    num_combinations_rejected_anchor += 1
                    continue

                # Build structural node map:
                # fan-in: a,b,c are sources, d is common destination.
                node_map = {
                    "a": int(edges[0].src),
                    "b": int(edges[1].src),
                    "c": int(edges[2].src),
                    "d": int(edges[0].dst),
                }

                # Build role map according to pattern roles.
                pattern_edges_sorted = sorted(pattern.edges, key=lambda x: x.order)

                role_map = {}
                for p_edge, real_edge in zip(pattern_edges_sorted, edges):
                    role = p_edge.role if p_edge.role else p_edge.name
                    role_map[role] = int(real_edge.edge_id)

                try:
                    motif_row = make_motif_instance_row(
                        window_id=window.window_id,
                        pattern=pattern,
                        edges=edges,
                        node_map=node_map,
                        role_map=role_map,
                        anchor_edge_id=anchor_edge.edge_id,
                        validate=True,
                    )
                except Exception as exc:
                    # This should be rare if checks above are correct.
                    # Keep the matcher robust during experimentation.
                    continue

                membership = make_edge_motif_membership_rows(
                    motif_row=motif_row,
                    pattern=pattern,
                    edges=edges,
                )

                motif_rows.append(motif_row)
                membership_rows.extend(membership)

                if len(motif_rows) >= self.max_instances_per_window:
                    break

            if len(motif_rows) >= self.max_instances_per_window:
                break

        motif_df = motif_instance_rows_to_polars(motif_rows)
        membership_df = membership_rows_to_polars(membership_rows)

        elapsed = time.time() - start_time

        stats = {
            "window_id": int(window.window_id),
            "motif_type": pattern.name,
            "matcher_type": pattern.matcher_type,
            "primary_start": int(window.primary_start),
            "primary_end": int(window.primary_end),
            "extended_start": int(window.extended_start),
            "extended_end": int(window.extended_end),
            "primary_num_edges": int(df_primary.height),
            "extended_num_edges": int(df_extended.height),
            "num_dst_groups_scanned": int(num_dst_groups_scanned),
            "num_dst_groups_capped": int(num_dst_groups_capped),
            "num_combinations_checked": int(num_combinations_checked),
            "num_instances": int(motif_df.height),
            "num_membership_rows": int(membership_df.height),
            "num_combinations_rejected_duration": int(num_combinations_rejected_duration),
            "num_combinations_rejected_delta_hop": int(num_combinations_rejected_delta_hop),
            "num_combinations_rejected_structure": int(num_combinations_rejected_structure),
            "num_combinations_rejected_amount": int(num_combinations_rejected_amount),
            "num_combinations_rejected_anchor": int(num_combinations_rejected_anchor),
            "hit_max_instances_per_window": int(len(motif_rows) >= self.max_instances_per_window),
            "elapsed_seconds": float(elapsed),
        }

        if write_output:
            motif_path, membership_path = write_motif_outputs(
                motif_rows=motif_rows,
                membership_rows=membership_rows,
                window_id=window.window_id,
                motif_type=pattern.name,
            )

            stats["motif_path"] = motif_path
            stats["membership_path"] = membership_path

        return motif_df, membership_df, stats


# # ------------------------------------------------------------
# # Smoke test FanInMatcher on first non-empty window
# # ------------------------------------------------------------

# print("Running Cell 9 FanInMatcher smoke test...")

fanin_matcher = FanInMatcher(
    max_in_candidates=MAX_IN_CANDIDATES,
    max_instances_per_window=MAX_INSTANCES_PER_WINDOW,
    amount_min=AMOUNT_MIN,
    delta_hop=DELTA_HOP,
)

# # Use the first non-empty test window from Cell 6.
# df_primary_test, df_extended_test = slice_window_edges(df_edges, test_window)
# test_index = build_temporal_index_from_polars(df_extended_test)

# fanin_motif_df, fanin_membership_df, fanin_stats = fanin_matcher.match(
#     df_primary=df_primary_test,
#     df_extended=df_extended_test,
#     index=test_index,
#     window=test_window,
#     pattern=fan_in_4,
#     write_output=False,
# )

# print("\nFanInMatcher smoke test stats:")
# for k, v in fanin_stats.items():
#     print(f"{k}: {v}")

# print("\nFan-in motif instances preview:")
# display(fanin_motif_df.head(10))

# print("\nFan-in edge membership preview:")
# display(fanin_membership_df.head(15))


# # ------------------------------------------------------------
# # Optional: write smoke test output to Drive
# # Set to True only if you want to persist the test shard.
# # ------------------------------------------------------------

# WRITE_CELL9_SMOKE_OUTPUT = False

# if WRITE_CELL9_SMOKE_OUTPUT:
#     motif_path, membership_path = write_motif_outputs(
#         motif_rows=fanin_motif_df.to_dicts(),
#         membership_rows=fanin_membership_df.to_dicts(),
#         window_id=test_window.window_id,
#         motif_type=fan_in_4.name,
#     )

#     print("\nSmoke test outputs written:")
#     print("motif_path:", motif_path)
#     print("membership_path:", membership_path)


# ------------------------------------------------------------
# Store matcher in registry for later pipeline cells
# ------------------------------------------------------------

MATCHER_REGISTRY = {
    "fan_in": fanin_matcher,
}

print("\nMATCHER_REGISTRY keys:", list(MATCHER_REGISTRY.keys()))
print("\nCell 9 completed.")



# ============================================================
# Cell 10: FanOutMatcher
# ============================================================

def passes_distinct_nodes_for_fanout(edges: List[EdgeRecord]) -> bool:
    """
    Fan-out structure:
        a -> b
        a -> c
        a -> d

    Conditions:
        all src are the same
        all dst are distinct
        dst nodes are different from src
        edge IDs are distinct
    """
    if len(edges) == 0:
        return False

    edge_ids = [e.edge_id for e in edges]
    if len(edge_ids) != len(set(edge_ids)):
        return False

    src_values = [e.src for e in edges]
    if len(set(src_values)) != 1:
        return False

    dst_values = [e.dst for e in edges]
    if len(dst_values) != len(set(dst_values)):
        return False

    center_src = src_values[0]

    if any(dst == center_src for dst in dst_values):
        return False

    return True


def passes_fanout_duration(edges: List[EdgeRecord], max_duration: int) -> bool:
    """
    Fan-out motif duration constraint:
        max(step_i) - min(step_i) <= max_duration
    """
    steps = [e.step for e in edges]
    return (max(steps) - min(steps)) <= max_duration

def passes_consecutive_delta_hop(
    edges: List[EdgeRecord],
    delta_hop: Optional[int],
) -> bool:
    """
    Check whether consecutive edges are temporally close enough.

    After edges are sorted by step, edge_id:
        step[i + 1] - step[i] <= delta_hop

    If delta_hop is None, this constraint is disabled.
    """
    if delta_hop is None:
        return True

    if len(edges) <= 1:
        return True

    steps = [int(e.step) for e in edges]

    for i in range(len(steps) - 1):
        if steps[i + 1] - steps[i] > delta_hop:
            return False

    return True


def canonicalize_fanout_edges(edges: List[EdgeRecord]) -> List[EdgeRecord]:
    """
    Canonical ordering for fan-out branches.

    Since fan-out branches are unordered, sort by step then edge_id.
    """
    return sorted(edges, key=lambda e: (e.step, e.edge_id))


class FanOutMatcher:
    """
    Specialized matcher for fan_out_4.

    Pattern:
        a -> b
        a -> c
        a -> d

    Matching strategy:
        1. Use outgoing index grouped by src.
        2. For each src, take candidate outgoing edges.
        3. Apply degree/candidate cap.
        4. Generate combinations of size 3.
        5. Keep only combinations satisfying:
            - same src
            - distinct destinations
            - duration <= max_duration
            - optional amount_min
            - anchor edge is in primary window
        6. Emit motif_instances and edge_motif_membership rows.

    This matcher allows overlap between motif instances.
    It avoids duplicate enumeration by using combinations and canonical ordering.
    """

    def __init__(
        self,
        max_out_candidates: int = MAX_OUT_CANDIDATES,
        max_instances_per_window: int = MAX_INSTANCES_PER_WINDOW,
        amount_min: Optional[float] = AMOUNT_MIN,
        delta_hop: Optional[int] = DELTA_HOP,
    ):
        self.max_out_candidates = max_out_candidates
        self.max_instances_per_window = max_instances_per_window
        self.amount_min = amount_min
        self.delta_hop = delta_hop

    def _select_candidate_edges_for_src(
        self,
        edges: List[EdgeRecord],
        max_candidates: int,
    ) -> List[EdgeRecord]:
        """
        Candidate selection for one source node.

        Current policy:
            sort by step, edge_id and keep earliest max_candidates.

        Later alternatives:
            top-K by amount
            top-K by local risk score
            hybrid earliest + high amount
        """
        edges_sorted = sorted(edges, key=lambda e: (e.step, e.edge_id))

        if max_candidates is not None and len(edges_sorted) > max_candidates:
            return edges_sorted[:max_candidates]

        return edges_sorted

    def match(
        self,
        df_primary: pl.DataFrame,
        df_extended: pl.DataFrame,
        index: TemporalIndex,
        window: WindowSpec,
        pattern: MotifPattern,
        write_output: bool = False,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, Any]]:
        """
        Run fan-out motif matching for one window.

        Returns:
            motif_df
            membership_df
            stats
        """

        if pattern.matcher_type != "fan_out":
            raise ValueError(
                f"FanOutMatcher can only handle matcher_type='fan_out'. "
                f"Got: {pattern.matcher_type}"
            )

        if len(pattern.edges) != 3:
            raise ValueError(
                f"Current FanOutMatcher expects exactly 3 outgoing edges. "
                f"Pattern {pattern.name} has {len(pattern.edges)} edges."
            )

        start_time = time.time()

        motif_rows = []
        membership_rows = []

        num_src_groups_scanned = 0
        num_src_groups_capped = 0
        num_combinations_checked = 0
        num_combinations_rejected_duration = 0
        num_combinations_rejected_structure = 0
        num_combinations_rejected_delta_hop = 0
        num_combinations_rejected_amount = 0
        num_combinations_rejected_anchor = 0

        # Iterate over outgoing groups from temporal index.
        # Each key is a source node.
        for src_node, outgoing_edges in index.out_edges.items():
            num_src_groups_scanned += 1

            candidates = self._select_candidate_edges_for_src(
                outgoing_edges,
                self.max_out_candidates,
            )

            if len(outgoing_edges) > len(candidates):
                num_src_groups_capped += 1

            if len(candidates) < 3:
                continue

            for edge_triplet in combinations(candidates, 3):
                num_combinations_checked += 1

                edges = canonicalize_fanout_edges(list(edge_triplet))

                if not passes_fanout_duration(edges, pattern.max_duration):
                    num_combinations_rejected_duration += 1
                    continue

                if not passes_consecutive_delta_hop(edges, self.delta_hop):
                    num_combinations_rejected_delta_hop += 1
                    continue

                if not passes_distinct_nodes_for_fanout(edges):
                    num_combinations_rejected_structure += 1
                    continue

                if not passes_amount_min(edges, self.amount_min):
                    num_combinations_rejected_amount += 1
                    continue

                anchor_edge = get_anchor_edge(edges)

                if not is_edge_in_primary_window(anchor_edge, window):
                    num_combinations_rejected_anchor += 1
                    continue

                # Build structural node map:
                # fan-out: a is common source, b/c/d are destinations.
                node_map = {
                    "a": int(edges[0].src),
                    "b": int(edges[0].dst),
                    "c": int(edges[1].dst),
                    "d": int(edges[2].dst),
                }

                # Build role map according to pattern roles.
                pattern_edges_sorted = sorted(pattern.edges, key=lambda x: x.order)

                role_map = {}
                for p_edge, real_edge in zip(pattern_edges_sorted, edges):
                    role = p_edge.role if p_edge.role else p_edge.name
                    role_map[role] = int(real_edge.edge_id)

                try:
                    motif_row = make_motif_instance_row(
                        window_id=window.window_id,
                        pattern=pattern,
                        edges=edges,
                        node_map=node_map,
                        role_map=role_map,
                        anchor_edge_id=anchor_edge.edge_id,
                        validate=True,
                    )
                except Exception as exc:
                    # This should be rare if checks above are correct.
                    continue

                membership = make_edge_motif_membership_rows(
                    motif_row=motif_row,
                    pattern=pattern,
                    edges=edges,
                )

                motif_rows.append(motif_row)
                membership_rows.extend(membership)

                if len(motif_rows) >= self.max_instances_per_window:
                    break

            if len(motif_rows) >= self.max_instances_per_window:
                break

        motif_df = motif_instance_rows_to_polars(motif_rows)
        membership_df = membership_rows_to_polars(membership_rows)

        elapsed = time.time() - start_time

        stats = {
            "window_id": int(window.window_id),
            "motif_type": pattern.name,
            "matcher_type": pattern.matcher_type,
            "primary_start": int(window.primary_start),
            "primary_end": int(window.primary_end),
            "extended_start": int(window.extended_start),
            "extended_end": int(window.extended_end),
            "primary_num_edges": int(df_primary.height),
            "extended_num_edges": int(df_extended.height),
            "num_src_groups_scanned": int(num_src_groups_scanned),
            "num_src_groups_capped": int(num_src_groups_capped),
            "num_combinations_checked": int(num_combinations_checked),
            "num_instances": int(motif_df.height),
            "num_membership_rows": int(membership_df.height),
            "num_combinations_rejected_duration": int(num_combinations_rejected_duration),
            "num_combinations_rejected_structure": int(num_combinations_rejected_structure),
            "num_combinations_rejected_delta_hop": int(num_combinations_rejected_delta_hop),
            "num_combinations_rejected_amount": int(num_combinations_rejected_amount),
            "num_combinations_rejected_anchor": int(num_combinations_rejected_anchor),
            "hit_max_instances_per_window": int(len(motif_rows) >= self.max_instances_per_window),
            "elapsed_seconds": float(elapsed),
        }

        if write_output:
            motif_path, membership_path = write_motif_outputs(
                motif_rows=motif_rows,
                membership_rows=membership_rows,
                window_id=window.window_id,
                motif_type=pattern.name,
            )

            stats["motif_path"] = motif_path
            stats["membership_path"] = membership_path

        return motif_df, membership_df, stats


# # ------------------------------------------------------------
# # Smoke test FanOutMatcher on first non-empty window
# # ------------------------------------------------------------

# print("Running Cell 10 FanOutMatcher smoke test...")

fanout_matcher = FanOutMatcher(
    max_out_candidates=MAX_OUT_CANDIDATES,
    max_instances_per_window=MAX_INSTANCES_PER_WINDOW,
    amount_min=AMOUNT_MIN,
    delta_hop=DELTA_HOP,

)

# # Use the first non-empty test window from Cell 6.
# df_primary_test, df_extended_test = slice_window_edges(df_edges, test_window)
# test_index = build_temporal_index_from_polars(df_extended_test)

# fanout_motif_df, fanout_membership_df, fanout_stats = fanout_matcher.match(
#     df_primary=df_primary_test,
#     df_extended=df_extended_test,
#     index=test_index,
#     window=test_window,
#     pattern=fan_out_4,
#     write_output=False,
# )

# print("\nFanOutMatcher smoke test stats:")
# for k, v in fanout_stats.items():
#     print(f"{k}: {v}")

# print("\nFan-out motif instances preview:")
# display(fanout_motif_df.head(10))

# print("\nFan-out edge membership preview:")
# display(fanout_membership_df.head(15))


# # ------------------------------------------------------------
# # Optional: write smoke test output to Drive
# # Set to True only if you want to persist the test shard.
# # ------------------------------------------------------------

# WRITE_CELL10_SMOKE_OUTPUT = False

# if WRITE_CELL10_SMOKE_OUTPUT:
#     motif_path, membership_path = write_motif_outputs(
#         motif_rows=fanout_motif_df.to_dicts(),
#         membership_rows=fanout_membership_df.to_dicts(),
#         window_id=test_window.window_id,
#         motif_type=fan_out_4.name,
#     )

#     print("\nSmoke test outputs written:")
#     print("motif_path:", motif_path)
#     print("membership_path:", membership_path)


# # ------------------------------------------------------------
# # Update matcher registry for later pipeline cells
# # ------------------------------------------------------------

if "MATCHER_REGISTRY" not in globals():
    MATCHER_REGISTRY = {}

MATCHER_REGISTRY["fan_out"] = fanout_matcher

print("\nMATCHER_REGISTRY keys:", list(MATCHER_REGISTRY.keys()))
print("\nCell 10 completed.")



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

PATTERNS_TO_RUN_CELL11 = [
    fan_in_4,
    fan_out_4,
]

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

MAX_WINDOWS_TO_RUN = 4

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