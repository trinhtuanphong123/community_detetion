


# ============================================================
# Cell 6: Temporal Window Manager
# ============================================================

@dataclass(frozen=True)
class WindowSpec:
    """
    Temporal processing window.

    primary_start, primary_end:
        Anchor edges are selected only from this range.

    extended_start, extended_end:
        Motif matching can use edges from this range.
        The lookahead region prevents losing motifs crossing window boundaries.
    """
    window_id: int
    primary_start: int
    primary_end: int
    extended_start: int
    extended_end: int


def get_step_bounds(df_edges: pl.DataFrame) -> Tuple[int, int]:
    """
    Return min and max step from edge dataframe.
    """

    bounds = df_edges.select([
        pl.col("step").min().alias("min_step"),
        pl.col("step").max().alias("max_step"),
    ])

    min_step = int(bounds["min_step"][0])
    max_step = int(bounds["max_step"][0])

    return min_step, max_step


def build_temporal_windows(
    min_step: int,
    max_step: int,
    window_size: int,
    stride: int,
    max_motif_duration: int,
) -> List[WindowSpec]:
    """
    Build primary + extended windows.

    Example:
        primary:  [s, s + window_size - 1]
        extended: [s, s + window_size - 1 + max_motif_duration]

    Only anchor edges from primary window will be used later.
    Edges in lookahead can be used to complete motifs.
    """

    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    if stride <= 0:
        raise ValueError("stride must be positive.")

    if max_motif_duration < 0:
        raise ValueError("max_motif_duration must be non-negative.")

    windows = []
    window_id = 0
    primary_start = min_step

    while primary_start <= max_step:
        primary_end = primary_start + window_size - 1
        extended_start = primary_start
        extended_end = primary_end + max_motif_duration

        # Cap at global max_step to avoid unnecessary empty lookahead.
        primary_end_capped = min(primary_end, max_step)
        extended_end_capped = min(extended_end, max_step)

        windows.append(
            WindowSpec(
                window_id=window_id,
                primary_start=int(primary_start),
                primary_end=int(primary_end_capped),
                extended_start=int(extended_start),
                extended_end=int(extended_end_capped),
            )
        )

        window_id += 1
        primary_start += stride

    return windows


def slice_window_edges(
    df_edges: pl.DataFrame,
    window: WindowSpec,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Return:
        df_primary: edges whose step is inside primary window.
        df_extended: edges whose step is inside extended window.
    """

    df_primary = df_edges.filter(
        (pl.col("step") >= window.primary_start) &
        (pl.col("step") <= window.primary_end)
    )

    df_extended = df_edges.filter(
        (pl.col("step") >= window.extended_start) &
        (pl.col("step") <= window.extended_end)
    )

    return df_primary, df_extended


def summarize_windows(
    df_edges: pl.DataFrame,
    windows: List[WindowSpec],
    preview_n: int = 10,
) -> pl.DataFrame:
    """
    Create a small summary table for windows.
    Counts are computed for all windows because current dataset is moderate size.
    """

    step_counts_df = (
        df_edges
        .group_by("step")
        .len()
        .sort("step")
    )

    steps = step_counts_df["step"].to_list()
    counts = step_counts_df["len"].to_list()
    prefix = [0]
    for count in counts:
        prefix.append(prefix[-1] + int(count))

    def count_in_range(start_step: int, end_step: int) -> int:
        left = bisect_right(steps, start_step - 1)
        right = bisect_right(steps, end_step)
        return prefix[right] - prefix[left]

    rows = []

    for w in windows:
        primary_count = count_in_range(w.primary_start, w.primary_end)
        extended_count = count_in_range(w.extended_start, w.extended_end)

        rows.append({
            "window_id": w.window_id,
            "primary_start": w.primary_start,
            "primary_end": w.primary_end,
            "extended_start": w.extended_start,
            "extended_end": w.extended_end,
            "primary_num_edges": primary_count,
            "extended_num_edges": extended_count,
            "lookahead_num_edges": extended_count - primary_count,
        })

    return pl.DataFrame(rows)


# ------------------------------------------------------------
# Build windows from df_edges
# ------------------------------------------------------------

min_step, max_step = get_step_bounds(df_edges)

windows = build_temporal_windows(
    min_step=min_step,
    max_step=max_step,
    window_size=WINDOW_SIZE,
    stride=STRIDE,
    max_motif_duration=MAX_MOTIF_DURATION,
)

print("Temporal windows created.")
print("min_step:", min_step)
print("max_step:", max_step)
print("num_windows:", len(windows))

window_summary = summarize_windows(df_edges, windows)

print("\nWindow summary preview:")
display(window_summary.head(10))

print("\nWindow summary tail:")
display(window_summary.tail(5))


# ------------------------------------------------------------
# Save window manifest for reproducibility
# ------------------------------------------------------------

WINDOW_MANIFEST_PATH = f"{LOG_DIR}/window_manifest.parquet"
window_summary.write_parquet(WINDOW_MANIFEST_PATH)

print("\nWindow manifest saved to:")
print(WINDOW_MANIFEST_PATH)


# ------------------------------------------------------------
# Quick test slicing first non-empty window
# ------------------------------------------------------------

non_empty_windows = [
    w for w in windows
    if window_summary.filter(pl.col("window_id") == w.window_id)["primary_num_edges"][0] > 0
]

if len(non_empty_windows) == 0:
    raise ValueError("No non-empty primary windows found. Check step range and window parameters.")

test_window = non_empty_windows[0]
df_primary_test, df_extended_test = slice_window_edges(df_edges, test_window)

print("\nFirst non-empty test window:")
print(test_window)
print("df_primary_test shape:", df_primary_test.shape)
print("df_extended_test shape:", df_extended_test.shape)

print("\nHead of df_primary_test:")
display(df_primary_test.head(5))

print("\nCell 6 completed.")


