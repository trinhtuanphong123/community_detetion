# ============================================================
# Cell 1: Install / Import / Mount Google Drive
# ============================================================

# Polars thường đã có sẵn trên Colab, nhưng dòng này giúp đảm bảo có bản mới đủ ổn định.
!pip -q install polars pyarrow

import os
import gc
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from bisect import bisect_right

import polars as pl

from google.colab import drive
drive.mount("/content/drive")

print("Environment ready.")
print("Polars version:", pl.__version__)



# ============================================================
# Cell 2: Config Paths and Parameters
# ============================================================

# ----------------------------
# Input path
# ----------------------------
DATA_PATH = "/content/drive/MyDrive/AML/dataset/subgraph_1hop_edges.parquet"

# ----------------------------
# Output directories
# ----------------------------
OUTPUT_DIR = "/content/drive/MyDrive/AML/outputs/motif_mining"

MOTIF_INSTANCE_DIR = f"{OUTPUT_DIR}/motif_instances"
MEMBERSHIP_DIR = f"{OUTPUT_DIR}/edge_motif_membership"
FEATURE_DIR = f"{OUTPUT_DIR}/edge_features"
LOG_DIR = f"{OUTPUT_DIR}/logs"

for path in [OUTPUT_DIR, MOTIF_INSTANCE_DIR, MEMBERSHIP_DIR, FEATURE_DIR, LOG_DIR]:
    os.makedirs(path, exist_ok=True)

# ----------------------------
# Windowing parameters
# ----------------------------
WINDOW_SIZE = 30
STRIDE = 30

# Maximum allowed gap between two consecutive edges in a temporal motif.
DELTA_HOP = 5

# Maximum total duration of a motif instance.
# This is also used as lookahead for extended windows.
MAX_MOTIF_DURATION = 20

# ----------------------------
# Candidate / output caps
# ----------------------------
MAX_IN_CANDIDATES = 100
MAX_OUT_CANDIDATES = 100
MAX_BRANCHING = 50

MAX_INSTANCES_PER_ANCHOR = 1_000
MAX_INSTANCES_PER_WINDOW = 1_000_000

# ----------------------------
# Amount constraints
# ----------------------------
AMOUNT_MIN = None

AMOUNT_RATIO_MIN = 0.3
AMOUNT_RATIO_MAX = 2.0

# ----------------------------
# Internal schema names
# ----------------------------
RAW_TO_INTERNAL_COLS = {
    "nameOrig": "src",
    "nameDest": "dst",
    "isSAR": "is_sar",
}

REQUIRED_RAW_COLUMNS = [
    "edge_id",
    "step",
    "type",
    "amount",
    "nameOrig",
    "nameDest",
    "isSAR",
]

MATCHER_COLUMNS = [
    "edge_id",
    "src",
    "dst",
    "step",
    "amount",
    "is_sar",
]

print("Config ready.")
print("DATA_PATH:", DATA_PATH)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("WINDOW_SIZE:", WINDOW_SIZE)
print("STRIDE:", STRIDE)
print("DELTA_HOP:", DELTA_HOP)
print("MAX_MOTIF_DURATION:", MAX_MOTIF_DURATION)



# ============================================================
# Cell 3: Load Parquet with Polars and Standardize Schema
# ============================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Input parquet file not found: {DATA_PATH}")

load_start = time.time()

df_raw = pl.read_parquet(DATA_PATH)

print("Loaded raw dataframe.")
print("Shape:", df_raw.shape)
print("Schema:")
print(df_raw.schema)

# ----------------------------
# Validate required columns
# ----------------------------
missing_cols = [col for col in REQUIRED_RAW_COLUMNS if col not in df_raw.columns]

if missing_cols:
    raise ValueError(
        f"Missing required columns: {missing_cols}. "
        f"Available columns: {df_raw.columns}"
    )

# ----------------------------
# Rename raw columns to internal schema
# ----------------------------
df = (
    df_raw
    .rename(RAW_TO_INTERNAL_COLS)
    .with_columns([
        pl.col("edge_id").cast(pl.UInt32),
        pl.col("src").cast(pl.Int64),
        pl.col("dst").cast(pl.Int64),
        pl.col("step").cast(pl.Int64),
        pl.col("amount").cast(pl.Float64),
        pl.col("is_sar").cast(pl.Int8),
    ])
    .sort(["step", "edge_id"])
)

# Minimal dataframe for motif matcher.
df_edges = df.select(MATCHER_COLUMNS)

load_end = time.time()

print("\nStandardized dataframe ready.")
print("df shape:", df.shape)
print("df_edges shape:", df_edges.shape)
print("Load + preprocess time:", round(load_end - load_start, 3), "seconds")

print("\nInternal schema:")
print(df_edges.schema)

print("\nHead of df_edges:")
display(df_edges.head(10))

# ----------------------------
# Basic sanity checks
# ----------------------------
summary = df_edges.select([
    pl.len().alias("num_edges"),
    pl.col("edge_id").n_unique().alias("n_unique_edges"),
    pl.col("src").n_unique().alias("n_unique_src"),
    pl.col("dst").n_unique().alias("n_unique_dst"),
    pl.col("step").min().alias("min_step"),
    pl.col("step").max().alias("max_step"),
    pl.col("amount").min().alias("min_amount"),
    pl.col("amount").max().alias("max_amount"),
    pl.col("is_sar").sum().alias("num_sar_edges"),
    pl.col("is_sar").mean().alias("sar_rate"),
])

print("\nBasic summary:")
display(summary)

# ----------------------------
# Data quality checks
# ----------------------------
duplicate_edge_ids = (
    df_edges
    .group_by("edge_id")
    .len()
    .filter(pl.col("len") > 1)
)

self_loops = df_edges.filter(pl.col("src") == pl.col("dst"))

non_positive_amount = df_edges.filter(pl.col("amount") <= 0)

print("\nData quality checks:")
print("Duplicate edge_id rows:", duplicate_edge_ids.height)
print("Self-loop rows:", self_loops.height)
print("Non-positive amount rows:", non_positive_amount.height)

if duplicate_edge_ids.height > 0:
    print("\nWarning: duplicate edge_id detected. Show first 10:")
    display(duplicate_edge_ids.head(10))

if self_loops.height > 0:
    print("\nWarning: self-loops detected. Show first 10:")
    display(self_loops.head(10))

if non_positive_amount.height > 0:
    print("\nWarning: non-positive amount detected. Show first 10:")
    display(non_positive_amount.head(10))

print("\nCell 3 completed.")


# ============================================================
# Cell 4: Standardize Schema
# ============================================================

def standardize_transaction_schema(
    df_raw: pl.DataFrame,
    raw_to_internal_cols: Dict[str, str],
    matcher_columns: List[str],
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Standardize raw transaction dataframe into internal edge schema.

    Input raw schema expected:
        edge_id, step, type, amount, nameOrig, nameDest, isSAR, ...

    Internal schema:
        edge_id, src, dst, step, amount, is_sar

    Returns:
        df_full: full dataframe with renamed columns and metadata preserved.
        df_edges: minimal edge dataframe for motif matching.
    """

    required_raw_cols = [
        "edge_id",
        "step",
        "amount",
        "nameOrig",
        "nameDest",
        "isSAR",
    ]

    missing_cols = [c for c in required_raw_cols if c not in df_raw.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required raw columns: {missing_cols}. "
            f"Available columns: {df_raw.columns}"
        )

    df_full = (
        df_raw
        .rename(raw_to_internal_cols)
        .with_columns([
            pl.col("edge_id").cast(pl.UInt32),
            pl.col("src").cast(pl.Int64),
            pl.col("dst").cast(pl.Int64),
            pl.col("step").cast(pl.Int64),
            pl.col("amount").cast(pl.Float64),
            pl.col("is_sar").cast(pl.Int8),
        ])
        .sort(["step", "edge_id"])
    )

    missing_internal_cols = [c for c in matcher_columns if c not in df_full.columns]
    if missing_internal_cols:
        raise ValueError(
            f"Missing internal matcher columns after standardization: {missing_internal_cols}"
        )

    df_edges = df_full.select(matcher_columns)

    return df_full, df_edges


# Re-standardize from df_raw loaded in Cell 3.
df_full, df_edges = standardize_transaction_schema(
    df_raw=df_raw,
    raw_to_internal_cols=RAW_TO_INTERNAL_COLS,
    matcher_columns=MATCHER_COLUMNS,
)

print("Schema standardization completed.")
print("df_full shape:", df_full.shape)
print("df_edges shape:", df_edges.shape)

print("\nInternal matcher schema:")
print(df_edges.schema)

print("\nHead of df_edges:")
display(df_edges.head(10))


# ------------------------------------------------------------
# Core sanity checks
# ------------------------------------------------------------

schema_summary = df_edges.select([
    pl.len().alias("num_edges"),
    pl.col("edge_id").n_unique().alias("n_unique_edges"),
    pl.col("src").n_unique().alias("n_unique_src_nodes"),
    pl.col("dst").n_unique().alias("n_unique_dst_nodes"),
    pl.concat([pl.col("src"), pl.col("dst")]).n_unique().alias("n_unique_all_nodes"),
    pl.col("step").min().alias("min_step"),
    pl.col("step").max().alias("max_step"),
    pl.col("amount").min().alias("min_amount"),
    pl.col("amount").max().alias("max_amount"),
    pl.col("amount").mean().alias("mean_amount"),
    pl.col("is_sar").sum().alias("num_sar_edges"),
    pl.col("is_sar").mean().alias("sar_rate"),
])

print("\nSchema summary:")
display(schema_summary)


# ------------------------------------------------------------
# Data quality checks
# ------------------------------------------------------------

duplicate_edge_ids = (
    df_edges
    .group_by("edge_id")
    .len()
    .filter(pl.col("len") > 1)
)

self_loops = df_edges.filter(pl.col("src") == pl.col("dst"))

non_positive_amount = df_edges.filter(pl.col("amount") <= 0)

null_check = df_edges.select([
    pl.col(c).null_count().alias(f"{c}_nulls")
    for c in df_edges.columns
])

print("\nData quality checks:")
print("Duplicate edge_id rows:", duplicate_edge_ids.height)
print("Self-loop rows:", self_loops.height)
print("Non-positive amount rows:", non_positive_amount.height)

print("\nNull check:")
display(null_check)

if duplicate_edge_ids.height > 0:
    print("\nWarning: duplicate edge_id detected. First 10 duplicated IDs:")
    display(duplicate_edge_ids.head(10))

if self_loops.height > 0:
    print("\nWarning: self-loops detected. First 10 rows:")
    display(self_loops.head(10))

if non_positive_amount.height > 0:
    print("\nWarning: non-positive amount detected. First 10 rows:")
    display(non_positive_amount.head(10))


# ------------------------------------------------------------
# Optional metadata dataframe for later joins
# ------------------------------------------------------------

EDGE_METADATA_COLUMNS = [
    c for c in [
        "edge_id",
        "type",
        "bankOrig",
        "bankDest",
        "daysInBankOrig",
        "daysInBankDest",
        "phoneChangesOrig",
        "phoneChangesDest",
        "oldbalanceOrig",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "alertID",
        "modelType",
    ]
    if c in df_full.columns
]

df_edge_metadata = df_full.select(EDGE_METADATA_COLUMNS)

print("\ndf_edge_metadata shape:", df_edge_metadata.shape)
print("Metadata columns:", EDGE_METADATA_COLUMNS)

print("\nCell 4 completed.")



# ============================================================
# Cell 5: Pattern Definitions
# ============================================================

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PatternEdge:
    """
    A pattern edge represents one abstract edge in a motif template.

    Example:
        PatternEdge("p1", "a", "d", 1, role="incoming_1")
    means:
        abstract edge p1 goes from abstract node a to abstract node d.
    """
    name: str
    src: str
    dst: str
    order: int
    role: str = ""


@dataclass(frozen=True)
class MotifPattern:
    """
    A motif template.

    nodes:
        Abstract node variables, not real account IDs.

    edges:
        Abstract directed edges.

    matcher_type:
        Used later to select a specialized matcher.
        Examples:
            fan_in
            fan_out
            path_cycle
            split_merge
            center_in_out
            two_stage_split

    time_order:
        strict means t_next > t_prev.
        nondecreasing means t_next >= t_prev.
    """
    name: str
    nodes: List[str]
    edges: List[PatternEdge]
    matcher_type: str
    max_duration: int
    time_order: str = "strict"
    distinct_nodes: bool = True
    amount_ratio_min: Optional[float] = None
    amount_ratio_max: Optional[float] = None
    description: str = ""


def validate_pattern(pattern: MotifPattern) -> None:
    """
    Basic validation for motif pattern definitions.
    """

    node_set = set(pattern.nodes)

    if len(node_set) != len(pattern.nodes):
        raise ValueError(f"Pattern {pattern.name} has duplicated node variables.")

    edge_names = [e.name for e in pattern.edges]
    if len(edge_names) != len(set(edge_names)):
        raise ValueError(f"Pattern {pattern.name} has duplicated edge names.")

    for e in pattern.edges:
        if e.src not in node_set:
            raise ValueError(f"Pattern {pattern.name}: edge {e.name} has unknown src node {e.src}.")
        if e.dst not in node_set:
            raise ValueError(f"Pattern {pattern.name}: edge {e.name} has unknown dst node {e.dst}.")

    orders = [e.order for e in pattern.edges]
    if len(orders) != len(set(orders)):
        raise ValueError(f"Pattern {pattern.name} has duplicated edge order values.")

    if pattern.time_order not in {"strict", "nondecreasing"}:
        raise ValueError(
            f"Pattern {pattern.name} has invalid time_order: {pattern.time_order}"
        )

    if pattern.max_duration <= 0:
        raise ValueError(f"Pattern {pattern.name} must have positive max_duration.")


# ------------------------------------------------------------
# Pattern 1: Fan-in size 4
# a -> d, b -> d, c -> d
# ------------------------------------------------------------

fan_in_4 = MotifPattern(
    name="fan_in_4",
    nodes=["a", "b", "c", "d"],
    edges=[
        PatternEdge("p1", "a", "d", 1, role="incoming_1"),
        PatternEdge("p2", "b", "d", 2, role="incoming_2"),
        PatternEdge("p3", "c", "d", 3, role="incoming_3"),
    ],
    matcher_type="fan_in",
    max_duration=MAX_MOTIF_DURATION,
    time_order="nondecreasing",
    distinct_nodes=True,
    description="Three distinct source nodes transfer to the same destination node.",
)


# ------------------------------------------------------------
# Pattern 2: Fan-out size 4
# a -> b, a -> c, a -> d
# ------------------------------------------------------------

fan_out_4 = MotifPattern(
    name="fan_out_4",
    nodes=["a", "b", "c", "d"],
    edges=[
        PatternEdge("p1", "a", "b", 1, role="outgoing_1"),
        PatternEdge("p2", "a", "c", 2, role="outgoing_2"),
        PatternEdge("p3", "a", "d", 3, role="outgoing_3"),
    ],
    matcher_type="fan_out",
    max_duration=MAX_MOTIF_DURATION,
    time_order="nondecreasing",
    distinct_nodes=True,
    description="One source node transfers to three distinct destination nodes.",
)


# ------------------------------------------------------------
# Pattern 3: Directed temporal cycles with length from 5 to 12
# cycle_k: v1 -> v2 -> ... -> vk -> v1
# ------------------------------------------------------------

def make_cycle_pattern(k: int) -> MotifPattern:
    """
    Create a directed temporal cycle pattern of length k.

    Example for k=5:
        v1 -> v2 -> v3 -> v4 -> v5 -> v1
    """

    if k < 3:
        raise ValueError("Cycle length k must be at least 3.")

    nodes = [f"v{i}" for i in range(1, k + 1)]

    edges = []

    for i in range(1, k):
        edges.append(
            PatternEdge(
                name=f"p{i}",
                src=f"v{i}",
                dst=f"v{i + 1}",
                order=i,
                role=f"step_{i}",
            )
        )

    edges.append(
        PatternEdge(
            name=f"p{k}",
            src=f"v{k}",
            dst="v1",
            order=k,
            role=f"step_{k}_close",
        )
    )

    return MotifPattern(
        name=f"cycle_{k}",
        nodes=nodes,
        edges=edges,
        matcher_type="path_cycle",
        max_duration=MAX_MOTIF_DURATION,
        time_order="strict",
        distinct_nodes=True,
        amount_ratio_min=AMOUNT_RATIO_MIN,
        amount_ratio_max=AMOUNT_RATIO_MAX,
        description=f"Directed temporal cycle with {k} nodes and {k} edges.",
    )


CYCLE_MIN_LEN = 5
CYCLE_MAX_LEN = 12

cycle_patterns = [
    make_cycle_pattern(k)
    for k in range(CYCLE_MIN_LEN, CYCLE_MAX_LEN + 1)
]

# Keep these variable names for compatibility with later cells if needed.
cycle_5 = cycle_patterns[0]
cycle_6 = cycle_patterns[1]
cycle_7 = cycle_patterns[2]
cycle_8 = cycle_patterns[3]
cycle_9 = cycle_patterns[4]
cycle_10 = cycle_patterns[5]
cycle_11 = cycle_patterns[6]
cycle_12 = cycle_patterns[7]



# ------------------------------------------------------------
# Pattern 4: Split-merge size 5
# a -> b, a -> c, a -> d, b -> e, c -> e, d -> e
# ------------------------------------------------------------

split_merge_5 = MotifPattern(
    name="split_merge_5",
    nodes=["a", "b", "c", "d", "e"],
    edges=[
        PatternEdge("p1", "a", "b", 1, role="split_1"),
        PatternEdge("p2", "a", "c", 2, role="split_2"),
        PatternEdge("p3", "a", "d", 3, role="split_3"),
        PatternEdge("p4", "b", "e", 4, role="merge_1"),
        PatternEdge("p5", "c", "e", 5, role="merge_2"),
        PatternEdge("p6", "d", "e", 6, role="merge_3"),
    ],
    matcher_type="split_merge",
    max_duration=MAX_MOTIF_DURATION,
    time_order="nondecreasing",
    distinct_nodes=True,
    amount_ratio_min=AMOUNT_RATIO_MIN,
    amount_ratio_max=AMOUNT_RATIO_MAX,
    description="One source splits value into three intermediates, then they merge into one sink.",
)


# ------------------------------------------------------------
# Pattern 5: Fan-in then fan-out
# a -> d, b -> d, c -> d, d -> e, d -> f
# ------------------------------------------------------------

fanin_fanout_6 = MotifPattern(
    name="fanin_fanout_6",
    nodes=["a", "b", "c", "d", "e", "f"],
    edges=[
        PatternEdge("p1", "a", "d", 1, role="incoming_1"),
        PatternEdge("p2", "b", "d", 2, role="incoming_2"),
        PatternEdge("p3", "c", "d", 3, role="incoming_3"),
        PatternEdge("p4", "d", "e", 4, role="outgoing_1"),
        PatternEdge("p5", "d", "f", 5, role="outgoing_2"),
    ],
    matcher_type="center_in_out",
    max_duration=MAX_MOTIF_DURATION,
    time_order="nondecreasing",
    distinct_nodes=True,
    amount_ratio_min=AMOUNT_RATIO_MIN,
    amount_ratio_max=AMOUNT_RATIO_MAX,
    description="A center node receives from three sources and then sends to two destinations.",
)


# ------------------------------------------------------------
# Pattern 6: Two-stage split
# a -> d, b -> d, b -> e, c -> e, d -> f, d -> g, e -> h
# ------------------------------------------------------------

two_stage_split_8 = MotifPattern(
    name="two_stage_split_8",
    nodes=["a", "b", "c", "d", "e", "f", "g", "h"],
    edges=[
        PatternEdge("p1", "a", "d", 1, role="stage1_in_1"),
        PatternEdge("p2", "b", "d", 2, role="stage1_in_2"),
        PatternEdge("p3", "b", "e", 3, role="stage1_in_3"),
        PatternEdge("p4", "c", "e", 4, role="stage1_in_4"),
        PatternEdge("p5", "d", "f", 5, role="stage2_out_1"),
        PatternEdge("p6", "d", "g", 6, role="stage2_out_2"),
        PatternEdge("p7", "e", "h", 7, role="stage2_out_3"),
    ],
    matcher_type="two_stage_split",
    max_duration=MAX_MOTIF_DURATION,
    time_order="nondecreasing",
    distinct_nodes=True,
    amount_ratio_min=AMOUNT_RATIO_MIN,
    amount_ratio_max=AMOUNT_RATIO_MAX,
    description="Two intermediate nodes receive from upstream nodes and then distribute downstream.",
)


# ------------------------------------------------------------
# Pattern registry
# ------------------------------------------------------------

ALL_PATTERNS = [
    fan_in_4,
    fan_out_4,
    *cycle_patterns,
    split_merge_5,
    fanin_fanout_6,
    two_stage_split_8,
]
# First implementation target.
# The first matcher implementation should start with these two.
ACTIVE_PATTERNS = [
    fan_in_4,
    fan_out_4,
]

for pattern in ALL_PATTERNS:
    validate_pattern(pattern)

print("Pattern definitions completed.")
print("Total patterns:", len(ALL_PATTERNS))
print("Active patterns for first implementation:", [p.name for p in ACTIVE_PATTERNS])

pattern_summary_rows = []

for p in ALL_PATTERNS:
    pattern_summary_rows.append({
        "name": p.name,
        "matcher_type": p.matcher_type,
        "num_nodes": len(p.nodes),
        "num_edges": len(p.edges),
        "max_duration": p.max_duration,
        "time_order": p.time_order,
        "distinct_nodes": p.distinct_nodes,
        "description": p.description,
    })

pattern_summary = pl.DataFrame(pattern_summary_rows)

display(pattern_summary)

print("\nCell 5 completed.")



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

    rows = []

    for w in windows:
        primary_count = df_edges.filter(
            (pl.col("step") >= w.primary_start) &
            (pl.col("step") <= w.primary_end)
        ).height

        extended_count = df_edges.filter(
            (pl.col("step") >= w.extended_start) &
            (pl.col("step") <= w.extended_end)
        ).height

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



# ============================================================
# Cell 7: Temporal Index
# ============================================================

@dataclass
class EdgeRecord:
    """
    Lightweight edge record used by temporal matchers.

    Keeping this as a dataclass makes matcher code clearer than using raw dicts.
    """
    edge_id: int
    src: int
    dst: int
    step: int
    amount: float
    is_sar: int


class TemporalIndex:
    """
    Temporal index for directed transaction edges.

    Indexes:
        outgoing_by_src[src] -> edges sorted by step
        incoming_by_dst[dst] -> edges sorted by step
        pair_by_src_dst[(src, dst)] -> edges sorted by step

    Query methods return candidate EdgeRecord objects within a step interval.
    This prevents scanning all edges in a delta_t range.
    """

    def __init__(self, edges: List[Dict[str, Any]]):
        self.out_edges = defaultdict(list)
        self.in_edges = defaultdict(list)
        self.pair_edges = defaultdict(list)

        self.num_edges = len(edges)

        # Convert dicts to EdgeRecord and sort once globally.
        edge_records = [
            EdgeRecord(
                edge_id=int(e["edge_id"]),
                src=int(e["src"]),
                dst=int(e["dst"]),
                step=int(e["step"]),
                amount=float(e["amount"]),
                is_sar=int(e["is_sar"]),
            )
            for e in edges
        ]

        edge_records.sort(key=lambda e: (e.step, e.edge_id))

        for e in edge_records:
            self.out_edges[e.src].append(e)
            self.in_edges[e.dst].append(e)
            self.pair_edges[(e.src, e.dst)].append(e)

        # Precompute step arrays for binary search.
        self.out_times = {
            key: [e.step for e in edge_list]
            for key, edge_list in self.out_edges.items()
        }

        self.in_times = {
            key: [e.step for e in edge_list]
            for key, edge_list in self.in_edges.items()
        }

        self.pair_times = {
            key: [e.step for e in edge_list]
            for key, edge_list in self.pair_edges.items()
        }

    def _range_query(
        self,
        edge_dict: Dict[Any, List[EdgeRecord]],
        time_dict: Dict[Any, List[int]],
        key: Any,
        t_min: int,
        t_max: int,
        include_left: bool = False,
        max_candidates: Optional[int] = None,
    ) -> List[EdgeRecord]:
        """
        Return edges with step in:
            (t_min, t_max] if include_left=False
            [t_min, t_max] if include_left=True

        By default, temporal motif expansion should use strict forward time:
            t_next > t_current
        """

        edges = edge_dict.get(key)
        if not edges:
            return []

        times = time_dict[key]

        if include_left:
            # First index with step >= t_min.
            # bisect_right(t_min - 1) works for integer steps.
            left = bisect_right(times, t_min - 1)
        else:
            # First index with step > t_min.
            left = bisect_right(times, t_min)

        # Last index with step <= t_max.
        right = bisect_right(times, t_max)

        result = edges[left:right]

        if max_candidates is not None and len(result) > max_candidates:
            # Keep earliest candidates by temporal order.
            # Later, this can be replaced by top-K amount/risk selection if needed.
            result = result[:max_candidates]

        return result

    def outgoing(
        self,
        src: int,
        t_min: int,
        t_max: int,
        include_left: bool = False,
        max_candidates: Optional[int] = None,
    ) -> List[EdgeRecord]:
        """
        Query src -> ? edges in temporal range.
        """
        return self._range_query(
            self.out_edges,
            self.out_times,
            src,
            t_min,
            t_max,
            include_left=include_left,
            max_candidates=max_candidates,
        )

    def incoming(
        self,
        dst: int,
        t_min: int,
        t_max: int,
        include_left: bool = False,
        max_candidates: Optional[int] = None,
    ) -> List[EdgeRecord]:
        """
        Query ? -> dst edges in temporal range.
        """
        return self._range_query(
            self.in_edges,
            self.in_times,
            dst,
            t_min,
            t_max,
            include_left=include_left,
            max_candidates=max_candidates,
        )

    def pair(
        self,
        src: int,
        dst: int,
        t_min: int,
        t_max: int,
        include_left: bool = False,
        max_candidates: Optional[int] = None,
    ) -> List[EdgeRecord]:
        """
        Query src -> dst edges in temporal range.
        """
        return self._range_query(
            self.pair_edges,
            self.pair_times,
            (src, dst),
            t_min,
            t_max,
            include_left=include_left,
            max_candidates=max_candidates,
        )

    def stats(self) -> Dict[str, int]:
        """
        Basic index statistics.
        """
        return {
            "num_edges": self.num_edges,
            "num_src_nodes": len(self.out_edges),
            "num_dst_nodes": len(self.in_edges),
            "num_pairs": len(self.pair_edges),
        }


def build_temporal_index_from_polars(df_window_edges: pl.DataFrame) -> TemporalIndex:
    """
    Build TemporalIndex from a Polars dataframe.

    Expected columns:
        edge_id, src, dst, step, amount, is_sar
    """

    required_cols = ["edge_id", "src", "dst", "step", "amount", "is_sar"]
    missing_cols = [c for c in required_cols if c not in df_window_edges.columns]

    if missing_cols:
        raise ValueError(f"Missing columns for TemporalIndex: {missing_cols}")

    # Convert only the current extended window to Python records.
    # This is acceptable because processing is windowed.
    edges = df_window_edges.select(required_cols).to_dicts()

    return TemporalIndex(edges)


# ------------------------------------------------------------
# Smoke test on first non-empty window from Cell 6
# ------------------------------------------------------------

index_build_start = time.time()

test_index = build_temporal_index_from_polars(df_extended_test)

index_build_end = time.time()

print("TemporalIndex smoke test completed.")
print("Index build time:", round(index_build_end - index_build_start, 3), "seconds")
print("Index stats:", test_index.stats())

# Pick one edge from primary test window and query possible next outgoing edges.
if df_primary_test.height > 0:
    anchor = df_primary_test.row(0, named=True)

    anchor_edge = EdgeRecord(
        edge_id=int(anchor["edge_id"]),
        src=int(anchor["src"]),
        dst=int(anchor["dst"]),
        step=int(anchor["step"]),
        amount=float(anchor["amount"]),
        is_sar=int(anchor["is_sar"]),
    )

    t_min = anchor_edge.step
    t_max = anchor_edge.step + DELTA_HOP

    next_candidates = test_index.outgoing(
        src=anchor_edge.dst,
        t_min=t_min,
        t_max=t_max,
        include_left=False,
        max_candidates=MAX_BRANCHING,
    )

    print("\nAnchor edge:")
    print(anchor_edge)

    print(f"\nOutgoing candidates from anchor.dst={anchor_edge.dst} in ({t_min}, {t_max}]:")
    print("num_candidates:", len(next_candidates))

    if len(next_candidates) > 0:
        for cand in next_candidates[:10]:
            print(cand)
else:
    print("df_primary_test is empty. No anchor query performed.")

print("\nCell 7 completed.")



# ============================================================
# Cell 8: Helper Functions for Motif Instance Output
# ============================================================

import json
import hashlib


def edge_record_to_dict(e: EdgeRecord) -> Dict[str, Any]:
    """
    Convert EdgeRecord to plain dict.
    Useful for debugging and JSON-safe serialization.
    """
    return {
        "edge_id": int(e.edge_id),
        "src": int(e.src),
        "dst": int(e.dst),
        "step": int(e.step),
        "amount": float(e.amount),
        "is_sar": int(e.is_sar),
    }


def make_canonical_key(
    motif_type: str,
    edge_ids: List[int],
    ordered: bool = True,
) -> str:
    """
    Canonical key used to avoid duplicate motif instances.

    For path/cycle motifs, ordered=True keeps edge order.
    For fan-in/fan-out motifs, ordered=False sorts edge IDs because branch order is not meaningful.
    """

    if ordered:
        canonical_edge_ids = [int(eid) for eid in edge_ids]
    else:
        canonical_edge_ids = sorted(int(eid) for eid in edge_ids)

    key_obj = {
        "motif_type": motif_type,
        "edge_ids": canonical_edge_ids,
    }

    return json.dumps(key_obj, sort_keys=True)


def make_motif_instance_id(
    window_id: int,
    motif_type: str,
    canonical_key: str,
) -> str:
    """
    Create stable motif_instance_id from window_id, motif_type, and canonical key.

    This is better than using a simple counter because it is reproducible.
    """

    digest = hashlib.md5(canonical_key.encode("utf-8")).hexdigest()[:12]
    return f"w{int(window_id):06d}_{motif_type}_{digest}"


def infer_ordered_flag(pattern: MotifPattern) -> bool:
    """
    Decide whether edge order should be meaningful for canonical key.

    fan_in and fan_out use unordered branch combinations.
    Path-like and composite motifs use ordered edge roles.
    """

    unordered_matcher_types = {"fan_in", "fan_out"}

    return pattern.matcher_type not in unordered_matcher_types


def validate_instance_edges(
    edges: List[EdgeRecord],
    pattern: MotifPattern,
) -> None:
    """
    Basic validation for a candidate motif instance.
    This does not fully validate structural correctness.
    Structural correctness belongs to each matcher.
    """

    if len(edges) != len(pattern.edges):
        raise ValueError(
            f"Pattern {pattern.name} expects {len(pattern.edges)} edges, "
            f"but got {len(edges)} edges."
        )

    edge_ids = [e.edge_id for e in edges]

    if len(edge_ids) != len(set(edge_ids)):
        raise ValueError(
            f"Duplicate edge_id inside motif instance for pattern {pattern.name}: {edge_ids}"
        )

    steps = [e.step for e in edges]

    if pattern.time_order == "strict":
        if any(steps[i + 1] <= steps[i] for i in range(len(steps) - 1)):
            raise ValueError(
                f"Strict time order violated for pattern {pattern.name}: steps={steps}"
            )

    elif pattern.time_order == "nondecreasing":
        if any(steps[i + 1] < steps[i] for i in range(len(steps) - 1)):
            raise ValueError(
                f"Nondecreasing time order violated for pattern {pattern.name}: steps={steps}"
            )

    duration = max(steps) - min(steps)

    if duration > pattern.max_duration:
        raise ValueError(
            f"Max duration violated for pattern {pattern.name}: "
            f"duration={duration}, max_duration={pattern.max_duration}"
        )


def make_motif_instance_row(
    window_id: int,
    pattern: MotifPattern,
    edges: List[EdgeRecord],
    node_map: Dict[str, int],
    role_map: Dict[str, int],
    anchor_edge_id: Optional[int] = None,
    validate: bool = True,
) -> Dict[str, Any]:
    """
    Create one row for motif_instances.

    Parameters:
        window_id:
            Temporal window ID.

        pattern:
            MotifPattern object.

        edges:
            Ordered list of EdgeRecord objects according to pattern edge order.
            For fan-in/fan-out, this should still be canonicalized by the matcher.

        node_map:
            Mapping from abstract pattern node to real node.
            Example:
                {"a": 1001, "b": 1002, "d": 9001}

        role_map:
            Mapping from pattern edge role/name to real edge_id.
            Example:
                {"incoming_1": 11, "incoming_2": 12, "incoming_3": 13}

        anchor_edge_id:
            Edge used as anchor. If None, use the first edge.

    Returns:
        A JSON/parquet-safe dictionary.
    """

    if validate:
        validate_instance_edges(edges, pattern)

    if len(node_map) == 0:
        raise ValueError("node_map must not be empty.")

    edge_ids = [int(e.edge_id) for e in edges]
    node_ids = sorted(set(int(v) for v in node_map.values()))

    steps = [int(e.step) for e in edges]
    amounts = [float(e.amount) for e in edges]
    sar_values = [int(e.is_sar) for e in edges]

    start_step = min(steps)
    end_step = max(steps)
    duration = end_step - start_step

    sar_edge_count = int(sum(sar_values))
    sar_ratio = float(sar_edge_count / len(edges)) if len(edges) > 0 else 0.0

    amount_sum = float(sum(amounts))
    amount_min = float(min(amounts)) if amounts else 0.0
    amount_max = float(max(amounts)) if amounts else 0.0
    amount_mean = float(amount_sum / len(amounts)) if amounts else 0.0

    ordered = infer_ordered_flag(pattern)
    canonical_key = make_canonical_key(
        motif_type=pattern.name,
        edge_ids=edge_ids,
        ordered=ordered,
    )

    motif_instance_id = make_motif_instance_id(
        window_id=window_id,
        motif_type=pattern.name,
        canonical_key=canonical_key,
    )

    if anchor_edge_id is None:
        anchor_edge_id = edge_ids[0]

    row = {
        "motif_instance_id": motif_instance_id,
        "motif_type": pattern.name,
        "matcher_type": pattern.matcher_type,
        "window_id": int(window_id),
        "anchor_edge_id": int(anchor_edge_id),

        # List columns.
        "edge_ids": edge_ids,
        "node_ids": node_ids,

        # JSON strings are more stable than nested dicts in early-stage parquet output.
        "node_map_json": json.dumps(
            {str(k): int(v) for k, v in node_map.items()},
            sort_keys=True,
        ),
        "role_map_json": json.dumps(
            {str(k): int(v) for k, v in role_map.items()},
            sort_keys=True,
        ),
        "canonical_key": canonical_key,

        "start_step": int(start_step),
        "end_step": int(end_step),
        "duration": int(duration),

        "num_edges": int(len(edges)),
        "num_nodes": int(len(node_ids)),

        "sar_edge_count": int(sar_edge_count),
        "sar_ratio": float(sar_ratio),
        "motif_is_sar_any": int(sar_edge_count > 0),
        "motif_is_sar_all": int(sar_edge_count == len(edges)),

        "amount_sum": float(amount_sum),
        "amount_min": float(amount_min),
        "amount_max": float(amount_max),
        "amount_mean": float(amount_mean),
    }

    return row


def make_edge_motif_membership_rows(
    motif_row: Dict[str, Any],
    pattern: MotifPattern,
    edges: List[EdgeRecord],
) -> List[Dict[str, Any]]:
    """
    Create edge_motif_membership rows from one motif instance.

    Each row says:
        edge_id belongs to motif_instance_id with a role.
    """

    if len(edges) != len(pattern.edges):
        raise ValueError(
            f"Pattern {pattern.name} expects {len(pattern.edges)} edges, "
            f"but got {len(edges)} edges."
        )

    pattern_edges_sorted = sorted(pattern.edges, key=lambda x: x.order)

    rows = []

    for p_edge, real_edge in zip(pattern_edges_sorted, edges):
        role = p_edge.role if p_edge.role else p_edge.name

        rows.append({
            "edge_id": int(real_edge.edge_id),
            "motif_instance_id": motif_row["motif_instance_id"],
            "motif_type": motif_row["motif_type"],
            "matcher_type": motif_row["matcher_type"],
            "window_id": int(motif_row["window_id"]),
            "role_in_motif": role,
            "pattern_edge_name": p_edge.name,
            "pattern_edge_order": int(p_edge.order),
            "edge_step": int(real_edge.step),
            "edge_src": int(real_edge.src),
            "edge_dst": int(real_edge.dst),
            "edge_amount": float(real_edge.amount),
            "edge_is_sar": int(real_edge.is_sar),
        })

    return rows


def motif_instance_rows_to_polars(rows: List[Dict[str, Any]]) -> pl.DataFrame:
    """
    Convert motif instance rows to Polars DataFrame with stable schema.
    """

    if len(rows) == 0:
        return pl.DataFrame(
            schema={
                "motif_instance_id": pl.String,
                "motif_type": pl.String,
                "matcher_type": pl.String,
                "window_id": pl.Int64,
                "anchor_edge_id": pl.Int64,
                "edge_ids": pl.List(pl.Int64),
                "node_ids": pl.List(pl.Int64),
                "node_map_json": pl.String,
                "role_map_json": pl.String,
                "canonical_key": pl.String,
                "start_step": pl.Int64,
                "end_step": pl.Int64,
                "duration": pl.Int64,
                "num_edges": pl.Int64,
                "num_nodes": pl.Int64,
                "sar_edge_count": pl.Int64,
                "sar_ratio": pl.Float64,
                "motif_is_sar_any": pl.Int8,
                "motif_is_sar_all": pl.Int8,
                "amount_sum": pl.Float64,
                "amount_min": pl.Float64,
                "amount_max": pl.Float64,
                "amount_mean": pl.Float64,
            }
        )

    return pl.DataFrame(rows)


def membership_rows_to_polars(rows: List[Dict[str, Any]]) -> pl.DataFrame:
    """
    Convert edge membership rows to Polars DataFrame with stable schema.
    """

    if len(rows) == 0:
        return pl.DataFrame(
            schema={
                "edge_id": pl.Int64,
                "motif_instance_id": pl.String,
                "motif_type": pl.String,
                "matcher_type": pl.String,
                "window_id": pl.Int64,
                "role_in_motif": pl.String,
                "pattern_edge_name": pl.String,
                "pattern_edge_order": pl.Int64,
                "edge_step": pl.Int64,
                "edge_src": pl.Int64,
                "edge_dst": pl.Int64,
                "edge_amount": pl.Float64,
                "edge_is_sar": pl.Int8,
            }
        )

    return pl.DataFrame(rows)


def write_motif_outputs(
    motif_rows: List[Dict[str, Any]],
    membership_rows: List[Dict[str, Any]],
    window_id: int,
    motif_type: str,
    motif_instance_dir: str = MOTIF_INSTANCE_DIR,
    membership_dir: str = MEMBERSHIP_DIR,
) -> Tuple[str, str]:
    """
    Write motif_instances and edge_motif_membership parquet shards.

    Returns:
        motif_path, membership_path
    """

    motif_df = motif_instance_rows_to_polars(motif_rows)
    membership_df = membership_rows_to_polars(membership_rows)

    motif_path = f"{motif_instance_dir}/window_{int(window_id):06d}_{motif_type}.parquet"
    membership_path = f"{membership_dir}/window_{int(window_id):06d}_{motif_type}.parquet"

    motif_df.write_parquet(motif_path)
    membership_df.write_parquet(membership_path)

    return motif_path, membership_path


# ------------------------------------------------------------
# Smoke test using a small synthetic fan-in instance
# ------------------------------------------------------------

print("Running Cell 8 smoke test...")

# Try to create a synthetic-like example from the first three primary edges
# only if they can be coerced into the fan_in_4 structure for output testing.
# This test is about output schema, not real motif correctness.
test_edges_raw = df_primary_test.head(3).to_dicts()

if len(test_edges_raw) >= 3:
    test_edges = [
        EdgeRecord(
            edge_id=int(e["edge_id"]),
            src=int(e["src"]),
            dst=int(e["dst"]),
            step=int(e["step"]),
            amount=float(e["amount"]),
            is_sar=int(e["is_sar"]),
        )
        for e in test_edges_raw
    ]

    # Sort by step, edge_id to satisfy fan_in_4 nondecreasing validation.
    test_edges = sorted(test_edges, key=lambda e: (e.step, e.edge_id))

    # For smoke test only, create an artificial node_map from the selected edges.
    # The matcher later will build a real structural node_map.
    test_node_map = {
        "a": test_edges[0].src,
        "b": test_edges[1].src,
        "c": test_edges[2].src,
        "d": test_edges[0].dst,
    }

    test_role_map = {
        "incoming_1": test_edges[0].edge_id,
        "incoming_2": test_edges[1].edge_id,
        "incoming_3": test_edges[2].edge_id,
    }

    try:
        test_motif_row = make_motif_instance_row(
            window_id=test_window.window_id,
            pattern=fan_in_4,
            edges=test_edges,
            node_map=test_node_map,
            role_map=test_role_map,
            anchor_edge_id=test_edges[0].edge_id,
            validate=True,
        )

        test_membership_rows = make_edge_motif_membership_rows(
            motif_row=test_motif_row,
            pattern=fan_in_4,
            edges=test_edges,
        )

        test_motif_df = motif_instance_rows_to_polars([test_motif_row])
        test_membership_df = membership_rows_to_polars(test_membership_rows)

        print("Motif instance output schema:")
        print(test_motif_df.schema)
        display(test_motif_df)

        print("\nMembership output schema:")
        print(test_membership_df.schema)
        display(test_membership_df)

    except Exception as exc:
        print("Smoke test could not create a validated output row.")
        print("Reason:", repr(exc))
        print("This is acceptable if the first three rows violate time duration/order constraints.")
else:
    print("Not enough rows in df_primary_test for smoke test.")

print("\nCell 8 completed.")