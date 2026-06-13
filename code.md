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

# Add to the existing import block:
from itertools import combinations, product
from dataclasses import dataclass, field    # already imported in Cell 5; add here too
                                            # so Cell 5 can stay self-contained
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
MAX_FAN_INSTANCES_PER_WINDOW = 10_000
MAX_FAN_INSTANCES_PER_CENTER = 20

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


# ----------------------------
# Fan-in / fan-out branch sizes
# ----------------------------
FAN_IN_SIZES  = [3, 4, 5, 6, 7]   # number of incoming branches per pattern
FAN_OUT_SIZES = [3, 4, 5, 6, 7]   # number of outgoing branches per pattern

# Candidate cap schedule keyed by branch count.
# C(cap, n) is bounded at roughly 250 000 for all n.
FAN_IN_CAP_SCHEDULE  = {3: 100, 4: 50, 5: 25, 6: 15, 7: 10}
FAN_OUT_CAP_SCHEDULE = {3: 100, 4: 50, 5: 25, 6: 15, 7: 10}

# ----------------------------
# Split-merge branch sizes
# ----------------------------
SPLIT_MERGE_SIZES = [3, 4, 5]     # number of branch pairs (split legs = merge legs)

SPLIT_MERGE_OUT_CAP   = {3: 12, 4: 8, 5: 6}
SPLIT_MERGE_MERGE_CAP = {3: 3,  4: 2, 5: 2}

# ----------------------------
# Center-in-out arm configs
# ----------------------------
# Each tuple is (n_in, n_out).
CENTER_INOUT_CONFIGS = [(3, 2), (3, 3), (4, 2), (4, 3), (5, 3)]

CENTER_INOUT_IN_CAP  = {3: 30, 4: 15, 5: 10}
CENTER_INOUT_OUT_CAP = {2: 20, 3: 12,  4: 8}

# ----------------------------
# Cycle sizes
# ----------------------------
CYCLE_SIZES           = list(range(5, 11))   # 5, 6, 7, 8, 9, 10
BIDIRECTIONAL_CYCLE_THRESHOLD = 999            # use bidirectional DFS for k >= this

CYCLE_MAX_BRANCHING = {5: 8, 6: 6, 7: 5, 8: 4, 9: 4, 10: 4}

# ----------------------------
# Stacked Bipartite layer configs
# ----------------------------
# Each entry is a list of intermediate layer widths.
STACKED_BIPARTITE_CONFIGS = [
    [2],        # equivalent to split_merge_4
    [3],        # equivalent to split_merge_6
    [2, 2],     # two-layer: 1→2→2→1
    [3, 2],     # two-layer: 1→3→2→1
]


print("Config ready.")
print("DATA_PATH:", DATA_PATH)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("WINDOW_SIZE:", WINDOW_SIZE)
print("STRIDE:", STRIDE)
print("DELTA_HOP:", DELTA_HOP)
print("MAX_MOTIF_DURATION:", MAX_MOTIF_DURATION)



# ============================================================
# Cell 3: Load Raw Parquet with Polars
# ============================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Input parquet file not found: {DATA_PATH}")

load_start = time.time()
df_raw = pl.read_parquet(DATA_PATH)
load_end = time.time()

print("Loaded raw transaction dataframe.")
print("Shape:", df_raw.shape)
print("Schema:")
print(df_raw.schema)
print("Load time:", round(load_end - load_start, 3), "seconds")

# Validate that required columns are in df_raw
missing_cols = [col for col in REQUIRED_RAW_COLUMNS if col not in df_raw.columns]
if missing_cols:
    raise ValueError(
        f"Missing required raw columns: {missing_cols}. "
        f"Available columns: {df_raw.columns}"
    )
print("All required raw columns are present.")
print("\nCell 3 completed.")


# ============================================================
# Cell 4: Standardize Schema and Execute Quality Checks
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


# Standardize raw transaction dataframe loaded in Cell 3
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

# ============================================================
# Factory: fan_in family
# ============================================================

def make_fan_in_pattern(n: int) -> MotifPattern:
    """
    n distinct source nodes each send one edge to one common destination.
    n = number of incoming branches (= number of source nodes).
    Total nodes: n + 1. Total edges: n.
    Minimum n: 3.
    """
    if n < 3:
        raise ValueError(f"fan_in requires n >= 3, got {n}")

    src_nodes = [f"src_{i}" for i in range(1, n + 1)]
    nodes     = src_nodes + ["dst"]

    edges = [
        PatternEdge(
            name  = f"p{i}",
            src   = f"src_{i}",
            dst   = "dst",
            order = i,
            role  = f"incoming_{i}",
        )
        for i in range(1, n + 1)
    ]

    return MotifPattern(
        name          = f"fan_in_{n + 1}",   # n+1 = total node count
        nodes         = nodes,
        edges         = edges,
        matcher_type  = "fan_in",
        max_duration  = MAX_MOTIF_DURATION,
        time_order    = "nondecreasing",
        distinct_nodes= True,
        description   = (
            f"Fan-in: {n} distinct source nodes each transfer to one destination."
        ),
    )

# ------------------------------------------------------------
# Pattern 2: Fan-out size 4
# a -> b, a -> c, a -> d
# ------------------------------------------------------------

def make_fan_out_pattern(n: int) -> MotifPattern:
    """
    One source node sends one edge to n distinct destinations.
    n = number of outgoing branches (= number of destination nodes).
    Total nodes: n + 1. Total edges: n.
    Minimum n: 3.
    """
    if n < 3:
        raise ValueError(f"fan_out requires n >= 3, got {n}")

    dst_nodes = [f"dst_{i}" for i in range(1, n + 1)]
    nodes     = ["src"] + dst_nodes

    edges = [
        PatternEdge(
            name  = f"p{i}",
            src   = "src",
            dst   = f"dst_{i}",
            order = i,
            role  = f"outgoing_{i}",
        )
        for i in range(1, n + 1)
    ]

    return MotifPattern(
        name          = f"fan_out_{n + 1}",
        nodes         = nodes,
        edges         = edges,
        matcher_type  = "fan_out",
        max_duration  = MAX_MOTIF_DURATION,
        time_order    = "nondecreasing",
        distinct_nodes= True,
        description   = (
            f"Fan-out: one source transfers to {n} distinct destination nodes."
        ),
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


# CYCLE_MIN_LEN = 5
# CYCLE_MAX_LEN = 12

# cycle_patterns = [
#     make_cycle_pattern(k)
#     for k in range(CYCLE_MIN_LEN, CYCLE_MAX_LEN + 1)
# ]

# # Keep these variable names for compatibility with later cells if needed.
# cycle_5 = cycle_patterns[0]
# cycle_6 = cycle_patterns[1]
# cycle_7 = cycle_patterns[2]
# cycle_8 = cycle_patterns[3]
# cycle_9 = cycle_patterns[4]
# cycle_10 = cycle_patterns[5]
# cycle_11 = cycle_patterns[6]
# cycle_12 = cycle_patterns[7]



# ------------------------------------------------------------
# Pattern 4: Split-merge size 5
# a -> b, a -> c, a -> d, b -> e, c -> e, d -> e
# ------------------------------------------------------------

def make_split_merge_pattern(n: int) -> MotifPattern:
    """
    One source fans out to n intermediates; all intermediates fan in to one sink.
    n = number of branch pairs.
    Total nodes: n + 2 (source, n intermediates, sink).
    Total edges: 2n.
    Minimum n: 3.
    """
    if n < 3:
        raise ValueError(f"split_merge requires n >= 3, got {n}")

    mid_nodes = [f"mid_{i}" for i in range(1, n + 1)]
    nodes     = ["src"] + mid_nodes + ["sink"]

    split_edges = [
        PatternEdge(
            name  = f"split_{i}",
            src   = "src",
            dst   = f"mid_{i}",
            order = i,
            role  = f"split_{i}",
        )
        for i in range(1, n + 1)
    ]
    merge_edges = [
        PatternEdge(
            name  = f"merge_{i}",
            src   = f"mid_{i}",
            dst   = "sink",
            order = n + i,
            role  = f"merge_{i}",
        )
        for i in range(1, n + 1)
    ]

    return MotifPattern(
        name             = f"split_merge_{2 * n}",   # 2n total edges
        nodes            = nodes,
        edges            = split_edges + merge_edges,
        matcher_type     = "split_merge",
        max_duration     = MAX_MOTIF_DURATION,
        time_order       = "nondecreasing",
        distinct_nodes   = True,
        amount_ratio_min = AMOUNT_RATIO_MIN,
        amount_ratio_max = AMOUNT_RATIO_MAX,
        description      = (
            f"Split-merge: source splits to {n} intermediates "
            f"that converge to one sink."
        ),
    )


# ------------------------------------------------------------
# Pattern 5: Fan-in then fan-out
# a -> d, b -> d, c -> d, d -> e, d -> f
# ------------------------------------------------------------

def make_center_inout_pattern(n_in: int, n_out: int) -> MotifPattern:
    """
    n_in sources feed into one center; center distributes to n_out destinations.
    All nodes must be distinct.
    Total nodes: n_in + 1 + n_out. Total edges: n_in + n_out.
    Minimum n_in: 2. Minimum n_out: 2.
    """
    if n_in < 2:
        raise ValueError(f"center_inout requires n_in >= 2, got {n_in}")
    if n_out < 2:
        raise ValueError(f"center_inout requires n_out >= 2, got {n_out}")

    in_nodes  = [f"in_{i}"  for i in range(1, n_in  + 1)]
    out_nodes = [f"out_{i}" for i in range(1, n_out + 1)]
    nodes     = in_nodes + ["center"] + out_nodes

    in_edges = [
        PatternEdge(
            name  = f"in_{i}",
            src   = f"in_{i}",
            dst   = "center",
            order = i,
            role  = f"incoming_{i}",
        )
        for i in range(1, n_in + 1)
    ]
    out_edges = [
        PatternEdge(
            name  = f"out_{i}",
            src   = "center",
            dst   = f"out_{i}",
            order = n_in + i,
            role  = f"outgoing_{i}",
        )
        for i in range(1, n_out + 1)
    ]

    return MotifPattern(
        name             = f"center_inout_{n_in}in_{n_out}out",
        nodes            = nodes,
        edges            = in_edges + out_edges,
        matcher_type     = "center_in_out",
        max_duration     = MAX_MOTIF_DURATION,
        time_order       = "nondecreasing",
        distinct_nodes   = True,
        amount_ratio_min = AMOUNT_RATIO_MIN,
        amount_ratio_max = AMOUNT_RATIO_MAX,
        description      = (
            f"Center-in-out: {n_in} sources feed center, "
            f"center distributes to {n_out} destinations."
        ),
    )



# ------------------------------------------------------------
# Pattern 6: Two-stage split (Stacked Bipartite) - Disabled/Removed
# a -> d, b -> d, b -> e, c -> e, d -> f, d -> g, e -> h
# ------------------------------------------------------------

# def make_stacked_bipartite_pattern(layer_sizes: List[int]) -> MotifPattern:
#     """
#     Layered bipartite chain.
# 
#     layer_sizes defines the width of each intermediate layer.
#     Example: layer_sizes=[3, 2] means:
#         layer 0: [L0_0]       (1 source node)
#         layer 1: [L1_0..L1_2] (3 intermediate nodes)
#         layer 2: [L2_0..L2_1] (2 intermediate nodes)
#         layer 3: [L3_0]       (1 sink node)
# 
#     Between each pair of adjacent layers, there is a full bipartite
#     edge set (every node in layer i connects to every node in layer i+1).
# 
#     Total edges = sum over adjacent layer pairs of (width_i * width_{i+1}),
#     with width_0 = width_last = 1.
#     """
#     if not layer_sizes:
#         raise ValueError("layer_sizes must have at least one element")
# 
#     all_layer_widths = [1] + list(layer_sizes) + [1]
# 
#     # Build node names per layer.
#     all_layers: List[List[str]] = []
#     for l_idx, width in enumerate(all_layer_widths):
#         all_layers.append([f"L{l_idx}_{j}" for j in range(width)])
# 
#     nodes = [n for layer in all_layers for n in layer]
# 
#     edges: List[PatternEdge] = []
#     order = 1
#     for l_idx in range(len(all_layers) - 1):
#         for src_node in all_layers[l_idx]:
#             for dst_node in all_layers[l_idx + 1]:
#                 edges.append(
#                     PatternEdge(
#                         name  = f"e{order}",
#                         src   = src_node,
#                         dst   = dst_node,
#                         order = order,
#                         role  = f"L{l_idx}_to_L{l_idx+1}_{src_node}_{dst_node}",
#                     )
#                 )
#                 order += 1
# 
#     size_str = "_".join(str(s) for s in layer_sizes)
# 
#     return MotifPattern(
#         name             = f"stacked_bipartite_{size_str}",
#         nodes            = nodes,
#         edges            = edges,
#         matcher_type     = "stacked_bipartite",
#         max_duration     = MAX_MOTIF_DURATION,
#         time_order       = "nondecreasing",
#         distinct_nodes   = True,
#         amount_ratio_min = AMOUNT_RATIO_MIN,
#         amount_ratio_max = AMOUNT_RATIO_MAX,
#         description      = (
#             f"Stacked bipartite with intermediate layer widths {layer_sizes}."
#         ),
#     )


# ============================================================
# Build pattern collections from factories
# ============================================================

fan_in_patterns        = [make_fan_in_pattern(n)      for n in FAN_IN_SIZES]
fan_out_patterns       = [make_fan_out_pattern(n)     for n in FAN_OUT_SIZES]
cycle_patterns         = [make_cycle_pattern(k)       for k in CYCLE_SIZES]
split_merge_patterns   = [make_split_merge_pattern(n) for n in SPLIT_MERGE_SIZES]
center_inout_patterns  = [make_center_inout_pattern(n_in, n_out)
                          for n_in, n_out in CENTER_INOUT_CONFIGS]
stacked_bipartite_patterns = []

# Convenience aliases for backward compatibility with later cells.
fan_in_4   = fan_in_patterns[0]    # n=3 branches, 4 total nodes
fan_out_4  = fan_out_patterns[0]   # n=3 branches, 4 total nodes
cycle_5    = cycle_patterns[0]
split_merge_6 = split_merge_patterns[0]   # 3 branch pairs, 5 nodes, 6 edges
split_merge_8 = split_merge_patterns[1]   # 4 branch pairs, 6 nodes, 8 edges
split_merge_10 = split_merge_patterns[2]  # 5 branch pairs, 7 nodes, 10 edges
split_merge_5 = split_merge_6              # backward compatibility alias
fanin_fanout_6 = center_inout_patterns[0] # (3 in, 2 out)

# 1. Core Patterns: Flow patterns (standard split-merge, standard center-in-out) + small fan patterns
CORE_PATTERNS = (
    [p for p in fan_in_patterns if len(p.edges) <= 4]        # fan_in_4, fan_in_5
    + [p for p in fan_out_patterns if len(p.edges) <= 4]    # fan_out_4, fan_out_5
    + [p for p in split_merge_patterns if len(p.edges) // 2 == 3]  # split_merge_6 (3 branch pairs)
    + [p for p in center_inout_patterns if p.name == "center_inout_3in_2out"]
)

# 2. Exploration Patterns: Large fan patterns + larger split-merge + larger center-in-out
EXPLORATION_PATTERNS = (
    [p for p in fan_in_patterns if len(p.edges) > 4]        # fan_in_6, fan_in_7, fan_in_8
    + [p for p in fan_out_patterns if len(p.edges) > 4]    # fan_out_6, fan_out_7, fan_out_8
    + [p for p in split_merge_patterns if len(p.edges) // 2 > 3]  # split_merge_7, split_merge_9
    + [p for p in center_inout_patterns if p.name != "center_inout_3in_2out"]
)

# 3. Diagnostic Patterns: Cycles
DIAGNOSTIC_PATTERNS = cycle_patterns

# For backward compatibility and dynamic validation of all configurations
ALL_PATTERNS = CORE_PATTERNS + EXPLORATION_PATTERNS + DIAGNOSTIC_PATTERNS
ACTIVE_PATTERNS = CORE_PATTERNS

for p in ALL_PATTERNS:
    validate_pattern(p)

print(f"Pattern definitions completed.")
print(f"Total patterns: {len(ALL_PATTERNS)}")
print(f"  fan_in family:          {len(fan_in_patterns)}")
print(f"  fan_out family:         {len(fan_out_patterns)}")
print(f"  cycle family:           {len(cycle_patterns)}")
print(f"  split_merge family:     {len(split_merge_patterns)}")
print(f"  center_inout family:    {len(center_inout_patterns)}")
print(f"  stacked_bipartite:      {len(stacked_bipartite_patterns)}")

pattern_summary = pl.DataFrame([
    {
        "name":          p.name,
        "matcher_type":  p.matcher_type,
        "num_nodes":     len(p.nodes),
        "num_edges":     len(p.edges),
        "max_duration":  p.max_duration,
        "time_order":    p.time_order,
    }
    for p in ALL_PATTERNS
])
display(pattern_summary)
print("\nCell 5 completed.")

 # ============================================================
# Cell 5a: Window Candidate Summary
# ============================================================

@dataclass
class WindowCandidateSummary:
    """
    Lightweight per-window structural summary.

    Computed once per window, shared across all matchers.
    Allows matchers to skip nodes that cannot satisfy minimum
    degree requirements before touching the temporal index.
    """
    total_edges: int
    node_in_degree:  Dict[int, int]
    node_out_degree: Dict[int, int]

    # Pre-filtered node lists for each matcher type.
    fan_in_candidate_dsts:   Dict[int, List[int]]   # dst -> [min_branches] for each size
    fan_out_candidate_srcs:  Dict[int, List[int]]   # src -> [min_branches] for each size
    center_candidate_nodes:  List[int]               # nodes with in >= 2 and out >= 2

    @property
    def high_in_degree_nodes(self) -> List[int]:
        return [n for n, d in self.node_in_degree.items()  if d >= 3]

    @property
    def high_out_degree_nodes(self) -> List[int]:
        return [n for n, d in self.node_out_degree.items() if d >= 3]


def build_window_candidate_summary(
    df_extended: pl.DataFrame,
    min_fan_branches: int = 3,
    min_center_in:    int = 2,
    min_center_out:   int = 2,
) -> WindowCandidateSummary:
    """
    Build a WindowCandidateSummary from the extended window edge dataframe.

    This is O(|edges|) and should be called once per window before
    any matcher is invoked.
    """
    in_degree:  Dict[int, int] = defaultdict(int)
    out_degree: Dict[int, int] = defaultdict(int)

    for row in df_extended.select(["src", "dst"]).iter_rows():
        src, dst = row
        out_degree[src] += 1
        in_degree[dst]  += 1

    # Pre-build candidate node lists per role.
    fan_in_dsts   = {n for n, d in in_degree.items()  if d >= min_fan_branches}
    fan_out_srcs  = {n for n, d in out_degree.items() if d >= min_fan_branches}
    center_nodes  = [
        n for n in set(in_degree) & set(out_degree)
        if in_degree[n] >= min_center_in and out_degree[n] >= min_center_out
    ]

    return WindowCandidateSummary(
        total_edges      = df_extended.height,
        node_in_degree   = dict(in_degree),
        node_out_degree  = dict(out_degree),
        fan_in_candidate_dsts  = {n: in_degree[n]  for n in fan_in_dsts},
        fan_out_candidate_srcs = {n: out_degree[n] for n in fan_out_srcs},
        center_candidate_nodes = center_nodes,
    )


# ------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------

# test_summary = build_window_candidate_summary(df_extended_test, min_fan_branches=3)

# print("WindowCandidateSummary smoke test:")
# print("  total_edges:           ", test_summary.total_edges)
# print("  nodes with in >= 3:    ", len(test_summary.fan_in_candidate_dsts))
# print("  nodes with out >= 3:   ", len(test_summary.fan_out_candidate_srcs))
# print("  center candidates:     ", len(test_summary.center_candidate_nodes))
print("\nCell 5a completed.")

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

    Query methods return candidate EdgeRecord objects within a step interval.
    This prevents scanning all edges in a delta_t range.
    """

    def __init__(self, edges: List[EdgeRecord]):
        import numpy as np

        self.out_edges = defaultdict(list)
        self.in_edges = defaultdict(list)

        self.num_edges = len(edges)

        # Sort once globally to ensure temporal order.
        edge_records = sorted(edges, key=lambda e: (e.step, e.edge_id))

        # Initialize metadata
        self.node_in_degree = defaultdict(int)
        self.node_out_degree = defaultdict(int)
        self.pair_count = defaultdict(int)

        for e in edge_records:
            self.out_edges[e.src].append(e)
            self.in_edges[e.dst].append(e)

            # Increment degrees and pair counts
            self.node_out_degree[e.src] += 1
            self.node_in_degree[e.dst] += 1
            self.pair_count[(e.src, e.dst)] += 1

        # Convert defaultdicts to regular dicts to save space
        self.node_in_degree = dict(self.node_in_degree)
        self.node_out_degree = dict(self.node_out_degree)
        self.pair_count = dict(self.pair_count)
        self.out_edges = dict(self.out_edges)
        self.in_edges = dict(self.in_edges)

        # Index-level metadata
        total_sar = sum(1 for e in edge_records if e.is_sar)
        self.sar_rate_window = float(total_sar / self.num_edges) if self.num_edges > 0 else 0.0

        amounts = [e.amount for e in edge_records]
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        if amounts:
            q_vals = np.quantile(amounts, quantiles)
            self.amount_quantiles = {q: float(v) for q, v in zip(quantiles, q_vals)}
        else:
            self.amount_quantiles = {q: 0.0 for q in quantiles}

    def _range_query(
        self,
        edge_dict: Dict[Any, List[EdgeRecord]],
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

        if include_left:
            # First index with step >= t_min.
            # bisect_right(t_min - 1) works for integer steps.
            left = bisect_right(edges, t_min - 1, key=lambda e: e.step)
        else:
            # First index with step > t_min.
            left = bisect_right(edges, t_min, key=lambda e: e.step)

        # Last index with step <= t_max.
        right = bisect_right(edges, t_max, key=lambda e: e.step)

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
        candidates = self.outgoing(src, t_min, t_max, include_left=include_left)
        result = [e for e in candidates if e.dst == dst]
        if max_candidates is not None and len(result) > max_candidates:
            result = result[:max_candidates]
        return result

    def stats(self) -> Dict[str, int]:
        """
        Basic index statistics.
        """
        return {
            "num_edges": self.num_edges,
            "num_src_nodes": len(self.out_edges),
            "num_dst_nodes": len(self.in_edges),
            "num_pairs": len(self.pair_count),
        }

    def has_any_outgoing(
        self,
        src: int,
        t_min: int,
        t_max: int,
    ) -> bool:
        """
        Return True if there is at least one outgoing edge from src
        with step in (t_min, t_max].
        Used for cheap return-to-start feasibility checks in cycle DFS.
        """
        edges = self.out_edges.get(src)
        if not edges:
            return False
        left  = bisect_right(edges, t_min, key=lambda e: e.step)
        right = bisect_right(edges, t_max, key=lambda e: e.step)
        return right > left

    def has_any_pair(
        self,
        src: int,
        dst: int,
        t_min: int,
        t_max: int,
    ) -> bool:
        """
        Return True if there is at least one edge src -> dst
        with step in (t_min, t_max].
        Used for cheap cycle-close feasibility checks.
        """
        edges = self.out_edges.get(src)
        if not edges:
            return False
        left  = bisect_right(edges, t_min, key=lambda e: e.step)
        right = bisect_right(edges, t_max, key=lambda e: e.step)
        if right <= left:
            return False
        for i in range(left, right):
            if edges[i].dst == dst:
                return True
        return False

    def has_any_incoming(
        self,
        dst: int,
        t_min: int,
        t_max: int,
    ) -> bool:
        """
        Return True if there is at least one incoming edge to dst
        with step in (t_min, t_max].
        Used for cheap feasibility checks in lookback queries.
        """
        edges = self.in_edges.get(dst)
        if not edges:
            return False
        left  = bisect_right(edges, t_min, key=lambda e: e.step)
        right = bisect_right(edges, t_max, key=lambda e: e.step)
        return right > left



def build_temporal_index_from_polars(df_window_edges: pl.DataFrame) -> TemporalIndex:
    required_cols = ["edge_id", "src", "dst", "step", "amount", "is_sar"]
    missing_cols = [c for c in required_cols if c not in df_window_edges.columns]
    if missing_cols:
        raise ValueError(f"Missing columns for TemporalIndex: {missing_cols}")
    # Tối ưu hoá tốc độ: chuyển đổi sang lists và zip thay vì dùng to_dicts() chậm
    edge_ids = df_window_edges["edge_id"].to_list()
    srcs = df_window_edges["src"].to_list()
    dsts = df_window_edges["dst"].to_list()
    steps = df_window_edges["step"].to_list()
    amounts = df_window_edges["amount"].to_list()
    is_sars = df_window_edges["is_sar"].to_list()
    edge_records = [
        EdgeRecord(
            edge_id=eid,
            src=s,
            dst=d,
            step=t,
            amount=a,
            is_sar=sar,
        )
        for eid, s, d, t, a, sar in zip(edge_ids, srcs, dsts, steps, amounts, is_sars)
    ]
    edge_records.sort(key=lambda e: (e.step, e.edge_id))
    return TemporalIndex(edge_records)


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
# Cell 7b: Common validation, scoring, and candidate selection helpers
# ============================================================

import json
import math
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

def has_unique_edge_ids(edges: List[EdgeRecord]) -> bool:
    edge_ids = [e.edge_id for e in edges]
    return len(edge_ids) == len(set(edge_ids))


def has_distinct_nodes(node_values: List[int]) -> bool:
    return len(node_values) == len(set(node_values))


def is_within_total_duration(edges: List[EdgeRecord], max_duration: int) -> bool:
    if not edges:
        return True
    steps = [e.step for e in edges]
    return (max(steps) - min(steps)) <= max_duration


def passes_consecutive_step_gap(edges: List[EdgeRecord], max_gap: Optional[int]) -> bool:
    if max_gap is None:
        return True
    if len(edges) <= 1:
        return True
    edges_sorted = sorted(edges, key=lambda e: (e.step, e.edge_id))
    steps = [int(e.step) for e in edges_sorted]
    for i in range(len(steps) - 1):
        if steps[i + 1] - steps[i] > max_gap:
            return False
    return True


def phase_gap_ok(
    earlier_edges: List[EdgeRecord],
    later_edges: List[EdgeRecord],
    max_gap: Optional[int],
) -> bool:
    if max_gap is None:
        return True
    if len(earlier_edges) == 0 or len(later_edges) == 0:
        return False
    earlier_end = max(int(e.step) for e in earlier_edges)
    later_start = min(int(e.step) for e in later_edges)
    return 0 < (later_start - earlier_end) <= max_gap


def amount_consistency(edges: List[EdgeRecord]) -> float:
    amounts = [float(e.amount) for e in edges if float(e.amount) > 0]
    if not amounts:
        return 0.0
    max_amount = max(amounts)
    if max_amount <= 0:
        return 0.0
    return float(min(amounts) / max_amount)

# Alias for backward compatibility
compute_amount_consistency = amount_consistency


def flow_ratio(in_edges: List[EdgeRecord], out_edges: List[EdgeRecord]) -> float:
    in_sum = float(sum(e.amount for e in in_edges))
    out_sum = float(sum(e.amount for e in out_edges))
    return out_sum / in_sum if in_sum > 0 else 0.0


def make_canonical_key(
    motif_type: str,
    edge_ids: List[int],
    ordered: bool = True,
) -> str:
    if ordered:
        canonical_edge_ids = [int(eid) for eid in edge_ids]
    else:
        canonical_edge_ids = sorted(int(eid) for eid in edge_ids)

    key_obj = {
        "motif_type": motif_type,
        "edge_ids": canonical_edge_ids,
    }
    return json.dumps(key_obj, sort_keys=True)


# ------------------------------------------------------------
# SUSPICIOUSNESS SCORING FUNCTIONS
# ------------------------------------------------------------

def score_temporal_compactness(edges: List[EdgeRecord], max_duration: int) -> float:
    if len(edges) <= 1:
        return 1.0
    if max_duration <= 0:
        return 0.0
    steps = [e.step for e in edges]
    duration = max(steps) - min(steps)
    return max(0.0, min(1.0, 1.0 - (duration / max_duration)))


def score_amount_consistency(edges: List[EdgeRecord]) -> float:
    return amount_consistency(edges)


def score_amount_scale(edges: List[EdgeRecord], amount_quantiles: Dict[float, float]) -> float:
    if not edges:
        return 0.0
    mean_amount = sum(e.amount for e in edges) / len(edges)
    
    q50 = amount_quantiles.get(0.5, 0.0)
    q90 = amount_quantiles.get(0.9, 0.0)
    q99 = amount_quantiles.get(0.99, 0.0)
    
    if mean_amount >= q99:
        return 1.0
    elif mean_amount >= q90:
        return 0.8 + 0.2 * (mean_amount - q90) / (q99 - q90) if q99 > q90 else 0.8
    elif mean_amount >= q50:
        return 0.5 + 0.3 * (mean_amount - q50) / (q90 - q50) if q90 > q50 else 0.5
    else:
        return 0.5 * mean_amount / q50 if q50 > 0 else 0.0


def score_degree_penalty(
    edges: List[EdgeRecord],
    node_in_degree: Dict[int, int],
    node_out_degree: Dict[int, int]
) -> float:
    nodes = set()
    for e in edges:
        nodes.add(e.src)
        nodes.add(e.dst)
        
    max_deg = 0
    for n in nodes:
        in_d = node_in_degree.get(n, 0)
        out_d = node_out_degree.get(n, 0)
        deg = in_d + out_d
        if deg > max_deg:
            max_deg = deg
            
    if max_deg <= 10:
        return 1.0
    elif max_deg <= 100:
        return 1.0 - 0.5 * (max_deg - 10) / 90
    else:
        return max(0.1, 0.5 * 100 / max_deg)


def score_flow_conservation(in_edges: List[EdgeRecord], out_edges: List[EdgeRecord]) -> float:
    ratio = flow_ratio(in_edges, out_edges)
    if ratio <= 0:
        return 0.0
    return math.exp(-((ratio - 1.0) ** 2) / 0.5)


def score_instance(edges: List[EdgeRecord], index: TemporalIndex, pattern: MotifPattern) -> float:
    if pattern.matcher_type == "split_merge":
        n = len(edges) // 2
        split_edges = edges[:n]
        merge_edges = edges[n:]
        
        split_comp = score_temporal_compactness(split_edges, pattern.max_duration)
        merge_comp = score_temporal_compactness(merge_edges, pattern.max_duration)
        
        in_end = max(e.step for e in split_edges)
        out_start = min(e.step for e in merge_edges)
        gap = max(0, out_start - in_end)
        gap_score = math.exp(-gap / 3.0)
        
        flow_score = score_flow_conservation(split_edges, merge_edges)
        consistency = score_amount_consistency(edges)
        penalty = score_degree_penalty(edges, index.node_in_degree, index.node_out_degree)
        
        overall_score = (split_comp * 0.15) + (merge_comp * 0.15) + (gap_score * 0.15) + (flow_score * 0.25) + (consistency * 0.20) + (penalty * 0.10)
        return float(overall_score)
        
    elif pattern.matcher_type == "center_in_out":
        n_in = sum(1 for e in pattern.edges if e.dst == "center")
        in_edges = edges[:n_in]
        out_edges = edges[n_in:]
        
        in_comp = score_temporal_compactness(in_edges, pattern.max_duration)
        out_comp = score_temporal_compactness(out_edges, pattern.max_duration)
        
        in_end = max(e.step for e in in_edges)
        out_start = min(e.step for e in out_edges)
        gap = max(0, out_start - in_end)
        gap_score = math.exp(-gap / 3.0)
        
        flow_score = score_flow_conservation(in_edges, out_edges)
        consistency = score_amount_consistency(edges)
        penalty = score_degree_penalty(edges, index.node_in_degree, index.node_out_degree)
        
        overall_score = (in_comp * 0.15) + (out_comp * 0.15) + (gap_score * 0.15) + (flow_score * 0.25) + (consistency * 0.20) + (penalty * 0.10)
        return float(overall_score)
        
    else:
        compactness = score_temporal_compactness(edges, pattern.max_duration)
        consistency = score_amount_consistency(edges)
        scale = score_amount_scale(edges, index.amount_quantiles)
        penalty = score_degree_penalty(edges, index.node_in_degree, index.node_out_degree)
        
        overall_score = (compactness * 0.25) + (consistency * 0.25) + (scale * 0.25) + (penalty * 0.25)
        return float(overall_score)


# ------------------------------------------------------------
# RANKING-BASED CANDIDATE SELECTION
# ------------------------------------------------------------

def cap_edges(
    edges: List[EdgeRecord],
    max_candidates: int,
    policy: str = "hybrid",
    anchor_edge: Optional[EdgeRecord] = None,
    amount_quantiles: Optional[Dict[float, float]] = None,
) -> List[EdgeRecord]:
    """
    Candidate cap helper implementing multiple policies.
    Policies:
        - earliest: earliest by step
        - top_amount: highest transaction amount
        - temporally_dense: closest in time to anchor_edge
        - amount_coherent: closest in amount to anchor_edge (or index median)
        - hybrid: a combination of earliest, top_amount, temporally_dense, and amount_coherent
    
    Important: is_sar is never used in candidate selection.
    """
    if max_candidates is None or len(edges) <= max_candidates:
        return sorted(edges, key=lambda e: (e.step, e.edge_id))

    if policy == "earliest":
        selected = sorted(edges, key=lambda e: (e.step, e.edge_id))[:max_candidates]
        return sorted(selected, key=lambda e: (e.step, e.edge_id))

    if policy == "top_amount":
        selected = sorted(edges, key=lambda e: (-e.amount, e.step, e.edge_id))[:max_candidates]
        return sorted(selected, key=lambda e: (e.step, e.edge_id))

    if policy == "temporally_dense" and anchor_edge is not None:
        selected = sorted(edges, key=lambda e: (abs(e.step - anchor_edge.step), e.step, e.edge_id))[:max_candidates]
        return sorted(selected, key=lambda e: (e.step, e.edge_id))

    if policy == "amount_coherent" and anchor_edge is not None:
        selected = sorted(edges, key=lambda e: (abs(e.amount - anchor_edge.amount), e.step, e.edge_id))[:max_candidates]
        return sorted(selected, key=lambda e: (e.step, e.edge_id))

    if anchor_edge is None:
        k_earliest = max_candidates // 2
        k_amount = max_candidates - k_earliest
        earliest = sorted(edges, key=lambda e: (e.step, e.edge_id))[:k_earliest]
        top_amount = sorted(edges, key=lambda e: (-e.amount, e.step, e.edge_id))[:k_amount]
        selected_dict = {e.edge_id: e for e in earliest + top_amount}
        if len(selected_dict) < max_candidates:
            remaining = sorted(edges, key=lambda e: (e.step, e.edge_id))
            for e in remaining:
                if e.edge_id not in selected_dict:
                    selected_dict[e.edge_id] = e
                    if len(selected_dict) >= max_candidates:
                        break
        return sorted(selected_dict.values(), key=lambda e: (e.step, e.edge_id))[:max_candidates]

    # Fallback to hybrid combination when anchor_edge is not None
    k_earliest = max(1, max_candidates // 4)
    k_amount = max(1, max_candidates // 4)
    k_dense = max(1, max_candidates // 4)
    k_coherent = max(1, max_candidates - (k_earliest + k_amount + k_dense))
    
    earliest = sorted(edges, key=lambda e: (e.step, e.edge_id))[:k_earliest]
    top_amount = sorted(edges, key=lambda e: (-e.amount, e.step, e.edge_id))[:k_amount]
    dense = sorted(edges, key=lambda e: (abs(e.step - anchor_edge.step), e.step, e.edge_id))[:k_dense]
    coherent = sorted(edges, key=lambda e: (abs(e.amount - anchor_edge.amount), e.step, e.edge_id))[:k_coherent]

    selected_dict = {}
    for e in earliest + top_amount + dense + coherent:
        selected_dict[e.edge_id] = e

    if len(selected_dict) < max_candidates:
        remaining = sorted(edges, key=lambda e: (e.step, e.edge_id))
        for e in remaining:
            if e.edge_id not in selected_dict:
                selected_dict[e.edge_id] = e
                if len(selected_dict) >= max_candidates:
                    break

    return sorted(selected_dict.values(), key=lambda e: (e.step, e.edge_id))[:max_candidates]


def select_edges_by_hybrid_policy(
    edges: List[EdgeRecord],
    max_candidates: int,
    early_ratio: float = 0.5,
) -> List[EdgeRecord]:
    return cap_edges(edges, max_candidates, "hybrid")


print("Cell 7b: Common validation, scoring, and candidate selection helpers ready.")


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
    validate: bool = False,
    index: Optional[TemporalIndex] = None,
    candidate_rank: int = -1,
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

        node_map:
            Mapping from abstract pattern node to real node.

        role_map:
            Mapping from pattern edge role/name to real edge_id.

        anchor_edge_id:
            Edge used as anchor. If None, use the first edge.

        validate:
            Whether to run structural validation.

        index:
            TemporalIndex object (optional, for scoring and degrees).

        candidate_rank:
            Rank of candidate anchor edge (optional).
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

    # Flow ratio, handoff_gap/flow_gap, in_sum, out_sum
    in_sum_val = 0.0
    out_sum_val = 0.0
    flow_ratio_val = 0.0
    flow_gap_val = 0.0

    if pattern.matcher_type in {"split_merge", "center_in_out"}:
        if pattern.matcher_type == "split_merge":
            n = len(edges) // 2
            in_edges = edges[:n]
            out_edges = edges[n:]
        else: # center_in_out
            n_in = sum(1 for e in pattern.edges if e.dst == "center")
            in_edges = edges[:n_in]
            out_edges = edges[n_in:]

        in_sum_val = float(sum(e.amount for e in in_edges))
        out_sum_val = float(sum(e.amount for e in out_edges))
        flow_ratio_val = out_sum_val / in_sum_val if in_sum_val > 0 else 0.0

        in_end = max(e.step for e in in_edges)
        out_start = min(e.step for e in out_edges)
        flow_gap_val = float(out_start - in_end)

    # Center degree
    center_degree_val = 0
    if index is not None:
        if pattern.matcher_type == "fan_in":
            center_node = node_map.get("dst")
            if center_node is not None:
                center_degree_val = index.node_in_degree.get(center_node, 0) + index.node_out_degree.get(center_node, 0)
        elif pattern.matcher_type == "fan_out":
            center_node = node_map.get("src")
            if center_node is not None:
                center_degree_val = index.node_in_degree.get(center_node, 0) + index.node_out_degree.get(center_node, 0)
        elif pattern.matcher_type == "center_in_out":
            center_node = node_map.get("center")
            if center_node is not None:
                center_degree_val = index.node_in_degree.get(center_node, 0) + index.node_out_degree.get(center_node, 0)
        else:
            center_degree_val = max(index.node_in_degree.get(n, 0) + index.node_out_degree.get(n, 0) for n in node_map.values()) if node_map else 0

    # Suspicion score and degree penalty
    if index is not None:
        instance_score_val = score_instance(edges, index, pattern)
        degree_penalty_val = score_degree_penalty(edges, index.node_in_degree, index.node_out_degree)
    else:
        # Fallback calculations without index
        compactness = score_temporal_compactness(edges, pattern.max_duration)
        consistency = amount_consistency(edges)
        overall = (compactness * 0.4) + (consistency * 0.4) + (flow_ratio_val * 0.2 if in_sum_val > 0 else 0.2)
        instance_score_val = float(overall)
        degree_penalty_val = 1.0

    time_compactness_val = score_temporal_compactness(edges, pattern.max_duration)
    amount_consistency_val = amount_consistency(edges)
    amount_sum_log_val = math.log1p(amount_sum) if amount_sum > 0 else 0.0

    # Compatibility features
    amount_coherence_val = float(min(amounts) / max(amounts)) if amounts else 1.0
    time_span_val = int(max(steps) - min(steps)) if steps else 0

    dst_in_degree_window_val = 0
    src_out_degree_window_val = 0
    if index is not None:
        if pattern.matcher_type == "fan_in":
            center_node = node_map.get("dst")
            if center_node is not None:
                dst_in_degree_window_val = index.node_in_degree.get(center_node, 0)
        elif pattern.matcher_type == "fan_out":
            center_node = node_map.get("src")
            if center_node is not None:
                src_out_degree_window_val = index.node_out_degree.get(center_node, 0)

    row = {
        "motif_instance_id": motif_instance_id,
        "motif_type": pattern.name,
        "matcher_type": pattern.matcher_type,
        "window_id": int(window_id),
        "anchor_edge_id": int(anchor_edge_id),

        "edge_ids": edge_ids,
        "node_ids": node_ids,

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

        # New ranking fields
        "instance_score": float(instance_score_val),
        "time_compactness": float(time_compactness_val),
        "amount_consistency": float(amount_consistency_val),
        "amount_sum_log": float(amount_sum_log_val),
        "degree_penalty": float(degree_penalty_val),
        "flow_ratio": float(flow_ratio_val),
        "flow_gap": float(flow_gap_val),
        "center_degree": int(center_degree_val),
        "candidate_rank": int(candidate_rank),

        # Compatibility fields
        "in_sum": float(in_sum_val),
        "out_sum": float(out_sum_val),
        "handoff_gap": float(flow_gap_val),
        "amount_coherence": float(amount_coherence_val),
        "time_span": int(time_span_val),
        "dst_in_degree_window": int(dst_in_degree_window_val),
        "src_out_degree_window": int(src_out_degree_window_val),
    }

    return row


def make_edge_motif_membership_rows(
    motif_row: Dict[str, Any],
    pattern: MotifPattern,
    edges: List[EdgeRecord],
) -> List[Dict[str, Any]]:
    """
    Create edge_motif_membership rows from one motif instance.
    """

    if len(edges) != len(pattern.edges):
        raise ValueError(
            f"Pattern {pattern.name} expects {len(pattern.edges)} edges, "
            f"but got {len(edges)} edges."
        )

    pattern_edges_sorted = sorted(pattern.edges, key=lambda x: x.order)

    # Reconstruct edges in the correct role order using role_map_json from motif_row if available
    ordered_edges = edges
    role_map_str = motif_row.get("role_map_json")
    if role_map_str:
        try:
            role_map = json.loads(role_map_str)
            edge_by_id = {e.edge_id: e for e in edges}
            ordered_edges = [edge_by_id[int(role_map[p_e.role])] for p_e in pattern_edges_sorted]
        except Exception:
            ordered_edges = edges

    rows = []

    for p_edge, real_edge in zip(pattern_edges_sorted, ordered_edges):
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
    schema = {
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
        
        # New ranking fields
        "instance_score": pl.Float64,
        "time_compactness": pl.Float64,
        "amount_consistency": pl.Float64,
        "amount_sum_log": pl.Float64,
        "degree_penalty": pl.Float64,
        "flow_ratio": pl.Float64,
        "flow_gap": pl.Float64,
        "center_degree": pl.Int64,
        "candidate_rank": pl.Int64,

        # Compatibility fields
        "in_sum": pl.Float64,
        "out_sum": pl.Float64,
        "handoff_gap": pl.Float64,
        "amount_coherence": pl.Float64,
        "time_span": pl.Int64,
        "dst_in_degree_window": pl.Int64,
        "src_out_degree_window": pl.Int64,
    }
    if len(rows) == 0:
        return pl.DataFrame(schema=schema)

    df = pl.DataFrame(rows)

    select_exprs = []
    for col_name, col_type in schema.items():
        if col_name in df.columns:
            select_exprs.append(pl.col(col_name).cast(col_type))
        else:
            select_exprs.append(pl.lit(None, dtype=col_type).alias(col_name))

    return df.select(select_exprs)


def membership_rows_to_polars(rows: List[Dict[str, Any]]) -> pl.DataFrame:
    """
    Convert edge membership rows to Polars DataFrame with stable schema.
    """
    schema = {
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
    if len(rows) == 0:
        return pl.DataFrame(schema=schema)

    df = pl.DataFrame(rows)

    select_exprs = []
    for col_name, col_type in schema.items():
        if col_name in df.columns:
            select_exprs.append(pl.col(col_name).cast(col_type))
        else:
            select_exprs.append(pl.lit(None, dtype=col_type).alias(col_name))

    return df.select(select_exprs)


def make_membership_df_from_motif_rows(
    motif_rows: List[Dict[str, Any]],
    pattern: MotifPattern,
    emitted_edges: List[List[EdgeRecord]],
) -> pl.DataFrame:
    """
    Build membership DataFrame from motif_rows and their corresponding EdgeRecords
    in a memory-efficient chunked manner to avoid RAM overflow.
    """
    if not motif_rows:
        return membership_rows_to_polars([])

    chunk_size = 10000
    membership_dfs = []
    current_chunk = []

    for motif_row, edges_val in zip(motif_rows, emitted_edges):
        current_chunk.extend(make_edge_motif_membership_rows(motif_row, pattern, edges_val))
        if len(current_chunk) >= chunk_size:
            membership_dfs.append(membership_rows_to_polars(current_chunk))
            current_chunk = []

    if current_chunk:
        membership_dfs.append(membership_rows_to_polars(current_chunk))

    return pl.concat(membership_dfs)


def write_motif_outputs(
    motif_rows: List[Dict[str, Any]],
    membership_rows: Any,
    window_id: int,
    motif_type: str,
    motif_instance_dir: str = MOTIF_INSTANCE_DIR,
    membership_dir: str = MEMBERSHIP_DIR,
) -> Tuple[str, str]:
    """
    Write motif_instances and edge_motif_membership parquet shards.
    """

    motif_df = motif_instance_rows_to_polars(motif_rows)
    if isinstance(membership_rows, pl.DataFrame):
        membership_df = membership_rows
    else:
        membership_df = membership_rows_to_polars(membership_rows)

    motif_path = f"{motif_instance_dir}/window_{int(window_id):06d}_{motif_type}.parquet"
    membership_path = f"{membership_dir}/window_{int(window_id):06d}_{motif_type}.parquet"

    motif_df.write_parquet(motif_path)
    membership_df.write_parquet(membership_path)

    return motif_path, membership_path


# End of code.md