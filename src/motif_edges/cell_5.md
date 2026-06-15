



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

# XÓA BỎ BÍ DANH fan_out_4 ĐỂ TRÁNH LỖI LOGIC
fan_in_4   = fan_in_patterns[0]    # n=3 branches, 4 total nodes
cycle_5    = cycle_patterns[0]
split_merge_6 = split_merge_patterns[0]   # 3 branch pairs, 5 nodes, 6 edges
split_merge_8 = split_merge_patterns[1]   # 4 branch pairs, 6 nodes, 8 edges
split_merge_10 = split_merge_patterns[2]  # 5 branch pairs, 7 nodes, 10 edges
split_merge_5 = split_merge_6              # backward compatibility alias
fanin_fanout_6 = center_inout_patterns[0] # (3 in, 2 out)

# 1. Core Patterns (Giữ nguyên các motif khác, nhưng fan_out nhỏ đã biến mất)
CORE_PATTERNS = (
    [p for p in fan_in_patterns if len(p.edges) <= 4]        
    + [p for p in split_merge_patterns if len(p.edges) // 2 == 3]  
    + [p for p in center_inout_patterns if p.name == "center_inout_3in_2out"]
)

# 2. Exploration Patterns (Bao gồm các motif khác cỡ lớn)
EXPLORATION_PATTERNS = (
    [p for p in fan_in_patterns if len(p.edges) > 4]        
    + [p for p in split_merge_patterns if len(p.edges) // 2 > 3]  
    + [p for p in center_inout_patterns if p.name != "center_inout_3in_2out"]
)

DIAGNOSTIC_PATTERNS = cycle_patterns
ALL_PATTERNS = CORE_PATTERNS + EXPLORATION_PATTERNS + DIAGNOSTIC_PATTERNS + fan_out_patterns

# 3. ĐIỂM QUYẾT ĐỊNH: Đưa họ fan_out mới vào bộ quét thực tế
# Nếu không cộng thêm fan_out_patterns vào đây, hệ thống sẽ bỏ qua toàn bộ fan_out.
ACTIVE_PATTERNS = CORE_PATTERNS + fan_out_patterns 

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




