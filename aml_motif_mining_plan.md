# AML Motif Mining — Detailed Improvement Plan

## Framing

The pipeline is a **pure structural pattern miner**. Its job is:

> Given a directed temporal transaction graph, find every instance of every defined
> motif pattern (fan-in, fan-out, cycle, split-merge, center-in-out, stacked bipartite)
> that satisfies structural, temporal, and amount constraints.

The `isSAR` column plays exactly one role: **post-hoc evaluation**. After mining completes,
it is used to check what fraction of SAR-labelled edges appear in at least one matched
motif instance. That fraction is the *structural recall* of the miner. Nothing in the
matching logic reads or filters by `isSAR`. The label does not exist at match time.

This document defines what needs to change and why, in four phases.

---

## Problem diagnosis

### 1. Fixed motif sizes

Every matcher is hard-coded to a single size. `FanInMatcher` requires exactly 3 incoming
edges. `FanOutMatcher` requires exactly 3 outgoing edges. `SplitMergeMatcher` requires
exactly 3 split + 3 merge edges. The pattern factory `make_cycle_pattern(k)` and
`CycleKMatcher` are already generalized, but the other families are not.

In practice, suspicious transaction clusters appear across a range of sizes. A 4-branch
fan-in is structurally more suspicious than a 3-branch one, but the current code misses it.
The families need to support 3 through 7 branches for fan types, and 3 through 10 nodes
for cycles.

### 2. Combination-first search

`FanInMatcher` and `FanOutMatcher` iterate over every destination (or source) node,
collect all incident edges, and call `combinations(candidates, 3)`. The structural and
temporal filters (`passes_distinct_nodes_for_fanin`, `passes_fanin_duration`, etc.) are
applied *after* the combination is materialized.

This order is wrong for large graphs. `combinations(n, 3)` is O(n³). Most combinations
are rejected by the first filter check. The correct order is to apply the cheapest
filters first, before generating the combination at all.

`SplitMergeMatcher` has the same problem compounded: it generates all split triplets,
then for each triplet calls `product(*lists)` over merge candidates. For a high-degree
source node, this is O(n³ × m³).

`CenterInOutMatcher` generates all inbound triplets, then for each triplet generates all
outbound pairs. That is O(in³ × out²) per center node, even if no valid instance exists.

### 3. DFS pruning is incomplete in `CycleKMatcher`

The cycle DFS is structurally correct, but partial paths are only rejected when:
- a node would be revisited
- total duration is exceeded

Missing guards that are cheap to add at each expansion step:
- **Remaining-hop feasibility**: if the remaining time budget divided by `delta_hop`
  is less than the remaining number of hops, no closing is possible. Reject immediately.
- **Return-to-start reachability**: before expanding depth `d < k-1`, check whether
  the current node has any outgoing edge in the pair index toward `start_node` within
  the remaining time window. If not, the cycle cannot close regardless of future hops.
  For short cycles this is cheap; for `cycle_7..10` it is essential.
- **Canonical rotation duplication**: the current check (`earliest_edge.edge_id !=
  anchor_edge.edge_id`) is correct but is only applied at cycle-close time. It should
  also prune at intermediate DFS steps where the accumulated path already contains an
  edge earlier than the anchor.

### 4. No Stacked Bipartite matcher

`two_stage_split_8` is a fixed-size, fixed-topology approximation of the stacked bipartite
pattern. It was deferred due to complexity. The general form — a layered structure where
one fan-out feeds into one fan-in, repeatedly — is not implemented.

### 5. Evaluation of structural coverage is not systematic

After a full run, there is no aggregate report answering:
- How many edges in the dataset appear in at least one matched instance, by motif type?
- What is the distribution of motif duration, amount, and branch count across instances?
- If `isSAR` labels are available, what fraction of SAR edges are covered by motif
  instances? (Post-hoc only, not used in matching.)

Without this, it is impossible to tell whether a change improved or harmed recall.

---

## Phase 1 — Pattern factory generalization

**Goal**: make every motif family support variable sizes without duplicating matcher code.

### 1.1 Fan-in family: size 3 to 7

Create `make_fan_in_pattern(n)`:

```python
def make_fan_in_pattern(n: int) -> MotifPattern:
    """
    fan_in_n: n distinct source nodes all transferring to one destination.
    n = number of incoming edges = number of source nodes.
    Minimum n = 3.
    """
    if n < 3:
        raise ValueError("fan_in requires n >= 3")
    nodes = [f"src_{i}" for i in range(1, n + 1)] + ["dst"]
    edges = [
        PatternEdge(
            name=f"p{i}",
            src=f"src_{i}",
            dst="dst",
            order=i,
            role=f"incoming_{i}",
        )
        for i in range(1, n + 1)
    ]
    return MotifPattern(
        name=f"fan_in_{n + 1}",   # n+1 = total nodes
        nodes=nodes,
        edges=edges,
        matcher_type="fan_in",
        max_duration=MAX_MOTIF_DURATION,
        time_order="nondecreasing",
        distinct_nodes=True,
        description=f"Fan-in with {n} distinct source nodes to one destination.",
    )

FAN_IN_SIZES = [3, 4, 5, 6, 7]
fan_in_patterns = [make_fan_in_pattern(n) for n in FAN_IN_SIZES]
```

Corresponding candidate cap schedule (combination count grows as C(cap, n)):

| n branches | max_in_candidates | max C(cap, n) |
|---|---|---|
| 3 | 100 | 161,700 |
| 4 | 50  | 230,300 |
| 5 | 25  | 53,130  |
| 6 | 15  | 5,005   |
| 7 | 10  | 120     |

The cap schedule ensures that no single destination node generates more than ~250k
combinations regardless of its true in-degree. This is a conservative default; tighter
caps should be applied if the dataset has very high-degree hub nodes.

### 1.2 Fan-out family: size 3 to 7

Identical factory logic, swapping src/dst roles:

```python
def make_fan_out_pattern(n: int) -> MotifPattern:
    """
    fan_out_n: one source node transferring to n distinct destination nodes.
    """
    nodes = ["src"] + [f"dst_{i}" for i in range(1, n + 1)]
    edges = [
        PatternEdge(
            name=f"p{i}",
            src="src",
            dst=f"dst_{i}",
            order=i,
            role=f"outgoing_{i}",
        )
        for i in range(1, n + 1)
    ]
    return MotifPattern(
        name=f"fan_out_{n + 1}",
        nodes=nodes,
        edges=edges,
        matcher_type="fan_out",
        max_duration=MAX_MOTIF_DURATION,
        time_order="nondecreasing",
        distinct_nodes=True,
        description=f"Fan-out from one source to {n} distinct destination nodes.",
    )
```

### 1.3 Cycle family: size 5 to 10

The factory `make_cycle_pattern(k)` already exists in Cell 5. No factory change is
needed. What changes is the matcher (Phase 2) and the active size range:

```python
CYCLE_SIZES = list(range(5, 11))   # 5, 6, 7, 8, 9, 10
cycle_patterns = [make_cycle_pattern(k) for k in CYCLE_SIZES]
```

`cycle_11` and `cycle_12` remain defined in `ALL_PATTERNS` for completeness but are
excluded from `ACTIVE_PATTERNS` because their runtime on realistic windows is not
yet validated.

### 1.4 Split-merge family: 3, 4, 5 branch pairs

```python
def make_split_merge_pattern(n: int) -> MotifPattern:
    """
    split_merge_n: one source splits to n intermediates,
    which all merge into one sink.
    Total nodes = n + 2 (source, n intermediates, sink).
    Total edges = 2n.
    """
    if n < 3:
        raise ValueError("split_merge requires n >= 3")
    nodes = ["src"] + [f"mid_{i}" for i in range(1, n + 1)] + ["sink"]
    split_edges = [
        PatternEdge(f"split_{i}", "src", f"mid_{i}", i, role=f"split_{i}")
        for i in range(1, n + 1)
    ]
    merge_edges = [
        PatternEdge(f"merge_{i}", f"mid_{i}", "sink", n + i, role=f"merge_{i}")
        for i in range(1, n + 1)
    ]
    return MotifPattern(
        name=f"split_merge_{n * 2}",   # 2n total edges
        nodes=nodes,
        edges=split_edges + merge_edges,
        matcher_type="split_merge",
        max_duration=MAX_MOTIF_DURATION,
        time_order="nondecreasing",
        distinct_nodes=True,
        amount_ratio_min=AMOUNT_RATIO_MIN,
        amount_ratio_max=AMOUNT_RATIO_MAX,
        description=f"One source splits to {n} intermediates that merge into one sink.",
    )

SPLIT_MERGE_SIZES = [3, 4, 5]
split_merge_patterns = [make_split_merge_pattern(n) for n in SPLIT_MERGE_SIZES]
```

### 1.5 Center-in-out family: scaling in and out arms

```python
def make_center_inout_pattern(n_in: int, n_out: int) -> MotifPattern:
    """
    center_in_out: n_in sources feed into one center node,
    which then distributes to n_out destinations.
    All source nodes, center, and destination nodes must be distinct.
    """
    if n_in < 2 or n_out < 2:
        raise ValueError("center_in_out requires n_in >= 2 and n_out >= 2")
    nodes = (
        [f"in_{i}" for i in range(1, n_in + 1)]
        + ["center"]
        + [f"out_{i}" for i in range(1, n_out + 1)]
    )
    in_edges = [
        PatternEdge(f"in_{i}", f"in_{i}", "center", i, role=f"incoming_{i}")
        for i in range(1, n_in + 1)
    ]
    out_edges = [
        PatternEdge(f"out_{i}", "center", f"out_{i}", n_in + i, role=f"outgoing_{i}")
        for i in range(1, n_out + 1)
    ]
    return MotifPattern(
        name=f"center_inout_{n_in}in_{n_out}out",
        nodes=nodes,
        edges=in_edges + out_edges,
        matcher_type="center_in_out",
        max_duration=MAX_MOTIF_DURATION,
        time_order="nondecreasing",
        distinct_nodes=True,
        amount_ratio_min=AMOUNT_RATIO_MIN,
        amount_ratio_max=AMOUNT_RATIO_MAX,
        description=f"Center node receives from {n_in} and distributes to {n_out}.",
    )

CENTER_INOUT_CONFIGS = [(3, 2), (3, 3), (4, 2), (4, 3), (5, 3)]
center_inout_patterns = [make_center_inout_pattern(n_in, n_out)
                         for n_in, n_out in CENTER_INOUT_CONFIGS]
```

### 1.6 Stacked Bipartite family (new)

This is a chain of alternating fan-out and fan-in layers. The simplest
instance is a 2-layer structure: one node fans out to k intermediates,
which then fan in to one sink. This is equivalent to `split_merge`. The
general form adds more layers.

```python
def make_stacked_bipartite_pattern(
    layer_sizes: list[int],
) -> MotifPattern:
    """
    Stacked bipartite chain.

    layer_sizes defines the width of each intermediate layer.
    Example: layer_sizes=[3, 2] means:
        - 1 source fans out to 3 intermediates (layer 1)
        - 3 intermediates fan in to 2 intermediates (layer 2)
        - 2 intermediates fan in to 1 sink

    This generalizes split_merge (layer_sizes=[n]) and
    two_stage_split (layer_sizes=[n, m]).
    """
    if len(layer_sizes) < 1:
        raise ValueError("layer_sizes must have at least 1 element")

    # Node naming: layer_0 = [src], layer_1..n = intermediates, layer_n+1 = [sink]
    all_layers = [[f"L0_0"]] + [[f"L{i+1}_{j}" for j in range(w)]
                                 for i, w in enumerate(layer_sizes)] + [[f"Lsink_0"]]

    nodes = [n for layer in all_layers for n in layer]

    edges = []
    order = 1
    for l_idx in range(len(all_layers) - 1):
        src_layer = all_layers[l_idx]
        dst_layer = all_layers[l_idx + 1]
        for src_node in src_layer:
            for dst_node in dst_layer:
                edges.append(PatternEdge(
                    name=f"e{order}",
                    src=src_node,
                    dst=dst_node,
                    order=order,
                    role=f"layer{l_idx}_to_{l_idx+1}_{src_node}_{dst_node}",
                ))
                order += 1

    size_str = "_".join(str(s) for s in layer_sizes)
    return MotifPattern(
        name=f"stacked_bipartite_{size_str}",
        nodes=nodes,
        edges=edges,
        matcher_type="stacked_bipartite",
        max_duration=MAX_MOTIF_DURATION,
        time_order="nondecreasing",
        distinct_nodes=True,
        amount_ratio_min=AMOUNT_RATIO_MIN,
        amount_ratio_max=AMOUNT_RATIO_MAX,
        description=f"Stacked bipartite with layer widths {layer_sizes}.",
    )
```

Start with small configs only: `[2]`, `[3]`, `[2, 2]`, `[3, 2]`. Do not
run `[3, 3]` or wider until single-layer configs are benchmarked.

---

## Phase 2 — Matcher redesign

### 2.1 Generalize `FanInMatcher` and `FanOutMatcher` to variable n

Change the size validation from a hard equality check to reading the pattern:

```python
# In FanInMatcher.match():
n_branches = len(pattern.edges)   # was hard-coded to == 3
# In combinations call:
for edge_combo in combinations(candidates, n_branches):
    ...
```

Add the cap schedule lookup:

```python
FAN_IN_CAP_SCHEDULE = {3: 100, 4: 50, 5: 25, 6: 15, 7: 10}

def get_fan_in_cap(n_branches: int) -> int:
    return FAN_IN_CAP_SCHEDULE.get(n_branches, 10)
```

The structural check `passes_distinct_nodes_for_fanin` already generalizes to any
list of edges — no change needed there.

### 2.2 Constraint-first ordering in `FanInMatcher` and `FanOutMatcher`

The current combination loop applies filters after generation. Reorder:

**Current (wrong) order**:
1. generate combination
2. canonicalize
3. check duration
4. check delta_hop
5. check distinct nodes
6. check amount
7. check anchor in primary window

**Correct order** (cheapest to most expensive):
1. check duration (just `max(steps) - min(steps)`, O(n))  
2. check delta_hop on sorted steps (O(n))
3. check distinct nodes (set construction, O(n))
4. check amount floor (O(n))
5. check anchor in primary window (O(1))
6. build node_map and role_map (O(n))
7. call `make_motif_instance_row` with `validate=False` (avoid re-checking)

Move `validate=False` once all pre-checks above pass — the validation inside
`make_motif_instance_row` is redundant at that point and wastes time.

### 2.3 Pre-filter candidate domain before combinations

Before calling `combinations(candidates, n)`, apply two fast filters to the candidate
list itself:

**Time-window filter**: for fan-in, all candidates must have
`step >= window.primary_start` and
`step <= window.primary_end + MAX_MOTIF_DURATION`. This is already satisfied by
using `df_extended`, but the candidate selection should explicitly enforce
`step <= first_candidate.step + MAX_MOTIF_DURATION`. Sort candidates by step,
then for each candidate at position i, binary-search to find the furthest j such
that `candidates[j].step - candidates[i].step <= MAX_MOTIF_DURATION`. Only
generate combinations within that window.

**Amount floor filter**: if `AMOUNT_MIN` is set, filter the candidate list once
before entering the combinations loop. This is O(n) and avoids re-checking the
floor on every combination.

### 2.4 Incremental DFS pruning in `CycleKMatcher`

Add three new guards inside the `while stack` loop, before expanding forward:

**Guard A — remaining-hop time feasibility**:
```python
remaining_hops = k - len(path_edges)
time_used = current_edge.step - path_edges[0].step
time_budget = pattern.max_duration - time_used

if delta_hop is not None:
    # Even if every remaining hop takes exactly 1 step,
    # we need at least remaining_hops steps left.
    if time_budget < remaining_hops:
        # Cannot fit remaining hops in budget; prune this path.
        continue
```

**Guard B — return-to-start reachability check** (for `len(path_edges) < k - 1`):
```python
# Check whether start_node is reachable from current_node
# within the remaining time budget, without spending all hops now.
# A sufficient condition: start_node appears in index.in_edges[start_node]
# with step in (current_edge.step, current_edge.step + remaining_hops * delta_hop].
# Use a fast has-any-incoming check via the temporal index.
reachable = len(index.pair(
    src=current_node,
    dst=start_node,
    t_min=current_edge.step,
    t_max=current_edge.step + (remaining_hops - 1) * (delta_hop or MAX_MOTIF_DURATION),
    include_left=False,
    max_candidates=1,   # only need to know if at least one exists
)) > 0
# If not reachable even in 1 hop, prune only at depth k-2 or later.
# Applying this at all depths may be too conservative for long paths.
# Apply conservatively: only at len(path_edges) >= k - 2.
if len(path_edges) >= k - 2 and not reachable:
    continue
```

**Guard C — early canonical rotation rejection**:
```python
# If any edge in the current partial path has a lower (step, edge_id)
# than the anchor edge, this path cannot produce a canonical instance.
# Prune immediately.
anchor_key = (anchor_edge.step, anchor_edge.edge_id)
if any((e.step, e.edge_id) < anchor_key for e in path_edges[1:]):
    continue
```

Guard C is the highest-value addition for long cycles: it eliminates entire DFS
subtrees as soon as a non-canonical edge is added, rather than waiting until
the cycle is fully closed.

### 2.5 Bidirectional search for `cycle_7` through `cycle_10`

For `k >= 7`, the DFS tree depth is large enough that bidirectional search reduces
the state space from O(b^k) to O(b^(k/2)) where b is the average branching factor.

**Architecture**:

```
forward frontier:  paths of length floor(k/2) starting from anchor_edge
backward frontier: paths of length ceil(k/2) ending at anchor_edge.src
                   (i.e., reverse edges from anchor.src backward)
```

A forward frontier state is a tuple `(endpoint, min_step, max_step, node_frozenset,
edge_frozenset)`. A backward frontier state is the same but built by traversing
incoming edges in reverse time order.

**Join condition**: a forward state `F` and backward state `B` can be joined when:
- `F.endpoint == B.endpoint` (they share a meeting node)
- `F.max_step < B.min_step` (forward path ends before backward path begins)
- `F.max_step + delta_hop >= B.min_step` (temporal continuity)
- `F.node_frozenset & B.node_frozenset == {anchor.src}` (only shared node is
  the start node, which closes the cycle)
- `F.edge_frozenset & B.edge_frozenset == set()` (no edge reuse)
- total duration `B.max_step - F.min_step <= MAX_MOTIF_DURATION`

Index the forward frontier by `endpoint` for O(1) join lookup.

**Implementation plan**:

```python
class BidirectionalCycleSearch:
    """
    Used only for k >= 7. For k <= 6, CycleKMatcher DFS is used unchanged.
    """
    def search(self, anchor_edge, k, index, pattern) -> List[List[EdgeRecord]]:
        half_fwd = k // 2
        half_bwd = k - half_fwd

        fwd_states = self._build_forward_states(anchor_edge, half_fwd, index, pattern)
        bwd_states = self._build_backward_states(anchor_edge, half_bwd, index, pattern)

        fwd_by_endpoint = defaultdict(list)
        for state in fwd_states:
            fwd_by_endpoint[state.endpoint].append(state)

        results = []
        for bwd_state in bwd_states:
            for fwd_state in fwd_by_endpoint.get(bwd_state.endpoint, []):
                if self._join_valid(fwd_state, bwd_state):
                    cycle_edges = fwd_state.edges + bwd_state.edges_reversed
                    results.append(cycle_edges)

        return results
```

This keeps `CycleKMatcher` as the dispatch point; it calls `BidirectionalCycleSearch`
when `k >= BIDIRECTIONAL_THRESHOLD` (recommended: 7) and falls back to the existing
DFS for smaller cycles.

### 2.6 Sink-intersection-first for `SplitMergeMatcher`

Replace the current `product(*lists)` approach with a sink-intersection step:

```python
# Current (wrong): for each split triplet, for each intermediate,
# get merge candidates, then product(*lists)

# Correct:
# 1. For each intermediate in the split triplet, build a set of reachable sinks.
# 2. Intersect those sets.
# 3. Only if intersection is non-empty, materialize edge assignments.

def get_reachable_sinks(intermediate, split_end_step, max_end_step, index, amount_min):
    cands = index.outgoing(
        src=intermediate,
        t_min=split_end_step,
        t_max=max_end_step,
        include_left=False,
    )
    if amount_min is not None:
        cands = [e for e in cands if e.amount >= amount_min]
    return {e.dst: e for e in cands}   # dst -> best EdgeRecord

# For each split triplet:
sink_dicts = [get_reachable_sinks(mid, ...) for mid in intermediates]

# Intersect by key (sink node id):
common_sinks = set(sink_dicts[0].keys())
for d in sink_dicts[1:]:
    common_sinks &= d.keys()

# Only enter product loop for real common sinks:
for sink in common_sinks:
    if sink in [source] + intermediates:
        continue
    edges_for_sink = [sink_dicts[i][sink] for i in range(len(intermediates))]
    # One merge edge per intermediate -> sink is already determined.
    # No product() needed when each intermediate has one best edge per sink.
```

When each intermediate has multiple edges toward the same sink, keep only the top 2–3
by amount or recency before entering any product. This caps the merge product at
`max_merge_per_pair^n` rather than the unrestricted `total_merge_candidates^n`.

### 2.7 Generalize `SplitMergeMatcher` and `CenterInOutMatcher` to variable n

**`SplitMergeMatcher`**: read branch count from pattern:
```python
n_branches = len(pattern.edges) // 2   # half are split, half are merge
for split_combo in combinations(split_candidates, n_branches):
    ...
```

Add a cap schedule:

```python
SPLIT_MERGE_OUT_CAP = {3: 12, 4: 8, 5: 6}
SPLIT_MERGE_MERGE_CAP = {3: 3, 4: 2, 5: 2}
```

**`CenterInOutMatcher`**: read arm counts from pattern:
```python
n_in  = sum(1 for e in pattern.edges if e.dst == "center")
n_out = sum(1 for e in pattern.edges if e.src == "center")
for in_combo  in combinations(incoming, n_in):
    for out_combo in combinations(outgoing, n_out):
        ...
```

For `center_in_out`, apply the selectivity-first rule: before generating any
combination, check whether the center node has at least `n_in` valid incoming
and `n_out` valid outgoing edges in the time window. Skip the center immediately
if either condition fails. This is an O(1) degree check that avoids entering
the combination loop for disqualified centers.

### 2.8 `StackedBipartiteMatcher` (new)

The stacked bipartite pattern is a sequence of bipartite layers. Each layer is
itself a fan-out (1→k) or fan-in (k→1) or full bipartite (m→n).

**Strategy**: phase-by-phase expansion, not full product.

```python
class StackedBipartiteMatcher:
    """
    Match stacked bipartite motifs layer by layer.

    For each anchor edge in the first layer:
        1. Find all nodes in layer 1 that were reached.
        2. For each such node, find all outgoing edges to layer 2.
        3. Continue until all layers are filled.
        4. Validate structural and temporal constraints at each step.
        5. Emit only when the final layer is complete.
    """

    def match(self, df_primary, df_extended, index, window, pattern, write_output=False):
        # Parse layer structure from pattern definition.
        layers = self._parse_layers(pattern)

        motif_rows = []
        membership_rows = []

        # Anchor = first edge in layer 0 -> layer 1 transition.
        for anchor_edge in self._get_primary_edges(df_primary):
            if not is_edge_in_primary_window(anchor_edge, window):
                continue

            partial_results = self._expand_layers(
                anchor_edge=anchor_edge,
                layers=layers,
                index=index,
                pattern=pattern,
            )

            for edge_assignment, node_map in partial_results:
                motif_row = make_motif_instance_row(
                    window_id=window.window_id,
                    pattern=pattern,
                    edges=edge_assignment,
                    node_map=node_map,
                    role_map=self._build_role_map(pattern, edge_assignment),
                    anchor_edge_id=anchor_edge.edge_id,
                    validate=True,
                )
                membership = make_edge_motif_membership_rows(motif_row, pattern,
                                                             edge_assignment)
                motif_rows.append(motif_row)
                membership_rows.extend(membership)

                if len(motif_rows) >= self.max_instances_per_window:
                    break

        ...

    def _expand_layers(self, anchor_edge, layers, index, pattern):
        """
        Recursive layer expansion.

        State: (partial_edge_list, partial_node_map, last_layer_nodes, last_step)
        At each layer transition, query the temporal index for valid edges
        from each node in the current layer to nodes in the next layer.
        """
        ...
```

The layer-by-layer approach avoids materializing the full edge product across all
layers. Each layer's candidate set is constrained by the previous layer's endpoint
nodes and the remaining time budget.

Start with `layer_sizes=[2]` (equivalent to `split_merge_4`) and `layer_sizes=[3]`
(equivalent to `split_merge_6`). Validate output against `SplitMergeMatcher` to
confirm correctness before adding multi-layer configs.

---

## Phase 3 — Temporal window and index improvements

### 3.1 Extended-window anchor separation

The current design slices `df_primary` and `df_extended` from the same call to
`slice_window_edges`. The extended window includes the lookahead region beyond the
primary window. This is correct for matching, but there is a subtle issue:

When `STRIDE < WINDOW_SIZE`, two adjacent windows share an overlap region. An anchor
edge in that overlap region may be processed twice: once as part of window W and once
as part of window W+1. The `canonical_key`-based deduplication handles this at
output time, but it does not prevent redundant computation.

**Fix**: enforce that anchor edges are selected only from the strict primary window
(`primary_start <= step <= primary_end`), and that this is checked before any DFS
or combination expansion begins. The existing `is_edge_in_primary_window` check
does this — verify it is applied at the outermost loop in all matchers, not only
at the final output step.

### 3.2 Per-window candidate domain pre-computation

Before running any matcher on a window, compute a window-level summary:

```python
@dataclass
class WindowCandidateSummary:
    node_in_degree: Dict[int, int]      # node -> count of incoming edges
    node_out_degree: Dict[int, int]     # node -> count of outgoing edges
    high_in_degree_nodes: List[int]     # nodes with in_degree >= 3
    high_out_degree_nodes: List[int]    # nodes with out_degree >= 3
    bipartite_center_nodes: List[int]   # nodes with in_degree >= 2 and out_degree >= 2
    total_edges: int
```

This summary is computed once per window and passed to all matchers. Matchers use it
to skip nodes that cannot possibly satisfy minimum degree requirements before touching
the temporal index. This eliminates the overhead of looking up empty index entries
for the majority of low-degree nodes.

### 3.3 Temporal index memory profile

For large datasets, the `TemporalIndex` builds three dictionaries (`out_edges`,
`in_edges`, `pair_edges`) over the extended window. When windows are large or
when many windows run in sequence, this can accumulate significant memory.

**Fix**: ensure `del temporal_index` is called immediately after each window's
matchers complete (already in `run_matchers_over_windows`). Additionally, add a
configurable `max_pair_index_nodes` parameter: if the number of distinct (src, dst)
pairs exceeds this threshold, skip building `pair_edges` and use `out_edges` with
a destination filter instead. For cycle closing, the `pair` index is heavily used;
prefer keeping it intact for cycle matchers and disabling it only for fan matchers
that never use it.

---

## Phase 4 — Output aggregation and evaluation

### 4.1 Cross-window deduplication

After all windows complete, merge the per-window motif shards and deduplicate by
`canonical_key`:

```python
def merge_and_deduplicate_motif_shards(
    motif_instance_dir: str,
    output_path: str,
) -> pl.DataFrame:
    shards = [
        pl.read_parquet(p)
        for p in sorted(Path(motif_instance_dir).glob("*.parquet"))
    ]
    df_all = pl.concat(shards)

    # Keep only the instance from the earliest window for each canonical_key.
    df_deduped = (
        df_all
        .sort(["canonical_key", "window_id"])
        .unique(subset=["canonical_key"], keep="first")
    )
    df_deduped.write_parquet(output_path)
    return df_deduped
```

The canonical key already encodes motif type and sorted edge IDs, so this is
exact deduplication. A motif instance found in window 3 and again in window 4
(due to window overlap) will be kept once, attributed to window 3.

### 4.2 Edge participation summary

```python
def build_edge_participation_summary(
    membership_dir: str,
    df_edges: pl.DataFrame,
) -> pl.DataFrame:
    """
    For each edge_id, summarize its motif participation.

    Columns:
        edge_id
        motif_type_counts: JSON string of {motif_type: count}
        num_motif_instances: total distinct instances this edge appears in
        num_motif_types: number of distinct motif types
        roles: list of distinct roles this edge played
        mean_motif_duration: mean duration of motifs this edge is part of
        min_motif_start_step: earliest motif start
        max_motif_end_step: latest motif end
    """
    shards = [pl.read_parquet(p) for p in Path(membership_dir).glob("*.parquet")]
    df_membership = pl.concat(shards)

    summary = (
        df_membership
        .group_by("edge_id")
        .agg([
            pl.col("motif_instance_id").n_unique().alias("num_motif_instances"),
            pl.col("motif_type").n_unique().alias("num_motif_types"),
            pl.col("role_in_motif").unique().alias("roles"),
        ])
    )

    # Left-join back to df_edges to include is_sar and other metadata.
    return df_edges.join(summary, on="edge_id", how="left")
```

### 4.3 Structural recall evaluation (post-hoc, isSAR used here only)

This is the only place `isSAR` enters the pipeline. It is an evaluation step,
not a mining step.

```python
def compute_structural_recall(
    edge_participation: pl.DataFrame,
) -> Dict[str, Any]:
    """
    Compute what fraction of SAR-labelled edges appear in at least one motif.

    This answers: "Is the structural miner recovering the edges that human
    analysts flagged as suspicious?"

    This is NOT used as a filter or target during mining.
    It is a diagnostic tool only.
    """
    if "is_sar" not in edge_participation.columns:
        return {"error": "is_sar column not found"}

    sar_edges = edge_participation.filter(pl.col("is_sar") == 1)
    total_sar = sar_edges.height

    if total_sar == 0:
        return {"total_sar": 0, "recall": None}

    sar_in_motif = sar_edges.filter(pl.col("num_motif_instances").is_not_null())
    covered_sar = sar_in_motif.height

    recall = covered_sar / total_sar

    # Break down by motif type.
    # Requires joining membership back to get per-type breakdown.
    return {
        "total_sar_edges": total_sar,
        "sar_edges_in_any_motif": covered_sar,
        "structural_recall": recall,
        "sar_edges_in_no_motif": total_sar - covered_sar,
    }
```

If recall is low, the correct response is to inspect the topology of uncovered SAR
edges: what is their in-degree, out-degree, and neighborhood structure? If they
form patterns that match none of the defined motif families, either the motif
definitions need to be extended or new pattern types are needed. The SAR label is
diagnostic evidence about which patterns to define, not a mining filter.

### 4.4 Runtime audit and performance tracking

Extend the existing stats schema to record the new pruning counters from Phase 2:

```python
# Add to all matcher stats dicts:
"num_partial_paths_pruned_by_hop_budget": int,
"num_partial_paths_pruned_by_return_feasibility": int,
"num_partial_paths_pruned_by_canonical_rotation": int,
"num_combinations_skipped_by_time_prefilter": int,
"num_sinks_skipped_by_intersection_empty": int,
```

Build a per-run aggregate that compares pruning efficiency across motif families
and window sizes. This makes it possible to verify that Phase 2 changes reduce
work rather than simply move it.

---

## Parameter thresholds and calibration

### `DELTA_HOP`

The current value is 5 (steps). This means consecutive edges in a motif must occur
within 5 time steps of each other. For cycles, this is the inter-hop gap; for fan
patterns it is the spread across branches.

**Calibration approach**: on a sample of 10–20 windows, plot the distribution of
inter-edge time gaps within known SAR subgraphs (if labels allow this) and within
the full edge set. Set `DELTA_HOP` to the 90th percentile of the SAR distribution.
If no SAR-specific calibration is available, start with 5 and try 3 and 10 as
sensitivity checks.

### `MAX_MOTIF_DURATION`

The current value is 20. This is the maximum total time span of a motif instance
(from first edge to last edge).

**Calibration approach**: same as above — use the distribution of total subgraph
duration in the data. If the step unit is days, 20 days is a reasonable upper bound
for a laundering cycle. If the step unit is hours or transactions-per-block, adjust
accordingly.

### Candidate cap schedule

The cap schedule in Phase 1 is conservative. After a first full run, inspect the
`num_dst_groups_capped` and `num_src_groups_capped` stats. If capping is frequent,
it means the dataset has many high-degree hub nodes. In that case, either tighten
caps further or add a hub-node exclusion filter (nodes with degree > threshold are
excluded as anchors because they participate in so many combinations that the output
becomes dominated by hub artifacts rather than suspicious substructures).

### Amount constraints

`AMOUNT_RATIO_MIN = 0.3` and `AMOUNT_RATIO_MAX = 2.0` allow a 3.3× ratio between
consecutive edge amounts. These are set in `Config` and applied to `split_merge`,
`center_in_out`, and `cycle` patterns.

For fan-in and fan-out, amount ratios between branches are not directly meaningful
(branches are parallel, not sequential). For these, consider an alternative:
**branch dispersion ratio** — the ratio of the maximum branch amount to the minimum
branch amount. A very uneven fan-in (one branch 1000×, others ~1) may be an
operational accident rather than a coordinated pattern. The threshold is dataset-specific.

---

## Implementation sequence

| Step | Scope | Prerequisite |
|---|---|---|
| 1 | Implement `make_fan_in_pattern(n)` and `make_fan_out_pattern(n)` factories | None |
| 2 | Generalize `FanInMatcher` and `FanOutMatcher` to variable n with cap schedule | Step 1 |
| 3 | Reorder constraint checks (cheapest first) in all fan matchers | Step 2 |
| 4 | Implement `WindowCandidateSummary` pre-computation | None |
| 5 | Add DFS pruning guards A, B, C to `CycleKMatcher` | None |
| 6 | Validate cycle pruning on `cycle_5` with 4 windows, compare instance counts | Step 5 |
| 7 | Extend `CYCLE_SIZES` to 5–10, benchmark per-k runtime | Step 6 |
| 8 | Implement `BidirectionalCycleSearch` for k ≥ 7 | Step 7 |
| 9 | Implement `make_split_merge_pattern(n)` factory | None |
| 10 | Apply sink-intersection-first in `SplitMergeMatcher`, generalize to variable n | Step 9 |
| 11 | Implement `make_center_inout_pattern(n_in, n_out)` and generalize matcher | None |
| 12 | Implement `make_stacked_bipartite_pattern(layer_sizes)` and `StackedBipartiteMatcher` | Steps 2, 10 |
| 13 | Implement cross-window deduplication | All matchers stable |
| 14 | Implement `build_edge_participation_summary` | Step 13 |
| 15 | Implement `compute_structural_recall` (evaluation only, uses isSAR) | Step 14 |
| 16 | Calibrate `DELTA_HOP`, `MAX_MOTIF_DURATION`, cap schedules from run stats | Step 15 |

Steps 1–3 and 5–7 can proceed in parallel. Steps 9–11 can proceed in parallel with
Steps 5–8. Steps 12–16 are sequential and require all matchers to be stable.

---

## What does not change

- The `MotifPattern` / `PatternEdge` dataclass structure
- The `TemporalIndex` and its binary-search range queries
- The `WindowSpec` and `build_temporal_windows` logic
- The `make_motif_instance_row` and `make_edge_motif_membership_rows` output format
- The `write_motif_outputs` and `run_matchers_over_windows` pipeline skeleton
- The parquet output schema for `motif_instances` and `edge_motif_membership`

These are correct and stable. All changes are confined to pattern factories,
matcher internals, and output aggregation.
