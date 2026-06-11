# Research Specification: `cycle_5` progressing to `cycle_6..10`

## Scope

This document defines the research specification for the retained temporal cycle motif family:

- `cycle_5`
- future extensions `cycle_6` through `cycle_10`

The pipeline will not include `two_stage_split_8`. That motif family is removed from scope and has no dependency on the cycle design.

This specification is intended to preserve exact cycle matching while reducing the search-space explosion that appears when cycle length grows.

## Research basis

This specification is grounded in two references:

1. Time-Constrained Continuous Subgraph Matching Using Temporal Information for Filtering and Backtracking
   Link: https://arxiv.org/abs/2312.10486

2. Temporal graph patterns by timed automata
   Link: https://arxiv.org/abs/2205.14269

Why these references matter here:

- Min et al. show that temporal information can be used both for filtering candidate edges and for pruning backtracking during time-constrained matching. That maps directly to DFS-based cycle expansion.
- Aghasadeghi et al. show that temporal constraint evaluation can be applied incrementally, and that cyclic patterns on sparse graphs benefit from on-demand processing over total matchings. That supports pruning partial cycle paths before complete closure.

## Problem statement

The current `cycle_k` matcher uses depth-first expansion from a primary anchor and closes the cycle at depth `k-1`.

This is the correct exact-search baseline, but the cost grows quickly as:

- cycle length increases
- branching factor increases
- time windows remain permissive

For `cycle_6..10`, the core challenge is no longer defining the pattern. The challenge is preventing DFS from exploring partial paths that cannot possibly close into a valid temporal cycle.

## Core design principle

The group applies Min/Aghasadeghi's DFS pruning algorithm and a bidirectional search mechanism.

Operational interpretation:

- DFS remains the base search engine because exact cycles require ordered path construction
- temporal feasibility must be evaluated at each partial step, not only when closing the cycle
- for longer cycles, the search must meet in the middle rather than always expanding from one side only

## Target motif family

Canonical cycle definition:

- directed edges
- strict temporal order
- distinct cycle nodes before closure
- final edge returns to the start node

Target cycle sizes:

- `cycle_5`
- `cycle_6`
- `cycle_7`
- `cycle_8`
- `cycle_9`
- `cycle_10`

`cycle_11` and `cycle_12` may remain defined in pattern metadata, but they are not part of this specification target set.

## Search architecture requirements

### 1. Time-constrained candidate filtering

Before DFS expansion, mark or derive only those candidate edges that can participate in at least one cycle under:

- strict time ordering
- `delta_hop`
- total motif duration
- distinct-node requirements

This follows the idea that temporal information should be used to reduce the edge domain before backtracking starts.

### 2. Incremental pruning during DFS

At every path extension, reject the partial path if any of the following fail:

- the next step violates strict time order
- the next step violates `delta_hop`
- the current partial path already exceeds total duration
- the next node repeats a previously used cycle node
- the remaining time budget cannot accommodate the remaining number of edges
- the current node cannot still reach the start node within the remaining hop budget and time budget

The last condition is essential for `cycle_7..10`.

### 3. Bidirectional search for longer cycles

For `cycle_7..10`, unidirectional DFS should no longer be the only execution plan.

Required alternative:

- forward search from the anchor edge
- backward or reverse-feasible search from a legal closing edge
- meet-in-the-middle join on compatible frontier states

This is relevant because a cycle is naturally decomposable into two directed temporal segments plus a closing compatibility check.

## DFS pruning rules

The following pruning rules are mandatory research targets.

### 1. Remaining-depth feasibility

Given current depth and remaining edges, reject a state when:

- even the earliest possible continuation would exceed the total duration
- even the latest allowed continuation cannot satisfy the remaining number of hops

### 2. Return-to-start feasibility

Reject a state when the current node has no feasible path back to the start node within:

- remaining hop count
- remaining time budget
- remaining distinct-node budget

### 3. Temporal edge-domain pruning

For a candidate next edge, reject it before recursion when:

- its timestamp leaves too little room for later edges
- its timestamp is too far from the previous edge under `delta_hop`
- closing after taking it would already be impossible

### 4. Canonical duplication control

Keep only one representation of the same directed temporal cycle.

Required canonical rules:

- anchor edge must be temporally canonical
- optional secondary canonical rule on start node or edge signature to reduce duplicate rotations

### 5. Optional amount pruning

If the cycle family is used for AML-specific motif extraction rather than pure structural mining, support:

- minimum amount floor
- bounded adjacent amount ratios
- bounded whole-cycle amount dispersion

These should remain optional because not every suspicious cycle has strong conservation behavior.

## Bidirectional search requirements

Bidirectional search is required for `cycle_7..10` research, not necessarily for `cycle_5`.

### Search split

For a target cycle of length `k`:

- build a forward partial path of roughly `floor(k / 2)` edges
- build a reverse-feasible partial path or closing-side partial of roughly `ceil(k / 2)` edges
- join only frontier states that are compatible in time, nodes, and closure feasibility

### Frontier state contents

Each frontier state should at minimum include:

- current endpoint
- ordered timestamp bounds
- used node set signature
- used edge set signature or sufficient duplicate-prevention state
- anchor identifier

### Join conditions

Two frontier states may be joined only if:

- they share a compatible meeting node or meeting edge relation
- their used node sets are disjoint except for permitted join points
- the merged path respects strict temporal order
- the merged path can still be closed into a full cycle under the duration budget

## Temporal automaton interpretation

The timed-automata reference is not being adopted as a full engine replacement here. It is used as a design principle.

Practical interpretation:

- treat the cycle matcher as a sequence of temporal states
- each DFS extension is a state transition
- if a transition makes final acceptance impossible, prune immediately

This matters because it formalizes why partial temporal validation belongs inside DFS rather than after a full path is assembled.

## Accuracy-oriented conditions

To keep cycle results useful for AML analysis, the following precision controls should be supported:

- all non-closing cycle nodes must be distinct
- repeated pair transfers should be optionally suppressible
- cycles with excessively loose time spacing should be rejected
- cycles with extreme amount volatility should be optionally rejected
- cycles confined to one benign operational cluster may be ranked lower than cross-cluster cycles

These are not required to define a valid cycle. They are required to keep large-cycle search from flooding the pipeline with low-value structural matches.

## Runtime metrics to collect

The redesigned cycle matcher must record:

- number of anchor edges
- number of DFS states expanded
- number of forward candidates filtered before recursion
- number of states rejected by time
- number of states rejected by distinct-node constraints
- number of states rejected by return-feasibility pruning
- number of close attempts
- number of bidirectional frontier states
- number of frontier joins attempted
- number of final cycle instances
- elapsed time per cycle length per window

These metrics are required so that `cycle_6..10` can be evaluated by search efficiency, not only by output count.

## Acceptance criteria

The research is considered successful when the future cycle matcher design demonstrates all of the following:

- lower DFS state expansion counts than a naive one-sided DFS baseline
- lower runtime for `cycle_6..10` on the same windows
- exact structural correctness is preserved
- duplicate rotations are controlled
- temporal pruning remains explainable and auditable

## In-scope and out-of-scope

In scope:

- directed temporal cycles of length 5 through 10
- exact search with aggressive pruning
- bidirectional search for larger cycles
- optional AML amount constraints

Out of scope:

- `two_stage_split_8`
- non-cycle motifs
- approximate-only cycle counting as the primary production path

## Relationship to the retained pipeline

This document assumes the retained motif families are:

- `fan_in`
- `fan_out`
- `cycle`
- `split_merge`
- `center_in_out`

Only the `cycle` family is specified here. `fan_in`, `fan_out`, `split_merge`, and `center_in_out` remain in the retained pipeline but are not the subject of this document.

## References

- Time-Constrained Continuous Subgraph Matching Using Temporal Information for Filtering and Backtracking. Min et al., 2023. https://arxiv.org/abs/2312.10486
- Temporal graph patterns by timed automata. Aghasadeghi, Van den Bussche, and Stoyanovich, 2022. https://arxiv.org/abs/2205.14269
