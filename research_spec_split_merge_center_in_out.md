# Research Specification: `split_merge` and `center_in_out`

## Scope

This document defines the research specification for the retained AML temporal motif families:

- `split_merge`
- `center_in_out`

The current concrete baselines are:

- `split_merge_5`
- `fanin_fanout_6` with matcher type `center_in_out`

The pipeline will not include `two_stage_split_8`. That motif family is removed from scope.

This specification is intended to guide future matcher redesign for larger motif sizes in the 6-10 edge or vertex range while preserving AML relevance and keeping exact search tractable.

## Research basis

This specification is grounded in two references:

1. XMiner: Efficient Directed Subgraph Matching with Pattern Reduction
   Link: https://arxiv.org/abs/2404.11105

2. Detecting Mixing Services via Mining Bitcoin Transaction Network with Hybrid Motifs
   Link: https://arxiv.org/abs/2001.05233

Why these references matter here:

- XMiner is relevant because `split_merge` and `center_in_out` are directed motifs with strong internal constraints. XMiner argues that search should be driven by constraint reduction and an execution plan rather than by naive expansion from the first edge.
- Wu et al. are relevant because AML-like transaction motifs need more than topology. Their work supports adding transaction-level and motif-level monetary constraints so that benign structural matches are filtered out earlier.

## Problem statement

The current matchers enumerate branch combinations first and validate later. That is acceptable for small motifs, but it does not scale well.

Current risk points:

- `split_merge` pays a large cost on split triplets and then a second cost on merge products.
- `center_in_out` pays a large cost on inbound combinations multiplied by outbound combinations.
- Larger motifs in these families will increase branching more quickly than cycles of similar size because the number of branch combinations grows combinatorially around high-degree nodes.

The research goal is to redesign search so that:

- search starts from the most selective phase or join variable
- full edge products are delayed as long as possible
- monetary constraints remove low-value candidates before deep enumeration
- exact matching remains possible for 6-10 edge variants on realistic AML windows

## Motif family definitions

### `split_merge`

Canonical form:

- one source splits to multiple intermediates
- intermediates later converge to one sink

Examples for future scaling:

- `split_merge_6`: 3 split edges and 3 merge edges
- `split_merge_8`: 4 split edges and 4 merge edges
- `split_merge_10`: 5 split edges and 5 merge edges

### `center_in_out`

Canonical form:

- one center receives from multiple upstream nodes
- the same center later sends to multiple downstream nodes

Examples for future scaling:

- `center_in_out_6`: 3 inbound and 2 outbound edges
- `center_in_out_8`: 4 inbound and 3 outbound edges
- `center_in_out_10`: 5 inbound and 4 outbound edges

## Core design principle

The group applies XMiner's join order optimization theory and Wu et al.'s monetary constraints.

Operational interpretation:

- join order optimization means execution begins from the rarest and most selective structural condition, not from a fixed hand-written edge order
- monetary constraints are not only post-validation rules; they are part of candidate filtering and partial-match pruning

## Search architecture requirements

### 1. Selectivity-first execution plan

For each motif instance search, compute an execution plan before expansion.

The plan should prefer:

- rare sinks over common sources when common-sink intersection is more selective
- rare centers over raw inbound or outbound combinations
- narrow time bands over wide time bands
- nodes with smaller eligible candidate domains

Required planning statistics per window:

- in-degree and out-degree by node
- distinct-neighbor counts by node
- count of eligible edges after time filtering
- count of eligible edges after amount filtering
- estimated common-sink or common-center candidate counts

### 2. Constraint reduction before materialization

The matcher should reason over candidate node sets and sink sets before materializing edge tuples.

For `split_merge`:

- build candidate intermediate-to-sink reachability sets inside the legal time band
- intersect sink sets first
- materialize edge-level combinations only after common sinks are identified

For `center_in_out`:

- build candidate centers that already satisfy inbound and outbound degree thresholds
- discard centers failing time-feasible handoff bounds before building combinations

### 3. Partial-match pruning

At every phase extension, reject a partial match when any of the following fail:

- remaining time budget cannot fit the unfinished motif
- required distinct nodes cannot still be satisfied
- required remaining in-degree or out-degree support does not exist
- monetary bounds are already violated

## Monetary constraints

The group applies Wu et al.'s monetary constraints as AML-specific precision controls.

These constraints are relevant because laundering-like motifs are rarely defined by structure alone. Amount behavior across branches and phases matters.

Required constraint categories:

### 1. Absolute transaction floor

Reject edges or phases below an amount threshold when the search objective is suspicious-flow triage rather than full enumeration.

### 2. Phase conservation

For `split_merge`:

- compare total split amount to total merge amount
- enforce a bounded conservation ratio

For `center_in_out`:

- compare total inbound amount to total outbound amount
- enforce a bounded conservation ratio

### 3. Branch coherence

Reject candidates where branch amounts are too dispersed to plausibly represent coordinated splitting or coordinated redistribution.

Examples:

- each branch must lie within a ratio band of the branch median
- no single branch may dominate the motif beyond a configured share

### 4. Phase monotonicity or bounded shrinkage

Where appropriate, require:

- merge totals not to exceed split totals beyond tolerance
- outbound totals not to exceed inbound totals beyond tolerance

This is useful for AML settings where fees, leakage, or partial withholding are plausible, but unexplained amplification is not.

## Temporal constraints

Required temporal controls:

- compact split phase
- compact merge phase
- compact inbound phase
- compact outbound phase
- bounded handoff gap between phases
- bounded total motif duration

Temporal constraints must be checked incrementally, not only after the final candidate is built.

## Precision-oriented structural conditions

To improve motif accuracy, the following conditions should be supported:

- all role nodes must be distinct unless the motif definition explicitly allows reuse
- branch roles must be unique by edge and by endpoint pair
- intermediates must not collapse to the source, center, or sink
- outbound destinations in `center_in_out` must be distinct from inbound sources unless a special motif variant explicitly allows overlap

Recommended optional AML filters when metadata is available:

- cross-bank requirement
- minimum number of distinct institutions or communities across branches
- account-age asymmetry filters
- repeated-pair suppression for operational transfers

## Search order recommendations by motif family

### `split_merge`

Preferred execution order:

1. choose candidate source or candidate sink based on lower estimated domain
2. construct eligible intermediate set inside time and amount bounds
3. intersect common sink candidates across intermediates
4. materialize merge edges only after sink intersection succeeds
5. validate conservation and branch coherence before full output emission

When scaling to 8-10 edges:

- avoid generating all split `k`-combinations first
- extend one branch at a time with immediate feasibility checks
- stop extension as soon as the common-sink domain becomes empty or too large

### `center_in_out`

Preferred execution order:

1. enumerate candidate centers satisfying minimum inbound and outbound support
2. prefilter by time-feasible handoff window
3. construct inbound partials only if outbound support remains feasible
4. construct outbound partials only if inbound totals and timing remain plausible
5. apply conservation and branch-coherence checks before final emission

When scaling to 8-10 edges:

- use incremental expansion rather than full inbound-by-outbound products
- prune centers whose support is high-degree but low-selectivity
- cap candidate domains adaptively by time plus amount, not by time alone

## Runtime metrics to collect

The redesigned matcher must record:

- number of candidate sources or centers scanned
- number of candidate sink intersections attempted
- number of partial matches extended
- number of products materialized
- number of candidates rejected by time
- number of candidates rejected by structure
- number of candidates rejected by monetary constraints
- number of candidates rejected by remaining-budget infeasibility
- number of final instances
- elapsed time per window and per motif type

These metrics are required to prove that join-order changes reduce enumeration rather than only moving work around.

## Acceptance criteria

The research is considered successful when the future matcher design demonstrates all of the following:

- lower candidate materialization counts than the current combination-first design
- lower runtime per window on the same motif family
- equal or better precision under analyst review or available labels
- no dependence on `two_stage_split_8`
- a clear path to 6-10 edge motif variants without rewriting the search strategy from scratch

## In-scope and out-of-scope

In scope:

- `split_merge` family scaling
- `center_in_out` family scaling
- join-order planning
- AML monetary constraints
- exact matching with strong pruning

Out of scope:

- `two_stage_split_8`
- generic unlabeled subgraph search detached from AML semantics
- replacing motif search with a learned detector

## Relationship to the retained pipeline

This document assumes the retained motif families are:

- `fan_in`
- `fan_out`
- `cycle`
- `split_merge`
- `center_in_out`

Only `split_merge` and `center_in_out` are specified here. `fan_in` and `fan_out` remain in the pipeline but are not the subject of this document.

## References

- XMiner: Efficient Directed Subgraph Matching with Pattern Reduction. Yuan et al., 2024. https://arxiv.org/abs/2404.11105
- Detecting Mixing Services via Mining Bitcoin Transaction Network with Hybrid Motifs. Wu et al., 2020, revised 2021. https://arxiv.org/abs/2001.05233
