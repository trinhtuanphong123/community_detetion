# Motif Pattern Search Guide

## 1. Goal

This guide defines how to implement the pattern search logic for temporal motif mining in AML.

The search must be:
- directed
- temporal
- bounded
- memory efficient
- configurable

## 2. Matching rules

A candidate motif instance must satisfy all of the following:

### 2.1 Direction
The edge direction must exactly match the template.

### 2.2 Time order
If the template order is:
`e1 < e2 < ... < ek`
then the observed transactions must satisfy:
`t1 < t2 < ... < tk`

### 2.3 Latency bound
For consecutive edges:
`0 < t(i+1) - t(i) <= Δ`

### 2.4 Amount preservation
For consecutive amounts:
`ρmin <= a(i+1) / a(i) <= ρmax`

These thresholds must be configurable.

### 2.5 Minimum repetition
A motif is suspicious only when its support reaches the minimum threshold:
`support(M) >= rmin`

## 3. Templates to implement

### 3.1 Fan-in
Many -> one.
Search for multiple incoming edges into the same target within the allowed window.

### 3.2 Fan-out
One -> many.
Search for multiple outgoing edges from the same source within the allowed window.

### 3.3 Cycle
Example:
`u -> v -> w -> u`

### 3.4 Relay / Path
Example:
`u -> v -> w -> x`

### 3.5 Split-merge
A split followed by a recombination.
Implement in a bounded and incremental way, not as a full brute-force search.

## 4. Search strategy

Use seed-based enumeration.

Recommended flow:
1. Select a seed edge.
2. Search only forward in time.
3. Expand only if direction, lag, and amount ratio still hold.
4. Stop immediately when any constraint fails.

Do not brute force the entire graph.
Do not search across the full transaction history in one pass.
Do not keep large candidate lists in memory if they can be streamed or filtered early.

## 5. Pruning rules

Prune a branch immediately when any of the following happens:
- direction mismatch
- lag exceeds Δ
- amount ratio outside `[ρmin, ρmax]`
- repeated nodes violate the template rules
- the branch exceeds the allowed motif size
- the branch moves outside the current window

## 6. Search indexes

Pattern search should use prebuilt indexes instead of repeated scans.
Recommended indexes:
- outgoing edges by node
- incoming edges by node
- edges by step
- edges after a given step

These indexes should support fast forward expansion.

## 7. Function boundaries

Each motif template should have its own matcher function.
Recommended function set:
- `find_fanin`
- `find_fanout`
- `find_cycle_3`
- `find_relay_4`
- `find_split_merge`

Supporting utilities:
- `count_support`
- `compute_null_zscore`
- `build_motif_features`

## 8. Output of the matcher

Each matcher should return motif instances with at least:
- motif type
- node sequence
- edge sequence
- step sequence
- amount sequence
- lag sequence
- ratio sequence
- window ID

## 9. Practical constraints

The implementation must stay usable in Google Colab.
That means:
- window-based processing
- sparse or index-based search where possible
- no full in-memory graph traversal for the entire dataset
- no unnecessary abstraction layers

## 10. Acceptance criteria

The search implementation is acceptable if it:
- finds directed temporal motif instances correctly
- supports pruning and bounded search
- respects amount and latency constraints
- produces motif instances and counts that can become ML features
