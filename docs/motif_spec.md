# Motif Specification

## 1. Purpose

This document defines the implementation contract for temporal motif mining in the AML Graph Mining project.

A motif is only meaningful when it satisfies all of the following at the same time:
- directed topology
- temporal order
- latency bound
- amount preservation
- repetition threshold
- AML context

The goal of motif mining is to:
- detect concentrated suspicious transaction patterns
- generate ML-ready features
- support later ranking and investigation

Motifs are signals, not final labels.

## 2. Scope

### In scope
- Event-level transaction tables
- Bounded temporal windows
- Local temporal subgraph search
- Pattern counting and scoring
- Feature extraction from matched motif instances

### Out of scope
- Full-graph brute force search
- Motifs without time constraints
- Motifs that ignore direction or amount
- Unbounded search across the entire history in one pass

## 3. Input and output

### 3.1 Input
The motif module should mainly use `event_df`.

Mandatory columns:
- `event_id`
- `step`
- `nameOrig`
- `nameDest`
- `amount`

Recommended optional columns:
- `type`
- `bankOrig`
- `bankDest`
- `oldbalanceOrig`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`

`event_df` must be sorted by:
- `step`
- `event_id`

### 3.2 Output
Motif mining should produce two layers of output.

#### Layer 1 — motif instances
Each successful match should store:
- motif type
- transaction IDs
- participating nodes
- step sequence
- amount sequence
- lag sequence
- ratio sequence
- window ID

#### Layer 2 — feature table
Aggregate by entity, window, or community when needed.
Typical features:
- motif count by type
- motif frequency normalized by degree or transaction volume
- average latency
- amount preservation statistics
- z-score against a null model

## 4. Motif library

Only AML-relevant templates should be implemented.

### 4.1 Fan-in
Many to one.
Useful for concentrated fund collection from multiple sources.

### 4.2 Fan-out
One to many.
Useful for dispersing funds and hiding traces.

### 4.3 Cycle
Example:
`u -> v -> w -> u`
Useful for circular flow and closed-loop laundering behavior.

### 4.4 Relay / Path
Example:
`u -> v -> w -> x`
Useful for layering across multiple hops.

### 4.5 Split-merge
A source splits into multiple branches and later recombines.
Useful for complex layering patterns.

## 5. Required pipeline

The motif module should follow this order:

1. Normalize event-level data.
2. Build fast search indexes.
3. Define motif templates.
4. Match each template.
5. Apply pruning.
6. Count support and filter strong motifs.
7. Build the feature table.

## 6. Indexes required for search

The matcher should not scan the full table repeatedly.
Create indexes such as:
- `out_edges_by_node`
- `in_edges_by_node`
- `edges_by_step`
- `edges_after_time`

Indexes must support forward-only temporal search.

## 7. Feature targets

Minimum motif features:
- count by motif type
- normalized frequency
- average lag
- max lag
- amount ratio mean
- amount ratio variance
- repetition count
- z-score per motif type

Feature aggregation targets:
- entity
- day or window
- community if required

## 8. Validation rules

A motif result is valid only if it satisfies:
- correct direction
- correct time order
- correct latency bound
- correct amount ratio
- sufficient repetition
- meaningful support or z-score

## 9. Acceptance criteria

Motif mining is successful if it:
- finds concentrated suspicious patterns
- produces useful downstream features
- runs within Colab memory limits
- avoids full-graph brute force
- preserves time and direction
