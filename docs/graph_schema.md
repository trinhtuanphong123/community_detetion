# graph_schema.md

## 1. Purpose

This document defines the data schema and standard graph schema for the **AML Graph Mining Pipeline** project.

The objective is to ensure that all subsequent processing steps correctly utilize:
- Transaction **direction**
- Transaction **weight**
- Transaction **time**
- **Limited memory** constraints of Colab

This document serves as the authoritative source for:
- Constructing the event graph for motif mining
- Constructing snapshot / rolling-window graphs for community detection
- Generating features for the ML model in later stages

---

## 2. Scope

This schema applies to AMLGentex transaction data and variants with the same structure.

**Permitted operations:**
- Reading raw transaction data
- Normalizing data types
- Mapping nodes to integer IDs
- Creating directed, weighted edges with timestamps
- Aggregating edges by day or by time window

**Prohibited operations:**
- Constructing a full MultiDiGraph for the entire dataset in memory
- Removing direction or time information
- Using SAR labels as input attributes for graph topology

---

## 3. Input Schema

### 3.1 Minimum Transaction Record

Each transaction must be represented as a record:

$$
e = (u, v, t, a)
$$

Where:
- `u`: sender
- `v`: receiver
- `t`: time point / step
- `a`: amount

### 3.2 Actual Data Columns of AMLGentex

| Column | Recommended Type | Meaning |
|---|---:|---|
| `step` | int32 | time index / day |
| `type` | category | transaction type |
| `amount` | float32 | transaction amount |
| `nameOrig` | int64 / string | sender identifier |
| `bankOrig` | category | source bank |
| `daysInBankOrig` | int32 | days at source bank |
| `phoneChangesOrig` | int32 | number of phone changes at source |
| `oldbalanceOrig` | float32 | old balance at source |
| `newbalanceOrig` | float32 | new balance at source |
| `nameDest` | int64 / string | receiver identifier |
| `bankDest` | category | destination bank |
| `daysInBankDest` | int32 | days at destination bank |
| `phoneChangesDest` | int32 | number of phone changes at destination |
| `oldbalanceDest` | float32 | old balance at destination |
| `newbalanceDest` | float32 | new balance at destination |
| `isSAR` | int8 | assessment label, not used for graph construction |
| `alertID` | int32 | alert metadata |
| `modelType` | int32 | model metadata |

### 3.3 Mandatory Columns for Graph Pipeline

The following are mandatory:
- `step`
- `nameOrig`
- `nameDest`
- `amount`

Recommended to keep:
- `type`
- `bankOrig`
- `bankDest`
- `oldbalanceOrig`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`

---

## 4. Node Schema

### 4.1 Node Identity

Each entity must be encoded into a stable integer ID:

- `node_id`: Internal ID
- `raw_id`: Original ID in the data

Mapping must be consistent throughout the entire pipeline.

### 4.2 Node Attributes

Each node may hold the following attributes:
- Total in-flow
- Total out-flow
- Weighted in-degree / out-degree
- Number of transaction counterparties
- Operational persistence over time
- Most frequent bank
- Number of observed active days

Do not include label attributes in the graph topology construction step unless they are necessary.

---

## 5. Edge Schema

Each edge must be directed and weighted.

### 5.1 Edge Record

A standard edge:

$$
e = (u, v, t, a)
$$

With:
- `u -> v` is the direction of money transfer
- `t` is the normalized step or timestamp
- `a` is the amount

### 5.2 Recommended Edge Attributes to Store

- `amount`
- `step`
- `tx_count` (if aggregating multiple transactions for the same pair)
- `first_event_id`
- `last_event_id`
- `window_start`
- `window_end`
- `type` (if deep analysis is required)

---

## 6. Graph Representations

### 6.1 Event Graph

Used for motif mining.

**Characteristics:**
- Retains each raw transaction
- Retains chronological order
- Retains direction and amount
- Does not lose ordering through premature aggregation

**Recommendation:**
- Store as an event table `event_df`
- Construct local graphs only when motif searching is required

### 6.2 Snapshot Graph

Used for community detection.

Defined by step:

$$
G^{(t)} = (V, E^{(t)}, W^{(t)})
$$

With:

$$
W^{(t)}_{uv} = \sum_{e=(u,v,t,a)} a
$$

If there are multiple transactions from `u` to `v` in the same step, they are aggregated into a single edge.

### 6.3 Rolling-window Graph

Used to track persistence.

Defined as:

$$
G_{[t-w+1, t]}
$$

With windows of:
- 7 days
- 14 days
- 30 days

### 6.4 Local Subgraph

Used for:
- Local motif matching
- Local community debugging
- Analysis of small suspicious groups

Do not construct a global graph unless mandatory.

---

## 7. Storage and Memory Rules

- Prioritize event tables and aggregated edge tables
- Prioritize sparse representation
- Do not duplicate the entire dataset in memory
- Do not hold multiple graph objects simultaneously
- Release intermediate variables after each window
- For Colab, only materialize the graph within the window being processed

---

## 8. Validation Rules

A graph is valid when:
- Direction is correct
- Weight is correct
- Time is correct
- Node mapping is stable
- Aggregated edges follow window rules

Minimum checks:
1. Number of nodes is non-negative
2. Number of edges is consistent with the number of transactions after aggregation
3. No reversed edges due to mapping errors
4. `amount` is not cast to the wrong type
5. Window graph only contains transactions within that window

---

## 9. Acceptance Criteria

The schema is considered successful if:
- It can be used directly to build an event graph
- It can be used directly to build a snapshot graph
- It supports motif mining and community detection without schema changes
- It runs in Colab within limited RAM
- It preserves the semantics of the AML data