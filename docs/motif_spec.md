# motif_spec.md

## 1. Purpose

This document describes the implementation standards for **temporal motif mining** within the AML Graph Mining project.

In this context, a motif is not a generic geometric pattern. A motif is only meaningful when it simultaneously satisfies:
- Topology
- Direction
- Timing
- Amount preservation
- Repetition
- AML context

The objectives of motif mining are:
- To detect concentrated suspicious transaction patterns.
- To generate features for downstream ML models.
- To support the differentiation between normal behavior and money laundering behavior.

---

## 2. Scope

**Applicable to:**
- Event graphs
- Local temporal subgraphs
- Bounded window searches

**Not applicable to:**
- Full graph brute force
- Motifs without temporal constraints
- Motifs based solely on topology while ignoring amount and repetition

---

## 3. Input Schema

### 3.1 Mandatory Input
- Raw transaction table `event_df`.
- Each row represents a transaction event.
- Minimum columns:
  - `event_id`
  - `step`
  - `nameOrig`
  - `nameDest`
  - `amount`

### 3.2 Recommended Additional Columns
- `type`
- `bankOrig`
- `bankDest`
- `oldbalanceOrig`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`

### 3.3 Mandatory Chronological Order
`event_df` must be sorted by:
- `step`
- `event_id`

---

## 4. Motif Library

Only templates with AML significance will be mined.

### 4.1 Fan-in
Multiple nodes transferring money into a single central node.
**AML Significance:**
- Consolidating funds from multiple unusual sources.

### 4.2 Fan-out
A single node transferring money to multiple nodes.
**AML Significance:**
- Dispersing funds to obscure the audit trail.

### 4.3 Cycle
Example:
$$u \to v \to w \to u$$
**AML Significance:**
- Circular flow of funds.
- Creating fictitious revenue.
- Closed-loop layering.

### 4.4 Relay / Path
Example:
$$u \to v \to w \to x$$
**AML Significance:**
- Transferring through multiple layers.

### 4.5 Split-merge
A single source splits into multiple paths and later recombines.
**AML Significance:**
- Complex layering techniques.

---

## 5. Exact Matching Rules

A valid motif instance must simultaneously satisfy:

### 5.1 Direction
Edge directions must match the template exactly.

### 5.2 Time Order
If the template has an order:
$$e_1 \prec e_2 \prec \dots \prec e_k$$
Then the actual data must satisfy:
$$t_1 < t_2 < \dots < t_k$$

### 5.3 Latency Bound
For consecutive edges:
$$0 < t_{i+1} - t_i \le \Delta$$

### 5.4 Money Preservation
The amount ratio must fall within a specific threshold:
$$\rho_{\min} \le \frac{a_{i+1}}{a_i} \le \rho_{\max}$$
Thresholds should be configurable, not hard-coded.

### 5.5 Minimum Repetition
A motif is only considered suspicious if:
$$\text{support}(M) \ge r_{\min}$$

---

## 6. Enumeration Strategy

### 6.1 Seed-based Search
Do not use brute force on the entire graph.
**Process:**
1. Select a seed edge.
2. Search only forward in time.
3. Extend the branch only if it maintains:
   - Direction
   - Lag
   - Amount ratio
4. Stop immediately when a condition is violated.

### 6.2 Pruning Rules
Prune branches when:
- Direction is incorrect.
- Lag exceeds $\Delta$.
- Amount ratio is outside thresholds.
- Nodes are incorrectly repeated.
- Motif exceeds defined size.

### 6.3 Bounded Windows
Searches should be performed within:
- 7 days
- 14 days
- 30 days
Do not search across the entire history in a single pass.

---

## 7. Outputs

### 7.1 Motif Instance Table
Store for each instance:
- `motif_type`
- `node_sequence`
- `edge_sequence`
- `step_sequence`
- `amount_sequence`
- `lag_sequence`
- `window_id`

### 7.2 Aggregated Feature Table
By node or window:
- Motif count by type.
- Frequency normalized by degree / volume.
- Average lag.
- Amount ratio statistics.
- Z-score against null model.

---

## 8. Scoring

A motif is not sufficient just because it "exists." It must be scored.

### 8.1 Raw Support
Number of occurrences within the window.

### 8.2 Normalized Frequency
Normalized by:
- Degree
- Transaction volume
- Node activity

### 8.3 Null-model Z-score
$$z(M)=\frac{C_{obs}(M)-\mu_{null}(M)}{\sigma_{null}(M)}$$

### 8.4 Suspicion Criteria
A high-quality motif must:
- Be concentrated in suspicious cases.
- Not just be generally common.
- Increase the predictive power of the ML model.

---

## 9. Feature Engineering

Minimum motif features should include:
- Count by motif type.
- Frequency normalized by transaction volume.
- Frequency normalized by degree.
- Average latency.
- Max latency.
- Amount preservation mean.
- Amount preservation variance.
- Z-score per motif type.
- Repetition count.

Features should be calculated by:
- Entity
- Window
- Community (if required)

---

## 10. Implementation Rules

- Prioritize index-based search.
- Prioritize local windows.
- Do not keep the entire instance list in RAM if it is too large.
- If necessary, write intermediate results to disk.
- Separate each template into its own function.

**Recommended functions:**
- `find_fanin`
- `find_fanout`
- `find_cycle_3`
- `find_relay_4`
- `find_split_merge`
- `count_support`
- `compute_null_zscore`
- `build_motif_features`

---

## 11. Validation Rules

A motif result is valid when it has:
- Correct direction.
- Correct time order.
- Correct lag.
- Correct amount ratio.
- Sufficient repetition.
- A meaningful z-score.

---

## 12. Acceptance Criteria

Motif mining is considered successful if:
- It discovers patterns concentrated in known suspicious cases.
- It generates features that increase the predictive power of the model.
- It runs within Colab with limited RAM.
- It avoids full-graph brute force.
- It preserves the temporal and directional nature of AML.