# community_spec.md

## 1. Purpose

This document describes the implementation standards for **community detection** within the AML Graph Mining project.

In an AML context, a community is not a typical "proximity group" but rather a **functional transaction cluster** with the potential to contain money laundering behavior.

Communities must help to:
- Narrow the investigation space.
- Group suspicious entities into a smaller, manageable subset.
- Preserve direction, weight, and time.
- Track persistence across multiple time windows.

---

## 2. Scope

**Applicable to:**
- Snapshot graphs by step.
- Rolling-window graphs.
- Directed weighted graphs.
- Local subgraph analysis.

**Not applicable to:**
- Full in-memory graphs of the entire dataset.
- Purely undirected clustering if it results in the loss of direction.
- Using SAR labels to generate communities.

---

## 3. Input Schema

### 3.1 Required Input
- `daily_edges` or `window_edges`.
- Node mapping `node2idx`.
- Sparse adjacency matrix $A^{(t)}$ or equivalent.

### 3.2 Graph Requirement

Each window must be represented as a directed and weighted adjacency matrix:

$$
A^{(t)}_{ij} = \sum a_{i \to j}
$$

Where:
- `i` is the sender node.
- `j` is the receiver node.
- The value is the total amount within the window.

### 3.3 Data Constraints

**Must retain:**
- Direction.
- Weight.
- Window time.
- Stable node IDs.

**Avoid:**
- Symmetrizing at the start if flow-based methods are still being used.
- Using large graph objects when sparse matrices are sufficient.

---

## 4. Community Definition

A community is considered to have AML significance when it exhibits the following characteristics:

1. High internal transfer intensity.
2. Repeated circular or layered flow.
3. Presence of suspicious bridge nodes.
4. Tight coordination in time.
5. Persistent existence across multiple windows.

A community is a **candidate suspicious subgraph**, not the final ground truth.

---

## 5. Core Mathematical Objectives

### 5.1 Directed Modularity

Optimize:

$$
Q^{(t)} = \frac{1}{m_t} \sum_{i,j}
\left(
A^{(t)}_{ij} - \frac{d^{out}_i d^{in}_j}{m_t}
\right)\mathbf{1}[c_i^{(t)} = c_j^{(t)}]
$$

Where:
- `m_t` is the total weight of edges in the window.
- `d_out` and `d_in` are the weighted out-degree and in-degree.
- `c_i^{(t)}` is the community label of node `i` at window `t`.

### 5.2 Temporal Regularization

Add a penalty for label changes over time:

$$
\max \sum_t Q^{(t)} - \lambda \sum_{t>1} \sum_i \mathbf{1}[c_i^{(t)} \neq c_i^{(t-1)}]
$$

The goal is to keep communities stable across consecutive windows.

### 5.3 Flow-based Community

If using Infomap, the objective is to minimize the description length:

$$
L(M)
$$

Meaning:
- A good community is one where money flow is "trapped" inside.
- Suitable for detecting circular flows, layering, and relay patterns.

---

## 6. Implementation Strategy

### 6.1 Primary Strategy
Priority:
1. Directed weighted modularity on window graphs.
2. Temporal tracking between windows.
3. Scoring suspicious communities.
4. Utilizing Infomap when flow-centric detection is required.

### 6.2 Fallback Strategy
If the primary library does not fully support direction:
- Use local directed projection.
- Or use Infomap with directed edges.
- Or use slight symmetrization on local subgraphs, but any information loss must be explicitly documented.

---

## 7. Code Plan

### 7.1 Build Sparse Adjacency
From `window_edges`, create:
- `row = u`
- `col = v`
- `data = amount`

Store using `csr_matrix`.

### 7.2 Compute Degrees
Calculate:
- `d_out`
- `d_in`
- `m`

Based on weights.

### 7.3 Run Clustering
Execute one of the following:
- Leiden with directed modularity if available.
- Infomap for flow-based communities.

### 7.4 Convert Labels to Communities
Convert labels into:
- `community_id`.
- List of nodes in the community.
- Community size.

### 7.5 Track Across Time
Match communities between windows using an overlap score:

$$
\text{overlap}(C_a, C_b) = \frac{|C_a \cap C_b|}{|C_a \cup C_b|}
$$

---

## 8. Suspicion Scoring

A community must be scored to prioritize investigation.

Suggested formula:

$$
S(C)=\alpha \cdot \text{InternalFlow}(C)
+\beta \cdot \text{Reciprocity}(C)
+\gamma \cdot \text{Persistence}(C)
+\delta \cdot \text{MotifEnrichment}(C)
-\eta \cdot \text{ExternalNoise}(C)
$$

### 8.1 InternalFlow
The ratio of internal money versus money flowing out.

### 8.2 Reciprocity
The degree to which money flows bidirectionally.

### 8.3 Persistence
The number of consecutive windows the community exists.

### 8.4 MotifEnrichment
The number of suspicious motifs found within the community.

### 8.5 ExternalNoise
If the flow outside the community is too large, the suspicion score is reduced.

---

## 9. Filtering Rules

**Keep a community if:**
- Size is large enough to be significant.
- Internal flow is high.
- Persistence is long enough.
- Motif enrichment is high.
- Score exceeds the configured threshold.

**Discard a community if:**
- It is merely a normal transaction hub.
- It is strong in volume only but has no closed flow.
- It appears once and then disappears.

---

## 10. Output Schema

Each window should output:

### 10.1 Community Assignment Table
- `window_id`
- `node_id`
- `community_id`
- `persistent_community_id`

### 10.2 Community Score Table
- `community_id`
- `window_id`
- `size`
- `internal_flow`
- `reciprocity`
- `persistence`
- `motif_enrichment`
- `suspicion_score`

### 10.3 Investigation Shortlist
A list of top communities for manual investigation or downstream model use.

---

## 11. Validation Rules

A community result is valid when:
- It retains direction and weight.
- It can be tracked across multiple windows.
- The score has meaningful discriminative power.
- It reduces the number of nodes requiring investigation.
- It is not dependent on a single fixed threshold.

---

## 12. Acceptance Criteria

Community detection is considered successful if:
- It narrows the inspection space for investigators.
- It groups suspicious entities into manageable clusters.
- It preserves temporal structure.
- It supports downstream motif enrichment.
- It runs within Colab with limited memory.