# Community Detection Guide

## 1. Detection Priority
Use the following order of methods:

1. Directed modularity on window graphs
2. Temporal regularization across windows
3. Directed spectral clustering / Directed Laplacian
4. Infomap for flow-based communities
5. Temporal Infomap / multilayer Infomap

## 2. Directed Weighted Window Graph
For each window `t`, define:

$$
A^{(t)}_{ij} = \sum a_{i \to j}
$$

Use weighted out-degree and in-degree:

$$
d_i^{out} = \sum_j A^{(t)}_{ij}, \quad d_j^{in} = \sum_i A^{(t)}_{ij}
$$

$$
m_t = \sum_{i,j} A^{(t)}_{ij}
$$

## 3. Directed Modularity
Optimize:

$$
Q^{(t)} = \frac{1}{m_t} \sum_{i,j}
\left(
A^{(t)}_{ij} - \frac{d_i^{out} d_j^{in}}{m_t}
\right) \mathbf{1}[c_i^{(t)} = c_j^{(t)}]
$$

Use this on each rolling window.

## 4. Temporal Stability
Add a penalty for label changes across windows:

$$
\max \sum_t Q^{(t)} - \lambda \sum_{t>1} \sum_i \mathbf{1}[c_i^{(t)} \ne c_i^{(t-1)}]
$$

This keeps communities stable unless the data strongly changes.

## 5. Directed Spectral / Directed Laplacian
Use a directed spectral representation only if it preserves direction and weight.
Do not silently symmetrize the graph as the default choice.

If an undirected approximation is required, document the loss of direction explicitly.

## 6. Infomap
Use Infomap when flow behavior is the main signal.
This is the preferred option when the goal is to trap money flow inside a community.

## 7. Multilayer / Temporal Infomap
Connect the same node across consecutive windows with inter-layer coupling.
Use this when you need persistent communities over time.

## 8. Implementation Rules
- Use sparse window graphs.
- Avoid full-graph in-memory community construction.
- Process window by window.
- Keep outputs small and reusable.
- Return labels plus node sets for later tracking.
