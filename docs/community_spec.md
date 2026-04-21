# Community Spec

In this project, a community is a candidate subgraph of entities with unusually dense internal flow or unusually low separation from the rest of the graph. Community detection is used to narrow the search space, not to produce the final AML label.

Primary metrics:
- Modularity: useful as a partition-quality signal, but not sufficient alone.
- Conductance: useful for measuring how well a subgraph is separated from the rest of the graph.
- Temporal stability: required for AML use across windows.

Ground truth:
- True AML community ground truth is usually absent.
- Analyst decisions, SAR/STR outcomes, and known cases are weak supervision or evaluation signals, not perfect ground truth.
- Metadata should not be treated as ground-truth community labels.

Project rule:
- Prefer communities that are interpretable, temporally stable, and enriched for suspicious motifs.

Implementation notes:
- Run detection on bounded windows or bounded candidate subgraphs.
- Keep community assignment versioned by window.
- Use community results as context for motif mining and feature aggregation.
