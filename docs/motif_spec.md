# Motif Spec

Temporal motifs are induced subgraphs over sequences of timestamped edges. For this project, motifs must capture both topology and time order; topology alone is not enough.

Required motif families:
- fan-in
- fan-out
- chain
- cycle
- split-merge

Default search limits:
- max nodes: 4
- max temporal edges: 5
- default windows: 1d / 3d / 7d / 30d
- expand beyond this only for seeded local search

Suspicious motif score should combine:
- repetition frequency
- short time span
- amount consistency or near-conservation
- role consistency of nodes
- rarity versus background behavior
- community risk context

Rule:
- A cycle is not suspicious by itself.
- A motif becomes suspicious only when time, amount, repetition, and context align.

Implementation notes:
- Prefer local extraction inside a window or community.
- Motif output must be convertible into numeric features.
- Keep the motif vocabulary small and explicit before adding new templates.
