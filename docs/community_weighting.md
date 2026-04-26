# Community Weighting Guide

## 1. Goal
Define how weights are used inside community detection and community scoring.

## 2. Weight Source
Use the transaction amount as the primary edge weight.
For a window graph, the directed weighted adjacency is:

$$
A^{(t)}_{ij} = \sum a_{i \to j}
$$

## 3. Weighting Rules
- Keep direction.
- Keep raw amount as the default weight.
- Only normalize weights if the config explicitly asks for it.
- Do not discard small values unless they are clearly noise for the task.

## 4. Useful Community Weights
Compute these from the weighted graph:
- internal flow
- external flow
- reciprocity
- bridge intensity
- closed-flow strength

## 5. Internal Flow
A community should have more weight inside than outside.
Use this as a key signal for suspicion scoring.

## 6. Bridge Nodes
A bridge node is a node that connects the community to the outside world.
Track bridge-like behavior because it often matters in laundering rings.

## 7. Implementation Rules
- Use sparse matrices.
- Keep all weight computations window-bound.
- Reuse the same node mapping across windows.
- Do not build heavy intermediate graph objects.
