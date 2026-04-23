# Project Context

## Project: AML Graph Mining Pipeline
Goal: detect suspicious money movement on temporal transaction data using two cooperating components:
1. Community Detection — narrow the investigation space.
2. Temporal Motif Mining — extract behavioral signals for feature engineering.

## Data
- AMLGentex synthetic benchmark
- Timestamped transaction events
- Daily and rolling-window views

## Required graph semantics
- Directed
- Weighted
- Time-aware
- Bounded-window processing only

## Current implementation focus
- Temporal graph creation is established.
- Window-based graph processing is the working unit.
- Community detection is being aligned with directed flow behavior.
- Motif mining is being aligned with temporal and monetary constraints.
- Feature extraction and evaluation remain downstream tasks.

## Design constraints
- Keep all processing memory-efficient for Colab.
- Avoid full in-memory graph construction.
- Preserve direction and time in every analysis step.
- Use graph structure only when necessary for bounded windows or local subgraphs.

## Outputs
- Suspicious entities
- Suspicious communities
- Temporal motifs
- ML-ready numeric features

## Success criteria
- Community detection must reduce the number of entities that need investigation.
- Motif mining must produce informative pattern features for later ML models.
- All methods must remain explainable, configurable, and feasible within Colab limits.
