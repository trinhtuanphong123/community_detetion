# Project Context

This project builds a temporal AML detection pipeline from transaction events.

Input:
- timestamped transactions between entities
- daily or windowed event data

Output:
- risky entities
- suspicious communities
- suspicious temporal motifs
- ML-ready features for downstream models

Core goals:
- narrow the search space with community detection
- extract temporal motifs as behavioral signals
- convert graph patterns into numeric features
- avoid full in-memory graph construction for large datasets
