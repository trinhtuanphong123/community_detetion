# Community Pipeline Guide

## 1. Execution Order
Use this order for the community pipeline:

1. Load the windowed graph data.
2. Build sparse directed weighted adjacency.
3. Run community detection.
4. Track communities across windows.
5. Score communities.
6. Export tables for downstream use.

## 2. Data Flow
Input:
- graph outputs from `src/graph`
- optional motif summaries for enrichment

Intermediate:
- sparse window adjacency
- community labels
- persistent community mapping

Output:
- community assignment table
- community score table
- suspicious community shortlist

## 3. File Responsibilities
- `config.py`: thresholds, method flags, and window settings
- `weighting.py`: weight preparation and graph weight metrics
- `detection.py`: directed modularity, spectral, and Infomap detection
- `tracking.py`: persistence and community matching across windows
- `scoring.py`: suspicion score and filtering
- `pipeline.py`: orchestration and export
- `__init__.py`: public exports

## 4. Memory Rules
- Process window by window.
- Use sparse matrices.
- Avoid keeping all windows in memory.
- Write intermediate results to disk if necessary.

## 5. Colab Rules
- Keep the implementation runnable in Colab.
- Prefer simple and bounded computations.
- Do not assume full-graph GPU community processing is available.
