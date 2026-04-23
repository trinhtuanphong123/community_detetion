# Skills / Tools Used

## Execution environment
- Google Colab
- T4 GPU available
- Python 3.x
- Limited RAM, so every step must be window-aware

## Primary stack
- pandas / polars — tabular preprocessing
- scipy.sparse — sparse adjacency and matrix operations
- networkx — bounded local graph debugging and visualization only
- python-igraph — directed community / flow analysis when needed
- scikit-learn — feature modeling and evaluation
- XGBoost / LightGBM — downstream ML
- Pydantic — schema validation for transaction records

## Optional acceleration
- cuDF may be used for tabular preprocessing only if the notebook path already supports it cleanly.
- Do not rely on GPU for full-graph construction or large graph algorithms.

## Graph processing
- Use windowed graphs as the default unit of work.
- Keep graphs directed, weighted, and time-aware.
- Use NetworkX only for small subgraphs and debug checks.
- Prefer sparse representations over dense matrices.
- Avoid full-dataset in-memory graph objects.

## Data handling
- Read and process only the columns needed for the current step.
- Prefer groupby / merge / filter operations over custom loops.
- Export intermediate results with parquet or csv when needed.

## Validation
- Validate transaction schema before graph building.
- Ensure time, direction, and amount fields are present and consistent.
