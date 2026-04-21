# Skills / Tools Used

This project runs primarily on Google Colab with limited RAM and optional T4 GPU.

## Core Environment
- Google Colab (primary execution environment)
- Python 3.x

## Data Processing (Primary)
- pandas (default)
- polars (preferred for large datasets due to lower memory usage)

## Graph Processing (Restricted Use Only)
- NetworkX (ONLY for small, bounded subgraphs or short time windows)
- DO NOT use NetworkX to build full event graphs for the entire dataset

## Modeling
- scikit-learn
- XGBoost / LightGBM (CPU-based by default)

## Schema & Validation
- Pydantic (lightweight usage)

## GPU Usage (T4)
- GPU is NOT used for graph processing
- GPU may be used for:
  - model training (if supported)
  - large matrix operations
- Do NOT assume GPU accelerates NetworkX or pandas

## Key Constraints
- RAM is limited → avoid large in-memory structures
- Prefer:
  - chunked processing
  - window-based computation
  - vectorized operations
  - joins/groupby instead of graph traversal