


## Environment Constraints
- Runs on Google Colab
- Limited RAM
- T4 GPU available but not for graph processing

## Input
- Timestamped transaction events
- Daily or windowed data

## Output
- Suspicious entities
- Suspicious communities
- Temporal motifs
- ML-ready features

## Core Technical Goals
- Avoid full in-memory graph construction
- Use window-based processing
- Extract behavioral patterns efficiently
- Convert patterns into numeric features

## Design Constraint
All processing must be memory-efficient and runnable within Colab limits.