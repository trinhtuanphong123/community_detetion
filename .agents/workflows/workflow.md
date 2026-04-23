---
description: 
---

# Workflow Rules

- Identify the task type before coding:
  - graph construction
  - community detection
  - motif mining
  - feature extraction

- Define input and output before implementation.
- Choose the correct data representation:
  - event-level for motif mining
  - window graph for community detection

- Implement in steps:
  1. preprocess
  2. graph/window construction
  3. algorithm
  4. output

- Validate after each step:
  - direction preserved
  - time ordering preserved
  - aggregation correct

- Process data in windows.
- Do not load the full dataset into complex structures.