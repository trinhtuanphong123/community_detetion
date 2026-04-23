---
trigger: always_on
---

# Coding Conventions

- Use vectorized operations (pandas, cuDF).
- Avoid iterrows().
- Use itertuples() only when necessary.

- Do not copy large DataFrames unnecessarily.
- Drop unused columns early.
- Use sparse structures for graphs.

- Keep functions small and focused.
- Use clear variable names: u, v, step, amount.

- Separate logic into:
  - preprocessing
  - graph building
  - motif mining
  - community detection

- Free large objects after use.