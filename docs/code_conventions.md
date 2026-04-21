

# Code Conventions

## Core Principle
Optimize for memory efficiency and scalability in a Colab environment.

## Required Practices
- Use vectorized operations (pandas/polars) whenever possible
- Prefer groupby, joins, and filtering over graph traversal
- Process data in time windows instead of full dataset
- Use chunking for large datasets

## Graph Usage Rules
- DO NOT build full MultiDiGraph on entire dataset
- ONLY build graph for:
  - small time windows
  - local subgraph analysis
- Always justify why a graph is needed before using it

## Performance Rules
- Avoid iterrows()
- Prefer itertuples() or vectorized operations
- Avoid storing large intermediate lists in memory
- Drop unused columns early

## Memory Rules
- Never duplicate full datasets in memory
- Avoid copying DataFrames unless necessary
- Use in-place operations when safe

## Change Discipline
- Minimal changes only
- No unnecessary refactoring
- Match existing code style