# Code Conventions

- Make the smallest change that solves the task.
- Do not refactor unrelated code.
- Do not build a full in-memory MultiDiGraph for the entire dataset unless explicitly required.
- Prefer window-based processing, joins, groupby, and local subgraph extraction.
- Keep changes aligned with the existing style.
- Remove only imports/variables/functions made unused by the change itself.
- Prefer explicit, readable code over speculative abstractions.
- Keep memory-heavy graph objects bounded to a window, a community, or a local candidate subgraph.
