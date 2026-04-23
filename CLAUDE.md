# CLAUDE.md

## Project: AML Graph Mining Pipeline

## Source of truth
Use these docs as the working reference, in this order when there is overlap:
1. `.cursorrule.md`
2. `.cursorrule.conventions.md`
3. `project_context.md`
4. `skills_tools.md`
5. Any task-specific spec or notebook already in the repo

## Operating principles
- Prefer correctness, explainability, and low memory use over cleverness.
- Keep changes minimal and local to the request.
- Do not refactor unrelated code.
- Do not invent abstractions that the task does not need.
- Surface uncertainty instead of assuming.

## AML domain rules
- AML is risk-based detection, not deterministic classification.
- Community detection produces candidate suspicious groups, not final truth.
- Motifs are suspicious only when topology, timing, amount, repetition, and context align.
- Thresholds must remain configurable and data-dependent.

## Graph and data rules
- Preserve time, direction, and weight.
- Prefer windowed processing, joins, groupby, and local subgraph extraction.
- Do not build a full dataset-wide MultiDiGraph.
- Use graph structures only for bounded windows or local analysis.
- Treat transactions as an event stream first.

## Environment rules
- Code must run in Google Colab with limited RAM.
- T4 GPU may be present, but graph processing must not depend on it.
- Keep notebook cells runnable independently when practical.

## Implementation guidance
- State assumptions when they matter.
- Prefer explicit code over speculative helpers.
- Remove only imports or variables that become unused because of the current change.
- If a requested design is likely to fail at scale, push back with a simpler alternative.

## Task execution style
For multi-step tasks, use a brief plan and verify each step against the requested outcome.

## Project modules
- `community/` for community detection logic
- `motif/` for temporal motif mining logic
- Feature extraction should consume outputs from both components

## Prohibited patterns
- Full in-memory graph construction on the whole dataset
- Dense matrix representations when sparse is enough
- Row-wise Python loops over large tables
- Hidden changes outside the requested scope
