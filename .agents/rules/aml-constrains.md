---
trigger: always_on
---

# AML Constraints

- Treat transactions as a temporal directed weighted system.
- Always preserve:
  - direction
  - timestamp
  - amount

- Community detection:
  - must reflect flow behavior
  - must use weighted edges
  - must consider temporal windows
  - must track stability across time

- Motif mining:
  - must enforce direction order
  - must enforce time ordering
  - must enforce latency constraints
  - must enforce amount ratio constraints
  - must enforce minimum repetition

- Do not treat a single pattern as money laundering.
- Motifs and communities are signals, not final decisions.

- Avoid full in-memory graph construction.
- Prefer window-based processing.# AML Constraints

- Treat transactions as a temporal directed weighted system.
- Always preserve:
  - direction
  - timestamp
  - amount

- Community detection:
  - must reflect flow behavior
  - must use weighted edges
  - must consider temporal windows
  - must track stability across time

- Motif mining:
  - must enforce direction order
  - must enforce time ordering
  - must enforce latency constraints
  - must enforce amount ratio constraints
  - must enforce minimum repetition

- Do not treat a single pattern as money laundering.
- Motifs and communities are signals, not final decisions.

- Avoid full in-memory graph construction.
- Prefer window-based processing.