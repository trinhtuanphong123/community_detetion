# Community Tracking Guide

## 1. Goal
Track communities across time windows so stable suspicious groups can be identified.

## 2. Window-to-Window Matching
Match communities using node overlap between consecutive windows.

A simple overlap score is:

$$
\text{overlap}(C_a, C_b) = \frac{|C_a \cap C_b|}{|C_a \cup C_b|}
$$

## 3. Persistent Community ID
Assign a persistent ID when overlap is high enough.
If overlap is too low, treat the community as new.

## 4. Tracking Outputs
Each window should keep:
- `window_id`
- `community_id`
- `persistent_community_id`
- community size
- overlap score with previous window

## 5. Split and Merge
Support the following simple cases:
- one community splitting into multiple smaller ones
- two communities merging into one

Do not overcomplicate the first version.
Track the dominant overlap path first.

## 6. Temporal Stability
A community is more suspicious if it appears across multiple windows.
Use persistence as a scoring input, not as a hard truth label.

## 7. Implementation Rules
- Compare only neighboring windows first.
- Keep the matching logic lightweight.
- Avoid keeping every history state in RAM.
- Save tracking tables incrementally if needed.
