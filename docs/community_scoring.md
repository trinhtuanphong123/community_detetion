# Community Scoring Guide

## 1. Goal
Score communities so investigators can prioritize which groups to inspect first.

## 2. Core Score
Use a configurable suspiciousness score:

$$
S(C)=\alpha \cdot \text{InternalFlow}(C)
+\beta \cdot \text{Reciprocity}(C)
+\gamma \cdot \text{Persistence}(C)
+\delta \cdot \text{MotifEnrichment}(C)
-\eta \cdot \text{ExternalNoise}(C)
$$

## 3. Score Components
### InternalFlow
Amount of money that stays inside the community.

### Reciprocity
How much money moves in both directions inside the group.

### Persistence
How many consecutive windows the community survives.

### MotifEnrichment
How many suspicious motifs appear inside the community.

### ExternalNoise
How much weight leaks outside the community.

## 4. Filtering
Keep a community if it:
- is large enough to matter
- has strong internal flow
- persists across windows
- has suspicious motif enrichment
- exceeds the configured score threshold

## 5. Output Fields
At minimum, output:
- `window_id`
- `community_id`
- `size`
- `internal_flow`
- `reciprocity`
- `persistence`
- `motif_enrichment`
- `external_noise`
- `suspicion_score`

## 6. Implementation Rules
- Keep the score configurable.
- Do not hard-code one global threshold.
- Keep scoring bounded to the current window or a small set of windows.
