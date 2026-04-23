# AMLGentex Dataset (sweden_100K_dist_change_difficult)

This project uses the dataset:

AMLGentex/sweden_100K_dist_change_difficult

## Dataset Nature

- Synthetic AML dataset (not real banking data)
- ~100K transactions
- Simulates financial activity in a Sweden-like environment
- Includes both normal and suspicious transaction patterns

## Key Characteristics

- Temporal transaction data (event-based)
- Directed transactions between entities
- Contains AML typologies (e.g., layering, structuring)
- Includes **distribution shift over time** (concept drift)
- “difficult” variant → patterns are less obvious and noisier

## Labels

- A binary suspicious label is provided (e.g., `isSAR`)
- Labels are **not perfect ground truth**
- Treat labels as:
  - training signal
  - evaluation reference
  - NOT absolute truth

## Canonical Schema Used in This Project

All data must be normalized into:

| field | type | description |
|------|------|------------|
| tx_id | string/int | unique transaction id |
| timestamp | datetime | event timestamp |
| step | int | discrete time index (e.g., day) |
| src_node | int | sender entity |
| dst_node | int | receiver entity |
| amount | float | transaction amount |
| normalized_amount | float | scaled amount (optional) |
| type | string | transaction type |
| is_sar | int | suspicious label (0/1) |

## Important Constraints

- Transactions must be processed in **time order**
- Do NOT shuffle data randomly (breaks temporal patterns)
- Distribution changes across time → models must be robust to drift
- Do NOT assume stationarity

## Usage in This Project

### For Community Detection
- Build graph per time window
- Detect dense or abnormal subgraphs
- Use communities as candidate suspicious regions

### For Motif Mining
- Extract temporal motifs within windows
- Use:
  - time ordering
  - amount relationships
  - repetition
- Do NOT rely on topology only

### For Feature Engineering
- Convert motifs and community signals into numeric features
- Aggregate per:
  - node
  - community
  - time window

## Processing Rules

- Prefer window-based processing (7d / 30d)
- Avoid building full graph in memory
- Use pandas/polars operations where possible
- Build graph objects ONLY for local analysis

## Notes for Claude

- This dataset contains concept drift → patterns change over time
- Suspicious behavior is not defined by a single rule
- A motif is only meaningful when:
  - timing + amount + repetition align
- Always prioritize memory-efficient processing (Colab environment)