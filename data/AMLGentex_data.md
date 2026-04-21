# AMLGentex Data Overview

AMLGentex is the synthetic AML benchmark data family used in this project. It is designed to generate realistic, configurable transaction data for anti-money-laundering research and evaluation.

## Purpose in this repo

Use AMLGentex as:
- source data for graph construction
- input for community detection
- input for temporal motif mining
- training / validation data for ML models

## Data philosophy

This project should treat AMLGentex as an event stream first, not as a full in-memory graph.

Recommended processing:
- read raw transactions
- normalize columns
- create time windows
- derive local graph views only when needed
- convert graph patterns into numeric features

## Recommended folder usage

- `data/raw/`  
  Unmodified AMLGentex exports.

- `data/processed/`  
  Cleaned tables with normalized schema.

- `data/sample/`  
  Small synthetic subset for quick tests and debugging.

## Canonical transaction schema

The repo should normalize AMLGentex transactions into a common schema like this:

| field | type | meaning |
|---|---|---|
| `tx_id` | string | unique transaction id |
| `timestamp` | datetime | event time in UTC or normalized local time |
| `step` | int | discrete time bucket, often daily |
| `src_node` | int/string | sender entity id |
| `dst_node` | int/string | receiver entity id |
| `amount` | float | transaction amount in source currency |
| `normalized_amount` | float | amount normalized to a common scale |
| `currency` | string | currency code |
| `type` | string | transaction type if available |
| `channel` | string | channel / rail if available |
| `label` | int/nullable | suspicious label if available |
| `scenario` | string/nullable | simulation or typology tag if available |

## Example record

```json
{
  "tx_id": "tx_000001",
  "timestamp": "2024-01-03T09:15:00Z",
  "step": 3,
  "src_node": 12,
  "dst_node": 47,
  "amount": 2500.0,
  "normalized_amount": 2500.0,
  "currency": "USD",
  "type": "transfer",
  "channel": "bank_transfer",
  "label": 0,
  "scenario": "baseline"
}
```

## Minimal processing rules

- Keep only fields needed for graph / motif / feature generation.
- Drop unused columns early.
- Never duplicate the full dataset in memory.
- Build graphs only inside a bounded window or a local subgraph.
- Use the processed table as the source for both community detection and motif mining.

## Notes for Claude

When working with AMLGentex data in this repo:
- prefer window-based logic over full-graph logic
- keep memory use low for Colab execution
- treat suspicious labels as training signals, not perfect truth
- preserve temporal order when deriving motifs
