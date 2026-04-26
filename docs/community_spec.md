# Community Detection Spec

## 1. Purpose
This document defines the community detection contract for the AML Graph Mining project.

A community is a candidate suspicious transaction cluster, not final truth.
It must help narrow the investigation space while preserving:
- direction
- weight
- time
- persistence across windows

## 2. Scope
Community detection applies to:
- snapshot graphs by step
- rolling-window graphs
- directed weighted graphs
- bounded local subgraphs

It does not apply to:
- full in-memory graphs for the full dataset
- SAR-driven label generation
- undirected clustering that loses flow meaning

## 3. Required Inputs
Use the graph outputs produced by `src/graph`:
- `daily_edges` or `window_edges`
- `node2idx`
- sparse adjacency matrix `A^(t)` or an equivalent bounded representation

The data must remain:
- time-aware
- directed
- weighted
- memory-efficient

## 4. Required Outputs
Each window should produce:
- community assignment table
- community score table
- persistent community mapping
- shortlist of suspicious communities

## 5. Primary Success Criteria
A community is useful if it:
- reduces the number of entities requiring review
- groups suspicious entities into smaller clusters
- preserves flow direction and time
- remains stable across windows
- supports later motif enrichment
