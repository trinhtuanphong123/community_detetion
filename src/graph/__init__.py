"""
src/graph/__init__.py

Public API for the graph module.

Components implemented:
  - GraphConfig           : pipeline-level hyperparameters
  - LoaderConfig          : data loading configuration
  - load_transactions     : load and normalize AMLGentex data
  - iter_windows          : windowed iteration over a transaction table
  - NodeEncoder           : raw → encoded integer ID mapping
  - encode_series         : module-level helper for single-Series encoding
  - build_temporal_edges  : relay self-join for motif mining
  - build_snapshot_edges  : directed weighted edge aggregation per window
  - build_second_order_edges : multi-hop relay aggregation
  - build_snapshot_graph  : directed sparse adjacency matrix for community detection
"""

from src.graph.config import GraphConfig, LoaderConfig
from src.graph.loader import load_transactions, iter_windows
from src.graph.encoder import NodeEncoder, encode_series
from src.graph.temporal import build_temporal_edges, build_snapshot_edges
from src.graph.second_order import build_second_order_edges, build_snapshot_graph

__all__ = [
    "GraphConfig",
    "LoaderConfig",
    "load_transactions",
    "iter_windows",
    "NodeEncoder",
    "encode_series",
    "build_temporal_edges",
    "build_snapshot_edges",
    "build_second_order_edges",
    "build_snapshot_graph",
]
