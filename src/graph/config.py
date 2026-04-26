"""
config.py — Configuration for the AML temporal graph pipeline.

All thresholds are configurable; none are hardcoded in logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class GraphConfig:
    """
    Hyperparameters for temporal graph construction.

    window_sizes : standard analysis windows (days/steps), per graph_schema §6.3
    window_stride : step increment between rolling windows (overlap = size - stride)
    delta_w : max step gap allowed between two linked temporal events
    min_amount : filter out transactions at or below this value
    """

    # Rolling-window sizes used by community detection (§6.3)
    window_sizes: List[int] = field(default_factory=lambda: [7, 14, 30])

    # Default window used by the loader iterator
    window_size: int = 30
    window_stride: int = 15

    # Max step gap for a temporal edge (motif / second-order links)
    delta_w: int = 5

    # Transactions with amount <= this are discarded
    min_amount: float = 0.0


@dataclass
class LoaderConfig:
    """
    Configuration for loading and normalizing AMLGentex raw data.

    column_map : rename raw columns → canonical names
    keep_cols  : columns to retain after normalization (drop everything else)
    dtypes     : cast canonical columns to memory-efficient types
    window_size, window_stride : forwarded to iter_windows default
    """

    # AMLGentex column names → canonical schema (graph_schema §3.2)
    column_map: dict = field(default_factory=lambda: {
        # AMLSim / IBM-style headers (CSV variant)
        "Timestamp":          "timestamp",
        "From Bank":          "src_bank",
        "Account":            "src_node",
        "To Bank":            "dst_bank",
        "Account.1":          "dst_node",
        "Amount Received":    "amount",
        "Receiving Currency": "currency",
        "Is Laundering":      "is_sar",
        # AMLGentex / PaySim-style headers (alternative CSV variant)
        "step":               "step",
        "nameOrig":           "src_node",
        "nameDest":           "dst_node",
        "amount":             "amount",
        "type":               "type",
        "isSAR":              "is_sar",
        "oldbalanceOrig":     "old_bal_src",
        "newbalanceOrig":     "new_bal_src",
        "oldbalanceDest":     "old_bal_dst",
        "newbalanceDest":     "new_bal_dst",
    })

    # Columns to keep in the normalized DataFrame (§3.3)
    keep_cols: List[str] = field(default_factory=lambda: [
        "src_node", "dst_node", "amount", "step", "is_sar",
    ])

    # Memory-efficient dtypes (§3.2 recommended types)
    dtypes: dict = field(default_factory=lambda: {
        "step":     "int32",
        "amount":   "float32",
        "is_sar":   "int8",
        "src_node": "int64",
        "dst_node": "int64",
    })

    window_size: int = 30
    window_stride: int = 15
