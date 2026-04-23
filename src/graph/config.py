"""
config.py — Configuration dataclasses for graph construction.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GraphConfig:
    """Hyperparameters for temporal graph construction."""
    delta_w: int = 5          # max step gap for temporal edges (4.1)
    window_size: int = 30     # steps per processing window
    window_stride: int = 15   # overlap = window_size - window_stride
