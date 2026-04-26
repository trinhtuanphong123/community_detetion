"""
config.py — Motif pipeline configuration.

All thresholds are configurable dataclass fields.
No threshold is hardcoded inside matchers.
Calibrate DELTA, RHO_MIN, RHO_MAX against the real amount distribution
before treating defaults as ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class MotifConfig:
    """
    Hyperparameters for temporal motif mining.

    Temporal constraints
    --------------------
    delta : int
        Maximum step gap between two consecutive transactions in a motif.
        AML relay chains typically close within a few days; default 3 is
        intentionally tight. Widen only if dataset step != 1 day.

    Amount ratio constraints
    ------------------------
    rho_min, rho_max : float
        Allowed ratio a(i+1) / a(i) for consecutive transactions.
        Layering keeps amounts close to 1.0.
        Structuring splits downward (ratio < 1).
        [0.5, 2.0] is a starting range; tighten toward [0.7, 1.4] for
        capital-preservation behavior typical of layering.

    Minimum support (repetition)
    ----------------------------
    r_min_fanin  : minimum number of incoming sources for a fan-in node.
    r_min_fanout : minimum number of outgoing targets for a fan-out node.
    r_min_cycle  : minimum occurrences of a cycle before it is flagged.
    r_min_relay  : minimum occurrences of a relay chain.
    r_min_split_merge : minimum occurrences of a split-merge path.

    Statistical filtering
    ---------------------
    n_permutations : shuffles for null-model z-score estimation.
        30 is sufficient for early screening; raise to 100+ for final runs.
    z_min : z-score threshold below which a motif is discarded.

    Search limits
    -------------
    max_nodes : maximum nodes in a single motif instance (prune guard).
    max_edges : maximum edges in a single motif instance (prune guard).

    Window sizes
    ------------
    window_sizes : step counts for windowed search, aligned with graph module.
        [7, 14, 30] matches graph_schema §6.3 and community_spec §6.
    """

    # ── Temporal ─────────────────────────────────────────────────────────────
    delta: int = 3

    # ── Amount ratio ─────────────────────────────────────────────────────────
    rho_min: float = 0.5
    rho_max: float = 2.0

    # ── Support thresholds (per motif type) ──────────────────────────────────
    r_min_fanin:      int = 3   # at least 3 distinct sources into one node
    r_min_fanout:     int = 3   # at least 3 distinct targets from one node
    r_min_cycle:      int = 1   # a single observed cycle is already a signal
    r_min_relay:      int = 1
    r_min_split_merge: int = 1

    # ── Statistical filtering ─────────────────────────────────────────────────
    n_permutations: int = 30
    z_min: float = 2.0

    # ── Search limits (prune guard) ───────────────────────────────────────────
    max_nodes: int = 4
    max_edges: int = 5

    # ── Window sizes (aligned with GraphConfig.window_sizes) ─────────────────
    window_sizes: List[int] = field(default_factory=lambda: [7, 14, 30])


__all__ = ["MotifConfig"]