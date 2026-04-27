"""
config.py — Community detection configuration.

All thresholds are configurable dataclass fields.
No threshold is hardcoded in detection, scoring, or tracking modules.

Parameter groups:
    Graph weighting    : alpha, beta, weight_filter_thresh, top_k_neighbors
    Community detection: method, min_comm_size, resolution, lambda_temporal
    Recursive split    : s_max, resolution_multiplier, max_recursion_depth
    Role classification: role_threshold, layering_consistency
    Temporal tracking  : window_size, stride, jaccard_thresh, tracking_memory
    Suspicion scoring  : score weights and filtering thresholds
    Output             : top_k_export
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class CommunityConfig:
    """
    All hyperparameters for the community detection pipeline.

    Change here, never hardcode in detection / scoring / tracking modules.

    Graph weighting
    ---------------
    alpha : float
        Monetary continuity penalty — exp(-alpha * |log(a_src / a_dst)|).
        Higher = stricter capital preservation requirement.
    beta : float
        Temporal decay penalty — exp(-beta * avg_gap / delta_w).
        Higher = edges further apart in time get lower weight.
    weight_filter_thresh : float
        Drop second-order edges with final weight below this value.
    top_k_neighbors : int
        Keep at most this many outgoing neighbours per node after weighting.
        0 = disabled (keep all).

    Community detection
    -------------------
    method : str
        Primary algorithm. Options: "leiden", "infomap".
        "leiden"  — directed modularity, good for structural communities.
        "infomap" — flow-based, good for detecting money-trapping patterns.
    min_comm_size : int
        Discard communities smaller than this. Prevents trivial singletons.
    resolution : float
        Leiden resolution parameter. Higher = more, smaller communities.
    lambda_temporal : float
        Temporal regularisation strength (λ).
        Penalty for label changes between consecutive windows.
        0 = no regularisation; 0.5 = moderate stability pressure.

    Recursive split
    ---------------
    s_max : int
        Communities larger than this are recursively split.
    resolution_multiplier : float
        Multiply resolution by this at each recursion level.
    max_recursion_depth : int
        Hard stop on recursion depth.

    Role classification
    -------------------
    role_threshold : float
        |net_flow_ratio| > threshold → node classified as source (>0) or sink (<0).
    layering_consistency : float
        flow_consistency > threshold → node classified as layering intermediary.

    Temporal tracking
    -----------------
    window_sizes : list[int]
        Rolling-window sizes in steps, aligned with GraphConfig.
    stride : int
        Step increment between windows.
    jaccard_thresh : float
        Minimum Jaccard overlap to consider two communities the same entity
        across consecutive windows.
    tracking_memory : int
        Number of past windows to match against during tracking.

    Suspicion scoring
    -----------------
    Weights for the composite suspicion score S(C):
        S(C) = w_internal_flow  * InternalFlow
             + w_reciprocity    * Reciprocity
             + w_persistence    * Persistence
             + w_motif          * MotifEnrichment
             - w_external_noise * ExternalNoise
    All weights should sum to approximately 1.0.

    alert_thresh : float
        Flag community if SAR-edge ratio exceeds this value.
    min_flow_for_scoring : float
        Skip scoring communities with total internal flow below this.
    global_susp_threshold : float
        Communities with S(C) above this are added to the investigation shortlist.

    size_penalty_gamma : float
        Log size-penalty coefficient in the suspicion score.
        Reduces the score of large communities to prevent size-bias.
        S_penalised = S - gamma * log(1 + |C|).
        Configurable here so experiments can turn it off (set to 0.0).

    Output
    ------
    top_k_export : int
        Number of highest-scoring communities to include in the export.
    """

    # ── Graph weighting ───────────────────────────────────────────────────────
    alpha: float = 1.0
    beta: float = 0.5
    weight_filter_thresh: float = 0.01
    top_k_neighbors: int = 10

    # ── Community detection ───────────────────────────────────────────────────
    method: str = "leiden"          # "leiden" or "infomap"
    min_comm_size: int = 3
    resolution: float = 1.0
    lambda_temporal: float = 0.3    # temporal regularisation strength

    # ── Recursive split ───────────────────────────────────────────────────────
    s_max: int = 50
    resolution_multiplier: float = 1.5
    max_recursion_depth: int = 3

    # ── Role classification ───────────────────────────────────────────────────
    role_threshold: float = 0.3
    layering_consistency: float = 0.8

    # ── Temporal tracking ─────────────────────────────────────────────────────
    window_sizes: List[int] = field(default_factory=lambda: [7, 14, 30])
    stride: int = 7
    jaccard_thresh: float = 0.25
    tracking_memory: int = 3

    # ── Suspicion scoring weights ─────────────────────────────────────────────
    w_internal_flow: float = 0.30
    w_reciprocity: float = 0.20
    w_persistence: float = 0.25
    w_motif: float = 0.15
    w_external_noise: float = 0.10

    alert_thresh: float = 0.30
    min_flow_for_scoring: float = 10_000.0
    global_susp_threshold: float = 0.45

    # ── Output ────────────────────────────────────────────────────────────────
    top_k_export: int = 50
    size_penalty_gamma: float = 0.05   # log size-penalty coefficient in score_communities


__all__ = ["CommunityConfig"]