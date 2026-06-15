# src/community/__init__.py
from .config import CommunityConfig
from .weighting import (
    WindowGraph,
    build_window_graph,
    apply_weighting,
    symmetrize,
    compute_degrees,
)
from .detection import (
    detect_communities,
    compute_node_roles,
    build_relay_edges,
    labels_to_dataframe,
    split_large_communities,
    run_recursive_leiden,   # backward-compat alias
)
from .tracking import (
    match_communities_jaccard,
    build_tracking_record,
    update_buffer,
)
from .scoring import (
    extract_community_features,
    score_communities,
    get_shortlist,
)
from .pipeline import run_community_pipeline, save_pipeline_outputs

__all__ = [
    "CommunityConfig",
    # weighting
    "WindowGraph",
    "build_window_graph",
    "apply_weighting",
    "symmetrize",
    "compute_degrees",
    # detection
    "detect_communities",
    "compute_node_roles",
    "build_relay_edges",
    "labels_to_dataframe",
    "split_large_communities",
    "run_recursive_leiden",
    # tracking
    "match_communities_jaccard",
    "build_tracking_record",
    "update_buffer",
    # scoring
    "extract_community_features",
    "score_communities",
    "get_shortlist",
    # pipeline
    "run_community_pipeline",
    "save_pipeline_outputs",
]