"""
tracking.py — Cross-window community identity tracking.

Assigns persistent community IDs across consecutive time windows using
Jaccard-based node overlap matching.

Public API:

    match_communities_jaccard(curr_labels, tracking_buffer, cfg, counter)
        Core matching function. Compares current window's communities
        against a rolling buffer of recent windows.
        Returns (new_labels_with_global_ids, updated_counter).

    build_tracking_record(global_labels, window_id, step_start, step_end)
        Produces the per-window tracking output table (tracking.md §4).

    update_buffer(buffer, record_df, cfg)
        Maintain the rolling tracking buffer (capped at tracking_memory).

Matching rules (tracking.md §2–5):
    - Jaccard(C_a, C_b) = |C_a ∩ C_b| / |C_a ∪ C_b|
    - If Jaccard >= cfg.jaccard_thresh  → inherit persistent ID from best match
    - Otherwise                         → assign new persistent ID
    - Buffer spans cfg.tracking_memory past windows (oldest first).
      A community that "disappears" for ≤ tracking_memory windows can still
      match when it re-appears.

Split / merge handling (tracking.md §5):
    - Dominant overlap path: a current community inherits the ID of
      whichever past community it overlaps with most strongly.
    - If two current communities both best-match the same past community,
      the one with higher Jaccard takes the ID; the other gets a new ID.

Guarantees:
    - Time      : processing is sequential; buffer contains only past windows.
    - Direction : node IDs come from detection.py which uses WindowGraph (directed).
    - Memory    : buffer is capped; only node→global_cid mappings retained (not full graphs).
"""

from __future__ import annotations

import gc
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import CommunityConfig


# ---------------------------------------------------------------------------
# Core matcher
# ---------------------------------------------------------------------------

def match_communities_jaccard(
    curr_labels: Dict[int, int],
    tracking_buffer: List[pd.DataFrame],
    cfg: CommunityConfig,
    global_cid_counter: int,
) -> Tuple[Dict[int, int], int]:
    """
    Assign persistent (global) community IDs to the current window's communities.

    Parameters
    ----------
    curr_labels : Dict[node_id, local_community_id]
        Output of detect_communities() for the current window.
        Local community IDs are window-local (not persistent).
    tracking_buffer : List[pd.DataFrame]
        Each entry is a DataFrame with columns [node, global_cid] from a
        past window. Oldest first; length capped at cfg.tracking_memory.
    cfg : CommunityConfig
        jaccard_thresh and tracking_memory are used here.
    global_cid_counter : int
        Next unused persistent community ID.

    Returns
    -------
    (global_labels, updated_counter)
        global_labels : Dict[node_id, global_community_id]
        updated_counter : int — new value after assigning any new IDs.

    Notes
    -----
    - Only plain Python sets and pandas operations; no cuDF.
    - When two current communities best-match the same past community,
      the higher-Jaccard one inherits the ID; the other gets a new one.
    - Nodes filtered out by detect_communities (cid == -1) keep cid == -1.
    """
    # Separate out noise / filtered nodes (community -1)
    active = {n: c for n, c in curr_labels.items() if c != -1}

    if not active:
        return curr_labels, global_cid_counter

    # Group current nodes by local community ID
    curr_sets: Dict[int, set] = {}
    for node, cid in active.items():
        curr_sets.setdefault(cid, set()).add(node)

    local_cids = list(curr_sets.keys())

    # No history → assign fresh IDs to all
    if not tracking_buffer:
        mapping = {}
        for cid in local_cids:
            mapping[cid] = global_cid_counter
            global_cid_counter += 1
        global_labels = {n: (mapping[c] if c != -1 else -1) for n, c in curr_labels.items()}
        return global_labels, global_cid_counter

    # Build node → global_cid lookup from buffer (most-recent window wins)
    prev_lookup: Dict[int, int] = {}
    for buf_df in tracking_buffer:           # oldest first
        for _, row in buf_df.iterrows():
            prev_lookup[int(row["node"])] = int(row["global_cid"])

    # Group previous nodes by global_cid
    prev_sets: Dict[int, set] = {}
    for node, gcid in prev_lookup.items():
        prev_sets.setdefault(gcid, set()).add(node)

    prev_gcids = list(prev_sets.keys())

    # Compute Jaccard for all (curr_cid, prev_gcid) pairs — vectorized
    best_match: Dict[int, Tuple[int, float]] = {}   # local_cid → (best_gcid, best_j)

    for local_cid, curr_nodes in curr_sets.items():
        best_j    = -1.0
        best_gcid = None

        for gcid, prev_nodes in prev_sets.items():
            inter = len(curr_nodes & prev_nodes)
            if inter == 0:
                continue
            union = len(curr_nodes | prev_nodes)
            j = inter / union
            if j > best_j:
                best_j    = j
                best_gcid = gcid

        if best_gcid is not None and best_j >= cfg.jaccard_thresh:
            best_match[local_cid] = (best_gcid, best_j)

    # Resolve conflicts: two local communities claim the same past global ID
    claimed: Dict[int, Tuple[int, float]] = {}     # gcid → (local_cid, jaccard)
    for local_cid, (gcid, j) in best_match.items():
        if gcid not in claimed or j > claimed[gcid][1]:
            claimed[gcid] = (local_cid, j)

    # Build local_cid → global_cid mapping
    winner_locals = {local_cid for (local_cid, _) in claimed.values()}
    local_to_global: Dict[int, int] = {}

    for local_cid in local_cids:
        if local_cid in winner_locals:
            # Find which gcid this local_cid won
            for gcid, (winning_local, _) in claimed.items():
                if winning_local == local_cid:
                    local_to_global[local_cid] = gcid
                    break
        else:
            # No match or lost conflict → new persistent ID
            local_to_global[local_cid] = global_cid_counter
            global_cid_counter += 1

    global_labels = {
        n: (local_to_global[c] if c != -1 else -1)
        for n, c in curr_labels.items()
    }
    return global_labels, global_cid_counter


# ---------------------------------------------------------------------------
# Tracking record builder
# ---------------------------------------------------------------------------

def build_tracking_record(
    global_labels: Dict[int, int],
    window_id: int,
    step_start: int = 0,
    step_end: int = 0,
    prev_buffer: List[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    Build the per-window tracking output table (tracking.md §4).

    Output columns:
        window_id, step_start, step_end,
        node, global_cid, community_size, overlap_score

    overlap_score is the max Jaccard between this community and any community
    in the previous buffer entry.  0.0 if no buffer provided or no match.

    Parameters
    ----------
    global_labels : Dict[node_id, global_community_id]
    window_id : int
    step_start, step_end : int
    prev_buffer : List[pd.DataFrame], optional
        Used only to compute overlap_score for reporting.

    Returns
    -------
    pd.DataFrame with columns:
        window_id, step_start, step_end, node, global_cid,
        community_size, overlap_score
    """
    if not global_labels:
        return pd.DataFrame(columns=[
            "window_id", "step_start", "step_end", "node",
            "global_cid", "community_size", "overlap_score",
        ])

    # Community sizes
    size_map: Dict[int, int] = {}
    for gcid in global_labels.values():
        if gcid != -1:
            size_map[gcid] = size_map.get(gcid, 0) + 1

    # Overlap scores (optional, reported per community)
    overlap_map: Dict[int, float] = {}
    if prev_buffer:
        last = prev_buffer[-1]  # most recent past window
        prev_lookup: Dict[int, int] = {}
        for _, row in last.iterrows():
            prev_lookup[int(row["node"])] = int(row["global_cid"])

        prev_sets: Dict[int, set] = {}
        for node, gcid in prev_lookup.items():
            prev_sets.setdefault(gcid, set()).add(node)

        curr_sets: Dict[int, set] = {}
        for node, gcid in global_labels.items():
            if gcid != -1:
                curr_sets.setdefault(gcid, set()).add(node)

        for gcid, curr_nodes in curr_sets.items():
            best_j = 0.0
            for prev_nodes in prev_sets.values():
                inter = len(curr_nodes & prev_nodes)
                if inter == 0:
                    continue
                j = inter / len(curr_nodes | prev_nodes)
                if j > best_j:
                    best_j = j
            overlap_map[gcid] = round(best_j, 4)

    rows = []
    for node, gcid in global_labels.items():
        rows.append({
            "window_id":      window_id,
            "step_start":     step_start,
            "step_end":       step_end,
            "node":           int(node),
            "global_cid":     int(gcid),
            "community_size": size_map.get(gcid, 0),
            "overlap_score":  overlap_map.get(gcid, 0.0),
        })

    return pd.DataFrame(rows).sort_values(["global_cid", "node"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Buffer management
# ---------------------------------------------------------------------------

def update_buffer(
    buffer: List[pd.DataFrame],
    record_df: pd.DataFrame,
    cfg: CommunityConfig,
) -> List[pd.DataFrame]:
    """
    Append the current window's node→global_cid mapping to the buffer,
    and trim to the last cfg.tracking_memory entries.

    Parameters
    ----------
    buffer : List[pd.DataFrame]
        Existing buffer (modified in-place conceptually, returned explicitly).
    record_df : pd.DataFrame
        Output of build_tracking_record() for the current window.
    cfg : CommunityConfig

    Returns
    -------
    Updated buffer list.
    """
    # Keep only [node, global_cid] for the buffer — discard window metadata
    if "node" in record_df.columns and "global_cid" in record_df.columns:
        slim = record_df[["node", "global_cid"]].copy()
        slim = slim[slim["global_cid"] != -1]   # skip noise nodes
    else:
        slim = pd.DataFrame(columns=["node", "global_cid"])

    buffer.append(slim)

    # Cap buffer size
    if len(buffer) > cfg.tracking_memory:
        removed = buffer.pop(0)
        del removed

    gc.collect()
    return buffer


__all__ = [
    "match_communities_jaccard",
    "build_tracking_record",
    "update_buffer",
]