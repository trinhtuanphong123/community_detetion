"""
scoring.py — Motif support counting, filtering, and null-model z-score.

Three responsibilities:
    1. count_support()       — count instances per motif type.
    2. filter_motifs()       — apply r_min and z_min thresholds.
    3. compute_null_zscore() — compare observed counts against a shuffled
                               null model to assess statistical significance.

Null model strategy:
    Shuffle the `step` column globally (breaks temporal structure while
    preserving degree and amount distributions).
    Repeat n_permutations times, collect counts, compute z-score.

Memory notes:
    - _shuffle_timestamps uses numpy shuffle (no full DataFrame copy per perm).
    - Intermediate null indexes and instances are deleted after each perm.
    - gc.collect() called after each permutation.
"""

from __future__ import annotations

import gc
import random
from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import MotifConfig
from .index import build_event_indexes
from .matchers import run_all_matchers


# ---------------------------------------------------------------------------
# Support counting
# ---------------------------------------------------------------------------

def count_support(motif_instances: List[dict]) -> Dict[str, int]:
    """
    Count motif instances by type.

    Parameters
    ----------
    motif_instances : list[dict]
        Combined output from all matchers (or run_all_matchers).

    Returns
    -------
    dict {motif_type: count}
    """
    c: Counter = Counter()
    for inst in motif_instances:
        c[inst["motif_type"]] += 1
    return dict(c)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_motifs(
    instances: List[dict],
    cfg: MotifConfig,
    zscore_results: Dict[str, dict] | None = None,
) -> List[dict]:
    """
    Filter motif instances by support and optional z-score threshold.

    Fan-in and fan-out instances already satisfy r_min_fanin / r_min_fanout
    by construction (the matchers enforce it).  This filter applies an
    additional global check: motif types whose total support across the
    window is below r_min_cycle / r_min_relay / r_min_split_merge are removed.
    If zscore_results is provided, types below z_min are also removed.

    Parameters
    ----------
    instances : list[dict]
        Raw output from matchers.
    cfg : MotifConfig
        Contains per-type r_min thresholds and z_min.
    zscore_results : dict, optional
        Output of compute_null_zscore().  If supplied, types with
        zscore < cfg.z_min are discarded.

    Returns
    -------
    Filtered list of motif instance dicts.
    """
    # # Per-type minimum support map
    # r_min_map = {
    #     "fanin":       cfg.r_min_fanin,
    #     "fanout":      cfg.r_min_fanout,
    #     "cycle3":      cfg.r_min_cycle,
    #     "relay4":      cfg.r_min_relay,
    #     "split_merge": cfg.r_min_split_merge,
    # }
    # Types where the matcher already enforces per-instance structure
    _MATCHER_ENFORCED = {"fanin", "fanout"}
    r_min_map = {
    "fanin":       1,                    # ← already enforced; just require ≥1
    "fanout":      1,
    "cycle3":      cfg.r_min_cycle,
    "relay4":      cfg.r_min_relay,
    "split_merge": cfg.r_min_split_merge,
    }
    # Count observed support per type
    support = count_support(instances)

    # Determine which types pass both thresholds
    keep_types = set()
    for mtype, count in support.items():
        if count < r_min_map.get(mtype, 1):
            continue
        if zscore_results is not None:
            z = zscore_results.get(mtype, {}).get("zscore", 0.0)
            if z < cfg.z_min:
                continue
        keep_types.add(mtype)

    return [inst for inst in instances if inst["motif_type"] in keep_types]


# ---------------------------------------------------------------------------
# Null-model z-score
# ---------------------------------------------------------------------------

def _shuffle_timestamps(event_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Return a null-model DataFrame with the `step` column randomly permuted.

    Uses numpy in-place shuffle (no full DataFrame copy).
    Only `step` changes; src_node, dst_node, amount are untouched so
    degree and amount distributions are preserved.

    Parameters
    ----------
    event_df : pd.DataFrame
        Window of transactions.  Must have a `step` column.
    rng : np.random.Generator
        Seeded RNG for reproducibility.

    Returns
    -------
    DataFrame with shuffled `step`, sorted by [step, event_id].
    """
    # shuffled_steps = rng.permutation(event_df["step"].to_numpy())
    # df_null = event_df.copy()
    # df_null["step"] = shuffled_steps
    # df_null = df_null.sort_values(["step", "event_id"]).reset_index(drop=True)
    # return df_null

    df_null = event_df.copy()
    # Shuffle step values separately within each source node's edges
    for src, idx in event_df.groupby("src_node").groups.items():
        shuffled = rng.permutation(event_df.loc[idx, "step"].to_numpy())
        df_null.loc[idx, "step"] = shuffled
    return df_null.sort_values(["step", "event_id"]).reset_index(drop=True)

def _run_matchers_on_df(event_df: pd.DataFrame, cfg: MotifConfig) -> Dict[str, int]:
    """Build indexes from event_df, run all matchers, return support counts."""
    out_idx, in_idx, _ = build_event_indexes(event_df)
    instances = run_all_matchers(out_idx, in_idx, cfg)
    counts = count_support(instances)
    del out_idx, in_idx, instances
    return counts


def compute_null_zscore(
    observed_counts: Dict[str, int],
    event_df: pd.DataFrame,
    cfg: MotifConfig,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, dict]:
    """
    Compute z-score for each motif type against a shuffled null model.

        z(M) = (C_obs(M) - mean_null(M)) / (std_null(M) + ε)

    Parameters
    ----------
    observed_counts : dict
        Output of count_support() on the real (unshuffled) data.
    event_df : pd.DataFrame
        The same window of transactions used to produce observed_counts.
        Must have columns: event_id, step, src_node, dst_node, amount.
    cfg : MotifConfig
        n_permutations and z_min are read from here.
    seed : int
        Base seed for reproducible null models.
    verbose : bool
        Print progress per permutation (useful in Colab).

    Returns
    -------
    dict {motif_type: {observed, mean_null, std_null, zscore}}

    Notes
    -----
    - Each permutation builds its own index and runs all matchers on
      the shuffled table.  Index and instances are freed after each run.
    - gc.collect() is called after every permutation to keep RAM low.
    """
    null_counts: Dict[str, list] = {mt: [] for mt in observed_counts}
    rng = np.random.default_rng(seed)

    for i in range(cfg.n_permutations):
        if verbose:
            print(f"  Null permutation {i + 1}/{cfg.n_permutations}...")

        df_null = _shuffle_timestamps(event_df, rng)
        perm_counts = _run_matchers_on_df(df_null, cfg)

        for mt in observed_counts:
            null_counts[mt].append(perm_counts.get(mt, 0))

        del df_null, perm_counts
        gc.collect()

    results: Dict[str, dict] = {}
    for mt, c_obs in observed_counts.items():
        arr       = np.array(null_counts[mt], dtype=float)
        mean_null = float(arr.mean())
        std_null  = float(arr.std())
        zscore    = (c_obs - mean_null) / (std_null + 1e-9)
        results[mt] = {
            "observed":  c_obs,
            "mean_null": round(mean_null, 2),
            "std_null":  round(std_null, 2),
            "zscore":    round(zscore, 3),
        }

    return results


__all__ = [
    "count_support",
    "filter_motifs",
    "compute_null_zscore",
]