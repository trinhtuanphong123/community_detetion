# ============================================================
# CELL 0 — Environment setup for 02_motif_pipeline.ipynb
#
# Run ONCE, restart the Colab runtime, then run Cells 1-6.
#
# Prerequisite:
#   01_graph_pipeline.ipynb Cell 6 must have saved
#   temporal_edges.parquet to OUTPUT_DIR on Google Drive.
# ============================================================


# 1. Install RAPIDS cuDF for T4 GPU

try:
    import cudf  # noqa: F401
    print("cuDF already available.")

except ImportError:
    import subprocess
    import sys

    print("Installing cuDF (CUDA 12 build)...")

    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "cudf-cu12",
        "--extra-index-url",
        "https://pypi.nvidia.com"
    ])

    print("\ncuDF installation completed.")
    print("IMPORTANT: Restart the Colab runtime now.")
    print("After restarting, re-run all notebook cells from the beginning.")


# ============================================================
# 2. Mount Google Drive
# ============================================================

import os

if not os.path.isdir("/content/drive/MyDrive"):

    from google.colab import drive

    drive.mount("/content/drive")

    print("Google Drive mounted successfully.")

else:
    print("Google Drive already mounted.")


# ============================================================
# 3. Optional sanity checks
# ============================================================

print("\nEnvironment check:")
print(f"Current working directory: {os.getcwd()}")

if os.path.isdir("/content/drive/MyDrive"):
    print("Drive access: OK")
else:
    print("Drive access: FAILED")


cell 1: 


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

        AML relay chains typically close within a few days.
        Default = 3 is intentionally tight.

        Widen only if dataset step != 1 day.

    Amount ratio constraints
    ------------------------
    rho_min, rho_max : float
        Allowed ratio:

            a(i+1) / a(i)

        for consecutive transactions.

        Layering keeps amounts close to 1.0.
        Structuring splits downward (ratio < 1).

        [0.5, 2.0] is a reasonable starting range.

        Tighten toward:
            [0.7, 1.4]

        for capital-preservation behavior typical of layering.

    Minimum support (repetition)
    ----------------------------
    r_min_fanin
        Minimum number of incoming sources
        for a fan-in node.

    r_min_fanout
        Minimum number of outgoing targets
        for a fan-out node.

    r_min_cycle
        Minimum occurrences of a cycle
        before it is flagged.

    r_min_relay
        Minimum occurrences of a relay chain.

    r_min_split_merge
        Minimum occurrences of a split-merge path.

    Statistical filtering
    ---------------------
    n_permutations : int
        Number of shuffles used for
        null-model z-score estimation.

        30:
            sufficient for early screening

        100+:
            recommended for final experiments

    z_min : float
        Minimum z-score threshold.
        Motifs below this value are discarded.

    Search limits
    -------------
    max_nodes : int
        Maximum nodes in a single motif instance.

    max_edges : int
        Maximum edges in a single motif instance.

    max_instances : int
        Hard cap on total matched instances
        kept in RAM.

        Prevents OOM on large windows.

        0 = disabled.

    Window sizes
    ------------
    window_sizes : List[int]
        Step counts for windowed search.

        Aligned with:
            - graph_schema §6.3
            - community_spec §6

        Default:
            [7, 14, 30]
    """

    # =========================================================
    # Temporal
    # =========================================================

    delta: int = 2

    # =========================================================
    # Amount ratio constraints
    # =========================================================

    rho_min: float = 0.3
    rho_max: float = 3.0

    # =========================================================
    # Support thresholds (per motif type)
    # =========================================================

    # at least 3 distinct sources into one node
    r_min_fanin: int = 3

    # at least 3 distinct targets from one node
    r_min_fanout: int = 3

    # a single observed cycle is already suspicious
    r_min_cycle: int = 1

    r_min_relay: int = 1

    r_min_split_merge: int = 1

    # =========================================================
    # Statistical filtering
    # =========================================================

    n_permutations: int = 5 # start here; raise to 30 only after pipeline completes


    z_min: float = 2.0

    # =========================================================
    # Search limits / prune guards
    # =========================================================

    max_nodes: int = 4

    max_edges: int = 5

    # total matched instance cap across all matchers
    # prevents unbounded RAM growth
    # 0 = disabled

    max_instances: int = 10_000   # đổi từ 50_000


    # =========================================================
    # Window sizes
    # =========================================================

    window_sizes: List[int] = field(
        default_factory=lambda: [7]
    )


__all__ = [
    "MotifConfig",
]


cell 2:

"""
index -  Event-level search indexes for motif matching.

Builds lightweight Python dict indexes from a window of transactions.
Matchers use these indexes instead of scanning the full DataFrame repeatedly.

Indexes produced by build_event_indexes():
    out_index  : {src_node -> [event_dict, ...]} sorted by step ascending
    in_index   : {dst_node -> [event_dict, ...]} sorted by step ascending
    step_index : {step     -> [event_dict, ...]}

Helper:
    edges_after_step(out_index, node, step) — forward-only edge lookup.
    filter_window(event_df, step_start, step_end) — slice a time window.

Guarantees:
- Time      : all buckets are sorted by step; edges_after_step enforces
              forward-only search (no backward traversal).
- Direction : out_index keys on src; in_index keys on dst; never merged.
- Memory    : plain Python dicts after indexing (no DataFrame held);
              gc.collect() called after build.
"""

from __future__ import annotations

import gc
from bisect import bisect_right
from collections import defaultdict
from typing import Dict, List

import pandas as pd
import os

# Type alias for an event record stored in the index


# Each event is a plain dict for zero-overhead lookup in matcher loops.
# Keys: event_id, step, src, dst, amount, is_sar
EventDict = Dict[str, object]


# ---------------------------------------------------------------------------
# Main index builder
# ---------------------------------------------------------------------------

def build_event_indexes(
    event_df: pd.DataFrame,
    src_col: str = "src_node",
    dst_col: str = "dst_node",
    step_col: str = "step",
    amount_col: str = "amount",
    alert_col: str = "is_sar",
) -> tuple[dict, dict, dict]:
    """
    Build three search indexes from a window of transactions.

    Parameters
    ----------
    event_df : pd.DataFrame
        A single time window of transactions.  Must be sorted by step
        (loader.py and iter_windows guarantee this).
        Supported column name variants:
          - canonical: src_node, dst_node (from loader.py)
          - legacy:    nameOrig, nameDest  (raw AMLGentex)
          - short:     src, dst           (old pipeline)
    src_col, dst_col, step_col, amount_col, alert_col : str
        Column name overrides.

    Returns
    -------
    out_index  : {src_node: [EventDict, ...]}  sorted by step ascending
    in_index   : {dst_node: [EventDict, ...]}  sorted by step ascending
    step_index : {step:     [EventDict, ...]}

    Notes
    -----
    - event_id is auto-assigned as the row index if absent (motif_spec §3.1).
    - itertuples() is used intentionally: the single-pass loop is O(n) and
      avoids creating three separate groupby-materialised DataFrames.
    - All three dicts are populated in one pass to minimise memory pressure.
    - Buckets are already in step order because event_df is pre-sorted.
    """
    # cuDF → pandas: motif index works on plain Python dicts after this point;
    # converting once here avoids per-row GPU tensor overhead in itertuples.

    if hasattr(event_df, "to_pandas"):
        event_df = event_df.to_pandas()

    # --- FIX WARNING 4: Defensive Integrity Guard ---
    # Validate monotonicity after pandas conversion but before indexing.
    if os.getenv("MOTIF_DEBUG") == "1":
        if not event_df[step_col].is_monotonic_increasing:
            raise ValueError(
                f"CRITICAL DATA ERROR: event_df must be sorted by '{step_col}' before indexing. "
                "Unsorted data will cause binary search (bisect) to return silent failures."
            )



    df = _normalize_columns(event_df, src_col, dst_col, step_col, amount_col, alert_col)
    df = _ensure_event_id(df)
    _validate_required(df)

    has_alert = "is_sar" in df.columns
    out_index:  dict = defaultdict(list)
    in_index:   dict = defaultdict(list)
    step_index: dict = defaultdict(list)

    # Single O(n) pass — itertuples permitted per coding conventions when
    # building an index that cannot be expressed as a vectorized operation.
    for row in df.itertuples(index=False):
        e: EventDict = {
            "event_id": int(row.event_id),
            "step":     int(row.step),
            "src":      int(row.src_node),
            "dst":      int(row.dst_node),
            "amount":   float(row.amount),
            "is_sar":   int(row.is_sar) if has_alert else 0,
        }
        out_index[e["src"]].append(e)
        in_index[e["dst"]].append(e)
        step_index[e["step"]].append(e)

    # Freeze to regular dicts: prevents accidental key creation on miss
    out_index  = dict(out_index)
    in_index   = dict(in_index)
    step_index = dict(step_index)
    # At the end of build_event_indexes, after populating out_index:
    out_steps = {node: [e["step"] for e in edges]
             for node, edges in out_index.items()}
    gc.collect()
    return out_index, in_index, step_index, out_steps


# ---------------------------------------------------------------------------
# Forward-only lookup helper
# ---------------------------------------------------------------------------


def edges_after_step(
    out_index: dict,
    node: int,
    step: int,
    out_steps: dict,
) -> List[EventDict]:
    """
    Return all outgoing edges from `node` with step > `step`.

    Used by matchers to expand a relay chain forward in time only.
    Binary search on the pre-sorted bucket avoids a linear scan.

    Parameters
    ----------
    out_index : dict
        Output of build_event_indexes().
    node : int
        Source node ID.
    step : int
        All returned edges have step strictly greater than this value.

    Returns
    -------
    List of EventDict, empty if node has no outgoing edges after step.
    """
    bucket = out_index.get(node)
    if not bucket:
        return []
    idx = bisect_right(out_steps[node], step)
    return bucket[idx:]



# ---------------------------------------------------------------------------
# Window slice helper
# ---------------------------------------------------------------------------

def filter_window(
    event_df: pd.DataFrame,
    step_start: int,
    step_end: int,
) -> pd.DataFrame:
    """
    Slice event_df to [step_start, step_end] (inclusive).

    Use before build_event_indexes() when working from a full loaded
    DataFrame rather than through iter_windows().

    Parameters
    ----------
    event_df : pd.DataFrame
        Must have a `step` column.
    step_start, step_end : int
        Inclusive step bounds.

    Returns
    -------
    Filtered DataFrame with reset index.  Direction and step order preserved.
    """
    mask = (event_df["step"] >= step_start) & (event_df["step"] <= step_end)
    return event_df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_columns(
    df: pd.DataFrame,
    src_col: str,
    dst_col: str,
    step_col: str,
    amount_col: str,
    alert_col: str,
) -> pd.DataFrame:
    """
    Rename columns to canonical names (src_node, dst_node, step, amount, is_sar).
    Supports three naming variants without copying the full DataFrame.
    """
    rename: dict = {}

    # Caller-specified override names
    if src_col != "src_node" and src_col in df.columns:
        rename[src_col] = "src_node"
    if dst_col != "dst_node" and dst_col in df.columns:
        rename[dst_col] = "dst_node"
    if step_col != "step" and step_col in df.columns:
        rename[step_col] = "step"
    if amount_col != "amount" and amount_col in df.columns:
        rename[amount_col] = "amount"
    if alert_col != "is_sar" and alert_col in df.columns:
        rename[alert_col] = "is_sar"

    # Legacy / raw AMLGentex column names
    cols = set(df.columns)
    if "src_node" not in cols and "nameOrig" in cols:
        rename["nameOrig"] = "src_node"
    if "dst_node" not in cols and "nameDest" in cols:
        rename["nameDest"] = "dst_node"
    # Short-form names (old pipeline)
    if "src_node" not in cols and "src" in cols:
        rename["src"] = "src_node"
    if "dst_node" not in cols and "dst" in cols:
        rename["dst"] = "dst_node"
    # is_sar / Is Laundering / is_laundering aliases
    if "is_sar" not in cols:
        for alias in ("is_laundering", "Is Laundering", "isSAR"):
            if alias in cols:
                rename[alias] = "is_sar"
                break

    if rename:
        df = df.rename(columns=rename)
    return df


def _ensure_event_id(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-assign event_id from row index if the column is absent."""
    if "event_id" not in df.columns:
        df = df.copy()
        df["event_id"] = range(len(df))
    return df


def _validate_required(df: pd.DataFrame) -> None:
    """Raise ValueError if mandatory columns are missing."""
    required = {"event_id", "step", "src_node", "dst_node", "amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"event_df is missing required columns: {sorted(missing)}. "
            f"Available columns: {sorted(df.columns)}"
        )


__all__ = [
    "build_event_indexes",
    "edges_after_step",
    "filter_window",
]



"""
matchers — Temporal motif matching for AML detection.

Five templates:
    fan_in      : many -> one within delta steps
    fan_out     : one -> many within delta steps
    cycle_3     : u -> v -> w -> u with strict time order
    relay_4     : u -> v -> w -> x with strict time order
    split_merge : u -> v1 -> z and u -> v2 -> z

General rules for every matcher (guide §4):
    1. Forward-only — step strictly increases at each hop.
    2. Early stop — prune immediately when any constraint fails.
    3. No side effects — indexes are never modified.
    4. Return list[dict] — each dict is one motif instance.

Instance dict keys (guide §8):
    motif_type, nodes, edges (event_id list),
    steps, amounts, lags, ratios, n_alert

No pandas / cuDF imports — only plain Python dicts and lists.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Primitive constraint helpers
# ---------------------------------------------------------------------------

def _ratio_ok(a_prev: float, a_curr: float, rho_min: float, rho_max: float) -> bool:
    """True if a_curr / a_prev is in [rho_min, rho_max]."""
    if a_prev <= 0:
        return False
    r = a_curr / a_prev
    return rho_min <= r <= rho_max


def _lag_ok(step_prev: int, step_curr: int, delta: int) -> bool:
    """True if 0 < step_curr - step_prev <= delta."""
    lag = step_curr - step_prev
    return 0 < lag <= delta


# ---------------------------------------------------------------------------
# Instance constructor
# ---------------------------------------------------------------------------

def _make_instance(
    motif_type: str,
    edges: list[dict],
    nodes: list[int],
) -> dict:
    """
    Build a standardised motif instance dict from an ordered edge list.

    `edges` must be in chronological order (step ascending).
    """
    steps   = [e["step"]   for e in edges]
    amounts = [e["amount"] for e in edges]
    lags    = [steps[i] - steps[i - 1] for i in range(1, len(steps))]
    ratios  = [
        round(amounts[i] / amounts[i - 1], 4) if amounts[i - 1] > 0 else 0.0
        for i in range(1, len(amounts))
    ]
    return {
        "motif_type": motif_type,
        "nodes":      nodes,
        "edges":      [e["event_id"] for e in edges],
        "steps":      steps,
        "amounts":    amounts,
        "lags":       lags,
        "ratios":     ratios,
        "n_alert":    sum(e.get("is_sar", 0) for e in edges),
    }


# ---------------------------------------------------------------------------
# Fan-in  — FIX R3: add seen-set dedup
# ---------------------------------------------------------------------------

def find_fanin(
    in_index: dict,
    cfg: MotifConfig,
) -> list[dict]:
    """
    Fan-in: r_min_fanin distinct sources -> same destination x, within delta steps.

        u1 -> x
        u2 -> x   (step_u2 - step_u1 <= delta)
        u3 -> x   ...

    Constraints:
        - All sources distinct.
        - All arrivals in [t0, t0 + delta].
        - Amount ratio of each edge vs seed in [rho_min, rho_max].
        - At least r_min_fanin sources found.

    FIX R3: A seen-set keyed by (frozenset(source_ids), destination) prevents
    the O(n^2) duplicate groups produced by seed-shifting without dedup.
    """
    results = []
    # seen: prevents re-emitting the same group under a different seed edge.
    seen: set = set()

    for x, incoming in in_index.items():
        n = len(incoming)
        if n < cfg.r_min_fanin:
            continue    # prune: not enough edges to form a group

        for i in range(n):
            seed   = incoming[i]
            t0     = seed["step"]
            a_prev = seed["amount"]
            seen_src = {seed["src"]}
            group  = [seed]

            for j in range(i + 1, n):
                e = incoming[j]
                # Prune: window exceeded — bucket is sorted, so break early
                if e["step"] - t0 > cfg.delta:
                    break
                # Prune: duplicate source — skip, do not break
                if e["src"] in seen_src:
                    continue
                # Prune: amount ratio
                if not _ratio_ok(a_prev, e["amount"], cfg.rho_min, cfg.rho_max):
                    continue

                seen_src.add(e["src"])
                group.append(e)
                a_prev = e["amount"]

            if len(group) >= cfg.r_min_fanin:
                # Canonical key: frozenset of participating source IDs + destination.
                # This collapses all seed permutations of the same source group.
                key = (frozenset(e["src"] for e in group), x)
                if key in seen:
                    continue        # already emitted this exact group
                seen.add(key)

                nodes = [e["src"] for e in group] + [x]
                results.append(_make_instance("fanin", group, nodes))

    return results


# ---------------------------------------------------------------------------
# Fan-out  — unchanged; correct as-is
# ---------------------------------------------------------------------------

def find_fanout(
    out_index: dict,
    cfg: MotifConfig,
    out_steps: dict,
) -> list[dict]:
    """
    Fan-out: source x -> r_min_fanout distinct destinations, within delta steps.

        x -> v1
        x -> v2   (step_v2 - step_v1 <= delta)
        x -> v3   ...

    Constraints mirror find_fanin (symmetric).
    Returns list of instance dicts.
    """
    results = []

    for x, outgoing in out_index.items():
        n = len(outgoing)
        if n < cfg.r_min_fanout:
            continue

        for i in range(n):
            seed   = outgoing[i]
            t0     = seed["step"]
            a0     = seed["amount"]
            seen   = {seed["dst"]}
            group  = [seed]
            a_prev = a0

            for j in range(i + 1, n):
                e = outgoing[j]

                if e["step"] - t0 > cfg.delta:
                    break

                if e["dst"] in seen:
                    continue

                if not _ratio_ok(a_prev, e["amount"], cfg.rho_min, cfg.rho_max):
                    continue

                seen.add(e["dst"])
                group.append(e)
                a_prev = e["amount"]

            if len(group) >= cfg.r_min_fanout:
                nodes = [x] + [e["dst"] for e in group]
                results.append(_make_instance("fanout", group, nodes))

    return results


# ---------------------------------------------------------------------------
# Cycle-3  — unchanged; has seen-set dedup
# ---------------------------------------------------------------------------

def find_cycle3(
    out_index: dict,
    cfg: MotifConfig,
    out_steps: dict,
) -> list[dict]:
    """
    Cycle-3: u -> v -> w -> u with strictly increasing steps.

    Constraints:
        - step_e1 < step_e2 < step_e3 (strict)
        - Each consecutive lag <= delta
        - Amount ratio at each hop in [rho_min, rho_max]
        - u, v, w are three distinct nodes

    Uses edges_after_step() for forward-only bucket access.
    Returns list of instance dicts.
    """
    results = []
    seen = set()
    for u, edges_u in out_index.items():
        for e1 in edges_u:
            v  = e1["dst"]
            t1 = e1["step"]
            a1 = e1["amount"]

            if v == u:
                continue

            # e2: v -> w, step in (t1, t1 + delta]
            for e2 in edges_after_step(out_index, v, t1, out_steps):
                if e2["step"] - t1 > cfg.delta:
                    break   # bucket sorted -> nothing after is valid

                w  = e2["dst"]
                a2 = e2["amount"]

                if w == u or w == v:
                    continue

                if not _ratio_ok(a1, a2, cfg.rho_min, cfg.rho_max):
                    continue

                # e3: w -> u, step in (t2, t2 + delta]
                for e3 in edges_after_step(out_index, w, e2["step"], out_steps):
                    if e3["step"] - e2["step"] > cfg.delta:
                        break

                    if e3["dst"] != u:
                        continue

                    if not _ratio_ok(a2, e3["amount"], cfg.rho_min, cfg.rho_max):
                        continue
                    key = frozenset([e1["event_id"],
                                     e2["event_id"],
                                     e3["event_id"]])
                    if key in seen:
                        continue
                    seen.add(key)

                    results.append(_make_instance(
                        "cycle3",
                        [e1, e2, e3],
                        [u, v, w, u],
                    ))

    return results


# ---------------------------------------------------------------------------
# Relay-4  — unchanged; has seen-set dedup
# ---------------------------------------------------------------------------

def find_relay4(
    out_index: dict,
    cfg: MotifConfig,
    out_steps: dict,
) -> list[dict]:
    """
    Relay-4: u -> v -> w -> x with strictly increasing steps.

    Constraints:
        - step_e1 < step_e2 < step_e3 (strict)
        - Each consecutive lag <= delta
        - Amount ratio at each hop in [rho_min, rho_max]
        - u, v, w, x are four distinct nodes

    Uses edges_after_step() for forward-only access.
    Returns list of instance dicts.
    """
    results = []
    seen = set()
    for u, edges_u in out_index.items():
        for e1 in edges_u:
            v  = e1["dst"]
            t1 = e1["step"]
            a1 = e1["amount"]

            if v == u:
                continue

            for e2 in edges_after_step(out_index, v, t1, out_steps):
                if e2["step"] - t1 > cfg.delta:
                    break

                w  = e2["dst"]
                a2 = e2["amount"]

                if w in (u, v):
                    continue

                if not _ratio_ok(a1, a2, cfg.rho_min, cfg.rho_max):
                    continue

                for e3 in edges_after_step(out_index, w, e2["step"], out_steps):
                    if e3["step"] - e2["step"] > cfg.delta:
                        break

                    x  = e3["dst"]
                    a3 = e3["amount"]

                    if x in (u, v, w):
                        continue

                    if not _ratio_ok(a2, a3, cfg.rho_min, cfg.rho_max):
                        continue

                    key = frozenset([e1["event_id"],
                                     e2["event_id"],
                                     e3["event_id"]])
                    if key in seen:
                        continue
                    seen.add(key)

                    results.append(_make_instance(
                        "relay4",
                        [e1, e2, e3],
                        [u, v, w, x],
                    ))

    return results


# ---------------------------------------------------------------------------
# Split-merge  — FIX R2: single-best-edge + dedup + remove dead code
# ---------------------------------------------------------------------------

def find_split_merge(
    out_index: dict,
    in_index: dict,
    cfg: MotifConfig,
    out_steps: dict,
) -> list[dict]:
    """
    Split-merge: source splits to two intermediaries that recombine at one target.

        u -> v1 -> z
        u -> v2 -> z

    Two-phase:
        Phase 1: find split pairs (u -> v1, u -> v2) within delta steps.
        Phase 2: for each pair, find a common target z that both v1 and v2
                 reach within 2 * delta steps of the split start.

    Constraints:
        - u, v1, v2, z are four distinct nodes
        - Amount ratio at every hop in [rho_min, rho_max]
        - All events in chronological order per hop

    FIX R2a — single-best-edge: v1_targets keeps only the FIRST (earliest)
        edge from v1 to each z, not all edges.  This avoids the Cartesian
        product explosion in the original code where v1_targets[z] was a list
        and every (e_v1z, e_v2z) combo was emitted as a separate instance.

    FIX R2b — seen-set: a frozenset over all four event_ids prevents the same
        (u, v1, v2, z) quad from being re-emitted via different traversal paths.

    FIX R2c — v1_targets cached per (u, v1): computed once before the v2 loop,
        not rebuilt for every (v1, v2) pair.
    """
    results = []
    seen: set = set()   # frozenset of 4 event_ids to deduplicate quads

    for u, outgoing_u in out_index.items():
        n = len(outgoing_u)

        for i in range(n):
            e_uv1 = outgoing_u[i]
            v1    = e_uv1["dst"]
            t0    = e_uv1["step"]
            a_uv1 = e_uv1["amount"]

            # FIX R2c — build v1_targets once per (u, v1) before the v2 loop.
            # Maps z -> the single earliest v1->z edge within 2*delta of t0.
            v1_targets: dict = {}
            for e in edges_after_step(out_index, v1, t0, out_steps):
                if e["step"] - t0 > 2 * cfg.delta:
                    break
                z = e["dst"]
                if z in (u, v1):
                    continue
                if not _ratio_ok(a_uv1, e["amount"], cfg.rho_min, cfg.rho_max):
                    continue
                # FIX R2a — keep only the first (earliest) edge to z.
                if z not in v1_targets:
                    v1_targets[z] = e

            # Skip this v1 entirely if it reaches no valid targets
            if not v1_targets:
                continue

            # Phase 1: iterate v2 candidates (outgoing edges from u after e_uv1)
            for j in range(i + 1, n):
                e_uv2 = outgoing_u[j]

                # Prune: split window exceeded
                if e_uv2["step"] - t0 > cfg.delta:
                    break

                v2 = e_uv2["dst"]
                if v2 == v1 or v2 == u:
                    continue

                # Prune: split amount ratio
                if not _ratio_ok(a_uv1, e_uv2["amount"], cfg.rho_min, cfg.rho_max):
                    continue

                # Phase 2: find v2 -> z where z is already reachable from v1
                for e_v2z in edges_after_step(out_index, v2, t0, out_steps):
                    if e_v2z["step"] - t0 > 2 * cfg.delta:
                        break
                    z = e_v2z["dst"]

                    # z must be in v1_targets and must be a new node
                    if z not in v1_targets or z in (u, v1, v2):
                        continue

                    # Amount ratio on the v2->z leg
                    if not _ratio_ok(e_uv2["amount"], e_v2z["amount"],
                                     cfg.rho_min, cfg.rho_max):
                        continue

                    e_v1z = v1_targets[z]   # single best edge

                    # FIX R2b — dedup by the four event_ids
                    key = frozenset([e_uv1["event_id"], e_uv2["event_id"],
                                     e_v1z["event_id"], e_v2z["event_id"]])
                    if key in seen:
                        continue
                    seen.add(key)

                    all_edges = sorted(
                        [e_uv1, e_uv2, e_v1z, e_v2z],
                        key=lambda e: e["step"],
                    )
                    results.append(
                        _make_instance("split_merge", all_edges, [u, v1, v2, z])
                    )

    return results



# Convenience: run all matchers on a pre-built index set
# FIX R1: max_instances cap prevents unbounded RAM accumulation.


def run_all_matchers(
    out_index: dict,
    in_index: dict,
    cfg: MotifConfig,
    out_steps: dict[int, list[dict]],
) -> list[dict]:
    """
    Run all five matchers and return a combined instance list.

    Parameters
    ----------
    out_index, in_index : dict
        Output of build_event_indexes().
    cfg : MotifConfig
        Motif configuration.  cfg.max_instances caps the total list size
        (0 = disabled).

    Returns
    -------
    Combined list of all matched motif instances across all types.
    If cfg.max_instances > 0, each matcher's result is truncated to
    cfg.max_instances before concatenation, and a warning is printed.
    """
    # Run each matcher independently so we can apply the cap per type.
    matchers = {
        "fanin":       find_fanin(in_index, cfg),
        "fanout":      find_fanout(out_index, cfg, out_steps),
        #"cycle3":      find_cycle3(out_index, cfg, out_steps),
        "relay4":      find_relay4(out_index, cfg, out_steps),
        #"split_merge": find_split_merge(out_index, in_index, cfg, out_steps),
    }

    combined = []
    cap = cfg.max_instances if cfg.max_instances > 0 else None
    for mtype, instances in matchers.items():
        if cap and len(instances) > cap:
            # Warn and truncate; do NOT silently skip — the signal is still present.
            print(f"  [WARN] {mtype}: {len(instances):,} instances exceed "
                  f"max_instances={cap:,}; truncating to {cap:,}.")
            instances = instances[:cap]
        combined.extend(instances)

    return combined


__all__ = [
    "find_fanin",
    "find_fanout",
    "find_cycle3",
    "find_relay4",
    "find_split_merge",
    "run_all_matchers",
]



# scoring — support counting, filtering, null-model z-score.
# FIX R4a: _run_matchers_on_df truncates instances per null pass (cap arg).
# FIX R4b: compute_null_zscore subsamples event_df per perm (sample_frac).
# FIX R4c: compute_null_zscore returns {} when observed_counts is empty.

from __future__ import annotations

import copy
import gc

from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd


# ============================================================
# Support counting
# ============================================================

def count_support(motif_instances: List[dict]) -> Dict[str, int]:
    """
    Count matched instances by motif type.

    Returns
    -------
    Dict[str, int]
        {motif_type: count}
    """
    c: Counter = Counter()

    for inst in motif_instances:
        c[inst["motif_type"]] += 1

    return dict(c)


# ============================================================
# Filtering
# ============================================================

def filter_motifs(
    instances: List[dict],
    cfg: MotifConfig,
    zscore_results: Dict[str, dict] | None = None,
) -> List[dict]:
    """
    Apply per-type r_min and optional z-score threshold.

    fan-in / fan-out r_min is enforced by the matchers;
    require >=1 here.
    """

    r_min_map = {
        "fanin":       1,
        "fanout":      1,
        "cycle3":      cfg.r_min_cycle,
        "relay4":      cfg.r_min_relay,
        "split_merge": cfg.r_min_split_merge,
    }

    support = count_support(instances)

    keep_types = set()

    for mtype, count in support.items():

        if count < r_min_map.get(mtype, 1):
            continue

        if zscore_results is not None:
            if zscore_results.get(mtype, {}).get("zscore", 0.0) < cfg.z_min:
                continue

        keep_types.add(mtype)

    return [
        inst
        for inst in instances
        if inst["motif_type"] in keep_types
    ]


# ============================================================
# Null-model helpers
# ============================================================

def _shuffle_timestamps(
    event_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Shuffle step values per source node, then re-sort globally.

    Preserves:
        - degree distribution
        - amount distribution

    Breaks:
        - sequential motif patterns
    """

    df_null = event_df.copy()

    for src, idx in event_df.groupby("src_node").groups.items():

        df_null.loc[idx, "step"] = rng.permutation(
            event_df.loc[idx, "step"].to_numpy()
        )

    return (
        df_null
        .sort_values(["step", "event_id"])
        .reset_index(drop=True)
    )


def _run_matchers_on_df(
    event_df: pd.DataFrame,
    cfg: MotifConfig,
    cap: int = 0,
) -> Dict[str, int]:
    """
    FIX R4a:
        Build indexes, run matchers with per-type cap,
        and return support counts.

    Parameters
    ----------
    cap : int
        Overrides cfg.max_instances for null runs,
        bounding RAM usage per permutation.
    """

    out_idx, in_idx, _, out_steps = build_event_indexes(event_df)

    cfg_null = copy.copy(cfg)

    if cap > 0:
        cfg_null.max_instances = (
            min(cap, cfg.max_instances)
            if cfg.max_instances > 0
            else cap
        )

    instances = run_all_matchers(
        out_idx,
        in_idx,
        cfg_null,
        out_steps,
    )

    counts = count_support(instances)

    del out_idx, in_idx, instances

    return counts


# ============================================================
# Null-model z-score
# ============================================================

def compute_null_zscore(
    observed_counts: Dict[str, int],
    event_df: pd.DataFrame,
    cfg: MotifConfig,
    seed: int = 42,
    verbose: bool = False,
    sample_frac: float = 1.0,
    null_cap: int = 10_000,
) -> Dict[str, dict]:
    """
    Compute motif z-score:

        z(M) = (C_obs - mean_null) / (std_null + eps)

    FIXES
    -----
    R4a:
        null_cap forwarded to matcher runs.

    R4b:
        sample_frac < 1.0 subsamples rows per permutation.

    R4c:
        Return {} immediately if observed_counts is empty.

    Parameters
    ----------
    sample_frac : float
        Fraction of rows sampled per permutation.

        Recommended:
            0.2 - 0.5 for large windows

        1.0 = no sampling.

    null_cap : int
        Per-type instance cap inside each null pass.
    """

    # --------------------------------------------------------
    # FIX R4c
    # --------------------------------------------------------

    if not observed_counts:
        return {}

    null_counts: Dict[str, list] = {
        mt: []
        for mt in observed_counts
    }

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Permutation loop
    # --------------------------------------------------------

    for i in range(cfg.n_permutations):

        if verbose:
            print(
                f"  Null permutation "
                f"{i + 1}/{cfg.n_permutations}..."
            )

        # ----------------------------------------------------
        # FIX R4b — optional row sampling
        # ----------------------------------------------------

        if sample_frac < 1.0:

            df_perm = (
                event_df
                .sample(
                    frac=sample_frac,
                    random_state=int(
                        rng.integers(1_000_000)
                    ),
                )
                .sort_values(["step", "event_id"])
                .reset_index(drop=True)
            )

        else:
            df_perm = event_df

        # ----------------------------------------------------
        # Timestamp shuffling
        # ----------------------------------------------------

        df_null = _shuffle_timestamps(df_perm, rng)

        # ----------------------------------------------------
        # FIX R4a
        # ----------------------------------------------------

        perm_counts = _run_matchers_on_df(
            df_null,
            cfg,
            cap=null_cap,
        )

        for mt in observed_counts:
            null_counts[mt].append(
                perm_counts.get(mt, 0)
            )

        del df_null, perm_counts

        if sample_frac < 1.0:
            del df_perm

        gc.collect()

    # --------------------------------------------------------
    # Aggregate z-score results
    # --------------------------------------------------------

    results: Dict[str, dict] = {}

    for mt, c_obs in observed_counts.items():

        arr = np.array(
            null_counts[mt],
            dtype=float,
        )

        mean_null = float(arr.mean())
        std_null  = float(arr.std())

        zscore = (
            (c_obs - mean_null)
            / (std_null + 1e-9)
        )

        results[mt] = {
            "observed":  c_obs,
            "mean_null": round(mean_null, 2),
            "std_null":  round(std_null, 2),
            "zscore":    round(zscore, 3),
        }

    return results


# ============================================================
# Public exports
# ============================================================

__all__ = [
    "count_support",
    "filter_motifs",
    "compute_null_zscore",
]

cell 4: 

"""
feature — Motif feature extraction for ML downstream use.

Two primary feature tables:
    1. Entity-level (per node, per motif type)
       Output: build_entity_motif_features()  → long format
               build_entity_feature_wide()     → wide format (one row per node)

    2. Window-level (per time bucket, per motif type)
       Output: build_window_motif_features()

Export:
    save_features() — writes parquet to a configurable path (Google Drive).

Feature columns produced (motif_spec §7 / guide §8):
    count             — raw instance count
    avg_amount        — mean amount across all edges in matched instances
    avg_ratio         — mean amount preservation ratio across hops
    ratio_std         — std of amount ratio (higher = more irregular)
    avg_lag           — mean step gap between consecutive hops
    max_lag           — worst-case lag (flags delayed relay)
    avg_n_alert_edges — mean flagged edges per instance
    zscore            — significance vs. null model (from compute_null_zscore)
    freq_by_degree    — count / node degree (passed in by caller)
    freq_by_volume    — count / total transaction volume in window

Guarantees:
    - Time   : window_start assigned from min(steps) of each instance.
    - Direction: node role (src/dst) preserved in entity table; not collapsed.
    - Memory : row-building is O(instances × nodes_per_instance), not O(all transactions).
               groupby aggregation is vectorized. No unnecessary intermediate copies.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Entity-level features
# ---------------------------------------------------------------------------

def build_entity_motif_features(
    motif_instances: List[dict],
    zscore_table: Optional[Dict[str, dict]] = None,
    node_degree: Optional[Dict[int, int]] = None,
    total_volume: float = 0.0,
) -> pd.DataFrame:
    """
    Aggregate motif instances into a feature table per (node, motif_type).

    For each node participating in at least one motif instance, computes:
        count             — number of instances the node appears in
        avg_amount        — mean of per-instance mean amounts
        avg_ratio         — mean amount preservation ratio across hops
        ratio_std         — std of ratio sequence (instability signal)
        avg_lag           — mean hop lag
        max_lag           — maximum hop lag seen
        avg_n_alert_edges — mean SAR-flagged edges per instance
        zscore            — motif significance (from compute_null_zscore)
        freq_by_degree    — count / node out-degree (0 if degree unknown)
        freq_by_volume    — count / total_volume (0 if volume == 0)

    Parameters
    ----------
    motif_instances : list[dict]
        Combined output of run_all_matchers() or filter_motifs().
    zscore_table : dict, optional
        Output of compute_null_zscore(). Keys are motif_type strings.
    node_degree : dict, optional
        {node_id: degree} mapping. Used to compute freq_by_degree.
        If None, freq_by_degree is set to 0 for all rows.
    total_volume : float
        Total transaction amount in the current window.
        Used to compute freq_by_volume. Set to 0 to skip.

    Returns
    -------
    pd.DataFrame with columns:
        node, motif_type, count, avg_amount, avg_ratio, ratio_std,
        avg_lag, max_lag, avg_n_alert_edges, zscore,
        freq_by_degree, freq_by_volume
    One row per (node, motif_type). Sorted by node, motif_type.
    """
    _EMPTY_COLS = [
        "node", "motif_type", "count",
        "avg_amount", "avg_ratio", "ratio_std",
        "avg_lag", "max_lag", "avg_n_alert_edges",
        "zscore", "freq_by_degree", "freq_by_volume",
    ]

    if not motif_instances:
        return pd.DataFrame(columns=_EMPTY_COLS)

    rows = []
    for inst in motif_instances:
        mt      = inst["motif_type"]
        amounts = inst.get("amounts", [])
        lags    = inst.get("lags", [])
        ratios  = inst.get("ratios", [])
        n_alert = inst.get("n_alert", 0)
        zs      = zscore_table[mt]["zscore"] if (zscore_table and mt in zscore_table) else np.nan

        avg_amt   = float(np.mean(amounts))   if amounts else 0.0
        avg_ratio = float(np.mean(ratios))    if ratios  else np.nan
        r_std     = float(np.std(ratios))     if ratios  else np.nan
        avg_lag   = float(np.mean(lags))      if lags    else 0.0
        max_lag   = float(max(lags))          if lags    else 0.0

        for node in set(inst.get("nodes", [])):
            rows.append({
                "node":             int(node),
                "motif_type":       mt,
                "avg_amount":       avg_amt,
                "avg_ratio":        avg_ratio,
                "ratio_std":        r_std,
                "avg_lag":          avg_lag,
                "max_lag":          max_lag,
                "n_alert_edges":    n_alert,
                "zscore":           zs,
            })

    if not rows:
        return pd.DataFrame(columns=_EMPTY_COLS)

    df = pd.DataFrame(rows)

    agg = (
        df.groupby(["node", "motif_type"], as_index=False)
        .agg(
            count               =("avg_amount",     "count"),
            avg_amount          =("avg_amount",      "mean"),
            avg_ratio           =("avg_ratio",       "mean"),
            ratio_std           =("ratio_std",       "mean"),
            avg_lag             =("avg_lag",         "mean"),
            max_lag             =("max_lag",         "max"),
            avg_n_alert_edges   =("n_alert_edges",   "mean"),
            zscore              =("zscore",          "first"),
        )
    )

    # Normalised frequency by node degree
    if node_degree:
        agg["freq_by_degree"] = agg.apply(
            lambda r: r["count"] / node_degree.get(int(r["node"]), 1),
            axis=1,
        )
    else:
        agg["freq_by_degree"] = 0.0

    # Normalised frequency by window transaction volume
    if total_volume > 0:
        agg["freq_by_volume"] = agg["count"] / total_volume
    else:
        agg["freq_by_volume"] = 0.0

    return agg.sort_values(["node", "motif_type"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Wide-format pivot (one row per node, all motif types as columns)
# ---------------------------------------------------------------------------

def build_entity_feature_wide(entity_features: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot entity_motif_features from long → wide format.

    Long:  [node, motif_type, count, avg_amount, ...]
    Wide:  [node, fanin_count, fanin_avg_amount, ..., cycle3_count, ...]

    Use when a single feature vector per node is needed for model input.
    Missing (node, motif_type) combinations are filled with 0.

    Parameters
    ----------
    entity_features : pd.DataFrame
        Output of build_entity_motif_features().

    Returns
    -------
    pd.DataFrame: one row per node.
    Column naming: {motif_type}_{metric}  e.g. fanin_count, relay4_avg_lag.
    """
    if entity_features.empty:
        return pd.DataFrame()

    metric_cols = [
        "count", "avg_amount", "avg_ratio", "ratio_std",
        "avg_lag", "max_lag", "avg_n_alert_edges", "zscore",
        "freq_by_degree", "freq_by_volume",
    ]
    available = [c for c in metric_cols if c in entity_features.columns]

    wide = entity_features.pivot_table(
        index="node",
        columns="motif_type",
        values=available,
        aggfunc="first",
    )

    # Flatten multi-level columns: (metric, motif_type) → "motif_type_metric"
    wide.columns = [f"{mt}_{metric}" for metric, mt in wide.columns]
    wide = wide.fillna(0).reset_index()

    return wide


# ---------------------------------------------------------------------------
# Window-level features
# ---------------------------------------------------------------------------

def build_window_motif_features(
    motif_instances: List[dict],
    window_size: int = 7,
) -> pd.DataFrame:
    """
    Aggregate motif instances into a feature table per (window_start, motif_type).

    For each time bucket, computes:
        count             — number of matched instances
        total_amount      — total transaction amount across all edges
        avg_lag           — mean hop lag across all instances
        n_alert_edges     — total SAR-flagged edges
        suspicious_ratio  — fraction of instances with at least one alert edge

    Parameters
    ----------
    motif_instances : list[dict]
        Output from run_all_matchers() or filter_motifs().
    window_size : int
        Number of steps per bucket. Default 7 = one week if step == 1 day.

    Returns
    -------
    pd.DataFrame with columns:
        window_start, motif_type, count, total_amount,
        avg_lag, n_alert_edges, suspicious_ratio
    Sorted by window_start, motif_type.
    """
    _EMPTY_COLS = [
        "window_start", "motif_type", "count",
        "total_amount", "avg_lag", "n_alert_edges", "suspicious_ratio",
    ]

    if not motif_instances:
        return pd.DataFrame(columns=_EMPTY_COLS)

    rows = []
    for inst in motif_instances:
        steps   = inst.get("steps", [])
        amounts = inst.get("amounts", [])
        lags    = inst.get("lags", [])
        n_alert = inst.get("n_alert", 0)

        if not steps:
            continue

        # Assign to bucket by first event's step — time-preserving
        window_start = (min(steps) // window_size) * window_size

        rows.append({
            "window_start": window_start,
            "motif_type":   inst["motif_type"],
            "total_amount": float(sum(amounts)) if amounts else 0.0,
            "avg_lag":      float(np.mean(lags)) if lags else 0.0,
            "n_alert_edges": n_alert,
            "has_alert":    int(n_alert > 0),
        })

    if not rows:
        return pd.DataFrame(columns=_EMPTY_COLS)

    df = pd.DataFrame(rows)

    agg = (
        df.groupby(["window_start", "motif_type"], as_index=False)
        .agg(
            count            =("total_amount",  "count"),
            total_amount     =("total_amount",  "sum"),
            avg_lag          =("avg_lag",        "mean"),
            n_alert_edges    =("n_alert_edges",  "sum"),
            suspicious_ratio =("has_alert",      "mean"),
        )
    )

    return agg.sort_values(["window_start", "motif_type"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Export to parquet / CSV (configurable path for Google Drive)
# ---------------------------------------------------------------------------

# Default export directory — override in Colab:
#   import src.motif.features as mf
#   mf.DEFAULT_EXPORT_DIR = "/content/drive/MyDrive/aml_outputs"
DEFAULT_EXPORT_DIR: str = "outputs/motif_features"


def save_features(
    df: pd.DataFrame,
    filename: str,
    export_dir: Optional[str] = None,
    fmt: str = "parquet",
) -> str:
    """
    Save a feature DataFrame to disk (parquet or CSV).

    Parameters
    ----------
    df : pd.DataFrame
        Any feature table produced by this module.
    filename : str
        Base filename without extension, e.g. "entity_features_w30".
    export_dir : str, optional
        Target directory.  Defaults to DEFAULT_EXPORT_DIR.
        Set to "/content/drive/MyDrive/<your_path>" in Colab.
    fmt : str
        "parquet" (default, smaller) or "csv".

    Returns
    -------
    str : full path of the saved file.

    Notes
    -----
    - Parquet is preferred for downstream pandas / spark reads.
    - CSV is a fallback for manual inspection or Drive sharing.
    - The directory is created if it does not exist.
    """
    out_dir = Path(export_dir or DEFAULT_EXPORT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    ext  = "parquet" if fmt == "parquet" else "csv"
    path = out_dir / f"{filename}.{ext}"

    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

    print(f"  Saved {len(df):,} rows → {path}")
    return str(path)


__all__ = [
    "build_entity_motif_features",
    "build_entity_feature_wide",
    "build_window_motif_features",
    "save_features",
    "DEFAULT_EXPORT_DIR",
]



cell 6:   

# ============================================================
# CELL 6  — Vectorized Graph Feature Engineering
#
# Unit of analysis : NODE
# Approach         : GFP-style statistical proxies via groupby
# Critical fix     : NO is_sar / SAR-based columns in features.
#                    The label is attached ONCE at the end,
#                    only as the target variable — never as input.
#
# Output: outputs/motif/node_features.parquet
#   One row per node, columns = structural + statistical features
#   derived purely from transaction topology and amounts.
# ============================================================

import os
import gc
import json
import numpy as np
import pandas as pd

OUTPUT_DIR       = "/content/drive/MyDrive/AML/outputs"
MOTIF_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "motif")
WINDOW_SIZE      = 7    # steps per rolling window (1 week)
os.makedirs(MOTIF_OUTPUT_DIR, exist_ok=True)


# ============================================================
# Step 1 — Load transactions
# ============================================================

print("[1/5] Loading transactions...")
df = pd.read_parquet(os.path.join(OUTPUT_DIR, "transactions.parquet"))
df["src_node"] = df["src_node"].astype(np.int64)
df["dst_node"] = df["dst_node"].astype(np.int64)
df["step"]     = df["step"].astype(np.int32)
df["amount"]   = df["amount"].astype(np.float32)

# is_sar is loaded but ONLY used at Step 4 to build the label.
# It is NEVER passed into any feature aggregation below.
if "is_sar" not in df.columns:
    df["is_sar"] = np.int8(0)
else:
    df["is_sar"] = df["is_sar"].astype(np.int8)

print(f"      {len(df):,} transactions  |  "
      f"steps {df['step'].min()}-{df['step'].max()}  |  "
      f"SAR rate: {df['is_sar'].mean()*100:.3f}%")

# Separate the label column so it cannot accidentally be
# included in any feature computation below.
labels_src = df[["src_node", "is_sar"]].copy()
labels_dst = df[["dst_node", "is_sar"]].copy()
df_feats   = df.drop(columns=["is_sar"])   # feature-safe view
del df
gc.collect()


# ============================================================
# Step 2 — Per-window node features (GFP-style proxies)
#
# For each rolling 7-step window, we compute per-node
# aggregations that proxy the structural patterns GFP searches
# for, but using only vectorized groupby — no subgraph search.
#
# Features computed (ALL label-free):
#
# Outgoing (fan-out proxy):
#   out_count, out_amount_{sum,mean,std,max,min}
#   out_n_unique_dst   ← fan-out cardinality
#   fanout_score       = out_n_unique_dst / out_count
#
# Incoming (fan-in proxy):
#   in_count, in_amount_{sum,mean,std,max,min}
#   in_n_unique_src    ← fan-in cardinality
#   fanin_score        = in_n_unique_src / in_count
#
# Relay / scatter-gather proxy (node is both sender & receiver):
#   relay_flag         = 1 if in_count > 0 and out_count > 0
#   relay_ratio        = out_amount_sum / in_amount_sum
#   amount_preservation= 1 - |relay_ratio - 1|  (≈1 for layering)
#   gather_scatter     = fanin_score * fanout_score
#
# Cycle proxy (sends to someone it received from):
#   cycle_proxy        = |sent-to ∩ received-from|  (set overlap)
#
# Temporal / velocity:
#   out_velocity       = out_count / WINDOW_SIZE
#   in_velocity        = in_count  / WINDOW_SIZE
#   out_step_std       = std of steps of outgoing tx (burstiness)
#   in_step_std        = std of steps of incoming tx (burstiness)
#
# Concentration (HHI-style):
#   out_concentration  = sum(amount_to_dst^2) / sum(amount)^2
#   in_concentration   = sum(amount_from_src^2) / sum(amount)^2
# ============================================================

print(f"\n[2/5] Computing per-window node features "
      f"(window={WINDOW_SIZE} steps)...")

step_min = int(df_feats["step"].min())
step_max = int(df_feats["step"].max())
all_parts = []


def _window_features(wdf: pd.DataFrame, ws: int) -> pd.DataFrame:
    """
    All feature computations for one time window.
    wdf must NOT contain is_sar.
    Returns one row per node active in this window.
    """

    # ── Outgoing aggregations ─────────────────────────────────
    out = wdf.groupby("src_node").agg(
        out_count        =("amount", "count"),
        out_amount_sum   =("amount", "sum"),
        out_amount_mean  =("amount", "mean"),
        out_amount_std   =("amount", "std"),
        out_amount_max   =("amount", "max"),
        out_amount_min   =("amount", "min"),
        out_n_unique_dst =("dst_node", "nunique"),
        out_step_std     =("step",   "std"),
    ).reset_index().rename(columns={"src_node": "node"})

    # ── Incoming aggregations ─────────────────────────────────
    inc = wdf.groupby("dst_node").agg(
        in_count         =("amount", "count"),
        in_amount_sum    =("amount", "sum"),
        in_amount_mean   =("amount", "mean"),
        in_amount_std    =("amount", "std"),
        in_amount_max    =("amount", "max"),
        in_amount_min    =("amount", "min"),
        in_n_unique_src  =("src_node", "nunique"),
        in_step_std      =("step",   "std"),
    ).reset_index().rename(columns={"dst_node": "node"})

    # ── Merge out + in ────────────────────────────────────────
    feat = pd.merge(out, inc, on="node", how="outer").fillna(0)

    # ── Structural proxy features ─────────────────────────────

    # Fan-out / fan-in scores (normalised cardinality)
    feat["fanout_score"] = (feat["out_n_unique_dst"]
                            / (feat["out_count"] + 1e-6))
    feat["fanin_score"]  = (feat["in_n_unique_src"]
                            / (feat["in_count"] + 1e-6))

    # Scatter-gather: high fan-in AND high fan-out at same node
    feat["gather_scatter_score"] = (feat["fanin_score"]
                                    * feat["fanout_score"])

    # Relay / layering
    feat["relay_flag"]          = (
        (feat["in_count"] > 0) & (feat["out_count"] > 0)
    ).astype(np.int8)
    feat["relay_ratio"]         = (feat["out_amount_sum"]
                                   / (feat["in_amount_sum"] + 1e-6))
    feat["amount_preservation"] = 1.0 - np.abs(feat["relay_ratio"] - 1.0)

    # Velocity
    feat["out_velocity"] = feat["out_count"] / WINDOW_SIZE
    feat["in_velocity"]  = feat["in_count"]  / WINDOW_SIZE

    # ── Concentration (HHI-style, outgoing) ───────────────────
    # Sum of squared per-dst amounts / total-out-amount²
    # High → one dominant recipient (suspicious structuring)
    out_by_dst = (wdf.groupby(["src_node", "dst_node"])["amount"]
                  .sum().reset_index())
    out_by_dst["sq"] = out_by_dst["amount"] ** 2
    hhi_num = (out_by_dst.groupby("src_node")["sq"]
               .sum().reset_index()
               .rename(columns={"src_node": "node", "sq": "_hhi_num"}))
    hhi_den = (wdf.groupby("src_node")["amount"]
               .sum() ** 2).reset_index()
    hhi_den.columns = ["node", "_hhi_den"]
    hhi = (hhi_num.merge(hhi_den, on="node", how="left")
           .assign(out_concentration=lambda x:
                   x["_hhi_num"] / (x["_hhi_den"] + 1e-9))
           [["node", "out_concentration"]])
    feat = feat.merge(hhi, on="node", how="left").fillna(0)
    del out_by_dst, hhi_num, hhi_den, hhi

    # Incoming concentration (HHI-style)
    inc_by_src = (wdf.groupby(["dst_node", "src_node"])["amount"]
                  .sum().reset_index())
    inc_by_src["sq"] = inc_by_src["amount"] ** 2
    ihhi_num = (inc_by_src.groupby("dst_node")["sq"]
                .sum().reset_index()
                .rename(columns={"dst_node": "node", "sq": "_ihhi_num"}))
    ihhi_den = (wdf.groupby("dst_node")["amount"]
                .sum() ** 2).reset_index()
    ihhi_den.columns = ["node", "_ihhi_den"]
    ihhi = (ihhi_num.merge(ihhi_den, on="node", how="left")
            .assign(in_concentration=lambda x:
                    x["_ihhi_num"] / (x["_ihhi_den"] + 1e-9))
            [["node", "in_concentration"]])
    feat = feat.merge(ihhi, on="node", how="left").fillna(0)
    del inc_by_src, ihhi_num, ihhi_den, ihhi

    # ── Cycle proxy (set intersection, label-free) ────────────
    # Counts how many nodes a given node BOTH sends to AND receives from.
    # A non-zero value is a necessary (not sufficient) condition for
    # participation in a cycle.
    src_to_dsts = wdf.groupby("src_node")["dst_node"].apply(set)
    dst_to_srcs = wdf.groupby("dst_node")["src_node"].apply(set)
    common = src_to_dsts.index.intersection(dst_to_srcs.index)
    if len(common) > 0:
        cycle_proxy = pd.Series(
            {n: len(src_to_dsts[n] & dst_to_srcs[n]) for n in common},
            name="cycle_proxy",
        ).astype(np.float32).reset_index()
        cycle_proxy.columns = ["node", "cycle_proxy"]
        feat = feat.merge(cycle_proxy, on="node", how="left")
    else:
        feat["cycle_proxy"] = 0.0
    feat["cycle_proxy"] = feat["cycle_proxy"].fillna(0)
    del src_to_dsts, dst_to_srcs

    # ── Window metadata ───────────────────────────────────────
    feat["window_start"] = ws

    return feat


for ws in range(step_min, step_max + 1, WINDOW_SIZE):
    we  = ws + WINDOW_SIZE - 1
    wdf = df_feats[(df_feats["step"] >= ws) & (df_feats["step"] <= we)]
    if wdf.empty:
        continue

    feat = _window_features(wdf, ws)
    all_parts.append(feat)

    del wdf, feat
    gc.collect()
    print(f"      window [{ws:3d}-{we:3d}] done", end="\r")

print(f"\n      {len(all_parts)} windows processed.")


# ============================================================
# Step 3 — Aggregate across windows → one row per node
#
# For each feature we compute: mean, max, std across windows.
# This captures temporal evolution without leaking labels.
# ============================================================

print("\n[3/5] Aggregating across windows...")

all_feats = pd.concat(all_parts, ignore_index=True)
del all_parts
gc.collect()

agg_cols = [c for c in all_feats.columns
            if c not in ("node", "window_start")]

node_features = (all_feats
                 .groupby("node")[agg_cols]
                 .agg(["mean", "max", "std"])
                 .fillna(0))
node_features.columns = [f"{col}_{stat}"
                          for col, stat in node_features.columns]
node_features = node_features.reset_index()

del all_feats
gc.collect()
print(f"      node_features shape: {node_features.shape}")


# ============================================================
# Step 4 — Attach label
#
# is_sar is used ONLY here, as the target variable.
# It is never part of the feature matrix.
#
# Label definition (AMLGentex §3):
#   A node is positive if it appears in ANY SAR-flagged
#   transaction (as sender OR receiver) within the full window.
# ============================================================

print("\n[4/5] Attaching label (is_sar)...")

sar_nodes = set(
    labels_src[labels_src["is_sar"] == 1]["src_node"].tolist()
    + labels_dst[labels_dst["is_sar"] == 1]["dst_node"].tolist()
)
node_features["label"] = (node_features["node"]
                           .isin(sar_nodes)
                           .astype(np.int8))

n_pos = node_features["label"].sum()
n_tot = len(node_features)
print(f"      Nodes     : {n_tot:,}")
print(f"      SAR nodes : {n_pos:,}  ({n_pos/n_tot*100:.2f}%)")
print(f"      Normal    : {n_tot-n_pos:,}  ({(n_tot-n_pos)/n_tot*100:.2f}%)")

del labels_src, labels_dst
gc.collect()


# ============================================================
# Step 5 — Save
# ============================================================

print("\n[5/5] Saving node_features.parquet...")

p_feats = os.path.join(MOTIF_OUTPUT_DIR, "node_features.parquet")
node_features.to_parquet(p_feats, index=False)

feature_cols = [c for c in node_features.columns
                if c not in ("node", "label")]
p_cols = os.path.join(MOTIF_OUTPUT_DIR, "feature_columns.json")
with open(p_cols, "w") as f:
    json.dump(feature_cols, f, indent=2)

size_kb = os.path.getsize(p_feats) / 1024
print(f"      node_features : {size_kb:.1f} KB  "
      f"({n_tot:,} nodes × {len(feature_cols)} features)")
print(f"      feature_cols  : {p_cols}")

del node_features, df_feats
gc.collect()
print("\nDone.")



cell 7: 



# ============================================================
# CELL 7 (corrected) — AMLGentex Features + XGBoost
#
# Inputs:
#   outputs/motif/node_features.parquet  ← Cell 6 graph features
#   outputs/transactions.parquet         ← raw transactions
#
# Critical fixes vs previous version:
#   FIX 1 — ALL is_sar / SAR-based sub-window features removed.
#            is_sar is used only to build the label at Step 3.
#   FIX 2 — Stratified random split (not time-based).
#            Matches AMLGentex §6 main protocol (transductive).
#   FIX 3 — Sub-window features use df_feats (no is_sar column)
#            so leakage is structurally impossible.
#
# Evaluation metric: Average Precision at Recall ≥ 0.6 (P@R>0.6)
# ============================================================

import os, gc, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    classification_report,
)
import xgboost as xgb

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# ============================================================
# USER CONFIG
# ============================================================

OUTPUT_DIR       = "/content/drive/MyDrive/AML/outputs"
MOTIF_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "motif")
MODEL_DIR        = os.path.join(OUTPUT_DIR, "model")

# AMLGentex sub-window setup (Appendix C / §4.3)
# 112 steps total, m=4 sub-windows of size 28.
N_SUBWINDOWS = 4
TOTAL_STEPS  = 112
SUB_WIN_SIZE = TOTAL_STEPS // N_SUBWINDOWS   # = 28

# Stratified split ratios (FIX 2)
TEST_SIZE = 0.20    # 80% train / 20% test
VAL_SIZE  = 0.15    # of the 80% train → validation for early stop
RANDOM_STATE = 42

# XGBoost
N_ESTIMATORS = 500
EARLY_STOP   = 30

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# Step 1 — Load inputs
# ============================================================

print("[1/6] Loading inputs...")

tx = pd.read_parquet(os.path.join(OUTPUT_DIR, "transactions.parquet"))
tx["src_node"] = tx["src_node"].astype(np.int64)
tx["dst_node"] = tx["dst_node"].astype(np.int64)
tx["step"]     = tx["step"].astype(np.int32)
tx["amount"]   = tx["amount"].astype(np.float32)
if "is_sar" not in tx.columns:
    tx["is_sar"] = np.int8(0)
else:
    tx["is_sar"] = tx["is_sar"].astype(np.int8)

graph_feats = pd.read_parquet(
    os.path.join(MOTIF_OUTPUT_DIR, "node_features.parquet")
)

# FIX 1 — isolate label column early.
# df_feats is the label-free view used for all feature construction.
# is_sar stays in tx only for label derivation at Step 3.
df_feats = tx.drop(columns=["is_sar"])

print(f"      transactions : {len(tx):,}")
print(f"      graph_feats  : {graph_feats.shape}")
print(f"      label column : isolated, NOT in df_feats")


# ============================================================
# Step 2 — AMLGentex Table-1 sub-window features
#
# Mirrors Appendix C, Table 1 exactly.
# Computed over m=4 non-overlapping sub-windows of 28 steps.
# Concatenated per node → model sees temporal evolution.
#
# df_feats has NO is_sar column, so leakage is structurally
# impossible regardless of what aggregations are applied.
# ============================================================

print(f"\n[2/6] AMLGentex sub-window features "
      f"(m={N_SUBWINDOWS} × {SUB_WIN_SIZE} steps)...")


def _subwindow_features(wdf: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    Compute Table-1 features for one sub-window.
    wdf must NOT contain is_sar (enforced by caller passing df_feats).
    """

    # ── Outgoing (spending) — Table 1 rows 1-7 ───────────────
    out = wdf.groupby("src_node")["amount"].agg(
        sum_spending   ="sum",
        mean_spending  ="mean",
        median_spending="median",
        std_spending   ="std",
        max_spending   ="max",
        min_spending   ="min",
        count_spending ="count",
    ).reset_index().rename(columns={"src_node": "node"})

    # ── All-transaction stats — Table 1 rows 8-13 ────────────
    both_amounts = pd.concat([
        wdf[["src_node", "amount"]].rename(columns={"src_node": "node"}),
        wdf[["dst_node", "amount"]].rename(columns={"dst_node": "node"}),
    ], ignore_index=True)
    total_stats = both_amounts.groupby("node")["amount"].agg(
        total_sum   ="sum",
        total_mean  ="mean",
        total_median="median",
        total_std   ="std",
        total_max   ="max",
        total_min   ="min",
    ).reset_index()

    # ── In/out counts — Table 1 rows 14-15 ───────────────────
    count_in  = (wdf.groupby("dst_node").size()
                 .rename("count_in").reset_index()
                 .rename(columns={"dst_node": "node"}))
    count_out = (wdf.groupby("src_node").size()
                 .rename("count_out").reset_index()
                 .rename(columns={"src_node": "node"}))

    # ── Unique counterparties — Table 1 rows 16-17 ───────────
    uniq_in  = (wdf.groupby("dst_node")["src_node"].nunique()
                .rename("count_unique_in").reset_index()
                .rename(columns={"dst_node": "node"}))
    uniq_out = (wdf.groupby("src_node")["dst_node"].nunique()
                .rename("count_unique_out").reset_index()
                .rename(columns={"src_node": "node"}))

    # ── Days active — proxy for count_days_in_bank (Table 1 row 18) ──
    step_range = pd.concat([
        wdf[["src_node", "step"]].rename(columns={"src_node": "node"}),
        wdf[["dst_node", "step"]].rename(columns={"dst_node": "node"}),
    ]).groupby("node")["step"].agg(
        step_first="min",
        step_last ="max",
    ).reset_index()
    step_range["days_active"] = (step_range["step_last"]
                                  - step_range["step_first"] + 1)

    # ── Structural ratios (not in Table 1 but aligned with GFP) ─
    # These are derived purely from counts/amounts, no label.

    # ── Merge all into one row per node ───────────────────────
    node_ids = pd.DataFrame(
        {"node": both_amounts["node"].unique()}, dtype=np.int64
    )
    merged = (node_ids
              .merge(out,                              on="node", how="left")
              .merge(total_stats,                      on="node", how="left")
              .merge(count_in,                         on="node", how="left")
              .merge(count_out,                        on="node", how="left")
              .merge(uniq_in,                          on="node", how="left")
              .merge(uniq_out,                         on="node", how="left")
              .merge(step_range[["node","days_active"]],
                     on="node", how="left")
              .fillna(0))

    # Derived ratios (pure structure, no label)
    merged["in_out_count_ratio"]  = (merged["count_in"]
                                     / (merged["count_out"] + 1e-6))
    merged["spend_total_ratio"]   = (merged["sum_spending"]
                                     / (merged["total_sum"] + 1e-6))
    merged["unique_in_out_ratio"] = (merged["count_unique_in"]
                                     / (merged["count_unique_out"] + 1e-6))

    # Rename with sub-window suffix
    rename = {c: f"{c}{suffix}" for c in merged.columns if c != "node"}
    return merged.rename(columns=rename)


all_sw = []
for sw in range(N_SUBWINDOWS):
    ws     = sw * SUB_WIN_SIZE
    we     = ws + SUB_WIN_SIZE - 1
    suffix = f"_w{sw}"

    # FIX 1: pass df_feats (no is_sar) to the function
    wdf = df_feats[(df_feats["step"] >= ws) & (df_feats["step"] <= we)]
    if wdf.empty:
        continue

    sw_feat = _subwindow_features(wdf, suffix)
    all_sw.append(sw_feat)
    del wdf, sw_feat
    gc.collect()
    print(f"      sub-window {sw} [{ws}-{we}] done", end="\r")

print()

sw_combined = all_sw[0]
for frame in all_sw[1:]:
    sw_combined = sw_combined.merge(frame, on="node", how="outer")
sw_combined = sw_combined.fillna(0)
del all_sw, df_feats
gc.collect()
print(f"      sub-window feature shape: {sw_combined.shape}")


# ============================================================
# Step 3 — Merge feature sets and attach label
# ============================================================

print("\n[3/6] Merging features + attaching label...")

# graph_feats from Cell 6 already has 'label' column.
# We use its label (derived the same way) and drop it from
# graph_feats before merging to avoid duplicate columns.
graph_feats_X = graph_feats.drop(columns=["label"], errors="ignore")

node_matrix = (sw_combined
               .merge(graph_feats_X, on="node", how="outer")
               .fillna(0))

# FIX 1 — Derive label from tx["is_sar"] directly (single source of truth)
# Never from any feature column.
sar_nodes = set(
    tx[tx["is_sar"] == 1]["src_node"].tolist()
    + tx[tx["is_sar"] == 1]["dst_node"].tolist()
)
node_matrix["label"] = (node_matrix["node"]
                         .isin(sar_nodes)
                         .astype(np.int8))

n_pos = int(node_matrix["label"].sum())
n_tot = len(node_matrix)
print(f"      Node matrix : {node_matrix.shape}")
print(f"      SAR nodes   : {n_pos:,}  ({n_pos/n_tot*100:.2f}%)")
print(f"      Normal      : {n_tot-n_pos:,}  ({(n_tot-n_pos)/n_tot*100:.2f}%)")

del sw_combined, graph_feats_X, tx
gc.collect()


# ============================================================
# Step 4 — Train / test split
#
# FIX 2 — Stratified random split on NODES.
# Matches AMLGentex main protocol (transductive features,
# node-level label stratification).
#
# Why NOT a time-based split here:
#   - Laundering nodes are active throughout all 112 steps.
#   - A cutoff at step 56 leaves almost no positives in train.
#   - Time-based split is only appropriate for the "changed
#     behavior" ablation experiment (AMLGentex §6 right panel).
# ============================================================

print("\n[4/6] Stratified train/test split...")

EXCLUDE = {"node", "label"}
feature_cols = [c for c in node_matrix.columns if c not in EXCLUDE]

X = node_matrix[feature_cols].values.astype(np.float32)
y = node_matrix["label"].values

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size    = TEST_SIZE,
    stratify     = y,
    random_state = RANDOM_STATE,
)

# Validation split from train (for early stopping)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size    = VAL_SIZE,
    stratify     = y_train_full,
    random_state = RANDOM_STATE,
)

print(f"      Train : {len(X_train):,}  "
      f"(pos={y_train.sum():,}, {y_train.mean()*100:.2f}%)")
print(f"      Val   : {len(X_val):,}  "
      f"(pos={y_val.sum():,}, {y_val.mean()*100:.2f}%)")
print(f"      Test  : {len(X_test):,}  "
      f"(pos={y_test.sum():,}, {y_test.mean()*100:.2f}%)")


# ============================================================
# Step 5 — XGBoost
# ============================================================

print("\n[5/6] Training XGBoost...")

n_pos_tr = y_train.sum()
n_neg_tr = len(y_train) - n_pos_tr
scale_pos = float(n_neg_tr) / float(n_pos_tr + 1e-9)
print(f"      scale_pos_weight = {scale_pos:.1f}")

model = xgb.XGBClassifier(
    n_estimators          = N_ESTIMATORS,
    max_depth             = 4,
    learning_rate         = 0.05,
    subsample             = 0.6,
    colsample_bytree      = 0.5,
    min_child_weight      = 20,
    scale_pos_weight      = scale_pos,
    eval_metric           = "aucpr",
    early_stopping_rounds = EARLY_STOP,
    random_state          = RANDOM_STATE,
    tree_method           = "hist",
    n_jobs                = -1,
    reg_alpha         = 0.1,    # L1 — adds sparsity
    reg_lambda        = 5.0,    # L2 — was 1.0 (default)
    gamma             = 1.0    # minimum split gain — new
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)
print(f"\n      Best iteration : {model.best_iteration}")


# ── Evaluation ───────────────────────────────────────────────

def p_at_recall(y_true, y_prob, recall_min=0.6):
    """Average precision for the PR-curve region where recall >= recall_min."""
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    mask = rec >= recall_min
    if mask.sum() < 2:
        return 0.0
    p_f = prec[mask]
    r_f = rec[mask]
    idx = np.argsort(r_f)
    denom = r_f[idx].max() - r_f[idx].min()
    if denom < 1e-9:
        return 0.0
    return float(np.trapz(p_f[idx], r_f[idx]) / denom)


y_prob_tr   = model.predict_proba(X_train)[:, 1]
y_prob_test = model.predict_proba(X_test)[:, 1]

metrics = {
    "train": {
        "roc_auc":  round(float(roc_auc_score(y_train, y_prob_tr)),   4),
        "pr_auc":   round(float(average_precision_score(y_train, y_prob_tr)), 4),
        "p_at_r06": round(p_at_recall(y_train, y_prob_tr), 4),
    },
    "test": {
        "roc_auc":  round(float(roc_auc_score(y_test, y_prob_test)),   4),
        "pr_auc":   round(float(average_precision_score(y_test, y_prob_test)), 4),
        "p_at_r06": round(p_at_recall(y_test, y_prob_test), 4),
    },
}

print(f"\n      {'Metric':<26} {'Train':>8}  {'Test':>8}")
print(f"      {'-'*44}")
for k in ("roc_auc", "pr_auc", "p_at_r06"):
    label = {"roc_auc":"ROC-AUC","pr_auc":"PR-AUC",
             "p_at_r06":"P@R>0.6 (paper metric)"}[k]
    print(f"      {label:<26} "
          f"{metrics['train'][k]:>8.4f}  {metrics['test'][k]:>8.4f}")

y_pred = (y_prob_test >= 0.5).astype(int)
print(f"\n      Classification report (test, threshold=0.5):")
print(classification_report(y_test, y_pred,
                             target_names=["normal","SAR"], digits=4))


# ── Optional PR-curve plot ────────────────────────────────────
if HAS_PLT:
    fig, ax = plt.subplots(figsize=(7, 5))
    for lbl, yt, yp in [("Train", y_train, y_prob_tr),
                          ("Test",  y_test,  y_prob_test)]:
        p, r, _ = precision_recall_curve(yt, yp)
        ap = average_precision_score(yt, yp)
        ax.plot(r, p, label=f"{lbl} (AP={ap:.3f})")
    ax.axvline(0.6, color="gray", linestyle="--",
               alpha=0.5, label="Recall threshold")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("PR Curve — AML Node Classification")
    ax.legend(); plt.tight_layout()
    pr_path = os.path.join(MODEL_DIR, "pr_curve.png")
    plt.savefig(pr_path, dpi=150); plt.show()
    print(f"      PR curve → {pr_path}")


# ============================================================
# Step 6 — Save artifacts
# ============================================================

print("\n[6/6] Saving...")

# Model
model_path = os.path.join(MODEL_DIR, "xgb_aml.json")
model.save_model(model_path)

# Feature importance (XGBoost gain)
score_dict = model.get_booster().get_score(importance_type="gain")
imp_df = pd.DataFrame([
    {"feature": feature_cols[int(k[1:])], "gain": v}
    for k, v in score_dict.items()
]).sort_values("gain", ascending=False).reset_index(drop=True)

print(f"\n      Top 20 features by gain:")
print(imp_df.head(20).to_string(index=False))

# SHAP
if HAS_SHAP:
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test[:2000])
    shap_df   = pd.DataFrame({
        "feature": feature_cols,
        "shap_mean_abs": np.abs(shap_vals).mean(axis=0),
    }).sort_values("shap_mean_abs", ascending=False).reset_index(drop=True)
    print(f"\n      Top 20 features by SHAP:")
    print(shap_df.head(20).to_string(index=False))
    if HAS_PLT:
        shap.summary_plot(shap_vals, X_test[:2000],
                          feature_names=feature_cols,
                          max_display=20, show=False)
        sp = os.path.join(MODEL_DIR, "shap_summary.png")
        plt.savefig(sp, dpi=150, bbox_inches="tight"); plt.show()
        print(f"      SHAP → {sp}")

# Test predictions
preds_df = pd.DataFrame({
    "node":     node_matrix["node"].values[
                    len(X_train_full):len(X_train_full)+len(X_test)],
    "label":    y_test,
    "prob_sar": y_prob_test,
    "pred_sar": y_pred,
})
preds_path = os.path.join(MODEL_DIR, "test_predictions.parquet")
preds_df.to_parquet(preds_path, index=False)

# Metrics JSON
metrics["n_features"]     = len(feature_cols)
metrics["n_train"]        = int(len(X_train))
metrics["n_test"]         = int(len(X_test))
metrics["n_pos_train"]    = int(n_pos_tr)
metrics["n_pos_test"]     = int(y_test.sum())
metrics["best_iteration"] = int(model.best_iteration)
m_path = os.path.join(MODEL_DIR, "metrics.json")
with open(m_path, "w") as f:
    json.dump(metrics, f, indent=2)

# Feature columns (needed to reload model)
fc_path = os.path.join(MODEL_DIR, "feature_columns.json")
with open(fc_path, "w") as f:
    json.dump(feature_cols, f, indent=2)

# Importance tables
imp_path = os.path.join(MODEL_DIR, "feature_importance.parquet")
imp_df.to_parquet(imp_path, index=False)

print("\nArtifacts saved:")
for lbl, path in [("xgb model",    model_path),
                   ("predictions",  preds_path),
                   ("metrics",      m_path),
                   ("feature_imp",  imp_path),
                   ("feature_cols", fc_path)]:
    print(f"   [{lbl:<14}]  {path}  "
          f"({os.path.getsize(path)/1024:.1f} KB)")

del X_train, X_val, X_test, y_train, y_val, y_test
del X_train_full, y_train_full
del node_matrix, graph_feats
gc.collect()
print("\nDone.")