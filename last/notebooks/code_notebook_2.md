from dataclasses import dataclass, field
from typing import List


@dataclass
class MotifConfig:
    """
    Runtime configuration for selective temporal motif mining.

    This config now serves two purposes:

    1. Motif definition
       Controls the temporal and amount-based constraints used by the
       exact matchers.

    2. Motif execution policy
       Controls how aggressively the exact branch is allowed to run,
       including support thresholds, statistical filtering, and memory caps.

    Notes
    -----
    - Exact motif mining is intended to run on candidate windows only,
      not on the full dataset globally.
    - `temporal_edges` from notebook 1 are the correct upstream substrate
      for exact or semi-exact motifs.
    - Aggregated graph objects such as snapshot edges or second-order edges
      are not substitutes for event-level motif search.

    Parameters
    ----------
    delta : int
        Maximum allowed step gap between consecutive events in one motif.

        Smaller values enforce tighter temporal continuity.
        Increase only if one step in the dataset is smaller than the
        intended business time unit.

    rho_min, rho_max : float
        Allowed range for consecutive amount ratios:

            a(i+1) / a(i)

        This is used to reject chains whose transferred amounts change too
        sharply from one hop to the next.

    r_min_fanin : int
        Minimum number of distinct incoming sources required for a fan-in motif.

    r_min_fanout : int
        Minimum number of distinct outgoing targets required for a fan-out motif.

    r_min_cycle : int
        Minimum support required to keep cycle motifs after matching.

    r_min_relay : int
        Minimum support required to keep relay motifs after matching.

    r_min_split_merge : int
        Minimum support required to keep split-merge motifs after matching.

    n_permutations : int
        Number of null-model permutations used when computing motif z-scores.

        Use small values for pipeline debugging.
        Increase only after the selective execution path is stable.

    z_min : float
        Minimum z-score required to keep a motif type when null-model
        filtering is enabled.

    max_nodes : int
        Soft design limit for the size of one motif instance.

    max_edges : int
        Soft design limit for the number of edges in one motif instance.

    max_instances : int
        Hard cap on total matched instances retained in memory per matcher run.

        This is a safety guard against RAM blow-up on dense candidate windows.
        Set to 0 to disable the cap.

    window_sizes : List[int]
        Window sizes retained for compatibility with older notebook logic.

        In the redesigned pipeline, notebook 1 is responsible for graph
        windowing and shard generation. Notebook 2 should usually inherit
        that windowing rather than redefining it here.
    """

    # =========================================================
    # Temporal constraint
    # =========================================================

    delta: int = 2

    # =========================================================
    # Amount ratio constraint
    # =========================================================

    rho_min: float = 0.3
    rho_max: float = 3.0

    # =========================================================
    # Minimum support by motif type
    # =========================================================

    # Require at least 3 distinct sources into one destination.
    r_min_fanin: int = 3

    # Require at least 3 distinct targets from one source.
    r_min_fanout: int = 3

    # A single observed cycle can already be suspicious.
    r_min_cycle: int = 1

    # A single observed relay chain can already be suspicious.
    r_min_relay: int = 1

    # A single observed split-merge can already be suspicious.
    r_min_split_merge: int = 1

    # =========================================================
    # Statistical filtering
    # =========================================================

    # Start small for debugging; increase only after the selective motif
    # pipeline is stable.
    n_permutations: int = 5

    z_min: float = 2.0

    # =========================================================
    # Search limits / memory guards
    # =========================================================

    max_nodes: int = 4
    max_edges: int = 5

    # Hard cap on retained instances across one matcher run.
    # Prevents unbounded RAM growth on dense candidate windows.
    # 0 = disabled.
    max_instances: int = 10_000

    # =========================================================
    # Legacy / compatibility window settings
    # =========================================================

    window_sizes: List[int] = field(
        default_factory=lambda: [7]
    )

    # New: explicit matcher execution control
    enabled_matchers: List[str] = field(
        default_factory=lambda: ["fanin", "fanout", "relay4"]
    )

    # New: candidate-window selection control
    candidate_window_mode: str = "top_k"
    top_k_windows: int = 10
    window_score_threshold: float = 0.0
    max_windows_exact: int = 10

    # New: export policy
    export_instances: bool = False


__all__ = [
    "MotifConfig",
]


"""
cell 2 — Event indexing + temporal shard loading for selective motif mining.

Redesign goals
--------------
1. Link notebook 2 to notebook 1's graph outputs.
2. Treat temporal_edges shards as the ONLY upstream object for exact / semi-exact motifs.
3. Reconstruct per-window canonical event tables from temporal relay shards.
4. Build local event indexes only for one selected candidate window at a time.

Expected upstream artifacts from notebook 1
-------------------------------------------
- windows_meta.parquet
- temporal_edges/w_{start}_{end}.parquet

This cell no longer assumes one monolithic temporal_edges.parquet file.
It works window-by-window from sharded graph outputs.
"""

from __future__ import annotations

import gc
import os
import re
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

EventDict = Dict[str, object]


# ---------------------------------------------------------------------------
# Defaults for graph-shard locations
# ---------------------------------------------------------------------------

GRAPH_OUTPUT_DIR = Path("/content/drive/MyDrive/AML/outputs")
TEMPORAL_SHARD_DIR = GRAPH_OUTPUT_DIR / "temporal_edges"
WINDOW_META_PATH = GRAPH_OUTPUT_DIR / "windows_meta.parquet"


# ---------------------------------------------------------------------------
# Canonical event index builder
# ---------------------------------------------------------------------------

def build_event_indexes(
    event_df: pd.DataFrame,
    src_col: str = "src_node",
    dst_col: str = "dst_node",
    step_col: str = "step",
    amount_col: str = "amount",
    alert_col: str = "is_sar",
) -> tuple[dict, dict, dict, dict]:
    """
    Build local search indexes from ONE event window.

    Parameters
    ----------
    event_df : pd.DataFrame
        Canonical per-window event table with one row per transaction event.
        Must be sorted by step ascending.
    src_col, dst_col, step_col, amount_col, alert_col : str
        Column name overrides.

    Returns
    -------
    out_index  : {src_node -> [EventDict, ...]}
    in_index   : {dst_node -> [EventDict, ...]}
    step_index : {step -> [EventDict, ...]}
    out_steps  : {src_node -> [step, ...]}

    Notes
    -----
    - This function now explicitly returns FOUR outputs.
    - It is intended for local candidate-window motif mining only.
    - It accepts pandas or cuDF input, but converts once to pandas.
    """
    if hasattr(event_df, "to_pandas"):
        event_df = event_df.to_pandas()

    df = _normalize_columns(event_df, src_col, dst_col, step_col, amount_col, alert_col)
    df = _ensure_event_id(df)
    _validate_required(df)

    if os.getenv("MOTIF_DEBUG") == "1":
        if not df["step"].is_monotonic_increasing:
            raise ValueError(
                "event_df must be sorted by 'step' before indexing. "
                "Binary search in edges_after_step() assumes monotonic order."
            )

    has_alert = "is_sar" in df.columns

    out_index: dict = defaultdict(list)
    in_index: dict = defaultdict(list)
    step_index: dict = defaultdict(list)

    for row in df.itertuples(index=False):
        e: EventDict = {
            "event_id": int(row.event_id),
            "step": int(row.step),
            "src": int(row.src_node),
            "dst": int(row.dst_node),
            "amount": float(row.amount),
            "is_sar": int(row.is_sar) if has_alert else 0,
        }
        out_index[e["src"]].append(e)
        in_index[e["dst"]].append(e)
        step_index[e["step"]].append(e)

    out_index = dict(out_index)
    in_index = dict(in_index)
    step_index = dict(step_index)
    out_steps = {
        node: [e["step"] for e in edges]
        for node, edges in out_index.items()
    }

    gc.collect()
    return out_index, in_index, step_index, out_steps


def edges_after_step(
    out_index: dict,
    node: int,
    step: int,
    out_steps: dict,
) -> List[EventDict]:
    """
    Return all outgoing edges from `node` with step > `step`.
    """
    bucket = out_index.get(node)
    if not bucket:
        return []
    idx = bisect_right(out_steps[node], step)
    return bucket[idx:]


# ---------------------------------------------------------------------------
# Temporal-edge shard loading
# ---------------------------------------------------------------------------

def load_windows_meta(meta_path: str | Path = WINDOW_META_PATH) -> pd.DataFrame:
    """
    Load notebook 1 window metadata.

    Expected columns
    ----------------
    window, start, end, n_tx, n_temporal, n_second, n_snapshot
    """
    meta_path = Path(meta_path)
    if not meta_path.exists():
        raise FileNotFoundError(
            f"windows_meta not found: {meta_path}. "
            "Run notebook 1 graph pipeline first."
        )

    meta = pd.read_parquet(meta_path)
    required = {"window", "start", "end"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(
            f"windows_meta is missing required columns: {sorted(missing)}. "
            f"Available columns: {sorted(meta.columns)}"
        )
    return meta.sort_values(["start", "end"]).reset_index(drop=True)


def temporal_shard_path(
    step_start: int,
    step_end: int,
    shard_dir: str | Path = TEMPORAL_SHARD_DIR,
) -> Path:
    """
    Resolve shard path for one temporal_edges window.
    """
    shard_dir = Path(shard_dir)
    return shard_dir / f"w_{step_start}_{step_end}.parquet"


def load_temporal_shard(
    step_start: int,
    step_end: int,
    shard_dir: str | Path = TEMPORAL_SHARD_DIR,
) -> pd.DataFrame:
    """
    Load one temporal_edges shard from notebook 1.
    """
    path = temporal_shard_path(step_start, step_end, shard_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Temporal shard not found: {path}. "
            "Check notebook 1 outputs and shard naming."
        )
    te = pd.read_parquet(path)
    _validate_temporal_edge_schema(te)
    return te


def iter_temporal_shards(
    meta_df: pd.DataFrame,
    shard_dir: str | Path = TEMPORAL_SHARD_DIR,
    min_temporal_edges: int = 1,
) -> Iterator[tuple[dict, pd.DataFrame]]:
    """
    Iterate over temporal_edges shards using windows_meta.

    Yields
    ------
    (window_info, temporal_edges_df)
    """
    shard_dir = Path(shard_dir)

    for row in meta_df.itertuples(index=False):
        n_temporal = int(getattr(row, "n_temporal", 0))
        if n_temporal < min_temporal_edges:
            continue

        step_start = int(row.start)
        step_end = int(row.end)
        te = load_temporal_shard(step_start, step_end, shard_dir)

        yield (
            {
                "window": int(row.window),
                "start": step_start,
                "end": step_end,
                "n_tx": int(getattr(row, "n_tx", 0)),
                "n_temporal": n_temporal,
                "n_second": int(getattr(row, "n_second", 0)),
                "n_snapshot": int(getattr(row, "n_snapshot", 0)),
            },
            te,
        )


# ---------------------------------------------------------------------------
# Convert temporal_edges shard -> canonical event window
# ---------------------------------------------------------------------------

def temporal_edges_to_event_df(temporal_edges: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct a canonical event table from one temporal_edges shard.

    Why this exists
    ---------------
    Notebook 1 emits temporal relay pairs, not one-row-per-event transaction
    windows. Exact matchers in notebook 2 expect canonical event rows with:

        event_id, src_node, dst_node, step, amount, is_sar

    This function recovers unique event rows from the pair table.

    Output columns
    --------------
    event_id, src_node, dst_node, step, amount, is_sar
    """
    _validate_temporal_edge_schema(temporal_edges)

    left = temporal_edges[
        ["src_1", "dst_1", "step_1", "amount_1", "alert_1"]
    ].rename(columns={
        "src_1": "src_node",
        "dst_1": "dst_node",
        "step_1": "step",
        "amount_1": "amount",
        "alert_1": "is_sar",
    })

    right = temporal_edges[
        ["src_2", "dst_2", "step_2", "amount_2", "alert_2"]
    ].rename(columns={
        "src_2": "src_node",
        "dst_2": "dst_node",
        "step_2": "step",
        "amount_2": "amount",
        "alert_2": "is_sar",
    })

    events = pd.concat([left, right], ignore_index=True)

    # Deduplicate identical event rows recovered from multiple relay pairs.
    events = (
        events
        .drop_duplicates(subset=["src_node", "dst_node", "step", "amount", "is_sar"])
        .sort_values(["step", "src_node", "dst_node", "amount"])
        .reset_index(drop=True)
    )

    events["event_id"] = range(len(events))
    events["src_node"] = events["src_node"].astype("int64")
    events["dst_node"] = events["dst_node"].astype("int64")
    events["step"] = events["step"].astype("int32")
    events["amount"] = events["amount"].astype("float32")
    events["is_sar"] = events["is_sar"].astype("int8")

    return events[
        ["event_id", "src_node", "dst_node", "step", "amount", "is_sar"]
    ]


def load_event_window_from_temporal_shard(
    step_start: int,
    step_end: int,
    shard_dir: str | Path = TEMPORAL_SHARD_DIR,
) -> pd.DataFrame:
    """
    One-step helper:
        temporal_edges shard -> canonical event window
    """
    te = load_temporal_shard(step_start, step_end, shard_dir)
    return temporal_edges_to_event_df(te)


# ---------------------------------------------------------------------------
# Legacy helper retained for compatibility
# ---------------------------------------------------------------------------

def filter_window(
    event_df: pd.DataFrame,
    step_start: int,
    step_end: int,
) -> pd.DataFrame:
    """
    Legacy slice helper retained for compatibility.
    Prefer notebook 1 shard loading over slicing a monolithic dataframe.
    """
    mask = (event_df["step"] >= step_start) & (event_df["step"] <= step_end)
    return event_df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal validators / normalizers
# ---------------------------------------------------------------------------

def _validate_temporal_edge_schema(te: pd.DataFrame) -> None:
    required = {
        "src_1", "dst_1", "step_1", "amount_1", "alert_1",
        "src_2", "dst_2", "step_2", "amount_2", "alert_2",
    }
    missing = required - set(te.columns)
    if missing:
        raise ValueError(
            f"temporal_edges shard is missing required columns: {sorted(missing)}. "
            f"Available columns: {sorted(te.columns)}"
        )


def _normalize_columns(
    df: pd.DataFrame,
    src_col: str,
    dst_col: str,
    step_col: str,
    amount_col: str,
    alert_col: str,
) -> pd.DataFrame:
    rename: dict = {}

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

    cols = set(df.columns)
    if "src_node" not in cols and "nameOrig" in cols:
        rename["nameOrig"] = "src_node"
    if "dst_node" not in cols and "nameDest" in cols:
        rename["nameDest"] = "dst_node"
    if "src_node" not in cols and "src" in cols:
        rename["src"] = "src_node"
    if "dst_node" not in cols and "dst" in cols:
        rename["dst"] = "dst_node"
    if "is_sar" not in cols:
        for alias in ("is_laundering", "Is Laundering", "isSAR"):
            if alias in cols:
                rename[alias] = "is_sar"
                break

    if rename:
        df = df.rename(columns=rename)
    return df


def _ensure_event_id(df: pd.DataFrame) -> pd.DataFrame:
    if "event_id" not in df.columns:
        df = df.copy()
        df["event_id"] = range(len(df))
    return df


def _validate_required(df: pd.DataFrame) -> None:
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
    "load_windows_meta",
    "load_temporal_shard",
    "iter_temporal_shards",
    "temporal_edges_to_event_df",
    "load_event_window_from_temporal_shard",
    "filter_window",
]



"""
cell 3 — Exact temporal motif matchers for AML detection.

This cell contains the exact matcher logic used AFTER candidate-window
selection. It is not intended to mine the full dataset globally.

Implemented motif families
--------------------------
- fanin       : many -> one within a tight time span
- fanout      : one -> many within a tight time span
- cycle3      : u -> v -> w -> u
- relay4      : u -> v -> w -> x
- split_merge : u -> v1 -> z and u -> v2 -> z

General design rules
--------------------
1. Forward-only:
   steps must strictly increase along the motif path.

2. Early pruning:
   stop as soon as any temporal, structural, or amount constraint fails.

3. No side effects:
   matchers treat indexes as read-only.

4. Standardized output:
   each matcher returns `list[dict]`, one dict per motif instance.

Expected instance keys
----------------------
motif_type, nodes, edges, steps, amounts, lags, ratios, n_alert

Dependencies
------------
These matchers assume Cell 2 already built local event indexes from one
candidate event window. They should not be run directly on aggregated
graph objects such as second-order edges or snapshot graphs.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Primitive constraint helpers
# ---------------------------------------------------------------------------

def _ratio_ok(a_prev: float, a_curr: float, rho_min: float, rho_max: float) -> bool:
    """Return True when a_curr / a_prev lies in [rho_min, rho_max]."""
    if a_prev <= 0:
        return False
    r = a_curr / a_prev
    return rho_min <= r <= rho_max


def _lag_ok(step_prev: int, step_curr: int, delta: int) -> bool:
    """Return True when 0 < step_curr - step_prev <= delta."""
    lag = step_curr - step_prev
    return 0 < lag <= delta


# ---------------------------------------------------------------------------
# Standard motif instance constructor
# ---------------------------------------------------------------------------

def _make_instance(
    motif_type: str,
    edges: list[dict],
    nodes: list[int],
) -> dict:
    """
    Build one standardized motif instance.

    Parameters
    ----------
    motif_type : str
        Name of the matched motif family.
    edges : list[dict]
        Edge records in chronological order.
    nodes : list[int]
        Ordered node sequence representing the motif.

    Returns
    -------
    dict
        Standardized motif instance payload.
    """
    steps   = [e["step"] for e in edges]
    amounts = [e["amount"] for e in edges]
    lags    = [steps[i] - steps[i - 1] for i in range(1, len(steps))]
    ratios  = [
        round(amounts[i] / amounts[i - 1], 4) if amounts[i - 1] > 0 else 0.0
        for i in range(1, len(amounts))
    ]
    return {
        "motif_type": motif_type,
        "nodes": nodes,
        "edges": [e["event_id"] for e in edges],
        "steps": steps,
        "amounts": amounts,
        "lags": lags,
        "ratios": ratios,
        "n_alert": sum(e.get("is_sar", 0) for e in edges),
    }


# ---------------------------------------------------------------------------
# Fan-in matcher
# ---------------------------------------------------------------------------

def find_fanin(
    in_index: dict,
    cfg: MotifConfig,
) -> list[dict]:
    """
    Match fan-in patterns:

        u1 -> x
        u2 -> x
        u3 -> x
        ...

    subject to:
    - distinct sources
    - arrivals within cfg.delta steps of the seed edge
    - consecutive accepted amounts within [rho_min, rho_max]

    Deduplication
    -------------
    A seen-set keyed by (frozenset(source_ids), destination) prevents
    the same source group from being emitted multiple times under
    different seed edges.
    """
    results = []
    seen: set = set()

    for x, incoming in in_index.items():
        n = len(incoming)
        if n < cfg.r_min_fanin:
            continue

        for i in range(n):
            seed = incoming[i]
            t0 = seed["step"]
            a_prev = seed["amount"]
            seen_src = {seed["src"]}
            group = [seed]

            for j in range(i + 1, n):
                e = incoming[j]

                if e["step"] - t0 > cfg.delta:
                    break

                if e["src"] in seen_src:
                    continue

                if not _ratio_ok(a_prev, e["amount"], cfg.rho_min, cfg.rho_max):
                    continue

                seen_src.add(e["src"])
                group.append(e)
                a_prev = e["amount"]

            if len(group) >= cfg.r_min_fanin:
                key = (frozenset(e["src"] for e in group), x)
                if key in seen:
                    continue
                seen.add(key)

                nodes = [e["src"] for e in group] + [x]
                results.append(_make_instance("fanin", group, nodes))

    return results


# ---------------------------------------------------------------------------
# Fan-out matcher
# ---------------------------------------------------------------------------

def find_fanout(
    out_index: dict,
    cfg: MotifConfig,
    out_steps: dict,
) -> list[dict]:
    """
    Match fan-out patterns:

        x -> v1
        x -> v2
        x -> v3
        ...

    subject to:
    - distinct destinations
    - departures within cfg.delta steps of the seed edge
    - consecutive accepted amounts within [rho_min, rho_max]
    """
    results = []

    for x, outgoing in out_index.items():
        n = len(outgoing)
        if n < cfg.r_min_fanout:
            continue

        for i in range(n):
            seed = outgoing[i]
            t0 = seed["step"]
            a0 = seed["amount"]
            seen = {seed["dst"]}
            group = [seed]
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
# Cycle-3 matcher
# ---------------------------------------------------------------------------

def find_cycle3(
    out_index: dict,
    cfg: MotifConfig,
    out_steps: dict,
) -> list[dict]:
    """
    Match 3-cycle patterns:

        u -> v -> w -> u

    subject to:
    - strict forward time order
    - each hop lag <= cfg.delta
    - hop-to-hop amount ratios within [rho_min, rho_max]
    - u, v, w all distinct
    """
    results = []
    seen = set()

    for u, edges_u in out_index.items():
        for e1 in edges_u:
            v = e1["dst"]
            t1 = e1["step"]
            a1 = e1["amount"]

            if v == u:
                continue

            for e2 in edges_after_step(out_index, v, t1, out_steps):
                if e2["step"] - t1 > cfg.delta:
                    break

                w = e2["dst"]
                a2 = e2["amount"]

                if w == u or w == v:
                    continue

                if not _ratio_ok(a1, a2, cfg.rho_min, cfg.rho_max):
                    continue

                for e3 in edges_after_step(out_index, w, e2["step"], out_steps):
                    if e3["step"] - e2["step"] > cfg.delta:
                        break

                    if e3["dst"] != u:
                        continue

                    if not _ratio_ok(a2, e3["amount"], cfg.rho_min, cfg.rho_max):
                        continue

                    key = frozenset([
                        e1["event_id"],
                        e2["event_id"],
                        e3["event_id"],
                    ])
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
# Relay-4 matcher
# ---------------------------------------------------------------------------

def find_relay4(
    out_index: dict,
    cfg: MotifConfig,
    out_steps: dict,
) -> list[dict]:
    """
    Match relay-4 patterns:

        u -> v -> w -> x

    subject to:
    - strict forward time order
    - each hop lag <= cfg.delta
    - hop-to-hop amount ratios within [rho_min, rho_max]
    - u, v, w, x all distinct
    """
    results = []
    seen = set()

    for u, edges_u in out_index.items():
        for e1 in edges_u:
            v = e1["dst"]
            t1 = e1["step"]
            a1 = e1["amount"]

            if v == u:
                continue

            for e2 in edges_after_step(out_index, v, t1, out_steps):
                if e2["step"] - t1 > cfg.delta:
                    break

                w = e2["dst"]
                a2 = e2["amount"]

                if w in (u, v):
                    continue

                if not _ratio_ok(a1, a2, cfg.rho_min, cfg.rho_max):
                    continue

                for e3 in edges_after_step(out_index, w, e2["step"], out_steps):
                    if e3["step"] - e2["step"] > cfg.delta:
                        break

                    x = e3["dst"]
                    a3 = e3["amount"]

                    if x in (u, v, w):
                        continue

                    if not _ratio_ok(a2, a3, cfg.rho_min, cfg.rho_max):
                        continue

                    key = frozenset([
                        e1["event_id"],
                        e2["event_id"],
                        e3["event_id"],
                    ])
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
# Split-merge matcher
# ---------------------------------------------------------------------------

def find_split_merge(
    out_index: dict,
    in_index: dict,
    cfg: MotifConfig,
    out_steps: dict,
) -> list[dict]:
    """
    Match split-merge patterns:

        u -> v1 -> z
        u -> v2 -> z

    Matching strategy
    -----------------
    Phase 1:
        find split pairs (u -> v1, u -> v2) within cfg.delta

    Phase 2:
        find a shared target z reachable from both v1 and v2 within
        2 * cfg.delta of the split start

    Important implementation notes
    ------------------------------
    - For each (u, v1), keep only the earliest v1 -> z edge.
    - Deduplicate using all four event_ids.
    - This is intentionally more restrictive than a full Cartesian expansion,
      to control runtime and duplicate explosion.
    """
    results = []
    seen: set = set()

    for u, outgoing_u in out_index.items():
        n = len(outgoing_u)

        for i in range(n):
            e_uv1 = outgoing_u[i]
            v1 = e_uv1["dst"]
            t0 = e_uv1["step"]
            a_uv1 = e_uv1["amount"]

            v1_targets: dict = {}
            for e in edges_after_step(out_index, v1, t0, out_steps):
                if e["step"] - t0 > 2 * cfg.delta:
                    break

                z = e["dst"]
                if z in (u, v1):
                    continue

                if not _ratio_ok(a_uv1, e["amount"], cfg.rho_min, cfg.rho_max):
                    continue

                if z not in v1_targets:
                    v1_targets[z] = e

            if not v1_targets:
                continue

            for j in range(i + 1, n):
                e_uv2 = outgoing_u[j]

                if e_uv2["step"] - t0 > cfg.delta:
                    break

                v2 = e_uv2["dst"]
                if v2 == v1 or v2 == u:
                    continue

                if not _ratio_ok(a_uv1, e_uv2["amount"], cfg.rho_min, cfg.rho_max):
                    continue

                for e_v2z in edges_after_step(out_index, v2, t0, out_steps):
                    if e_v2z["step"] - t0 > 2 * cfg.delta:
                        break

                    z = e_v2z["dst"]
                    if z not in v1_targets or z in (u, v1, v2):
                        continue

                    if not _ratio_ok(
                        e_uv2["amount"],
                        e_v2z["amount"],
                        cfg.rho_min,
                        cfg.rho_max,
                    ):
                        continue

                    e_v1z = v1_targets[z]

                    key = frozenset([
                        e_uv1["event_id"],
                        e_uv2["event_id"],
                        e_v1z["event_id"],
                        e_v2z["event_id"],
                    ])
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


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_all_matchers(
    out_index: dict,
    in_index: dict,
    cfg: MotifConfig,
    out_steps: dict[int, list[dict]],
) -> list[dict]:
    enabled = set(getattr(cfg, "enabled_matchers", ["fanin", "fanout", "relay4"]))

    available = {
        "fanin": lambda: find_fanin(in_index, cfg),
        "fanout": lambda: find_fanout(out_index, cfg, out_steps),
        "cycle3": lambda: find_cycle3(out_index, cfg, out_steps),
        "relay4": lambda: find_relay4(out_index, cfg, out_steps),
        "split_merge": lambda: find_split_merge(out_index, in_index, cfg, out_steps),
    }

    unknown = enabled - set(available.keys())
    if unknown:
        raise ValueError(f"Unknown matchers in cfg.enabled_matchers: {sorted(unknown)}")

    combined = []
    cap = cfg.max_instances if cfg.max_instances > 0 else None

    for mtype in ["fanin", "fanout", "cycle3", "relay4", "split_merge"]:
        if mtype not in enabled:
            continue

        instances = available[mtype]()

        if cap and len(instances) > cap:
            print(
                f"  [WARN] {mtype}: {len(instances):,} instances exceed "
                f"max_instances={cap:,}; truncating to {cap:,}."
            )
            instances = instances[:cap]

        combined.extend(instances)

    return combined



"""
cell 4 — Candidate selection + motif scoring + selective null-model evaluation.

Redesign goals
--------------
1. Stage 3:
   Exact motif mining is no longer run on every window by default.
   We score windows cheaply first, then run exact search only on candidates.

2. Stage 4:
   Motifs are supporting structural evidence.
   This cell prepares support counts, optional z-score filtering,
   and candidate-window mining outputs that feed feature extraction downstream.

Assumptions
-----------
- Cell 1 defines MotifConfig
- Cell 2 defines:
    load_windows_meta
    load_event_window_from_temporal_shard
    build_event_indexes
- Cell 3 defines:
    run_all_matchers
"""

from __future__ import annotations

import copy
import gc
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd




# ---------------------------------------------------------------------------
# Support counting
# ---------------------------------------------------------------------------

def count_support(motif_instances: List[dict]) -> Dict[str, int]:
    """
    Count matched instances by motif type.
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
    Apply per-type support and optional z-score filtering.

    Notes
    -----
    fanin / fanout minimum support is mostly enforced structurally by the matchers,
    but we still keep a consistent per-type gate here.
    """
    r_min_map = {
        "fanin": 1,
        "fanout": 1,
        "cycle3": cfg.r_min_cycle,
        "relay4": cfg.r_min_relay,
        "split_merge": cfg.r_min_split_merge,
    }

    support = count_support(instances)
    keep_types = set()

    for mtype, count in support.items():
        if count < r_min_map.get(mtype, 1):
            continue

        if zscore_results is not None:
            z = zscore_results.get(mtype, {}).get("zscore", 0.0)
            if z < cfg.z_min:
                continue

        keep_types.add(mtype)

    return [
        inst
        for inst in instances
        if inst["motif_type"] in keep_types
    ]


# ---------------------------------------------------------------------------
# Candidate window scoring
# ---------------------------------------------------------------------------

def score_candidate_windows(
    meta_df: pd.DataFrame,
    window_risk_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute cheap candidate scores BEFORE exact motif mining.

    Base score uses notebook 1 metadata only:
        - n_temporal
        - n_tx

    Optional extra signal:
        window_risk_df with columns:
            start, end, window_risk_score

    Returns
    -------
    DataFrame with columns:
        window, start, end, n_tx, n_temporal, candidate_score
    """
    df = meta_df.copy()

    if "n_temporal" not in df.columns:
        df["n_temporal"] = 0
    if "n_tx" not in df.columns:
        df["n_tx"] = 0

    # Base score:
    # favor windows with many relay pairs, lightly adjusted by tx volume
    df["candidate_score"] = (
        np.log1p(df["n_temporal"].astype(float))
        + 0.25 * np.log1p(df["n_tx"].astype(float))
    )

    if window_risk_df is not None and len(window_risk_df) > 0:
        required = {"start", "end", "window_risk_score"}
        missing = required - set(window_risk_df.columns)
        if missing:
            raise ValueError(
                f"window_risk_df is missing required columns: {sorted(missing)}"
            )

        df = df.merge(
            window_risk_df[["start", "end", "window_risk_score"]],
            on=["start", "end"],
            how="left",
        )
        df["window_risk_score"] = df["window_risk_score"].fillna(0.0)
        df["candidate_score"] = df["candidate_score"] + df["window_risk_score"]

    return df.sort_values(
        ["candidate_score", "n_temporal", "n_tx"],
        ascending=False,
    ).reset_index(drop=True)


def select_candidate_windows(
    meta_df: pd.DataFrame,
    cfg: MotifConfig,
    window_risk_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Select candidate windows for exact motif mining.

    Selection policy is controlled by config-like attributes.
    Uses getattr() so the code remains backward-compatible if older
    MotifConfig does not yet define these fields.

    Supported modes
    ---------------
    candidate_window_mode:
        - "top_k"
        - "threshold"
        - "all"

    Other config fields used if present
    -----------------------------------
    top_k_windows
    window_score_threshold
    max_windows_exact
    """
    scored = score_candidate_windows(meta_df, window_risk_df)

    mode = getattr(cfg, "candidate_window_mode", "top_k")
    top_k = int(getattr(cfg, "top_k_windows", 10))
    score_threshold = float(getattr(cfg, "window_score_threshold", 0.0))
    max_windows_exact = int(getattr(cfg, "max_windows_exact", top_k))

    # Always exclude windows with zero temporal relay pairs for exact motifs
    scored = scored[scored["n_temporal"] > 0].reset_index(drop=True)

    if mode == "all":
        selected = scored
    elif mode == "threshold":
        selected = scored[scored["candidate_score"] >= score_threshold]
    else:
        selected = scored.head(top_k)

    if max_windows_exact > 0:
        selected = selected.head(max_windows_exact)

    return selected.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Null-model helpers
# ---------------------------------------------------------------------------

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
    Build local indexes, run enabled matchers, return support counts.
    """
    out_idx, in_idx, _, out_steps = build_event_indexes(event_df)

    cfg_local = copy.copy(cfg)
    if cap > 0:
        cfg_local.max_instances = (
            min(cap, cfg.max_instances)
            if cfg.max_instances > 0
            else cap
        )

    instances = run_all_matchers(
        out_idx,
        in_idx,
        cfg_local,
        out_steps,
    )

    counts = count_support(instances)

    del out_idx, in_idx, instances
    gc.collect()

    return counts


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

    This function is now intended for SELECTED candidate windows only.
    """
    if not observed_counts:
        return {}

    null_counts: Dict[str, list] = {
        mt: [] for mt in observed_counts
    }

    rng = np.random.default_rng(seed)

    for i in range(cfg.n_permutations):
        if verbose:
            print(f"  Null permutation {i + 1}/{cfg.n_permutations}...")

        if sample_frac < 1.0:
            df_perm = (
                event_df
                .sample(
                    frac=sample_frac,
                    random_state=int(rng.integers(1_000_000)),
                )
                .sort_values(["step", "event_id"])
                .reset_index(drop=True)
            )
        else:
            df_perm = event_df

        df_null = _shuffle_timestamps(df_perm, rng)

        perm_counts = _run_matchers_on_df(
            df_null,
            cfg,
            cap=null_cap,
        )

        for mt in observed_counts:
            null_counts[mt].append(perm_counts.get(mt, 0))

        del df_null, perm_counts
        if sample_frac < 1.0:
            del df_perm

        gc.collect()

    results: Dict[str, dict] = {}

    for mt, c_obs in observed_counts.items():
        arr = np.array(null_counts[mt], dtype=float)
        mean_null = float(arr.mean())
        std_null = float(arr.std())

        zscore = (c_obs - mean_null) / (std_null + 1e-9)

        results[mt] = {
            "observed": c_obs,
            "mean_null": round(mean_null, 2),
            "std_null": round(std_null, 2),
            "zscore": round(zscore, 3),
        }

    return results


# ---------------------------------------------------------------------------
# Candidate-window exact mining
# ---------------------------------------------------------------------------

def mine_candidate_windows(
    cfg: MotifConfig,
    meta_path: str | Path = WINDOW_META_PATH,
    temporal_shard_dir: str | Path = TEMPORAL_SHARD_DIR,
    window_risk_df: pd.DataFrame | None = None,
    run_null_model: bool = False,
    null_sample_frac: float = 1.0,
    null_cap: int = 10_000,
    verbose: bool = True,
) -> tuple[list[dict], pd.DataFrame]:
    """
    Main Stage-3 exact motif execution.

    Pipeline
    --------
    1. Load windows_meta
    2. Score windows cheaply
    3. Select candidate windows
    4. For each candidate window:
         - load temporal shard
         - reconstruct canonical event_df
         - run exact matchers locally
         - optionally run null-model z-score
         - filter instances
    5. Return:
         - combined filtered motif instances
         - per-window summary table

    Returns
    -------
    all_instances : list[dict]
    summary_df    : pd.DataFrame
    """
    meta_df = load_windows_meta(meta_path)
    selected = select_candidate_windows(meta_df, cfg, window_risk_df)

    if verbose:
        print(f"Selected {len(selected)} candidate windows for exact motif mining.")

    all_instances: list[dict] = []
    summaries: list[dict] = []

    for row in selected.itertuples(index=False):
        step_start = int(row.start)
        step_end = int(row.end)

        if verbose:
            print(
                f"\n[Candidate window {int(row.window)}] "
                f"{step_start}-{step_end} | "
                f"score={float(row.candidate_score):.3f} | "
                f"n_temporal={int(row.n_temporal)}"
            )

        event_df = load_event_window_from_temporal_shard(
            step_start=step_start,
            step_end=step_end,
            shard_dir=temporal_shard_dir,
        )

        out_idx, in_idx, _, out_steps = build_event_indexes(event_df)
        raw_instances = run_all_matchers(out_idx, in_idx, cfg, out_steps)
        observed_counts = count_support(raw_instances)

        zscore_results = None
        if run_null_model and observed_counts:
            zscore_results = compute_null_zscore(
                observed_counts=observed_counts,
                event_df=event_df,
                cfg=cfg,
                sample_frac=null_sample_frac,
                null_cap=null_cap,
                verbose=verbose,
            )

        filtered_instances = filter_motifs(
            raw_instances,
            cfg,
            zscore_results=zscore_results,
        )

        all_instances.extend(filtered_instances)

        summaries.append({
            "window": int(row.window),
            "start": step_start,
            "end": step_end,
            "candidate_score": float(row.candidate_score),
            "n_events": int(len(event_df)),
            "n_instances_raw": int(len(raw_instances)),
            "n_instances_filtered": int(len(filtered_instances)),
            "support_raw": observed_counts,
            "zscores": zscore_results if zscore_results is not None else {},
        })

        del out_idx, in_idx, out_steps, event_df, raw_instances, filtered_instances
        gc.collect()

    summary_df = pd.DataFrame(summaries)
    return all_instances, summary_df

def run_exact_motif_feature_pipeline(
    cfg: MotifConfig,
    meta_path: str | Path = WINDOW_META_PATH,
    temporal_shard_dir: str | Path = TEMPORAL_SHARD_DIR,
    window_risk_df: pd.DataFrame | None = None,
    run_null_model: bool = False,
    null_sample_frac: float = 1.0,
    null_cap: int = 10_000,
    export_dir: str | Path | None = None,
    feature_format: str = "parquet",
    instance_format: str = "parquet",
    window_size: int = 7,
    verbose: bool = True,
):
    """
    Complete Stage B exact motif branch:
      1. select candidate windows
      2. run exact motif mining
      3. convert motif instances into Cell 5 feature outputs
      4. export feature tables as the primary artifact
    """
    all_instances, summary_df = mine_candidate_windows(
        cfg=cfg,
        meta_path=meta_path,
        temporal_shard_dir=temporal_shard_dir,
        window_risk_df=window_risk_df,
        run_null_model=run_null_model,
        null_sample_frac=null_sample_frac,
        null_cap=null_cap,
        verbose=verbose,
    )

    motif_instances_by_window: dict[str, list[dict]] = {}
    zscore_by_window: dict[str, dict] = {}

    # Re-run summary collection by window for export compatibility
    selected = select_candidate_windows(
        load_windows_meta(meta_path),
        cfg=cfg,
        window_risk_df=window_risk_df,
    )

    for row in selected.itertuples(index=False):
        step_start = int(row.start)
        step_end = int(row.end)
        window_key = f"w_{step_start}_{step_end}"

        event_df = load_event_window_from_temporal_shard(
            step_start=step_start,
            step_end=step_end,
            shard_dir=temporal_shard_dir,
        )

        out_idx, in_idx, _, out_steps = build_event_indexes(event_df)
        raw_instances = run_all_matchers(out_idx, in_idx, cfg, out_steps)
        observed_counts = count_support(raw_instances)

        zscore_results = None
        if run_null_model and observed_counts:
            zscore_results = compute_null_zscore(
                observed_counts=observed_counts,
                event_df=event_df,
                cfg=cfg,
                sample_frac=null_sample_frac,
                null_cap=null_cap,
                verbose=False,
            )

        filtered_instances = filter_motifs(
            raw_instances,
            cfg,
            zscore_results=zscore_results,
        )

        motif_instances_by_window[window_key] = filtered_instances
        zscore_by_window[window_key] = zscore_results or {}

        del event_df, out_idx, in_idx, out_steps, raw_instances, filtered_instances
        gc.collect()

    export_paths = export_motif_feature_outputs(
        window_summary_df=summary_df,
        motif_instances_by_window=motif_instances_by_window,
        zscore_by_window=zscore_by_window,
        node_degree_by_window=None,
        total_volume_by_window=None,
        export_dir=export_dir,
        export_instances=getattr(cfg, "export_instances", False),
        instance_format=instance_format,
        feature_format=feature_format,
        window_size=window_size,
    )

    return {
        "all_instances": all_instances,
        "summary_df": summary_df,
        "export_paths": export_paths,
    }
def narrow_candidate_event_df(
    event_df: pd.DataFrame,
    top_nodes: set[int],
    ) -> pd.DataFrame:
    """
    Keep only events touching the selected high-risk nodes.
    """
    mask = event_df["src_node"].isin(top_nodes) | event_df["dst_node"].isin(top_nodes)
    narrowed = event_df.loc[mask].copy()

    if not narrowed.empty:
        narrowed = narrowed.sort_values(["step", "event_id"]).reset_index(drop=True)

    return narrowed


__all__ = [
    "count_support",
    "filter_motifs",
    "score_candidate_windows",
    "select_candidate_windows",
    "compute_null_zscore",
    "mine_candidate_windows",
]



from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


DEFAULT_EXPORT_DIR: str = "outputs/motif_features"


def build_entity_motif_features(
    motif_instances: List[dict],
    zscore_table: Optional[Dict[str, dict]] = None,
    node_degree: Optional[Dict[int, int]] = None,
    total_volume: float = 0.0,
) -> pd.DataFrame:
    """
    Build node-level motif features from matched motif instances.

    This is the primary node-level output contract of the exact motif branch.
    Each row represents one (node, motif_type) pair and is intended for
    downstream model ingestion after optional wide-format pivoting.
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
        mt = inst["motif_type"]
        amounts = inst.get("amounts", [])
        lags = inst.get("lags", [])
        ratios = inst.get("ratios", [])
        n_alert = inst.get("n_alert", 0)
        zs = (
            zscore_table[mt]["zscore"]
            if (zscore_table and mt in zscore_table)
            else np.nan
        )

        avg_amt = float(np.mean(amounts)) if amounts else 0.0
        avg_ratio = float(np.mean(ratios)) if ratios else np.nan
        r_std = float(np.std(ratios)) if ratios else np.nan
        avg_lag = float(np.mean(lags)) if lags else 0.0
        max_lag = float(max(lags)) if lags else 0.0

        for node in set(inst.get("nodes", [])):
            rows.append({
                "node": int(node),
                "motif_type": mt,
                "avg_amount": avg_amt,
                "avg_ratio": avg_ratio,
                "ratio_std": r_std,
                "avg_lag": avg_lag,
                "max_lag": max_lag,
                "n_alert_edges": n_alert,
                "zscore": zs,
            })

    if not rows:
        return pd.DataFrame(columns=_EMPTY_COLS)

    df = pd.DataFrame(rows)

    agg = (
        df.groupby(["node", "motif_type"], as_index=False)
        .agg(
            count=("avg_amount", "count"),
            avg_amount=("avg_amount", "mean"),
            avg_ratio=("avg_ratio", "mean"),
            ratio_std=("ratio_std", "mean"),
            avg_lag=("avg_lag", "mean"),
            max_lag=("max_lag", "max"),
            avg_n_alert_edges=("n_alert_edges", "mean"),
            zscore=("zscore", "first"),
        )
    )

    if node_degree:
        agg["freq_by_degree"] = agg.apply(
            lambda r: r["count"] / node_degree.get(int(r["node"]), 1),
            axis=1,
        )
    else:
        agg["freq_by_degree"] = 0.0

    if total_volume > 0:
        agg["freq_by_volume"] = agg["count"] / total_volume
    else:
        agg["freq_by_volume"] = 0.0

    return agg.sort_values(["node", "motif_type"]).reset_index(drop=True)


def build_entity_feature_wide(entity_features: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot node-level motif features from long format to wide format.

    This is the preferred output when the downstream model expects one row
    per node and one feature vector per node.
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

    wide.columns = [f"{mt}_{metric}" for metric, mt in wide.columns]
    wide = wide.fillna(0).reset_index()

    return wide


def build_window_motif_features(
    motif_instances: List[dict],
    window_size: int = 7,
) -> pd.DataFrame:
    """
    Build window-level motif features from matched motif instances.

    This is the primary time-window output contract of the exact motif branch.
    It is useful for candidate-window diagnostics, temporal summaries, and
    optional downstream window scoring.
    """
    _EMPTY_COLS = [
        "window_start", "motif_type", "count",
        "total_amount", "avg_lag", "n_alert_edges", "suspicious_ratio",
    ]

    if not motif_instances:
        return pd.DataFrame(columns=_EMPTY_COLS)

    rows = []
    for inst in motif_instances:
        steps = inst.get("steps", [])
        amounts = inst.get("amounts", [])
        lags = inst.get("lags", [])
        n_alert = inst.get("n_alert", 0)

        if not steps:
            continue

        window_start = (min(steps) // window_size) * window_size

        rows.append({
            "window_start": window_start,
            "motif_type": inst["motif_type"],
            "total_amount": float(sum(amounts)) if amounts else 0.0,
            "avg_lag": float(np.mean(lags)) if lags else 0.0,
            "n_alert_edges": n_alert,
            "has_alert": int(n_alert > 0),
        })

    if not rows:
        return pd.DataFrame(columns=_EMPTY_COLS)

    df = pd.DataFrame(rows)

    agg = (
        df.groupby(["window_start", "motif_type"], as_index=False)
        .agg(
            count=("total_amount", "count"),
            total_amount=("total_amount", "sum"),
            avg_lag=("avg_lag", "mean"),
            n_alert_edges=("n_alert_edges", "sum"),
            suspicious_ratio=("has_alert", "mean"),
        )
    )

    return agg.sort_values(["window_start", "motif_type"]).reset_index(drop=True)


def save_features(
    df: pd.DataFrame,
    filename: str,
    export_dir: Optional[str] = None,
    fmt: str = "parquet",
) -> str:
    """
    Save one motif feature table.

    Feature tables are the primary artifacts of the exact motif branch.
    """
    out_dir = Path(export_dir or DEFAULT_EXPORT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = "parquet" if fmt == "parquet" else "csv"
    path = out_dir / f"{filename}.{ext}"

    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

    print(f"  Saved {len(df):,} rows -> {path}")
    return str(path)


def export_motif_feature_outputs(
    window_summary_df: pd.DataFrame,
    motif_instances_by_window: Dict[str, List[dict]],
    zscore_by_window: Optional[Dict[str, Dict[str, dict]]] = None,
    node_degree_by_window: Optional[Dict[str, Dict[int, int]]] = None,
    total_volume_by_window: Optional[Dict[str, float]] = None,
    export_dir: Optional[str] = None,
    export_instances: bool = False,
    instance_format: str = "parquet",
    feature_format: str = "parquet",
    window_size: int = 7,
) -> Dict[str, str]:
    """
    Export motif feature outputs as the main artifact of the exact motif branch.

    Required behavior
    -----------------
    - Per-window entity feature shards
    - Per-window entity-wide feature shards
    - Per-window window-level feature shards
    - Merged entity long table
    - Merged entity wide table
    - Merged window-level table

    Optional debug behavior
    -----------------------
    Raw motif instance tables are exported only if export_instances=True.
    They are not the primary saved artifact.
    """
    out_root = Path(export_dir or DEFAULT_EXPORT_DIR)
    entity_dir = out_root / "entity_features"
    entity_wide_dir = out_root / "entity_feature_wide"
    window_dir = out_root / "window_features"
    instance_dir = out_root / "instances_debug"
    meta_dir = out_root / "meta"

    for d in [entity_dir, entity_wide_dir, window_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if export_instances:
        instance_dir.mkdir(parents=True, exist_ok=True)

    merged_entity_long = []
    merged_entity_wide = []
    merged_window = []

    for window_key, motif_instances in motif_instances_by_window.items():
        zscore_table = zscore_by_window.get(window_key) if zscore_by_window else None
        node_degree = node_degree_by_window.get(window_key) if node_degree_by_window else None
        total_volume = total_volume_by_window.get(window_key, 0.0) if total_volume_by_window else 0.0

        entity_long = build_entity_motif_features(
            motif_instances=motif_instances,
            zscore_table=zscore_table,
            node_degree=node_degree,
            total_volume=total_volume,
        )
        entity_wide = build_entity_feature_wide(entity_long)
        window_feat = build_window_motif_features(
            motif_instances=motif_instances,
            window_size=window_size,
        )

        if not entity_long.empty:
            entity_long = entity_long.copy()
            entity_long["window_key"] = window_key
            merged_entity_long.append(entity_long)

        # SAU KHI FIX — Cell 5
        if not entity_wide.empty:
            merged_entity_wide.append(entity_wide)   # không thêm window_key vào wide

        if not window_feat.empty:
            window_feat = window_feat.copy()
            window_feat["window_key"] = window_key
            merged_window.append(window_feat)

        save_features(
            entity_long,
            filename=f"entity_features_{window_key}",
            export_dir=entity_dir,
            fmt=feature_format,
        )
        save_features(
            entity_wide,
            filename=f"entity_feature_wide_{window_key}",
            export_dir=entity_wide_dir,
            fmt=feature_format,
        )
        save_features(
            window_feat,
            filename=f"window_features_{window_key}",
            export_dir=window_dir,
            fmt=feature_format,
        )

        if export_instances:
            inst_df = pd.DataFrame(motif_instances)
            save_features(
                inst_df,
                filename=f"motif_instances_{window_key}",
                export_dir=instance_dir,
                fmt=instance_format,
            )

    merged_entity_long_df = (
        pd.concat(merged_entity_long, ignore_index=True)
        if merged_entity_long else
        pd.DataFrame()
    )
    merged_entity_wide_df = (
        pd.concat(merged_entity_wide, ignore_index=True)
        if merged_entity_wide else
        pd.DataFrame()
    )
    merged_window_df = (
        pd.concat(merged_window, ignore_index=True)
        if merged_window else
        pd.DataFrame()

    )

    merged_entity_long_path = save_features(
        merged_entity_long_df,
        filename="entity_features_merged",
        export_dir=out_root,
        fmt=feature_format,
    )
    merged_entity_wide_path = save_features(
        merged_entity_wide_df,
        filename="entity_feature_wide_merged",
        export_dir=out_root,
        fmt=feature_format,
    )
    merged_window_path = save_features(
        merged_window_df,
        filename="window_features_merged",
        export_dir=out_root,
        fmt=feature_format,
    )

    summary_path = meta_dir / "window_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            window_summary_df.to_dict(orient="records"),
            f,
            indent=2,
        )

    return {
        "entity_features_merged": merged_entity_long_path,
        "entity_feature_wide_merged": merged_entity_wide_path,
        "window_features_merged": merged_window_path,
        "window_summary": str(summary_path),
    }


__all__ = [
    "build_entity_motif_features",
    "build_entity_feature_wide",
    "build_window_motif_features",
    "save_features",
    "export_motif_feature_outputs",
    "DEFAULT_EXPORT_DIR",
]



import os
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# USER CONFIG
# ============================================================

OUTPUT_DIR = Path("/content/drive/MyDrive/AML/outputs")
SCREEN_DIR = OUTPUT_DIR / "screening"
MOTIF_FEATURE_DIR = OUTPUT_DIR / "motif_features"

TX_PATH = OUTPUT_DIR / "transactions.parquet"

WINDOW_SIZE = 7
MERGE_MOTIF_FEATURES = False

MOTIF_WIDE_PATH = MOTIF_FEATURE_DIR / "entity_feature_wide_merged.parquet"

SCREEN_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Step 1 — Load transactions safely
# ============================================================

print("[1/6] Loading transactions...")

tx = pd.read_parquet(TX_PATH)

required_cols = {"src_node", "dst_node", "step", "amount"}
missing_cols = required_cols - set(tx.columns)
if missing_cols:
    raise ValueError(
        f"transactions.parquet is missing required columns: {sorted(missing_cols)}"
    )

tx["src_node"] = tx["src_node"].astype(np.int64)
tx["dst_node"] = tx["dst_node"].astype(np.int64)
tx["step"] = tx["step"].astype(np.int32)
tx["amount"] = tx["amount"].astype(np.float32)

if "is_sar" not in tx.columns:
    tx["is_sar"] = np.int8(0)
else:
    tx["is_sar"] = tx["is_sar"].astype(np.int8)

print(
    f"      rows={len(tx):,} | "
    f"steps={int(tx['step'].min())}-{int(tx['step'].max())} | "
    f"sar_rate={tx['is_sar'].mean()*100:.3f}%"
)

labels_src = tx[["src_node", "is_sar"]].copy()
labels_dst = tx[["dst_node", "is_sar"]].copy()
df_feats = tx.drop(columns=["is_sar"]).copy()

step_min = int(df_feats["step"].min())
step_max = int(df_feats["step"].max())


# ============================================================
# Step 2 — Window-level proxy features
# ============================================================

print(f"\n[2/6] Computing scalable proxy features (window={WINDOW_SIZE})...")

all_node_parts = []


def _window_node_features(wdf: pd.DataFrame, ws: int, we: int) -> pd.DataFrame:
    out = (
        wdf.groupby("src_node")
        .agg(
            out_count=("amount", "count"),
            out_amount_sum=("amount", "sum"),
            out_amount_mean=("amount", "mean"),
            out_amount_std=("amount", "std"),
            out_amount_max=("amount", "max"),
            out_amount_min=("amount", "min"),
            out_n_unique_dst=("dst_node", "nunique"),
            out_step_std=("step", "std"),
        )
        .reset_index()
        .rename(columns={"src_node": "node"})
    )

    inc = (
        wdf.groupby("dst_node")
        .agg(
            in_count=("amount", "count"),
            in_amount_sum=("amount", "sum"),
            in_amount_mean=("amount", "mean"),
            in_amount_std=("amount", "std"),
            in_amount_max=("amount", "max"),
            in_amount_min=("amount", "min"),
            in_n_unique_src=("src_node", "nunique"),
            in_step_std=("step", "std"),
        )
        .reset_index()
        .rename(columns={"dst_node": "node"})
    )

    feat = pd.merge(out, inc, on="node", how="outer").fillna(0)

    feat["fanout_score"] = feat["out_n_unique_dst"] / (feat["out_count"] + 1e-6)
    feat["fanin_score"] = feat["in_n_unique_src"] / (feat["in_count"] + 1e-6)
    feat["gather_scatter_score"] = feat["fanin_score"] * feat["fanout_score"]

    feat["relay_flag"] = (
        (feat["in_count"] > 0) & (feat["out_count"] > 0)
    ).astype(np.int8)

    feat["relay_ratio"] = feat["out_amount_sum"] / (feat["in_amount_sum"] + 1e-6)
    feat["amount_preservation"] = 1.0 - np.abs(feat["relay_ratio"] - 1.0)

    feat["out_velocity"] = feat["out_count"] / WINDOW_SIZE
    feat["in_velocity"] = feat["in_count"] / WINDOW_SIZE

    out_by_dst = (
        wdf.groupby(["src_node", "dst_node"])["amount"]
        .sum()
        .reset_index()
    )
    out_by_dst["sq"] = out_by_dst["amount"] ** 2

    hhi_num = (
        out_by_dst.groupby("src_node")["sq"]
        .sum()
        .reset_index()
        .rename(columns={"src_node": "node", "sq": "_hhi_num"})
    )
    hhi_den = (wdf.groupby("src_node")["amount"].sum() ** 2).reset_index()
    hhi_den.columns = ["node", "_hhi_den"]

    hhi = (
        hhi_num.merge(hhi_den, on="node", how="left")
        .assign(out_concentration=lambda x: x["_hhi_num"] / (x["_hhi_den"] + 1e-9))
        [["node", "out_concentration"]]
    )
    feat = feat.merge(hhi, on="node", how="left").fillna(0)

    del out_by_dst, hhi_num, hhi_den, hhi

    inc_by_src = (
        wdf.groupby(["dst_node", "src_node"])["amount"]
        .sum()
        .reset_index()
    )
    inc_by_src["sq"] = inc_by_src["amount"] ** 2

    ihhi_num = (
        inc_by_src.groupby("dst_node")["sq"]
        .sum()
        .reset_index()
        .rename(columns={"dst_node": "node", "sq": "_ihhi_num"})
    )
    ihhi_den = (wdf.groupby("dst_node")["amount"].sum() ** 2).reset_index()
    ihhi_den.columns = ["node", "_ihhi_den"]

    ihhi = (
        ihhi_num.merge(ihhi_den, on="node", how="left")
        .assign(in_concentration=lambda x: x["_ihhi_num"] / (x["_ihhi_den"] + 1e-9))
        [["node", "in_concentration"]]
    )
    feat = feat.merge(ihhi, on="node", how="left").fillna(0)

    del inc_by_src, ihhi_num, ihhi_den, ihhi

    src_to_dsts = wdf.groupby("src_node")["dst_node"].apply(set)
    dst_to_srcs = wdf.groupby("dst_node")["src_node"].apply(set)
    common_nodes = src_to_dsts.index.intersection(dst_to_srcs.index)

    if len(common_nodes) > 0:
        cycle_proxy = pd.Series(
            {n: len(src_to_dsts[n] & dst_to_srcs[n]) for n in common_nodes},
            name="cycle_proxy",
        ).astype(np.float32).reset_index()
        cycle_proxy.columns = ["node", "cycle_proxy"]
        feat = feat.merge(cycle_proxy, on="node", how="left")
    else:
        feat["cycle_proxy"] = 0.0

    feat["cycle_proxy"] = feat["cycle_proxy"].fillna(0)
    feat["window_start"] = ws
    feat["window_end"] = we

    return feat


for ws in range(step_min, step_max + 1, WINDOW_SIZE):
    we = ws + WINDOW_SIZE - 1
    wdf = df_feats[(df_feats["step"] >= ws) & (df_feats["step"] <= we)]

    if wdf.empty:
        continue

    feat = _window_node_features(wdf, ws, we)
    all_node_parts.append(feat)

    del wdf, feat
    gc.collect()
    print(f"      processed window [{ws:3d}-{we:3d}]", end="\r")

print(f"\n      windows processed: {len(all_node_parts):,}")


# ============================================================
# Step 3 — Train lightweight Stage A screening model first
# ============================================================

print("\n[3/6] Training lightweight Stage A screening model...")

all_node_feats = pd.concat(all_node_parts, ignore_index=True)
del all_node_parts
gc.collect()

labels_src_node = tx[["src_node", "is_sar"]].rename(columns={"src_node": "node"})
labels_dst_node = tx[["dst_node", "is_sar"]].rename(columns={"dst_node": "node"})

node_labels = (
    pd.concat([labels_src_node, labels_dst_node], ignore_index=True)
    .groupby("node", as_index=False)["is_sar"]
    .max()
    .rename(columns={"is_sar": "label"})
)

screen_train_df = all_node_feats.merge(node_labels, on="node", how="left")
screen_train_df["label"] = screen_train_df["label"].fillna(0).astype(np.int8)

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

screen_exclude = {"node", "window_start", "window_end", "label"}
screen_feature_cols = [c for c in screen_train_df.columns if c not in screen_exclude]

X_screen = screen_train_df[screen_feature_cols].values.astype(np.float32)
y_screen = screen_train_df["label"].values.astype(np.int8)

X_sc_tr, X_sc_va, y_sc_tr, y_sc_va = train_test_split(
    X_screen,
    y_screen,
    test_size=0.20,
    stratify=y_screen,
    random_state=42,
)

n_pos_sc = int(y_sc_tr.sum())
n_neg_sc = int(len(y_sc_tr) - n_pos_sc)
scale_pos_sc = float(n_neg_sc) / float(n_pos_sc + 1e-9)

screen_model = xgb.XGBClassifier(
    n_estimators=250,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=10,
    scale_pos_weight=scale_pos_sc,
    eval_metric="aucpr",
    early_stopping_rounds=20,
    random_state=42,
    tree_method="hist",
    n_jobs=-1,
    reg_alpha=0.1,
    reg_lambda=2.0,
)

screen_model.fit(
    X_sc_tr,
    y_sc_tr,
    eval_set=[(X_sc_va, y_sc_va)],
    verbose=False,
)

y_sc_va_prob = screen_model.predict_proba(X_sc_va)[:, 1]
print(
    f"      Stage A val ROC-AUC={roc_auc_score(y_sc_va, y_sc_va_prob):.4f} | "
    f"PR-AUC={average_precision_score(y_sc_va, y_sc_va_prob):.4f}"
)

all_node_feats["node_screening_score"] = screen_model.predict_proba(
    all_node_feats[screen_feature_cols].values.astype(np.float32)
)[:, 1]


# ============================================================
# Step 3.5 — Aggregate node features across windows
# ============================================================

print("\n[3.5/6] Aggregating node-level screening features across windows...")

agg_exclude = {"node", "window_start", "window_end"}
agg_cols = [c for c in all_node_feats.columns if c not in agg_exclude]

node_proxy_features = (
    all_node_feats
    .groupby("node")[agg_cols]
    .agg(["mean", "max", "std"])
    .fillna(0)
)

node_proxy_features.columns = [
    f"{col}_{stat}" for col, stat in node_proxy_features.columns
]
node_proxy_features = node_proxy_features.reset_index()

sar_nodes = set(
    labels_src_node[labels_src_node["is_sar"] == 1]["node"].tolist()
    + labels_dst_node[labels_dst_node["is_sar"] == 1]["node"].tolist()
)
node_proxy_features["label"] = (
    node_proxy_features["node"].isin(sar_nodes).astype(np.int8)
)

print(f"      node_proxy_features shape: {node_proxy_features.shape}")


# ============================================================
# Step 4 — Build window risk summary for Cell 4 candidate selection
# ============================================================

print("\n[4/6] Building window risk summary...")

window_risk_summary = (
    all_node_feats.groupby(["window_start", "window_end"], as_index=False)
    .agg(
        n_nodes=("node", "nunique"),
        mean_node_score=("node_screening_score", "mean"),
        max_node_score=("node_screening_score", "max"),
        p95_node_score=("node_screening_score", lambda x: float(np.quantile(x, 0.95))),
        mean_relay_flag=("relay_flag", "mean"),
        mean_cycle_proxy=("cycle_proxy", "mean"),
        mean_gather_scatter=("gather_scatter_score", "mean"),
    )
)

window_tx = (
    tx.assign(window_start=(tx["step"] // WINDOW_SIZE) * WINDOW_SIZE)
      .groupby("window_start", as_index=False)
      .agg(n_tx=("amount", "count"))
)
window_tx["window_end"] = window_tx["window_start"] + WINDOW_SIZE - 1

window_risk_summary = window_risk_summary.merge(
    window_tx[["window_start", "window_end", "n_tx"]],
    on=["window_start", "window_end"],
    how="left",
).fillna({"n_tx": 0})

window_risk_summary["window_risk_score"] = (
    0.50 * window_risk_summary["mean_node_score"]
    + 0.35 * window_risk_summary["p95_node_score"]
    + 0.15 * np.log1p(window_risk_summary["n_tx"])
)

window_risk_summary = (
    window_risk_summary.rename(columns={"window_start": "start", "window_end": "end"})
    .sort_values(["start", "end"])
    .reset_index(drop=True)
)

print(f"      window_risk_summary shape: {window_risk_summary.shape}")
print("      top risk windows:")
print(
    window_risk_summary.sort_values("window_risk_score", ascending=False)
    .head(10)
    .to_string(index=False)
)


# ============================================================
# Step 5 — Optional merge with exact motif features from Cell 5
# ============================================================

print("\n[5/6] Optional merge with exact motif features...")

node_screening_features = node_proxy_features.copy()

motif_merge_done = False
if MERGE_MOTIF_FEATURES and MOTIF_WIDE_PATH.exists():
    motif_wide = pd.read_parquet(MOTIF_WIDE_PATH)

    if "node" not in motif_wide.columns:
        raise ValueError(
            f"Motif wide feature file does not contain 'node': {MOTIF_WIDE_PATH}"
        )

    before_cols = set(node_screening_features.columns)

    node_screening_features = (
        node_screening_features
        .merge(motif_wide, on="node", how="left")
    )

    # Handle 'window_key' column specifically if it exists in the merged DataFrame
    if "window_key" in node_screening_features.columns:
        # Fill NaN with an empty string and ensure the column is of string type
        node_screening_features["window_key"] = node_screening_features["window_key"].fillna("").astype(str)

    # Fill remaining numeric NaNs with 0
    numeric_cols = node_screening_features.select_dtypes(include=np.number).columns
    node_screening_features[numeric_cols] = node_screening_features[numeric_cols].fillna(0)

    added_cols = [c for c in node_screening_features.columns if c not in before_cols]
    motif_merge_done = True

    print(f"      merged motif features from: {MOTIF_WIDE_PATH}")
    print(f"      added motif columns: {len(added_cols)}")
else:
    print("      motif merge skipped (merged motif feature file not found or disabled).")


# ============================================================
# Step 6 — Save screening artifacts
# ============================================================

print("\n[6/6] Saving screening artifacts...")

# NEW: save per-window node rows for Stage B narrowing in Cell 7
node_window_path = SCREEN_DIR / "node_window_features.parquet"

node_proxy_path = SCREEN_DIR / "node_proxy_features.parquet"
node_screening_path = SCREEN_DIR / "node_screening_features.parquet"
window_risk_path = SCREEN_DIR / "window_risk_summary.parquet"
feature_cols_path = SCREEN_DIR / "feature_columns_screening.json"

all_node_feats.to_parquet(node_window_path, index=False)
node_proxy_features.to_parquet(node_proxy_path, index=False)
node_screening_features.to_parquet(node_screening_path, index=False)
window_risk_summary.to_parquet(window_risk_path, index=False)

feature_cols = [
    c for c in node_screening_features.columns
    if c not in ("node", "label")
]
with open(feature_cols_path, "w") as f:
    json.dump(feature_cols, f, indent=2)

print("Artifacts saved:")
for lbl, path in [
    ("node_window_features", node_window_path),
    ("node_proxy_features", node_proxy_path),
    ("node_screening_features", node_screening_path),
    ("window_risk_summary", window_risk_path),
    ("feature_columns", feature_cols_path),
]:
    print(f"   [{lbl:<24}] {path} ({os.path.getsize(path)/1024:.1f} KB)")

print("\nSummary:")
print(f"   node-window rows     : {len(all_node_feats):,}")
print(f"   nodes total          : {len(node_screening_features):,}")
print(f"   positives            : {int(node_screening_features['label'].sum()):,}")
print(f"   motif merge applied  : {motif_merge_done}")

del tx, df_feats, labels_src, labels_dst
del all_node_feats, node_proxy_features, node_screening_features, window_risk_summary
gc.collect()

print("\nDone.")





import gc
import json
import os
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# ============================================================
# USER CONFIG
# ============================================================

OUTPUT_DIR   = Path("/content/drive/MyDrive/AML/outputs")
SCREEN_DIR   = OUTPUT_DIR / "screening"
MOTIF_DIR    = OUTPUT_DIR / "motif_features"
MODEL_DIR    = OUTPUT_DIR / "model"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

TX_PATH                 = OUTPUT_DIR / "transactions.parquet"
SCREEN_FEATURE_PATH     = SCREEN_DIR / "node_screening_features.parquet"
WINDOW_RISK_PATH        = SCREEN_DIR / "window_risk_summary.parquet"
NODE_WINDOW_PATH        = SCREEN_DIR / "node_window_features.parquet"
MOTIF_WIDE_PATH         = MOTIF_DIR  / "entity_feature_wide_merged.parquet"

# Stage B
RUN_STAGE_B             = True
RUN_NULL_MODEL          = False

# Narrowing controls for Stage B
TOP_K_WINDOWS_STAGE_B   = 7
TOP_NODES_PER_WINDOW    = 500
EXACT_WINDOW_SIZE       = 7
EXACT_WINDOW_STRIDE     = 7

# Safer matcher execution defaults
DEFAULT_ENABLED_MATCHERS = ["fanin", "fanout", "relay4"]
MAX_INSTANCES_STAGE_B    = 3000

# Subwindow config for transaction features
N_SUBWINDOWS            = 4

# Split
TEST_SIZE               = 0.20
VAL_SIZE                = 0.15
RANDOM_STATE            = 42

# XGBoost
N_ESTIMATORS            = 1500
EARLY_STOP              = 30

# Operating point
RECALL_TARGET           = 0.60


# ============================================================
# Internal helpers
# ============================================================

def _partition_instances_by_window(
    instances: list,
    summary_df: pd.DataFrame,
) -> dict:
    windows = [
        (f"{int(r.start)}_{int(r.end)}", int(r.start), int(r.end))
        for r in summary_df.itertuples(index=False)
    ]

    partitioned = defaultdict(list)

    for inst in instances:
        steps = inst.get("steps", [])
        first_step = int(min(steps)) if steps else -1
        assigned = False

        for wk, ws, we in windows:
            if ws <= first_step <= we:
                partitioned[wk].append(inst)
                assigned = True
                break

        if not assigned:
            partitioned["unassigned"].append(inst)

    if partitioned.get("unassigned"):
        print(
            f"  [WARN] {len(partitioned['unassigned'])} instances could not be "
            "assigned to any candidate window and were placed under 'unassigned'."
        )

    return dict(partitioned)


def _node_degree_from_tx(tx_df: pd.DataFrame) -> dict:
    out_deg = tx_df.groupby("src_node").size().rename("out")
    in_deg  = tx_df.groupby("dst_node").size().rename("in")
    deg = (
        pd.concat([out_deg, in_deg], axis=1)
        .fillna(0)
        .assign(degree=lambda d: d["out"] + d["in"])["degree"]
    )
    return deg.astype(int).to_dict()


def _subwindow_features(wdf: pd.DataFrame, suffix: str) -> pd.DataFrame:
    out = (
        wdf.groupby("src_node")["amount"]
        .agg(
            sum_spending="sum",
            mean_spending="mean",
            median_spending="median",
            std_spending="std",
            max_spending="max",
            min_spending="min",
            count_spending="count",
        )
        .reset_index()
        .rename(columns={"src_node": "node"})
    )

    both_amounts = pd.concat([
        wdf[["src_node", "amount"]].rename(columns={"src_node": "node"}),
        wdf[["dst_node", "amount"]].rename(columns={"dst_node": "node"}),
    ], ignore_index=True)

    total_stats = (
        both_amounts.groupby("node")["amount"]
        .agg(
            total_sum="sum",
            total_mean="mean",
            total_median="median",
            total_std="std",
            total_max="max",
            total_min="min",
        )
        .reset_index()
    )

    count_in = (
        wdf.groupby("dst_node").size()
        .rename("count_in").reset_index()
        .rename(columns={"dst_node": "node"})
    )
    count_out = (
        wdf.groupby("src_node").size()
        .rename("count_out").reset_index()
        .rename(columns={"src_node": "node"})
    )
    uniq_in = (
        wdf.groupby("dst_node")["src_node"].nunique()
        .rename("count_unique_in").reset_index()
        .rename(columns={"dst_node": "node"})
    )
    uniq_out = (
        wdf.groupby("src_node")["dst_node"].nunique()
        .rename("count_unique_out").reset_index()
        .rename(columns={"src_node": "node"})
    )

    step_range = (
        pd.concat([
            wdf[["src_node", "step"]].rename(columns={"src_node": "node"}),
            wdf[["dst_node", "step"]].rename(columns={"dst_node": "node"}),
        ])
        .groupby("node")["step"]
        .agg(step_first="min", step_last="max")
        .reset_index()
    )
    step_range["days_active"] = step_range["step_last"] - step_range["step_first"] + 1

    node_ids = pd.DataFrame({"node": both_amounts["node"].unique()}, dtype=np.int64)

    merged = (
        node_ids
        .merge(out, on="node", how="left")
        .merge(total_stats, on="node", how="left")
        .merge(count_in, on="node", how="left")
        .merge(count_out, on="node", how="left")
        .merge(uniq_in, on="node", how="left")
        .merge(uniq_out, on="node", how="left")
        .merge(step_range[["node", "days_active"]], on="node", how="left")
        .fillna(0)
    )

    merged["in_out_count_ratio"]  = merged["count_in"] / (merged["count_out"] + 1e-6)
    merged["spend_total_ratio"]   = merged["sum_spending"] / (merged["total_sum"] + 1e-6)
    merged["unique_in_out_ratio"] = merged["count_unique_in"] / (merged["count_unique_out"] + 1e-6)

    rename = {c: f"{c}{suffix}" for c in merged.columns if c != "node"}
    return merged.rename(columns=rename)


def _threshold_at_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    recall_target: float = 0.60,
) -> tuple[float, float, float]:
    prec, rec, thr = precision_recall_curve(y_true, y_prob)

    if len(thr) == 0:
        return 0.5, 0.0, 0.0

    mask = rec[:-1] >= recall_target

    if not np.any(mask):
        print(
            f"  [WARN] Recall target {recall_target:.2f} is unreachable. "
            "Falling back to the threshold that maximizes recall."
        )
        best = int(np.argmax(rec[:-1]))
        best = min(best, len(thr) - 1)
        return float(thr[best]), float(prec[:-1][best]), float(rec[:-1][best])

    valid = np.where(mask)[0]
    best  = valid[np.argmax(prec[:-1][valid])]
    return float(thr[best]), float(prec[:-1][best]), float(rec[:-1][best])


def _compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    recall_target: float = 0.60,
) -> dict:
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    mask = rec[:-1] >= recall_target
    p_at_r = float(np.max(prec[:-1][mask])) if np.any(mask) else 0.0
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc":  float(average_precision_score(y_true, y_prob)),
        f"p_at_r{int(recall_target*100)}": p_at_r,
    }


def _select_top_nodes_for_window(
    screen_window_df: pd.DataFrame,
    top_n: int = 500,
) -> set[int]:
    if screen_window_df.empty:
        return set()

    ranked = (
        screen_window_df
        .sort_values("node_screening_score", ascending=False)
        .head(top_n)
    )

    return set(ranked["node"].astype(int).tolist())


def _narrow_candidate_event_df(
    event_df: pd.DataFrame,
    top_nodes: set[int],
) -> pd.DataFrame:
    if not top_nodes:
        return pd.DataFrame(columns=event_df.columns)

    mask = event_df["src_node"].isin(top_nodes) | event_df["dst_node"].isin(top_nodes)
    narrowed = event_df.loc[mask].copy()

    if not narrowed.empty:
        narrowed = narrowed.sort_values(["step", "event_id"]).reset_index(drop=True)

    return narrowed


def _iter_exact_subwindows(
    event_df: pd.DataFrame,
    exact_window_size: int = 7,
    exact_stride: int = 7,
):
    if event_df.empty:
        return

    step_min = int(event_df["step"].min())
    step_max = int(event_df["step"].max())

    for ws in range(step_min, step_max + 1, exact_stride):
        we = ws + exact_window_size - 1
        sub = event_df[(event_df["step"] >= ws) & (event_df["step"] <= we)]
        if not sub.empty:
            yield ws, we, sub.reset_index(drop=True)


def _get_super_hubs(event_df: pd.DataFrame, max_degree: int = 1500) -> set[int]:
    """Tìm các nút siêu trung tâm (ví sàn, cổng thanh toán) có số lượng giao dịch quá lớn."""
    if event_df.empty:
        return set()
    # Đếm số giao dịch gửi đi và nhận về
    out_counts = event_df["src_node"].value_counts()
    in_counts = event_df["dst_node"].value_counts()
    total_counts = out_counts.add(in_counts, fill_value=0)

    # Trả về các nút có tổng giao dịch vượt ngưỡng max_degree
    return set(total_counts[total_counts > max_degree].index)


# ============================================================
# Step 1 — Load base inputs
# ============================================================

print("[1/8] Loading base inputs...")

tx = pd.read_parquet(TX_PATH)
tx["src_node"] = tx["src_node"].astype(np.int64)
tx["dst_node"] = tx["dst_node"].astype(np.int64)
tx["step"]     = tx["step"].astype(np.int32)
tx["amount"]   = tx["amount"].astype(np.float32)

if "is_sar" not in tx.columns:
    tx["is_sar"] = np.int8(0)
else:
    tx["is_sar"] = tx["is_sar"].astype(np.int8)

screen_feats = pd.read_parquet(SCREEN_FEATURE_PATH)

if "node" not in screen_feats.columns:
    raise ValueError("node_screening_features.parquet must contain a 'node' column.")
if "label" not in screen_feats.columns:
    raise ValueError("node_screening_features.parquet must contain a 'label' column.")

screen_X  = screen_feats.drop(columns=["label"])
label_map = screen_feats[["node", "label"]].drop_duplicates()
df_feats  = tx.drop(columns=["is_sar"]).copy()

step_min  = int(df_feats["step"].min())
step_max  = int(df_feats["step"].max())
total_vol = float(tx["amount"].sum())

print(f"      transactions        : {len(tx):,} rows")
print(f"      step range          : {step_min} – {step_max}")
print(f"      SAR rate            : {tx['is_sar'].mean()*100:.3f}%")
print(f"      screen_feats shape  : {screen_feats.shape}")


# ============================================================
# Step 2 — Stage B: narrowed exact motif mining
# ============================================================

print(f"\n[2/8] Stage B — exact motif mining (RUN_STAGE_B={RUN_STAGE_B})...")

motif_wide = None
stage_b_used = False

if RUN_STAGE_B:
    if not WINDOW_RISK_PATH.exists():
        raise FileNotFoundError(
            f"window_risk_summary.parquet not found: {WINDOW_RISK_PATH}. "
            "Run Cell 6 before Cell 7."
        )

    if not NODE_WINDOW_PATH.exists():
        raise FileNotFoundError(
            f"node_window_features.parquet not found: {NODE_WINDOW_PATH}. "
            "Cell 6 should save per-window node screening rows."
        )

    window_risk_df = pd.read_parquet(WINDOW_RISK_PATH)
    node_window_df = pd.read_parquet(NODE_WINDOW_PATH)

    print(f"      window_risk_df loaded : {len(window_risk_df)} windows")
    print(f"      node_window_df loaded : {len(node_window_df):,} node-window rows")

    cfg = MotifConfig()
    cfg.candidate_window_mode = "top_k"
    cfg.top_k_windows         = TOP_K_WINDOWS_STAGE_B
    cfg.max_windows_exact     = TOP_K_WINDOWS_STAGE_B
    cfg.enabled_matchers      = DEFAULT_ENABLED_MATCHERS
    cfg.export_instances      = False
    cfg.max_instances         = MAX_INSTANCES_STAGE_B
    cfg.n_permutations        = 5 if RUN_NULL_MODEL else 0

    selected = select_candidate_windows(
        load_windows_meta(WINDOW_META_PATH),
        cfg=cfg,
        window_risk_df=window_risk_df,
    )

    print(f"      Selected {len(selected)} candidate windows")

    all_instances = []
    summary_rows = []

    for row in selected.itertuples(index=False):
        step_start = int(row.start)
        step_end   = int(row.end)

        print(f"\n[Candidate window {int(row.window)}] {step_start}-{step_end} "
              f"| score={float(row.candidate_score):.3f} | n_temporal={int(row.n_temporal)}")

        event_df = load_event_window_from_temporal_shard(
            step_start=step_start,
            step_end=step_end,
            shard_dir=TEMPORAL_SHARD_DIR,
        )


        # 1. Nhận diện và cô lập các Siêu trung tâm (> 1500 giao dịch/cửa sổ)
        super_hubs = _get_super_hubs(event_df, max_degree=1500)

        # 2. Lấy dữ liệu screening cho khoảng thời gian này
        screen_slice = node_window_df[
            (node_window_df["window_start"] >= step_start) &
            (node_window_df["window_end"] <= step_end)
        ]

        if not screen_slice.empty:
            # Gom nhóm tính điểm trung bình (tránh trùng nút vì 1 candidate window chứa nhiều subwindow 7 bước)
            agg_screen = screen_slice.groupby("node")["node_screening_score"].mean().reset_index()

            # KIỂM DUYỆT TÀN NHẪN: Bắn bỏ các siêu trung tâm khỏi danh sách nghi ngờ
            agg_screen = agg_screen[~agg_screen["node"].isin(super_hubs)]

            # Chọn top nodes thực sự
            top_nodes = _select_top_nodes_for_window(agg_screen, top_n=TOP_NODES_PER_WINDOW)
        else:
            top_nodes = set()

        # 3. Thu hẹp sự kiện dựa trên top_nodes đã làm sạch
        narrowed_df = _narrow_candidate_event_df(event_df, top_nodes)

        # 4. CHỐT CHẶN AN TOÀN CUỐI CÙNG (Hard Cap)
        MAX_SAFE_EVENTS = 50_000
        if len(narrowed_df) > MAX_SAFE_EVENTS:
            print(f"      [WARN] narrowed_df quá lớn ({len(narrowed_df):,} edges). Bỏ qua để tránh nổ RAM.")
            narrowed_df = pd.DataFrame(columns=narrowed_df.columns) # Làm rỗng để skip bên dưới

        print(
            f"      original events={len(event_df):,} | "
            f"narrowed events={len(narrowed_df):,} | "
            f"top_nodes={len(top_nodes):,}"
        )

        # screen_slice = node_window_df[
        #     (node_window_df["window_start"] >= step_start) &
        #      (node_window_df["window_end"] <= step_end)
        # ]

        # top_nodes = _select_top_nodes_for_window(
        #     screen_slice,
        #     top_n=TOP_NODES_PER_WINDOW,
        # )

        # narrowed_df = _narrow_candidate_event_df(event_df, top_nodes)

        # print(
        #     f"      original events={len(event_df):,} | "
        #     f"narrowed events={len(narrowed_df):,} | "
        #     f"top_nodes={len(top_nodes):,}"
        # )

        if narrowed_df.empty:
            summary_rows.append({
                "window": int(row.window),
                "start": step_start,
                "end": step_end,
                "candidate_score": float(row.candidate_score),
                "n_events_original": int(len(event_df)),
                "n_events_narrowed": 0,
                "n_instances": 0,
                "zscores": {},
            })
            del event_df, screen_slice, top_nodes, narrowed_df
            gc.collect()
            continue

        window_instances = []
        window_zscores = {}

        for sub_start, sub_end, sub_df in _iter_exact_subwindows(
            narrowed_df,
            exact_window_size=EXACT_WINDOW_SIZE,
            exact_stride=EXACT_WINDOW_STRIDE,
        ):
            out_idx, in_idx, _, out_steps = build_event_indexes(sub_df)
            raw_instances = run_all_matchers(out_idx, in_idx, cfg, out_steps)
            observed_counts = count_support(raw_instances)

            zscore_results = None
            if RUN_NULL_MODEL and observed_counts:
                zscore_results = compute_null_zscore(
                    observed_counts=observed_counts,
                    event_df=sub_df,
                    cfg=cfg,
                    sample_frac=1.0,
                    null_cap=MAX_INSTANCES_STAGE_B,
                    verbose=False,
                )

            filtered_instances = filter_motifs(
                raw_instances,
                cfg,
                zscore_results=zscore_results,
            )

            if filtered_instances:
                window_instances.extend(filtered_instances)

            if zscore_results:
                for mt, val in zscore_results.items():
                    window_zscores[mt] = val

            del out_idx, in_idx, out_steps, sub_df, raw_instances, filtered_instances
            gc.collect()

        summary_rows.append({
            "window": int(row.window),
            "start": step_start,
            "end": step_end,
            "candidate_score": float(row.candidate_score),
            "n_events_original": int(len(event_df)),
            "n_events_narrowed": int(len(narrowed_df)),
            "n_instances": int(len(window_instances)),
            "zscores": window_zscores,
        })

        all_instances.extend(window_instances)

        del event_df, screen_slice, top_nodes, narrowed_df, window_instances
        gc.collect()

    summary_df = pd.DataFrame(summary_rows)

    print(f"\n      Total motif instances: {len(all_instances):,}")
    print(f"      Candidate windows processed: {len(summary_df)}")

    if len(all_instances) > 0:
        instances_by_window = _partition_instances_by_window(
            all_instances,
            summary_df.rename(columns={"n_events_original": "n_events"})
        )

        node_degree = _node_degree_from_tx(tx)

        zscore_by_window = {
            f"{int(r.start)}_{int(r.end)}": r.zscores
            for r in summary_df.itertuples(index=False)
            if isinstance(r.zscores, dict) and len(r.zscores) > 0
        }

        volume_by_window = {
            f"{int(r.start)}_{int(r.end)}": total_vol
            for r in summary_df.itertuples(index=False)
        }

        print("      Exporting motif features...")
        export_paths = export_motif_feature_outputs(
            window_summary_df=summary_df,
            motif_instances_by_window=instances_by_window,
            zscore_by_window=zscore_by_window if zscore_by_window else None,
            node_degree_by_window={wk: node_degree for wk in instances_by_window},
            total_volume_by_window=volume_by_window,
            export_dir=str(MOTIF_DIR),
            export_instances=False,
            feature_format="parquet",
        )

        for k, v in export_paths.items():
            print(f"         {k:<30} -> {v}")

        if MOTIF_WIDE_PATH.exists():
            motif_wide = pd.read_parquet(MOTIF_WIDE_PATH)
            if "node" not in motif_wide.columns:
                raise ValueError("entity_feature_wide_merged.parquet must contain a 'node' column.")
            print(f"      motif_wide loaded: {motif_wide.shape}")
            stage_b_used = True
        else:
            print("      [WARN] motif_wide export missing after Stage B.")
    else:
        print("      No motif instances found. Proceeding with Stage A only.")

    del window_risk_df, node_window_df
    gc.collect()

else:
    if MOTIF_WIDE_PATH.exists():
        motif_wide = pd.read_parquet(MOTIF_WIDE_PATH)
        if "node" not in motif_wide.columns:
            raise ValueError("entity_feature_wide_merged.parquet must contain a 'node' column.")
        print(f"      Stage B skipped. Loaded pre-existing motif_wide: {motif_wide.shape}")
        stage_b_used = True
    else:
        print("      Stage B skipped. No pre-existing motif features found.")


# ============================================================
# Step 3 — Subwindow transaction features
# ============================================================

print(f"\n[3/8] Building subwindow transaction features...")

TOTAL_STEPS  = step_max - step_min + 1
SUB_WIN_SIZE = max(1, TOTAL_STEPS // N_SUBWINDOWS)

print(
    f"      TOTAL_STEPS={TOTAL_STEPS} | "
    f"N_SUBWINDOWS={N_SUBWINDOWS} | "
    f"SUB_WIN_SIZE={SUB_WIN_SIZE}"
)

all_sw = []

for sw in range(N_SUBWINDOWS):
    ws = step_min + sw * SUB_WIN_SIZE
    we = ws + SUB_WIN_SIZE - 1
    suffix = f"_w{sw}"

    wdf = df_feats[(df_feats["step"] >= ws) & (df_feats["step"] <= we)]
    if wdf.empty:
        print(f"      sub-window {sw} [{ws}-{we}]: empty, skipped")
        continue

    sw_feat = _subwindow_features(wdf, suffix)
    all_sw.append(sw_feat)

    del wdf, sw_feat
    gc.collect()
    print(f"      sub-window {sw} [{ws}-{we}]: done", end="\r")

print()

if not all_sw:
    raise RuntimeError("All subwindows were empty. Check step range and N_SUBWINDOWS.")

sw_combined = all_sw[0]
for frame in all_sw[1:]:
    sw_combined = sw_combined.merge(frame, on="node", how="outer")
sw_combined = sw_combined.fillna(0)

print(f"      subwindow features shape: {sw_combined.shape}")

del all_sw, df_feats
gc.collect()


# ============================================================
# Step 4 — Assemble node-level feature matrix
# ============================================================

print("\n[4/8] Assembling feature matrix...")

node_matrix = screen_X.merge(sw_combined, on="node", how="left")

# SAU KHI FIX — Cell 7, Step 4
if motif_wide is not None:
    # Loại bỏ các cột đã tồn tại trong node_matrix trước khi merge
    # để tránh pandas tạo ra _x và _y suffixes
    overlap = [
        c for c in motif_wide.columns
        if c != "node" and c in node_matrix.columns
    ]
    if overlap:
        print(f"      [INFO] Dropping {len(overlap)} overlapping columns "
              f"from node_matrix before motif merge: {overlap[:5]}...")
        node_matrix = node_matrix.drop(columns=overlap)
    
    node_matrix = node_matrix.merge(motif_wide, on="node", how="left")
    print(
        f"      motif columns added: "
        f"{len([c for c in motif_wide.columns if c != 'node'])}"
    )

node_matrix = node_matrix.fillna(0)

node_matrix = node_matrix.merge(label_map, on="node", how="left")
node_matrix["label"] = node_matrix["label"].fillna(0).astype(np.int8)

n_pos = int(node_matrix["label"].sum())
n_tot = len(node_matrix)

print(f"      node_matrix shape : {node_matrix.shape}")
print(f"      SAR nodes         : {n_pos:,} ({n_pos/n_tot*100:.2f}%)")
print(f"      Normal nodes      : {n_tot-n_pos:,} ({(n_tot-n_pos)/n_tot*100:.2f}%)")
print(
    f"      Feature sources   : screening={screen_X.shape[1]-1} cols | "
    f"subwindow={sw_combined.shape[1]-1} cols | "
    f"motif={motif_wide.shape[1]-1 if motif_wide is not None else 0} cols"
)

del sw_combined, screen_X
gc.collect()


<!-- # ============================================================
# Step 5 — Train / validation / test split
# ============================================================

print("\n[5/8] Stratified train / val / test split...")

EXCLUDE = {"node", "label"}
feature_cols = [c for c in node_matrix.columns if c not in EXCLUDE]

X        = node_matrix[feature_cols].values.astype(np.float32)
y        = node_matrix["label"].values.astype(np.int8)
node_ids = node_matrix["node"].values.astype(np.int64)

X_trainval, X_test, y_trainval, y_test, nodes_trainval, nodes_test = train_test_split(
    X, y, node_ids,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)

X_train, X_val, y_train, y_val, nodes_train, nodes_val = train_test_split(
    X_trainval, y_trainval, nodes_trainval,
    test_size=VAL_SIZE,
    stratify=y_trainval,
    random_state=RANDOM_STATE,
)

print(f"      Train : {len(X_train):,} (pos={int(y_train.sum()):,}, {y_train.mean()*100:.2f}%)")
print(f"      Val   : {len(X_val):,} (pos={int(y_val.sum()):,}, {y_val.mean()*100:.2f}%)")
print(f"      Test  : {len(X_test):,} (pos={int(y_test.sum()):,}, {y_test.mean()*100:.2f}%)")
print(f"      Features: {len(feature_cols)}")

del X_trainval, y_trainval, nodes_trainval
gc.collect() -->
# ============================================================
# Step 5 — Train / validation / test split by first appearance time
# ============================================================

print("\n[5/8] Time-based train / val / test split by node first_step...")

# ------------------------------------------------------------
# 1. Feature columns
# ------------------------------------------------------------
EXCLUDE = {"node", "label"}
feature_cols = [c for c in node_matrix.columns if c not in EXCLUDE]

# ------------------------------------------------------------
# 2. Compute first_step for each node from raw transactions
#    first_step[node] = earliest step where node appears
# ------------------------------------------------------------
src_first = (
    tx.groupby("src_node", as_index=False)["step"]
    .min()
    .rename(columns={"src_node": "node", "step": "first_step_src"})
)

dst_first = (
    tx.groupby("dst_node", as_index=False)["step"]
    .min()
    .rename(columns={"dst_node": "node", "step": "first_step_dst"})
)

first_step_df = (
    src_first.merge(dst_first, on="node", how="outer")
)

first_step_df["first_step"] = first_step_df[
    ["first_step_src", "first_step_dst"]
].min(axis=1)

first_step_df = first_step_df[["node", "first_step"]].copy()
first_step_df["first_step"] = first_step_df["first_step"].astype(np.int32)

# ------------------------------------------------------------
# 3. Attach first_step to node_matrix
# ------------------------------------------------------------
node_matrix = node_matrix.merge(first_step_df, on="node", how="left")

if node_matrix["first_step"].isna().any():
    missing_nodes = int(node_matrix["first_step"].isna().sum())
    raise ValueError(
        f"{missing_nodes} nodes in node_matrix do not have first_step. "
        "This indicates a mismatch between node_matrix and transactions."
    )

node_matrix["first_step"] = node_matrix["first_step"].astype(np.int32)

# ------------------------------------------------------------
# 4. Define temporal split ranges
#    Train: step 0  - 78
#    Val  : step 79 - 89
#    Test : step 90 - 111
# ------------------------------------------------------------
TRAIN_END = 78
VAL_START = 79
VAL_END = 89
TEST_START = 90

train_mask = node_matrix["first_step"] <= TRAIN_END
val_mask   = (node_matrix["first_step"] >= VAL_START) & (node_matrix["first_step"] <= VAL_END)
test_mask  = node_matrix["first_step"] >= TEST_START

train_df = node_matrix.loc[train_mask].copy()
val_df   = node_matrix.loc[val_mask].copy()
test_df  = node_matrix.loc[test_mask].copy()

# ------------------------------------------------------------
# 5. Basic safety checks
# ------------------------------------------------------------
if train_df.empty:
    raise RuntimeError("Train set is empty after time-based split.")
if val_df.empty:
    raise RuntimeError("Validation set is empty after time-based split.")
if test_df.empty:
    raise RuntimeError("Test set is empty after time-based split.")

if train_df["label"].sum() == 0:
    raise RuntimeError("Train set has no positive (SAR) nodes.")
if val_df["label"].sum() == 0:
    raise RuntimeError("Validation set has no positive (SAR) nodes.")
if test_df["label"].sum() == 0:
    raise RuntimeError("Test set has no positive (SAR) nodes.")

# Optional: enforce disjoint node sets
train_nodes = set(train_df["node"].tolist())
val_nodes   = set(val_df["node"].tolist())
test_nodes  = set(test_df["node"].tolist())

assert train_nodes.isdisjoint(val_nodes), "Train and Val node sets overlap."
assert train_nodes.isdisjoint(test_nodes), "Train and Test node sets overlap."
assert val_nodes.isdisjoint(test_nodes), "Val and Test node sets overlap."

# ------------------------------------------------------------
# 6. Build numpy arrays
# ------------------------------------------------------------
X_train = train_df[feature_cols].values.astype(np.float32)
y_train = train_df["label"].values.astype(np.int8)
nodes_train = train_df["node"].values.astype(np.int64)

X_val = val_df[feature_cols].values.astype(np.float32)
y_val = val_df["label"].values.astype(np.int8)
nodes_val = val_df["node"].values.astype(np.int64)

X_test = test_df[feature_cols].values.astype(np.float32)
y_test = test_df["label"].values.astype(np.int8)
nodes_test = test_df["node"].values.astype(np.int64)

# ------------------------------------------------------------
# 7. Reporting
# ------------------------------------------------------------
print(f"      Train range : step <= {TRAIN_END}")
print(f"      Val range   : step {VAL_START}-{VAL_END}")
print(f"      Test range  : step >= {TEST_START}")

print(f"      Train : {len(X_train):,} (pos={int(y_train.sum()):,}, {y_train.mean()*100:.2f}%)")
print(f"      Val   : {len(X_val):,} (pos={int(y_val.sum()):,}, {y_val.mean()*100:.2f}%)")
print(f"      Test  : {len(X_test):,} (pos={int(y_test.sum()):,}, {y_test.mean()*100:.2f}%)")
print(f"      Features: {len(feature_cols)}")

print(
    f"      first_step ranges -> "
    f"train[{train_df['first_step'].min()}-{train_df['first_step'].max()}], "
    f"val[{val_df['first_step'].min()}-{val_df['first_step'].max()}], "
    f"test[{test_df['first_step'].min()}-{test_df['first_step'].max()}]"
)

# ------------------------------------------------------------
# 8. Cleanup
# ------------------------------------------------------------
del src_first, dst_first, first_step_df
del train_df, val_df, test_df
gc.collect()

# ============================================================
# Step 6 — Train XGBoost
# ============================================================

print("\n[6/8] Training XGBoost...")

n_pos_tr = int(y_train.sum())
n_neg_tr = int(len(y_train) - n_pos_tr)

if n_pos_tr == 0:
    raise RuntimeError("Training fold contains no positive samples.")

scale_pos = float(n_neg_tr) / float(n_pos_tr)
print(f"      scale_pos_weight = {scale_pos:.2f}")

model = xgb.XGBClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.6,
    colsample_bytree=0.5,
    min_child_weight=20,
    scale_pos_weight=scale_pos,
    eval_metric="aucpr",
    early_stopping_rounds=EARLY_STOP,
    random_state=RANDOM_STATE,
    tree_method="hist",
    n_jobs=-1,
    reg_alpha=0.1,
    reg_lambda=5.0,
    gamma=1.0,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)

print(f"\n      Best iteration : {model.best_iteration}")

y_prob_train = model.predict_proba(X_train)[:, 1]
y_prob_val   = model.predict_proba(X_val)[:, 1]
y_prob_test  = model.predict_proba(X_test)[:, 1]

val_threshold, val_prec, val_rec = _threshold_at_recall(
    y_val, y_prob_val, recall_target=RECALL_TARGET
)

print(f"\n      Validation threshold = {val_threshold:.4f}")
print(f"      Validation precision = {val_prec:.4f}")
print(f"      Validation recall    = {val_rec:.4f}")

train_metrics = _compute_metrics(y_train, y_prob_train, RECALL_TARGET)
val_metrics   = _compute_metrics(y_val,   y_prob_val,   RECALL_TARGET)
test_metrics  = _compute_metrics(y_test,  y_prob_test,  RECALL_TARGET)

p_key = f"p_at_r{int(RECALL_TARGET*100)}"

print(f"\n      {'Metric':<22} {'Train':>8} {'Val':>8} {'Test':>8}")
print(f"      {'-'*52}")
for k, label in [
    ("roc_auc", "ROC-AUC"),
    ("pr_auc",  "PR-AUC"),
    (p_key,     f"P@R≥{RECALL_TARGET:.0%}"),
]:
    print(
        f"      {label:<22} "
        f"{train_metrics.get(k, 0):>8.4f} "
        f"{val_metrics.get(k, 0):>8.4f} "
        f"{test_metrics.get(k, 0):>8.4f}"
    )


# ============================================================
# Step 7 — Test-set evaluation
# ============================================================

print(f"\n[7/8] Test-set evaluation (threshold={val_threshold:.4f})...")

y_pred_test = (y_prob_test >= val_threshold).astype(np.int8)

print(classification_report(
    y_test, y_pred_test,
    target_names=["normal", "SAR"],
    digits=4,
))

cm = confusion_matrix(y_test, y_pred_test)
print("      Confusion matrix [TN FP; FN TP]:")
print(cm)

if HAS_PLT:
    p_curve, r_curve, _ = precision_recall_curve(y_test, y_prob_test)
    ap = average_precision_score(y_test, y_prob_test)

    plt.figure(figsize=(7, 5))
    plt.plot(r_curve, p_curve, label=f"Test (AP={ap:.3f})")
    plt.axvline(RECALL_TARGET, linestyle="--", alpha=0.6,
                label=f"Recall target = {RECALL_TARGET:.0%}")
    plt.scatter([val_rec], [val_prec], zorder=5,
                label=f"Operating point (P={val_prec:.2f}, R={val_rec:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve — Stage A + Stage B")
    plt.legend()
    plt.tight_layout()
    pr_path = MODEL_DIR / "pr_curve.png"
    plt.savefig(pr_path, dpi=150)
    plt.show()
    print(f"      PR curve -> {pr_path}")


# ============================================================
# Step 8 — Save model artifacts
# ============================================================

print("\n[8/8] Saving model artifacts...")

model_path = MODEL_DIR / "xgb_model.json"
model.save_model(model_path)

score_dict = model.get_booster().get_score(importance_type="gain")
gain_values = [score_dict.get(f"f{i}", 0.0) for i in range(len(feature_cols))]
imp_df = (
    pd.DataFrame({"feature": feature_cols, "gain": gain_values})
    .sort_values("gain", ascending=False)
    .reset_index(drop=True)
)
imp_path = MODEL_DIR / "feature_importance.parquet"
imp_df.to_parquet(imp_path, index=False)

preds_df = pd.DataFrame({
    "node": nodes_test,
    "label": y_test,
    "prob_sar": y_prob_test,
    "pred_sar": y_pred_test,
})
preds_path = MODEL_DIR / "test_predictions.parquet"
preds_df.to_parquet(preds_path, index=False)

metrics = {
    "recall_target": RECALL_TARGET,
    "threshold_from_validation": float(val_threshold),
    "validation_operating_point": {
        "precision": float(val_prec),
        "recall": float(val_rec),
    },
    "train": train_metrics,
    "val": val_metrics,
    "test": test_metrics,
    "n_features": len(feature_cols),
    "n_train": int(len(X_train)),
    "n_val": int(len(X_val)),
    "n_test": int(len(X_test)),
    "n_pos_train": int(y_train.sum()),
    "n_pos_val": int(y_val.sum()),
    "n_pos_test": int(y_test.sum()),
    "best_iteration": int(model.best_iteration),
    "stage_b_ran": bool(RUN_STAGE_B and stage_b_used),
    "stage_b_config": {
        "top_k_windows": TOP_K_WINDOWS_STAGE_B,
        "top_nodes_per_window": TOP_NODES_PER_WINDOW,
        "exact_window_size": EXACT_WINDOW_SIZE,
        "exact_window_stride": EXACT_WINDOW_STRIDE,
        "enabled_matchers": DEFAULT_ENABLED_MATCHERS,
        "max_instances_stage_b": MAX_INSTANCES_STAGE_B,
    },
    "subwindow_config": {
        "n_subwindows": N_SUBWINDOWS,
        "total_steps": TOTAL_STEPS,
        "sub_win_size": SUB_WIN_SIZE,
        "step_min": step_min,
        "step_max": step_max,
    },
    "feature_sources": {
        "screening": str(SCREEN_FEATURE_PATH),
        "transactions": str(TX_PATH),
        "motif_wide": str(MOTIF_WIDE_PATH) if stage_b_used else None,
    },
    "test_confusion_matrix": {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    },
}

metrics_path = MODEL_DIR / "metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

feature_cols_path = MODEL_DIR / "feature_columns.json"
with open(feature_cols_path, "w") as f:
    json.dump(feature_cols, f, indent=2)

print("\nArtifacts written:")
for label, path in [
    ("model", model_path),
    ("test_predictions", preds_path),
    ("metrics", metrics_path),
    ("feature_importance", imp_path),
    ("feature_columns", feature_cols_path),
]:
    size_kb = os.path.getsize(path) / 1024
    print(f"   [{label:<22}] {path} ({size_kb:.1f} KB)")

print("\nTop 20 features by gain:")
print(imp_df.head(20).to_string(index=False))

del tx, screen_feats, node_matrix
del X, y, node_ids
del X_train, X_val, X_test
del y_train, y_val, y_test
del nodes_train, nodes_val, nodes_test
gc.collect()

print("\nDone.")