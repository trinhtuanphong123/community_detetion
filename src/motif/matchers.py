"""
matchers.py — Temporal motif matching for AML detection.

Five templates:
    fan_in      : many → one within delta steps
    fan_out     : one → many within delta steps
    cycle_3     : u → v → w → u with strict time order
    relay_4     : u → v → w → x with strict time order
    split_merge : u → v1 → z and u → v2 → z

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

from .config import MotifConfig
from .index import edges_after_step


# ---------------------------------------------------------------------------
# Primitive constraint helpers
# ---------------------------------------------------------------------------

def _ratio_ok(a_prev: float, a_curr: float, rho_min: float, rho_max: float) -> bool:
    """True if a_curr / a_prev ∈ [rho_min, rho_max]."""
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
# Fan-in
# ---------------------------------------------------------------------------

def find_fanin(
    in_index: dict,
    cfg: MotifConfig,
) -> list[dict]:
    """
    Fan-in: r_min_fanin distinct sources → same destination x, within delta steps.

        u1 → x
        u2 → x   (step_u2 - step_u1 <= delta)
        u3 → x   ...

    Constraints:
        - All sources distinct.
        - All arrivals in [t0, t0 + delta].
        - Amount ratio of each edge vs seed in [rho_min, rho_max].
        - At least r_min_fanin sources found.

    Returns list of instance dicts.
    """
    results = []

    for x, incoming in in_index.items():
        n = len(incoming)
        if n < cfg.r_min_fanin:
            continue

        for i in range(n):
            seed   = incoming[i]
            t0     = seed["step"]
            a0     = seed["amount"]
            seen   = {seed["src"]}
            group  = [seed]

            for j in range(i + 1, n):
                e = incoming[j]

                # Prune: window exceeded — bucket is sorted, so break early
                if e["step"] - t0 > cfg.delta:
                    break

                # Prune: duplicate source — skip, do not break
                if e["src"] in seen:
                    continue

                # Prune: amount ratio
                if not _ratio_ok(a0, e["amount"], cfg.rho_min, cfg.rho_max):
                    continue

                seen.add(e["src"])
                group.append(e)

            if len(group) >= cfg.r_min_fanin:
                nodes = [e["src"] for e in group] + [x]
                results.append(_make_instance("fanin", group, nodes))

    return results


# ---------------------------------------------------------------------------
# Fan-out
# ---------------------------------------------------------------------------

def find_fanout(
    out_index: dict,
    cfg: MotifConfig,
) -> list[dict]:
    """
    Fan-out: source x → r_min_fanout distinct destinations, within delta steps.

        x → v1
        x → v2   (step_v2 - step_v1 <= delta)
        x → v3   ...

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

            for j in range(i + 1, n):
                e = outgoing[j]

                if e["step"] - t0 > cfg.delta:
                    break

                if e["dst"] in seen:
                    continue

                if not _ratio_ok(a0, e["amount"], cfg.rho_min, cfg.rho_max):
                    continue

                seen.add(e["dst"])
                group.append(e)

            if len(group) >= cfg.r_min_fanout:
                nodes = [x] + [e["dst"] for e in group]
                results.append(_make_instance("fanout", group, nodes))

    return results


# ---------------------------------------------------------------------------
# Cycle-3
# ---------------------------------------------------------------------------

def find_cycle3(
    out_index: dict,
    cfg: MotifConfig,
) -> list[dict]:
    """
    Cycle-3: u → v → w → u with strictly increasing steps.

    Constraints:
        - step_e1 < step_e2 < step_e3 (strict)
        - Each consecutive lag <= delta
        - Amount ratio at each hop in [rho_min, rho_max]
        - u, v, w are three distinct nodes

    Uses edges_after_step() for forward-only bucket access.
    Returns list of instance dicts.
    """
    results = []

    for u, edges_u in out_index.items():
        for e1 in edges_u:
            v  = e1["dst"]
            t1 = e1["step"]
            a1 = e1["amount"]

            if v == u:
                continue

            # e2: v → w, step in (t1, t1 + delta]
            for e2 in edges_after_step(out_index, v, t1):
                if e2["step"] - t1 > cfg.delta:
                    break  # bucket sorted → nothing after is valid

                w  = e2["dst"]
                a2 = e2["amount"]

                if w == u or w == v:
                    continue

                if not _ratio_ok(a1, a2, cfg.rho_min, cfg.rho_max):
                    continue

                # e3: w → u, step in (t2, t2 + delta]
                for e3 in edges_after_step(out_index, w, e2["step"]):
                    if e3["step"] - e2["step"] > cfg.delta:
                        break

                    if e3["dst"] != u:
                        continue

                    if not _ratio_ok(a2, e3["amount"], cfg.rho_min, cfg.rho_max):
                        continue

                    results.append(_make_instance(
                        "cycle3",
                        [e1, e2, e3],
                        [u, v, w, u],
                    ))

    return results


# ---------------------------------------------------------------------------
# Relay-4 (chain)
# ---------------------------------------------------------------------------

def find_relay4(
    out_index: dict,
    cfg: MotifConfig,
) -> list[dict]:
    """
    Relay-4: u → v → w → x with strictly increasing steps.

    Constraints:
        - step_e1 < step_e2 < step_e3 (strict)
        - Each consecutive lag <= delta
        - Amount ratio at each hop in [rho_min, rho_max]
        - u, v, w, x are four distinct nodes

    Uses edges_after_step() for forward-only access.
    Returns list of instance dicts.
    """
    results = []

    for u, edges_u in out_index.items():
        for e1 in edges_u:
            v  = e1["dst"]
            t1 = e1["step"]
            a1 = e1["amount"]

            if v == u:
                continue

            for e2 in edges_after_step(out_index, v, t1):
                if e2["step"] - t1 > cfg.delta:
                    break

                w  = e2["dst"]
                a2 = e2["amount"]

                if w in (u, v):
                    continue

                if not _ratio_ok(a1, a2, cfg.rho_min, cfg.rho_max):
                    continue

                for e3 in edges_after_step(out_index, w, e2["step"]):
                    if e3["step"] - e2["step"] > cfg.delta:
                        break

                    x  = e3["dst"]
                    a3 = e3["amount"]

                    if x in (u, v, w):
                        continue

                    if not _ratio_ok(a2, a3, cfg.rho_min, cfg.rho_max):
                        continue

                    results.append(_make_instance(
                        "relay4",
                        [e1, e2, e3],
                        [u, v, w, x],
                    ))

    return results


# ---------------------------------------------------------------------------
# Split-merge
# ---------------------------------------------------------------------------

def find_split_merge(
    out_index: dict,
    in_index: dict,
    cfg: MotifConfig,
) -> list[dict]:
    """
    Split-merge: source splits to two intermediaries that recombine at one target.

        u → v1 → z
        u → v2 → z

    Two-phase:
        Phase 1: find split pairs (u → v1, u → v2) within delta steps.
        Phase 2: for each pair, find a common target z that both v1 and v2
                 reach within 2 * delta steps of the split start.

    Constraints:
        - u, v1, v2, z are four distinct nodes
        - Amount ratio at every hop in [rho_min, rho_max]
        - All events in chronological order per hop

    Returns list of instance dicts, each with 4 edges sorted by step.
    """
    results = []

    for u, outgoing_u in out_index.items():
        n = len(outgoing_u)

        for i in range(n):
            e_uv1 = outgoing_u[i]
            v1    = e_uv1["dst"]
            t0    = e_uv1["step"]
            a_uv1 = e_uv1["amount"]

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

                # Phase 2: collect targets reachable from v1 in (t0, t0 + 2*delta]
                v1_targets: dict = {}  # z → edge dict
                for e in edges_after_step(out_index, v1, t0):
                    if e["step"] - t0 > 2 * cfg.delta:
                        break
                    z = e["dst"]
                    if z in (u, v1, v2):
                        continue
                    if not _ratio_ok(a_uv1, e["amount"], cfg.rho_min, cfg.rho_max):
                        continue
                    v1_targets[z] = e

                if not v1_targets:
                    continue

                # Find v2 → z where z ∈ v1_targets
                for e_v2z in edges_after_step(out_index, v2, t0):
                    if e_v2z["step"] - t0 > 2 * cfg.delta:
                        break

                    z = e_v2z["dst"]
                    if z not in v1_targets:
                        continue
                    if z in (u, v1, v2):
                        continue

                    e_v1z = v1_targets[z]

                    if not _ratio_ok(e_uv2["amount"], e_v2z["amount"], cfg.rho_min, cfg.rho_max):
                        continue

                    # Sort all four edges chronologically for the instance
                    all_edges = sorted(
                        [e_uv1, e_uv2, e_v1z, e_v2z],
                        key=lambda e: e["step"],
                    )
                    results.append(_make_instance(
                        "split_merge",
                        all_edges,
                        [u, v1, v2, z],
                    ))

    return results


# ---------------------------------------------------------------------------
# Convenience: run all matchers on a pre-built index set
# ---------------------------------------------------------------------------

def run_all_matchers(
    out_index: dict,
    in_index: dict,
    cfg: MotifConfig,
) -> list[dict]:
    """
    Run all five matchers and return a combined instance list.

    Parameters
    ----------
    out_index, in_index : dict
        Output of build_event_indexes().
    cfg : MotifConfig
        Motif configuration.

    Returns
    -------
    Combined list of all matched motif instances across all types.
    """
    return (
        find_fanin(in_index, cfg)
        + find_fanout(out_index, cfg)
        + find_cycle3(out_index, cfg)
        + find_relay4(out_index, cfg)
        + find_split_merge(out_index, in_index, cfg)
    )


__all__ = [
    "find_fanin",
    "find_fanout",
    "find_cycle3",
    "find_relay4",
    "find_split_merge",
    "run_all_matchers",
]