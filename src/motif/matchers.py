# src/motif/matchers.py
# Motif matchers cho 5 template AML.
#
# Nguyên tắc chung cho mọi matcher:
#   1. Search forward only — không quay lui
#   2. Stop early khi vi phạm bất kỳ điều kiện nào (pruning)
#   3. Trả về list[dict] — mỗi dict là một motif instance
#   4. Không side effects — không modify index
#
# Mỗi instance dict có keys:
#   motif_type, nodes, edges (event_id list), steps, amounts, lags
#
# Lưu ý: matchers không import pandas/cuDF —
# chỉ làm việc với dict và list thuần Python.

from .config import MotifConfig


def _ratio_ok(a_prev: float, a_curr: float, rho_min: float, rho_max: float) -> bool:
    """Kiểm tra tỷ lệ amount có trong ngưỡng [rho_min, rho_max]."""
    if a_prev <= 0:
        return False
    ratio = a_curr / a_prev
    return rho_min <= ratio <= rho_max


def _lag_ok(step_prev: int, step_curr: int, delta: int) -> bool:
    """Kiểm tra step_curr > step_prev và lag <= delta."""
    lag = step_curr - step_prev
    return 0 < lag <= delta


def _make_instance(motif_type: str, edges: list[dict], nodes: list[int]) -> dict:
    """Tạo instance dict chuẩn từ danh sách edge dicts."""
    steps   = [e["step"] for e in edges]
    amounts = [e["amount"] for e in edges]
    lags    = [steps[i] - steps[i - 1] for i in range(1, len(steps))]
    return {
        "motif_type": motif_type,
        "nodes":      nodes,
        "edges":      [e["event_id"] for e in edges],
        "steps":      steps,
        "amounts":    amounts,
        "lags":       lags,
        "n_alert":    sum(e["is_laundering"] for e in edges),
    }


# ── Fan-in ────────────────────────────────────────────────────────────────

def find_fanin(
    in_index: dict,
    cfg: MotifConfig,
) -> list[dict]:
    """
    Fan-in: nhiều nguồn khác nhau chuyển vào cùng một đích x
    trong khoảng DELTA steps.

        u1 → x
        u2 → x   (step_u2 - step_u1 <= DELTA)
        u3 → x   ...

    Điều kiện:
        - Các nguồn (u_i) phải khác nhau
        - Tất cả nằm trong cửa sổ [t0, t0 + DELTA]
        - Tỷ lệ amount so với seed trong [RHO_MIN, RHO_MAX]
        - Số nguồn >= R_MIN_FANIN

    Returns:
        list[dict] — mỗi dict là một fan-in instance.
    """
    results = []

    for x, incoming in in_index.items():
        n = len(incoming)
        if n < cfg.R_MIN_FANIN:
            continue

        for i in range(n):
            seed   = incoming[i]
            t0     = seed["step"]
            a0     = seed["amount"]
            seen   = {seed["src"]}
            group  = [seed]

            for j in range(i + 1, n):
                e = incoming[j]

                # Pruning 1: step đã vượt cửa sổ → dừng (đã sort theo step)
                if e["step"] - t0 > cfg.DELTA:
                    break

                # Pruning 2: cùng nguồn → bỏ qua
                if e["src"] in seen:
                    continue

                # Pruning 3: amount ratio
                if not _ratio_ok(a0, e["amount"], cfg.RHO_MIN, cfg.RHO_MAX):
                    continue

                seen.add(e["src"])
                group.append(e)

            if len(group) >= cfg.R_MIN_FANIN:
                nodes = [e["src"] for e in group] + [x]
                results.append(_make_instance("fanin", group, nodes))

    return results


# ── Fan-out ───────────────────────────────────────────────────────────────

def find_fanout(
    out_index: dict,
    cfg: MotifConfig,
) -> list[dict]:
    """
    Fan-out: một nguồn x gửi đến nhiều đích khác nhau
    trong khoảng DELTA steps.

        x → v1
        x → v2   (step_v2 - step_v1 <= DELTA)
        x → v3   ...

    Điều kiện:
        - Các đích (v_i) phải khác nhau
        - Tất cả nằm trong cửa sổ [t0, t0 + DELTA]
        - Tỷ lệ amount so với seed trong [RHO_MIN, RHO_MAX]
        - Số đích >= R_MIN_FANOUT

    Returns:
        list[dict]
    """
    results = []

    for x, outgoing in out_index.items():
        n = len(outgoing)
        if n < cfg.R_MIN_FANOUT:
            continue

        for i in range(n):
            seed   = outgoing[i]
            t0     = seed["step"]
            a0     = seed["amount"]
            seen   = {seed["dst"]}
            group  = [seed]

            for j in range(i + 1, n):
                e = outgoing[j]

                if e["step"] - t0 > cfg.DELTA:
                    break

                if e["dst"] in seen:
                    continue

                if not _ratio_ok(a0, e["amount"], cfg.RHO_MIN, cfg.RHO_MAX):
                    continue

                seen.add(e["dst"])
                group.append(e)

            if len(group) >= cfg.R_MIN_FANOUT:
                nodes = [x] + [e["dst"] for e in group]
                results.append(_make_instance("fanout", group, nodes))

    return results


# ── Cycle-3 ───────────────────────────────────────────────────────────────

def find_cycle3(
    out_index: dict,
    cfg: MotifConfig,
) -> list[dict]:
    """
    Cycle-3: tiền đi vòng qua 3 node rồi quay lại điểm đầu.

        u → v → w → u

    Điều kiện:
        - Đúng thứ tự thời gian: step1 < step2 < step3
        - Lag từng bước <= DELTA
        - Amount ratio từng bước trong [RHO_MIN, RHO_MAX]
        - u, v, w phải là 3 node khác nhau

    Returns:
        list[dict]
    """
    results = []

    for u, edges_uv in out_index.items():
        for e1 in edges_uv:
            v  = e1["dst"]
            t1 = e1["step"]
            a1 = e1["amount"]

            if v == u:
                continue  # self-loop không phải cycle

            edges_vw = out_index.get(v)
            if not edges_vw:
                continue

            for e2 in edges_vw:
                # Pruning: thứ tự thời gian + lag
                if not _lag_ok(t1, e2["step"], cfg.DELTA):
                    if e2["step"] <= t1:
                        continue  # chưa đến, tiếp tục
                    break         # đã vượt DELTA, dừng

                w  = e2["dst"]
                a2 = e2["amount"]

                if w == u or w == v:
                    continue

                # Pruning: amount ratio bước 1→2
                if not _ratio_ok(a1, a2, cfg.RHO_MIN, cfg.RHO_MAX):
                    continue

                edges_wu = out_index.get(w)
                if not edges_wu:
                    continue

                for e3 in edges_wu:
                    if not _lag_ok(e2["step"], e3["step"], cfg.DELTA):
                        if e3["step"] <= e2["step"]:
                            continue
                        break

                    # Phải quay về u
                    if e3["dst"] != u:
                        continue

                    a3 = e3["amount"]

                    # Pruning: amount ratio bước 2→3
                    if not _ratio_ok(a2, a3, cfg.RHO_MIN, cfg.RHO_MAX):
                        continue

                    results.append(_make_instance(
                        "cycle3",
                        [e1, e2, e3],
                        [u, v, w, u],
                    ))

    return results


# ── Relay-4 (chain) ───────────────────────────────────────────────────────

def find_relay4(
    out_index: dict,
    cfg: MotifConfig,
) -> list[dict]:
    """
    Relay-4 (chain): tiền đi qua 4 node liên tiếp.

        u → v → w → x

    Điều kiện:
        - Đúng thứ tự thời gian, lag từng bước <= DELTA
        - Amount ratio từng bước trong [RHO_MIN, RHO_MAX]
        - u, v, w, x là 4 node khác nhau

    Returns:
        list[dict]
    """
    results = []

    for u, edges_uv in out_index.items():
        for e1 in edges_uv:
            v  = e1["dst"]
            t1 = e1["step"]
            a1 = e1["amount"]

            if v == u:
                continue

            edges_vw = out_index.get(v)
            if not edges_vw:
                continue

            for e2 in edges_vw:
                if not _lag_ok(t1, e2["step"], cfg.DELTA):
                    if e2["step"] <= t1:
                        continue
                    break

                w  = e2["dst"]
                a2 = e2["amount"]

                if w in (u, v):
                    continue

                if not _ratio_ok(a1, a2, cfg.RHO_MIN, cfg.RHO_MAX):
                    continue

                edges_wx = out_index.get(w)
                if not edges_wx:
                    continue

                for e3 in edges_wx:
                    if not _lag_ok(e2["step"], e3["step"], cfg.DELTA):
                        if e3["step"] <= e2["step"]:
                            continue
                        break

                    x  = e3["dst"]
                    a3 = e3["amount"]

                    if x in (u, v, w):
                        continue

                    if not _ratio_ok(a2, a3, cfg.RHO_MIN, cfg.RHO_MAX):
                        continue

                    results.append(_make_instance(
                        "relay4",
                        [e1, e2, e3],
                        [u, v, w, x],
                    ))

    return results


# ── Split-merge ───────────────────────────────────────────────────────────

def find_split_merge(
    out_index: dict,
    in_index: dict,
    cfg: MotifConfig,
) -> list[dict]:
    """
    Split-merge: một nguồn tách ra 2 nhánh, cả 2 nhánh nhập lại cùng một đích.

        u → v1 → z
        u → v2 → z

    Điều kiện:
        - u gửi đến v1 và v2 trong cùng DELTA steps
        - v1 và v2 đều gửi đến z trong DELTA steps sau đó
        - u, v1, v2, z là 4 node khác nhau
        - Amount ratio trong [RHO_MIN, RHO_MAX] tại mỗi bước

    Hai-phase:
        Phase 1: tìm split (u → v1, u → v2)
        Phase 2: tìm merge (v1 → z, v2 → z)

    Returns:
        list[dict] — mỗi instance có 4 edges: e_uv1, e_uv2, e_v1z, e_v2z
    """
    results = []

    for u, outgoing_u in out_index.items():
        n = len(outgoing_u)

        # Phase 1: tìm tất cả cặp split (v1, v2) từ u
        for i in range(n):
            e_uv1 = outgoing_u[i]
            v1    = e_uv1["dst"]
            t0    = e_uv1["step"]
            a_uv1 = e_uv1["amount"]

            for j in range(i + 1, n):
                e_uv2 = outgoing_u[j]

                # Pruning: cửa sổ split
                if e_uv2["step"] - t0 > cfg.DELTA:
                    break

                v2 = e_uv2["dst"]
                if v2 == v1 or v2 == u:
                    continue

                # Pruning: amount ratio uv1 vs uv2
                if not _ratio_ok(a_uv1, e_uv2["amount"], cfg.RHO_MIN, cfg.RHO_MAX):
                    continue

                # Phase 2: tìm z nhận từ cả v1 và v2
                edges_v1 = out_index.get(v1, [])
                edges_v2 = out_index.get(v2, [])

                # Lấy tập đích của v1 trong [t0, t0 + 2*DELTA]
                v1_targets: dict = {}  # dst → edge dict
                for e in edges_v1:
                    lag = e["step"] - t0
                    if lag <= 0:
                        continue
                    if lag > 2 * cfg.DELTA:
                        break
                    # Pruning: amount ratio uv1 → v1z
                    if not _ratio_ok(a_uv1, e["amount"], cfg.RHO_MIN, cfg.RHO_MAX):
                        continue
                    z = e["dst"]
                    if z not in (u, v1, v2):
                        v1_targets[z] = e

                if not v1_targets:
                    continue

                # Tìm v2 → z với z ∈ v1_targets
                for e in edges_v2:
                    lag = e["step"] - t0
                    if lag <= 0:
                        continue
                    if lag > 2 * cfg.DELTA:
                        break

                    z = e["dst"]
                    if z not in v1_targets:
                        continue
                    if z in (u, v1, v2):
                        continue

                    e_v1z = v1_targets[z]
                    e_v2z = e

                    # Pruning: amount ratio uv2 → v2z
                    if not _ratio_ok(e_uv2["amount"], e_v2z["amount"], cfg.RHO_MIN, cfg.RHO_MAX):
                        continue

                    # Giữ thứ tự thời gian trong instance
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


__all__ = [
    "find_fanin",
    "find_fanout",
    "find_cycle3",
    "find_relay4",
    "find_split_merge",
]