# src/community/weighting.py
# Bước 4.3: Tính trọng số cho 2nd order graph.
#
# W_final = W_co-occurrence × Money_Factor × Time_Factor
#
#   W_co       = max(P_source, P_target)
#   Money_Factor = exp(-α · |log(amount_src / amount_dst)|)
#     → phạt nặng khi tiền vào/ra nút trung gian lệch nhau
#   Time_Factor  = exp(-β · avg_time_gap / Δw)
#     → phạt khi khoảng cách thời gian xa
#
# Sau đó: sparsification (base threshold + top-K neighbors)
# Cuối: áp dụng weight ngược lại vào temporal edges gốc.

import gc

from .config import CommunityConfig


def _cudf():
    import cudf  # noqa: PLC0415
    return cudf


def _cp():
    import cupy as cp  # noqa: PLC0415
    return cp


def compute_weights(
    second_order_edges,
    temporal_edges,
    cfg: CommunityConfig,
    delta_w: int,
):
    """
    Bước 4.3 — Tính trọng số nâng cấp cho temporal graph.

    Args:
        second_order_edges: cuDF DataFrame output của create_second_order_graph()
            columns: [src_2nd, dst_2nd, count, total_amount_src,
                      total_amount_dst, avg_time_gap, n_alert, _max_node_id]
        temporal_edges: cuDF DataFrame output của create_temporal_graph()
        cfg:     CommunityConfig
        delta_w: DELTA_W từ graph Config (cần cho Time_Factor normalization)

    Returns:
        weighted_temporal_edges (cudf.DataFrame):
            temporal_edges filtered + joined với weight column
    """
    cudf = _cudf()
    cp   = _cp()

    print("=" * 60)
    print("  BƯỚC 4.3: Tính trọng số")
    print(f"  α={cfg.ALPHA}, β={cfg.BETA}, threshold={cfg.WEIGHT_FILTER_THRESH}")
    print(f"  Top-K neighbors={cfg.TOP_K_NEIGHBORS}")
    print("=" * 60)

    if len(second_order_edges) == 0:
        print("  ⚠️  Không có cạnh bậc 2. Bỏ qua.")
        return temporal_edges

    # ── Phần 1: Co-occurrence weight ──────────────────────────────────────
    src_total = second_order_edges.groupby("src_2nd", as_index=False).agg(
        total_out=("count", "sum")
    )
    dst_total = second_order_edges.groupby("dst_2nd", as_index=False).agg(
        total_in=("count", "sum")
    )

    w = second_order_edges.merge(src_total, on="src_2nd", how="left")
    w = w.merge(dst_total, on="dst_2nd", how="left")

    w["p_source"] = (
        w["count"].astype("float32") / (w["total_out"].astype("float32") + 1e-9)
    )
    w["p_target"] = (
        w["count"].astype("float32") / (w["total_in"].astype("float32") + 1e-9)
    )

    # W_co = max(p_source, p_target)
    mask_higher = w["p_target"] > w["p_source"]
    w["w_co"] = (
        w["p_source"] * (~mask_higher).astype("float32")
        + w["p_target"] * mask_higher.astype("float32")
    )

    # ── Phần 2: Monetary Continuity Factor ────────────────────────────────
    # exp(-α · |log(amount_src / amount_dst)|)
    log_src = cudf.Series(
        cp.log(cp.asarray(w["total_amount_src"].astype("float32").values) + 1e-9)
    )
    log_dst = cudf.Series(
        cp.log(cp.asarray(w["total_amount_dst"].astype("float32").values) + 1e-9)
    )
    log_ratio = (log_src - log_dst).abs()
    w["money_factor"] = cudf.Series(
        cp.exp(-cfg.ALPHA * cp.asarray(log_ratio.values))
    )

    # ── Phần 3: Temporal Decay Factor ─────────────────────────────────────
    # exp(-β · avg_time_gap / Δw)
    w["time_factor"] = cudf.Series(
        cp.exp(
            -cfg.BETA
            * cp.asarray(w["avg_time_gap"].astype("float32").values)
            / delta_w
        )
    )

    # ── Phần 4: W_final = W_co × money × time ─────────────────────────────
    w["weight"] = (w["w_co"] * w["money_factor"] * w["time_factor"]).astype("float32")

    n_before = len(w)

    # ── Phần 5: Sparsification — base threshold ───────────────────────────
    w = w[w["weight"] >= cfg.WEIGHT_FILTER_THRESH].reset_index(drop=True)

    # Top-K neighbors per src_2nd
    if cfg.TOP_K_NEIGHBORS > 0 and len(w) > 0:
        w = w.sort_values(["src_2nd", "weight"], ascending=[True, False])
        w["_rank"] = w.groupby("src_2nd").cumcount() + 1
        w = w[w["_rank"] <= cfg.TOP_K_NEIGHBORS].reset_index(drop=True)
        w = w.drop(columns=["_rank"])

    n_after = len(w)
    print(f"  Cạnh bậc 2: {n_before:,} → {n_after:,} (loại {n_before - n_after:,})")

    # ── Phần 6: Áp weight ngược lại vào temporal edges ────────────────────
    max_node_id = int(w["_max_node_id"].iloc[0]) if "_max_node_id" in w.columns else 1

    # Tính src_2nd/dst_2nd cho temporal_edges nếu chưa có
    if "src_2nd" not in temporal_edges.columns:
        temporal_edges = temporal_edges.copy()
        temporal_edges["src_2nd"] = (
            temporal_edges["src_1"].astype("int64") * max_node_id
            + temporal_edges["dst_1"].astype("int64")
        )
        temporal_edges["dst_2nd"] = (
            temporal_edges["src_2"].astype("int64") * max_node_id
            + temporal_edges["dst_2"].astype("int64")
        )

    weight_map = w[["src_2nd", "dst_2nd", "weight"]].copy()
    weighted = temporal_edges.merge(weight_map, on=["src_2nd", "dst_2nd"], how="inner")

    print(f"  Temporal edges: {len(temporal_edges):,} → {len(weighted):,} sau lọc weight")
    print(f"  ✅ Trọng số đã áp dụng.\n")

    del w, weight_map
    gc.collect()
    return weighted


__all__ = ["compute_weights"]