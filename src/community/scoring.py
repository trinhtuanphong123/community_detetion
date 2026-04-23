# src/community/scoring.py
# Bước 4.5: Feature extraction và soft suspicion scoring.
#
# extract_community_features: tính các đặc trưng cho mỗi community
#   trong một snapshot (directed features + size-normalized).
#
# score_communities: tính suspicion_score tổng hợp bằng weighted sum:
#   S = W_C2·f(C2) + W_C3·f(C3) + W_VELOCITY·velocity
#     + W_ALERT·alert_ratio + W_STRUCTURE·structural_bonus
#   Sau đó: size-penalty để tránh ưu thế community lớn.

import gc

from tqdm.auto import tqdm

from .config import CommunityConfig


def _cudf():
    import cudf  # noqa: PLC0415
    return cudf


def _cp():
    import cupy as cp  # noqa: PLC0415
    return cp


def extract_community_features(snapshots: list, cfg: CommunityConfig):
    """
    Trích xuất features cho tất cả community trong tất cả snapshots.

    Mỗi snapshot là dict với keys:
        window_id, steps, step_start, step_end,
        edge_feat, node_feat, partition_df, n_edges, n_nodes

    Features trích xuất:
        total_volume, n_internal_edges, alert_ratio,
        flow_ratio, internal_recirc, sink_concentration,
        source_concentration, max_single_flow, size,
        n_sources, n_sinks, n_layering, avg_layering_score,
        vol_density, edge_density, max_flow_norm

    Returns:
        cudf.DataFrame — tất cả community records qua tất cả windows
    """
    cudf = _cudf()
    records = []

    for snap in tqdm(snapshots, desc="Community feature extraction"):
        edge_df  = snap["edge_feat"]
        part_df  = snap["partition_df"]
        node_df  = snap["node_feat"]

        if len(part_df) == 0:
            continue

        # Ánh xạ cạnh vào community
        part_src = part_df[["node", "global_cid"]].rename(
            columns={"node": "src", "global_cid": "src_cid"}
        )
        part_dst = part_df[["node", "global_cid"]].rename(
            columns={"node": "dst", "global_cid": "dst_cid"}
        )

        edges_mapped = edge_df.merge(part_src, on="src", how="inner")
        edges_mapped = edges_mapped.merge(part_dst, on="dst", how="inner")

        # Chỉ giữ cạnh nội bộ (src và dst cùng community)
        internal = edges_mapped[
            edges_mapped["src_cid"] == edges_mapped["dst_cid"]
        ].rename(columns={"src_cid": "global_cid"}).drop(columns=["dst_cid"])

        if len(internal) == 0:
            continue

        internal["is_alert_edge"] = (internal["n_alert"] > 0).astype("float32")

        # ── Basic features ────────────────────────────────────────────────
        comm_basic = internal.groupby("global_cid", as_index=False).agg(
            total_volume      =("total_amount", "sum"),
            n_internal_edges  =("total_amount", "count"),
            alert_ratio       =("is_alert_edge", "mean"),
        )

        # ── Flow ratio (in / out) ─────────────────────────────────────────
        node_with_cid = part_df[["node", "global_cid"]].merge(
            node_df[[
                "node", "total_volume_out", "total_volume_in",
                "role", "flow_consistency", "layering_score",
            ]],
            on="node", how="left"
        ).fillna(0)

        comm_flow = node_with_cid.groupby("global_cid", as_index=False).agg(
            comm_total_out=("total_volume_out", "sum"),
            comm_total_in =("total_volume_in",  "sum"),
        )
        comm_flow["flow_ratio"] = (
            comm_flow["comm_total_in"] / (comm_flow["comm_total_out"] + 1e-9)
        ).astype("float32")

        comm_basic = comm_basic.merge(
            comm_flow[["global_cid", "flow_ratio", "comm_total_out", "comm_total_in"]],
            on="global_cid", how="left"
        )

        # ── Internal recirculation ────────────────────────────────────────
        # Tỷ lệ volume có cả chiều A→B và B→A trong community
        reverse = internal[["src", "dst", "global_cid", "total_amount"]].rename(
            columns={"src": "dst_r", "dst": "src_r", "total_amount": "rev_amt"}
        )
        recirc = internal[["src", "dst", "global_cid", "total_amount"]].merge(
            reverse,
            left_on  =["src", "dst", "global_cid"],
            right_on =["src_r", "dst_r", "global_cid"],
            how="inner",
        )
        if len(recirc) > 0:
            recirc_vol = recirc.groupby("global_cid", as_index=False).agg(
                recirc_volume=("total_amount", "sum")
            )
        else:
            recirc_vol = cudf.DataFrame({
                "global_cid":   cudf.Series(dtype="int32"),
                "recirc_volume": cudf.Series(dtype="float32"),
            })

        comm_basic = comm_basic.merge(recirc_vol, on="global_cid", how="left")
        comm_basic["recirc_volume"] = comm_basic["recirc_volume"].fillna(0)
        comm_basic["internal_recirc"] = (
            comm_basic["recirc_volume"] / (comm_basic["total_volume"] + 1e-9)
        ).astype("float32")
        comm_basic = comm_basic.drop(columns=["recirc_volume"])

        # ── Sink concentration ────────────────────────────────────────────
        node_in_vol = internal.groupby(["global_cid", "dst"], as_index=False).agg(
            in_vol=("total_amount", "sum")
        )
        max_in    = node_in_vol.groupby("global_cid", as_index=False).agg(max_sink_in=("in_vol", "max"))
        total_in  = node_in_vol.groupby("global_cid", as_index=False).agg(total_in_vol=("in_vol", "sum"))
        sink_conc = max_in.merge(total_in, on="global_cid")
        sink_conc["sink_concentration"] = (
            sink_conc["max_sink_in"] / (sink_conc["total_in_vol"] + 1e-9)
        ).astype("float32")
        comm_basic = comm_basic.merge(
            sink_conc[["global_cid", "sink_concentration"]], on="global_cid", how="left"
        )

        # ── Source concentration ──────────────────────────────────────────
        node_out_vol = internal.groupby(["global_cid", "src"], as_index=False).agg(
            out_vol=("total_amount", "sum")
        )
        max_out      = node_out_vol.groupby("global_cid", as_index=False).agg(max_source_out=("out_vol", "max"))
        total_out    = node_out_vol.groupby("global_cid", as_index=False).agg(total_out_vol=("out_vol", "sum"))
        source_conc  = max_out.merge(total_out, on="global_cid")
        source_conc["source_concentration"] = (
            source_conc["max_source_out"] / (source_conc["total_out_vol"] + 1e-9)
        ).astype("float32")
        comm_basic = comm_basic.merge(
            source_conc[["global_cid", "source_concentration"]], on="global_cid", how="left"
        )

        # ── Max single flow ───────────────────────────────────────────────
        max_flow = internal.groupby("global_cid", as_index=False).agg(
            max_single_flow=("total_amount", "max")
        )
        comm_basic = comm_basic.merge(max_flow, on="global_cid", how="left")

        # ── Size + filter ─────────────────────────────────────────────────
        sizes = part_df.groupby("global_cid", as_index=False).agg(size=("node", "count"))
        comm_basic = comm_basic.merge(sizes, on="global_cid")
        comm_basic = comm_basic[comm_basic["size"] >= cfg.MIN_COMM_SIZE]

        # ── Role counts ───────────────────────────────────────────────────
        node_with_cid["is_source"]   = (node_with_cid["role"] == 1).astype("int32")
        node_with_cid["is_sink"]     = (node_with_cid["role"] == 2).astype("int32")
        node_with_cid["is_layering"] = (node_with_cid["role"] == 3).astype("int32")

        role_counts = node_with_cid.groupby("global_cid", as_index=False).agg(
            n_sources          =("is_source",       "sum"),
            n_sinks            =("is_sink",          "sum"),
            n_layering         =("is_layering",      "sum"),
            avg_layering_score =("layering_score",   "mean"),
        )
        comm_basic = comm_basic.merge(role_counts, on="global_cid", how="left")

        # ── Size-normalized features ──────────────────────────────────────
        comm_basic["vol_density"]   = (comm_basic["total_volume"]   / (comm_basic["size"] + 1e-9)).astype("float32")
        comm_basic["edge_density"]  = (
            comm_basic["n_internal_edges"].astype("float32")
            / (comm_basic["size"] * (comm_basic["size"] - 1) + 1e-9)
        ).astype("float32")
        comm_basic["max_flow_norm"] = (comm_basic["max_single_flow"] / (comm_basic["size"] + 1e-9)).astype("float32")

        # Metadata
        comm_basic["window_id"]  = snap["window_id"]
        comm_basic["step_start"] = snap["step_start"]
        comm_basic["step_end"]   = snap["step_end"]

        # Fill NaN
        fill_cols = [
            "flow_ratio", "internal_recirc", "sink_concentration",
            "source_concentration", "max_single_flow", "n_sources", "n_sinks",
            "n_layering", "avg_layering_score", "vol_density",
            "edge_density", "max_flow_norm", "comm_total_out", "comm_total_in",
        ]
        for col in fill_cols:
            if col in comm_basic.columns:
                comm_basic[col] = comm_basic[col].fillna(0)

        records.append(comm_basic)
        gc.collect()

    if records:
        return _cudf().concat(records, ignore_index=True)
    return _cudf().DataFrame()


def score_communities(comm_df, cfg: CommunityConfig, n_alert: int, n_rows: int):
    """
    Bước 4.5 — Soft suspicion scoring.

    S = W_C2·c2_score + W_C3·c3_score + W_VELOCITY·velocity_score
      + W_ALERT·alert_ratio + W_STRUCTURE·structural_score
    Sau đó: size-penalty = γ·log(|C|)

    Args:
        comm_df:  cuDF DataFrame từ extract_community_features()
        cfg:      CommunityConfig
        n_alert:  số alert rows trong dataset gốc (để tính AER)
        n_rows:   tổng số rows trong dataset gốc

    Returns:
        comm_df với thêm các cột:
            velocity, velocity_score, c2_score, c3_score,
            structural_score, suspicion_score, is_suspicious,
            c1_flag, c2_flag, c3_flag
    """
    cudf = _cudf()
    cp   = _cp()

    if len(comm_df) == 0:
        print("  ⚠️  Không có community để score.")
        return comm_df

    print("=" * 60)
    print("  BƯỚC 4.5: Soft Suspicion Scoring")
    print(f"  Weights: C2={cfg.W_C2}, C3={cfg.W_C3}, Vel={cfg.W_VELOCITY}, "
          f"Alert={cfg.W_ALERT}, Struct={cfg.W_STRUCTURE}")
    print(f"  Threshold: {cfg.GLOBAL_SUSP_THRESHOLD}")
    print("=" * 60)

    # ── Velocity (temporal burstiness) ────────────────────────────────────
    duration = (comm_df["step_end"] - comm_df["step_start"] + 1).astype("float32")
    comm_df["velocity"] = comm_df["total_volume"] / (duration + 1e-9)
    mean_vel = float(comm_df["velocity"].mean())
    comm_df["velocity_score"] = (
        comm_df["velocity"] / (mean_vel + 1e-9)
    ).clip(upper=1.0).astype("float32")

    # ── C2 score (sink concentration trong optimal range) ─────────────────
    mid_c2 = (cfg.C2_ALLOC_MIN + cfg.C2_ALLOC_MAX) / 2
    comm_df["c2_score"] = (
        1 - (comm_df["sink_concentration"] - mid_c2).abs()
    ).clip(lower=0.0).astype("float32")

    # ── C3 score (max single flow normalized) ─────────────────────────────
    comm_df["c3_score"] = (
        comm_df["max_single_flow"] / (cfg.C3_MIN_FLOW + 1e-9)
    ).clip(upper=1.0).astype("float32")

    # ── Structural score (C1: has source + sink; layering bonus) ──────────
    has_roles      = ((comm_df["n_sources"] >= 1) & (comm_df["n_sinks"] >= 1)).astype("float32")
    layering_bonus = (comm_df["n_layering"] >= 1).astype("float32") * 0.5
    comm_df["structural_score"] = (has_roles * 0.5 + layering_bonus).clip(upper=1.0).astype("float32")

    # ── Suspicion score tổng hợp ──────────────────────────────────────────
    comm_df["suspicion_score"] = (
        cfg.W_C2        * comm_df["c2_score"]
        + cfg.W_C3      * comm_df["c3_score"]
        + cfg.W_VELOCITY * comm_df["velocity_score"]
        + cfg.W_ALERT   * comm_df["alert_ratio"]
        + cfg.W_STRUCTURE * comm_df["structural_score"]
    ).astype("float32")

    # Size-penalty: giảm ưu thế community lớn
    GAMMA = 0.05
    size_penalty = GAMMA * cudf.Series(
        cp.log(cp.asarray(comm_df["size"].astype("float32").values) + 1)
    )
    comm_df["suspicion_score"] = (
        comm_df["suspicion_score"] - size_penalty
    ).clip(lower=0.0)

    # Binary flag
    comm_df["is_suspicious"] = (
        comm_df["suspicion_score"] >= cfg.GLOBAL_SUSP_THRESHOLD
    ).astype("int8")

    # Component flags cho explainability
    comm_df["c1_flag"] = has_roles.astype("int8")
    comm_df["c2_flag"] = (comm_df["c2_score"] >= 0.3).astype("int8")
    comm_df["c3_flag"] = (comm_df["c3_score"] >= 0.5).astype("int8")

    # ── Summary ───────────────────────────────────────────────────────────
    total   = len(comm_df)
    n_susp  = int(comm_df["is_suspicious"].sum())
    global_alert_rate = n_alert / n_rows if n_rows > 0 else 0
    susp_alert_rate   = float(comm_df[comm_df["is_suspicious"] == 1]["alert_ratio"].mean()) if n_susp > 0 else 0
    AER = susp_alert_rate / (global_alert_rate + 1e-9)

    print(f"\n  Tổng community: {total:,}")
    print(f"  Suspicious (score ≥ {cfg.GLOBAL_SUSP_THRESHOLD}): {n_susp:,} ({n_susp / total * 100:.1f}%)")
    print(f"  Mean score: {float(comm_df['suspicion_score'].mean()):.4f}")
    print(f"  AER: {AER:.2f}×  (global={global_alert_rate * 100:.2f}%, susp={susp_alert_rate * 100:.2f}%)")

    # Unknown unknowns
    if n_susp > 0:
        unknown = comm_df[
            (comm_df["is_suspicious"] == 1)
            & (comm_df["alert_ratio"] < cfg.ALERT_THRESH)
        ]
        if len(unknown) > 0:
            print(f"  🔍 Unknown unknowns (suspicious nhưng alert_ratio < {cfg.ALERT_THRESH}): {len(unknown)}")

    print(f"  ✅ Scoring hoàn tất.\n")
    gc.collect()
    return comm_df


__all__ = ["extract_community_features", "score_communities"]