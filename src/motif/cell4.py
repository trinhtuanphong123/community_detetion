# ============================================================================
# 4.1  TẠO ĐỒ THỊ THỜI GIAN (Temporal Graph Creation)
# ============================================================================

def build_temporal_graph_for_day(df_day: cudf.DataFrame, delta_w: int) -> cudf.DataFrame:
    """
    Tạo cạnh thời gian cho giao dịch trong một batch.

    Cạnh (tx_i → tx_j) được hình thành khi:
      1. dst_i == src_j (tài khoản nhận tx_i = tài khoản gửi tx_j)
      2. step_j > step_i (tuần tự thời gian)
      3. step_j - step_i <= delta_w (trong khoảng Δw)

    Returns:
        temporal_edges (cudf.DataFrame): [src_1, dst_1, step_1, amount_1, alert_1,
                                           src_2, dst_2, step_2, amount_2, alert_2]
    """
    if len(df_day) == 0:
        return cudf.DataFrame()

    tx1 = df_day[["src", "dst", "step", "amount", "is_laundering"]].rename(columns={
        "src": "src_1", "dst": "dst_1", "step": "step_1",
        "amount": "amount_1", "is_laundering": "alert_1"
    })

    tx2 = df_day[["src", "dst", "step", "amount", "is_laundering"]].rename(columns={
        "src": "src_2", "dst": "dst_2", "step": "step_2",
        "amount": "amount_2", "is_laundering": "alert_2"
    })

    # Inner join: dst_1 == src_2
    joined = tx1.merge(tx2, left_on="dst_1", right_on="src_2", how="inner")

    # Lọc: tuần tự + khoảng Δw
    mask = (joined["step_2"] > joined["step_1"]) & \
           ((joined["step_2"] - joined["step_1"]) <= delta_w)
    temporal_edges = joined[mask].reset_index(drop=True)

    return temporal_edges


def create_temporal_graph(df: cudf.DataFrame, cfg: Config) -> cudf.DataFrame:
    """
    4.1 — Tạo đồ thị thời gian T bằng cách xử lý từng batch.
    Xử lý lặp để giảm thiểu bộ nhớ.
    """
    print("=" * 60)
    print("  BƯỚC 4.1: Tạo Đồ thị Thời gian (Temporal Graph)")
    print(f"  Δw = {cfg.DELTA_W} steps")
    print("=" * 60)

    all_steps = df["step"].unique().to_pandas().sort_values().tolist()

    temporal_edge_batches = []
    batch_size = max(cfg.DELTA_W * 2, 10)

    for i in tqdm(range(0, len(all_steps), batch_size), desc="Temporal graph (per-batch)"):
        batch_steps = all_steps[max(0, i - cfg.DELTA_W) : i + batch_size + cfg.DELTA_W]

        mask = df["step"].isin(batch_steps)
        df_batch = df[mask]

        if len(df_batch) == 0:
            continue

        edges = build_temporal_graph_for_day(df_batch, cfg.DELTA_W)

        if len(edges) > 0:
            # drop_duplicates ngay trong batch để tiết kiệm memory (từ update.md mục 3)
            edges = edges.drop_duplicates().reset_index(drop=True)
            temporal_edge_batches.append(edges)

        del df_batch, edges
        gc.collect()

    if temporal_edge_batches:
        all_temporal_edges = cudf.concat(temporal_edge_batches, ignore_index=True)
        all_temporal_edges = all_temporal_edges.drop_duplicates().reset_index(drop=True)
    else:
        all_temporal_edges = cudf.DataFrame()

    n_edges = len(all_temporal_edges)
    print(f"\n  ✅ Đồ thị thời gian T: {n_edges:,} cạnh")

    gc.collect()
    return all_temporal_edges



# ============================================================================
# 4.2  TẠO ĐỒ THỊ BẬC 2 (2nd Order Graph — Line Graph)
# ============================================================================

def create_second_order_graph(temporal_edges: cudf.DataFrame) -> cudf.DataFrame:
    """
    4.2 — Tạo đồ thị bậc 2 S từ đồ thị thời gian T.

    Mỗi nút trong S là một cạnh của T: x = (u → v)
    Cạnh trong S tồn tại nếu có chuỗi: (u → v) → (v → z)

    Returns:
        second_order_edges (cudf.DataFrame): [src_2nd, dst_2nd, count,
            total_amount_src, total_amount_dst, avg_time_gap, n_alert]
    """
    print("\n" + "=" * 60)
    print("  BƯỚC 4.2: Tạo Đồ thị Bậc 2 (2nd Order / Line Graph)")
    print("=" * 60)

    if len(temporal_edges) == 0:
        print("  ⚠️ Không có cạnh thời gian. Bỏ qua.")
        return cudf.DataFrame()

    # Encode nút bậc 2: node_2nd = src * MAX_ID + dst
    max_node_id = max(
        int(temporal_edges[["src_1", "dst_1", "src_2", "dst_2"]].max().max()),
        1
    ) + 1

    temporal_edges["src_2nd"] = temporal_edges["src_1"].astype("int64") * max_node_id + \
                                 temporal_edges["dst_1"].astype("int64")
    temporal_edges["dst_2nd"] = temporal_edges["src_2"].astype("int64") * max_node_id + \
                                 temporal_edges["dst_2"].astype("int64")

    # Tính time gap cho mỗi cạnh temporal (cần cho weight calculation 4.3)
    temporal_edges["time_gap"] = (temporal_edges["step_2"] - temporal_edges["step_1"]).astype("float32")

    # Tổng hợp cạnh bậc 2: đếm, tổng amount, trung bình time gap
    second_order_edges = temporal_edges.groupby(
        ["src_2nd", "dst_2nd"], as_index=False
    ).agg(
        count=("amount_1", "count"),
        total_amount_src=("amount_1", "sum"),
        total_amount_dst=("amount_2", "sum"),
        avg_time_gap=("time_gap", "mean"),
        n_alert=("alert_1", "sum"),
    )

    second_order_edges["_max_node_id"] = max_node_id

    n_nodes_2nd = cudf.concat([
        second_order_edges["src_2nd"], second_order_edges["dst_2nd"]
    ]).nunique()

    print(f"  Nút bậc 2: {n_nodes_2nd:,}")
    print(f"  Cạnh bậc 2: {len(second_order_edges):,}")
    print(f"  ✅ Đồ thị bậc 2 hoàn tất.")

    gc.collect()
    return second_order_edges


# ============================================================================
# 4.3  TÍNH TOÁN TRỌNG SỐ (Monetary Continuity + Co-occurrence + Time Decay)
# ============================================================================

def compute_refined_weights(
    second_order_edges: cudf.DataFrame,
    temporal_edges: cudf.DataFrame,
    cfg: Config,
) -> cudf.DataFrame:
    """
    4.3 — Tính trọng số nâng cấp theo Advisor (update.md):

    W_final = W_co-occurrence × Money_Factor × Time_Factor

    Trong đó:
      W_co = max(P_source, P_target)
      Money_Factor = exp(-α · |log(amount_src / amount_dst)|)
        → Phạt nặng nếu tiền vào/ra không khớp (dấu hiệu structuring)
      Time_Factor = exp(-β · avg_time_gap / Δw)
        → Phạt nếu khoảng nghỉ quá lâu (phá vỡ tính liên tục layering)

    Sau đó:
      Layer 1 Sparsification: top-K neighbors + base threshold
      Áp dụng trọng số ngược lại cho temporal graph T

    Returns:
        weighted_temporal_edges (cudf.DataFrame)
    """
    print("\n" + "=" * 60)
    print("  BƯỚC 4.3: Tính trọng số (Monetary Continuity + Co-occurrence)")
    print(f"  α={cfg.ALPHA}, β={cfg.BETA}, threshold={cfg.WEIGHT_FILTER_THRESH}")
    print(f"  Top-K neighbors={cfg.TOP_K_NEIGHBORS}")
    print("=" * 60)

    if len(second_order_edges) == 0:
        print("  ⚠️ Không có cạnh bậc 2. Bỏ qua.")
        return temporal_edges

    # ══════════════════════════════════════════════════════════
    # PHẦN 1: Co-occurrence Weight (giữ nguyên logic gốc)
    # ══════════════════════════════════════════════════════════

    # Đếm tổng outgoing/incoming
    src_total = second_order_edges.groupby("src_2nd", as_index=False).agg(
        total_out=("count", "sum")
    )
    dst_total = second_order_edges.groupby("dst_2nd", as_index=False).agg(
        total_in=("count", "sum")
    )

    weighted = second_order_edges.merge(src_total, on="src_2nd", how="left")
    weighted = weighted.merge(dst_total, on="dst_2nd", how="left")

    # P_source, P_target
    weighted["p_source"] = (
        weighted["count"].astype("float32") /
        (weighted["total_out"].astype("float32") + 1e-9)
    )
    weighted["p_target"] = (
        weighted["count"].astype("float32") /
        (weighted["total_in"].astype("float32") + 1e-9)
    )

    # W_co = max(P_source, P_target)
    mask_target_higher = weighted["p_target"] > weighted["p_source"]
    weighted["w_co"] = (
        weighted["p_source"] * (~mask_target_higher).astype("float32") +
        weighted["p_target"] * mask_target_higher.astype("float32")
    )

    # ══════════════════════════════════════════════════════════
    # PHẦN 2: Monetary Continuity Factor (MỚI — update.md)
    # ══════════════════════════════════════════════════════════
    # exp(-α · |log(amount_src / amount_dst)|)
    # Phạt nặng nếu tiền vào và ra khỏi nút trung gian v có chênh lệch lớn
    # (layering thường giữ nguyên ~volume qua các bước)
    log_src = cudf.Series(cp.log(cp.asarray(weighted["total_amount_src"].astype("float32").values) + 1e-9))
    log_dst = cudf.Series(cp.log(cp.asarray(weighted["total_amount_dst"].astype("float32").values) + 1e-9))
    log_ratio = (log_src - log_dst).abs()
    weighted["money_factor"] = cudf.Series(
        cp.exp(-cfg.ALPHA * cp.asarray(log_ratio.values))
    )

    # ══════════════════════════════════════════════════════════
    # PHẦN 3: Temporal Decay Factor (MỚI — update.md)
    # ══════════════════════════════════════════════════════════
    # exp(-β · avg_time_gap / Δw)
    # Phạt nếu khoảng cách thời gian trung bình quá xa
    weighted["time_factor"] = cudf.Series(
        cp.exp(-cfg.BETA * cp.asarray(weighted["avg_time_gap"].astype("float32").values) / cfg.DELTA_W)
    )

    # ══════════════════════════════════════════════════════════
    # PHẦN 4: Trọng số cuối cùng = W_co × Money × Time
    # ══════════════════════════════════════════════════════════
    weighted["weight"] = (
        weighted["w_co"] * weighted["money_factor"] * weighted["time_factor"]
    ).astype("float32")

    n_before = len(weighted)

    # ══════════════════════════════════════════════════════════
    # PHẦN 5: Layer 1 Sparsification (từ 08_community_oversegmentation.md)
    # ══════════════════════════════════════════════════════════
    # Base threshold
    weighted = weighted[weighted["weight"] >= cfg.WEIGHT_FILTER_THRESH].reset_index(drop=True)

    # Top-K neighbors: cho mỗi nút src_2nd, chỉ giữ K cạnh mạnh nhất
    if cfg.TOP_K_NEIGHBORS > 0 and len(weighted) > 0:
        # Rank trong mỗi nhóm src_2nd
        weighted = weighted.sort_values(["src_2nd", "weight"], ascending=[True, False])
        weighted["_rank"] = weighted.groupby("src_2nd").cumcount() + 1
        weighted = weighted[weighted["_rank"] <= cfg.TOP_K_NEIGHBORS].reset_index(drop=True)
        weighted = weighted.drop(columns=["_rank"])

    n_after = len(weighted)
    n_removed = n_before - n_after

    print(f"  Cạnh bậc 2 trước lọc: {n_before:,}")
    print(f"  Cạnh bậc 2 sau lọc (threshold + top-K): {n_after:,} (loại bỏ {n_removed:,})")

    # ══════════════════════════════════════════════════════════
    # PHẦN 6: Áp dụng trọng số ngược lại cho temporal graph T
    # ══════════════════════════════════════════════════════════
    max_node_id = int(weighted["_max_node_id"].iloc[0]) if "_max_node_id" in weighted.columns else 1

    if "src_2nd" not in temporal_edges.columns:
        temporal_edges["src_2nd"] = temporal_edges["src_1"].astype("int64") * max_node_id + \
                                     temporal_edges["dst_1"].astype("int64")
        temporal_edges["dst_2nd"] = temporal_edges["src_2"].astype("int64") * max_node_id + \
                                     temporal_edges["dst_2"].astype("int64")

    weight_map = weighted[["src_2nd", "dst_2nd", "weight"]].copy()
    temporal_weighted = temporal_edges.merge(
        weight_map, on=["src_2nd", "dst_2nd"], how="inner"
    )

    n_temporal_before = len(temporal_edges)
    n_temporal_after = len(temporal_weighted)

    print(f"\n  Cạnh thời gian trước lọc: {n_temporal_before:,}")
    print(f"  Cạnh thời gian sau lọc:   {n_temporal_after:,}")
    print(f"  ✅ Trọng số (co-occurrence × monetary × temporal) đã áp dụng.")

    gc.collect()
    return temporal_weighted


# ============================================================================
# 4.4  PHÁT HIỆN CỘNG ĐỒNG (Relay-preserving + Recursive Leiden)
# ============================================================================

def build_relay_preserving_edge_list(
    weighted_temporal_edges: cudf.DataFrame,
) -> cudf.DataFrame:
    """
    Tạo edge list cho Leiden BẢO TOÀN RELAY STRUCTURE.

    THAY ĐỔI QUAN TRỌNG (update.md):
    Code cũ: gom src_1 → dst_2 (nối tắt, mất nút trung gian)
    Code mới: giữ cả 2 cạnh thực tế:
      - Cạnh 1: src_1 → dst_1 (Nguồn → Trung gian)
      - Cạnh 2: src_2 → dst_2 (Trung gian → Đích)

    Lý do: Leiden cần nhìn thấy nút trung gian để xác định
    "trung tâm điều phối" dòng tiền (layering hub).

    Returns:
        final_graph (cudf.DataFrame): [src, dst, weighted_amount, total_amount,
                                        n_tx, n_alert, alert_ratio]
    """
    if len(weighted_temporal_edges) == 0:
        return cudf.DataFrame(columns=["src", "dst", "weighted_amount", "total_amount",
                                         "n_tx", "n_alert", "alert_ratio"])

    # Cạnh 1: Nguồn → Trung gian
    e1 = weighted_temporal_edges[["src_1", "dst_1", "weight", "amount_1", "alert_1"]].copy()
    e1.columns = ["src", "dst", "w", "amt", "alert"]

    # Cạnh 2: Trung gian → Đích
    e2 = weighted_temporal_edges[["src_2", "dst_2", "weight", "amount_2", "alert_2"]].copy()
    e2.columns = ["src", "dst", "w", "amt", "alert"]

    full_edges = cudf.concat([e1, e2], ignore_index=True)
    del e1, e2

    # Groupby để gộp cạnh trùng lặp
    final_graph = full_edges.groupby(["src", "dst"], as_index=False).agg(
        weighted_amount=("w", "max"),     # Giữ trọng số cao nhất (max flow signal)
        total_amount=("amt", "sum"),       # Tổng volume
        n_tx=("amt", "count"),
        n_alert=("alert", "sum"),
    )

    final_graph["weighted_amount"] = (
        final_graph["weighted_amount"].astype("float32") *
        final_graph["total_amount"].astype("float32")
    )
    final_graph["total_amount"] = final_graph["total_amount"].astype("float32")
    final_graph["n_tx"]         = final_graph["n_tx"].astype("int32")
    final_graph["n_alert"]      = final_graph["n_alert"].astype("int32")
    final_graph["alert_ratio"]  = (
        final_graph["n_alert"].astype("float32") / (final_graph["n_tx"].astype("float32") + 1e-9)
    )

    del full_edges
    gc.collect()
    return final_graph


def run_leiden_community(edge_df: cudf.DataFrame, resolution: float = 1.0) -> cudf.DataFrame:
    """
    Chạy Leiden community detection trên GPU (undirected — cuGraph constraint).

    Nâng cấp (update.md): thêm Reciprocity bonus
    → Nếu A→B và B→A đều tồn tại, edge weight × 2 (vòng lặp cực ngắn)
    → Leiden buộc phải giữ chúng cùng cluster

    Returns:
        partitions_df (cudf.DataFrame): ['node', 'partition']
    """
    if len(edge_df) == 0:
        return cudf.DataFrame(columns=["node", "partition"])

    edges = edge_df[["src", "dst", "weighted_amount"]].copy()

    # ── Reciprocity bonus (MỚI — update.md) ──
    # Tìm cặp (A,B) mà cả A→B và B→A tồn tại
    reverse = edges[["dst", "src", "weighted_amount"]].rename(
        columns={"dst": "src", "src": "dst", "weighted_amount": "rev_w"}
    )
    edges_with_rev = edges.merge(reverse, on=["src", "dst"], how="left")
    # Nếu có reciprocal edge: w * 2, nếu không: giữ nguyên
    has_reciprocal = edges_with_rev["rev_w"].notna()
    edges["weighted_amount"] = (
        edges["weighted_amount"] * (1 + has_reciprocal.astype("float32"))
    )
    del edges_with_rev, reverse

    # ── Canonical pair ──
    edges["u"] = edges[["src", "dst"]].min(axis=1)
    edges["v"] = edges[["src", "dst"]].max(axis=1)

    undirected = edges.groupby(["u", "v"], as_index=False).agg(
        weight=("weighted_amount", "sum")
    )
    undirected["weight"] = undirected["weight"].astype("float32")

    # ── Build undirected graph ──
    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(
        undirected, source="u", destination="v",
        edge_attr="weight", renumber=True,
    )

    # ── Leiden ──
    parts, modularity = cugraph.leiden(G, resolution=resolution)
    parts = parts.rename(columns={"vertex": "node"})

    print(f"    Leiden modularity: {modularity:.4f} (resolution={resolution:.2f})")

    del G, undirected, edges
    gc.collect()

    return parts[["node", "partition"]]


def run_recursive_leiden(
    edge_df: cudf.DataFrame,
    cfg: Config,
    depth: int = 0,
) -> cudf.DataFrame:
    """
    Recursive Leiden (update.md Layer 2 + 08_oversegmentation.md):
    1. Chạy Leiden
    2. Nếu có community > S_MAX nodes → chạy lại trên subgraph với resolution tăng
    3. Dừng khi tất cả <= S_MAX hoặc đạt max depth

    Returns:
        partitions_df (cudf.DataFrame): ['node', 'partition'] — globally unique partition IDs
    """
    resolution = cfg.RESOLUTION * (cfg.RESOLUTION_MULTIPLIER ** depth)

    # Chạy Leiden ban đầu
    parts = run_leiden_community(edge_df, resolution=resolution)

    if len(parts) == 0 or depth >= cfg.MAX_RECURSION_DEPTH:
        return parts

    # Kiểm tra oversized communities
    comm_sizes = parts.groupby("partition", as_index=False).agg(
        size=("node", "count")
    )
    oversized = comm_sizes[comm_sizes["size"] > cfg.S_MAX]["partition"]

    if len(oversized) == 0:
        return parts  # Không có community quá khổ

    print(f"    ⚠️ Depth {depth}: {len(oversized)} communities > {cfg.S_MAX} nodes → recursive split")

    # Tách partition offset cho sub-communities
    max_partition = int(parts["partition"].max()) + 1
    final_parts_list = []

    # Giữ nguyên các community đã đủ nhỏ
    normal_parts = parts[~parts["partition"].isin(oversized)]
    if len(normal_parts) > 0:
        final_parts_list.append(normal_parts)

    # Recursive split cho từng oversized community
    for cid in oversized.to_pandas().tolist():
        sub_nodes = parts[parts["partition"] == cid]["node"]
        # Lọc edges thuộc subgraph
        sub_edges = edge_df[
            edge_df["src"].isin(sub_nodes) & edge_df["dst"].isin(sub_nodes)
        ]

        if len(sub_edges) == 0:
            # Không có cạnh nội bộ → giữ nguyên partition
            final_parts_list.append(parts[parts["partition"] == cid])
            continue

        # Recursive call
        sub_parts = run_recursive_leiden(sub_edges, cfg, depth=depth + 1)

        if len(sub_parts) > 0:
            # Offset partition IDs để tránh collision
            sub_parts["partition"] = sub_parts["partition"] + max_partition
            max_partition = int(sub_parts["partition"].max()) + 1
            final_parts_list.append(sub_parts)

    if final_parts_list:
        result = cudf.concat(final_parts_list, ignore_index=True)
    else:
        result = parts

    return result


def compute_refined_node_roles(
    df_window: cudf.DataFrame,
    cfg: Config,
) -> cudf.DataFrame:
    """
    Tính node features với Layering role (MỚI — update.md).

    Role classification:
      0: Neutral (intermediary)
      1: Source (gửi >> nhận)
      2: Sink (nhận >> gửi)
      3: Layering (Vol_In ≈ Vol_Out + volume cao) — VIP target

    Layering Score = min(Vol_In, Vol_Out) / max(Vol_In, Vol_Out) × (1 - |net_flow_ratio|)

    Returns:
        node_features (cudf.DataFrame): [node, total_volume, ..., role, layering_score]
    """
    node_stats_src = df_window.groupby("src", as_index=False).agg(
        total_volume_out=("amount", "sum"),
        n_tx_out=("amount", "count"),
        n_alert_tx_out=("is_laundering", "sum")
    ).rename(columns={"src": "node"})

    node_stats_dst = df_window.groupby("dst", as_index=False).agg(
        total_volume_in=("amount", "sum"),
        n_tx_in=("amount", "count"),
        n_alert_tx_in=("is_laundering", "sum")
    ).rename(columns={"dst": "node"})

    node_features = node_stats_src.merge(node_stats_dst, on="node", how="outer").fillna(0)

    node_features["total_volume"] = node_features["total_volume_out"] + node_features["total_volume_in"]
    node_features["n_tx"]         = node_features["n_tx_out"] + node_features["n_tx_in"]
    node_features["n_alert_tx"]   = node_features["n_alert_tx_out"] + node_features["n_alert_tx_in"]
    node_features["alert_rate"]   = node_features["n_alert_tx"] / (node_features["n_tx"] + 1e-9)

    # Net flow ratio
    node_features["net_flow_ratio"] = (
        (node_features["total_volume_out"] - node_features["total_volume_in"])
        / (node_features["total_volume_out"] + node_features["total_volume_in"] + 1e-9)
    ).astype('float32')

    # ── Flow Consistency (MỚI — update.md) ──
    # = 2 * min(in, out) / (in + out + ε)
    # Phạm vi [0, 1]: 1 = volume vào/ra cân bằng hoàn hảo
    vol_min_col = node_features[["total_volume_in", "total_volume_out"]].min(axis=1)
    node_features["flow_consistency"] = (
        2 * vol_min_col /
        (node_features["total_volume_in"] + node_features["total_volume_out"] + 1e-9)
    ).astype("float32")

    # ── Layering Score (MỚI — update.md) ──
    # Layering Score = flow_consistency × (1 - |net_flow_ratio|)
    # Nút Layering: consistency cao + net_flow thấp + tổng volume lớn
    node_features["layering_score"] = (
        node_features["flow_consistency"] *
        (1 - node_features["net_flow_ratio"].abs())
    ).astype("float32")

    # ── Role Assignment ──
    # cuDF không hỗ trợ .loc assignment → dùng arithmetic vectorized
    is_source  = node_features["net_flow_ratio"] > cfg.ROLE_THRESHOLD
    is_sink    = node_features["net_flow_ratio"] < -cfg.ROLE_THRESHOLD
    is_layering = (
        (node_features["flow_consistency"] > cfg.LAYERING_CONSISTENCY) &
        (node_features["net_flow_ratio"].abs() < cfg.ROLE_THRESHOLD * 0.67)
    )

    # Priority: layering > source/sink > neutral
    # role = 3 nếu layering, else 1/2/0
    node_features["role"] = (
        is_layering.astype("int8") * 3 +                      # layering = 3
        (~is_layering & is_source).astype("int8") * 1 +        # source = 1
        (~is_layering & is_sink).astype("int8") * 2            # sink = 2
    )  # neutral = 0

    node_features = node_features[[
        "node", "total_volume", "total_volume_out", "total_volume_in",
        "n_tx", "n_alert_tx", "alert_rate", "net_flow_ratio",
        "flow_consistency", "layering_score", "role"
    ]]

    return node_features


def _vectorized_jaccard_tracking_multi(
    curr_partitions: cudf.DataFrame,   # ['node', 'partition']
    prev_partitions_buffer: list,      # list of cuDF ['node', 'global_cid'] — last N windows
    jaccard_thresh: float,
    global_cid_counter: int,
):
    """
    Cross-window community tracking với MULTI-WINDOW MEMORY (MỚI — update.md).

    Thay vì chỉ so sánh với window T-1, so sánh với buffer N windows gần nhất.
    Cho phép community "tắt đèn" 1-2 windows mà không mất ID.

    Returns:
        mapping (cudf.DataFrame): ['partition', 'global_cid']
        global_cid_counter (int): updated counter
    """
    # Merge tất cả previous windows vào 1 DataFrame
    if not prev_partitions_buffer:
        n_local = curr_partitions["partition"].nunique()
        mapping = curr_partitions[["partition"]].drop_duplicates()
        mapping["global_cid"] = cudf.Series(
            range(global_cid_counter, global_cid_counter + len(mapping)),
            dtype="int32",
        ).values
        return mapping, global_cid_counter + int(n_local)

    # Stack all previous windows (giữ global_cid nhất quán)
    all_prev = cudf.concat(prev_partitions_buffer, ignore_index=True)
    # De-duplicate: mỗi node chỉ giữ 1 global_cid (lấy từ window gần nhất)
    all_prev = all_prev.drop_duplicates(subset=["node"], keep="last")

    # Kích thước community
    curr_sizes = (
        curr_partitions.groupby("partition", as_index=False)
        .agg(curr_size=("node", "count"))
    )
    prev_sizes = (
        all_prev.groupby("global_cid", as_index=False)
        .agg(prev_size=("node", "count"))
    )

    # Overlap
    overlap = curr_partitions[["node", "partition"]].merge(
        all_prev[["node", "global_cid"]],
        on="node",
        how="inner",
    )

    if len(overlap) == 0:
        n_local = curr_partitions["partition"].nunique()
        mapping = curr_partitions[["partition"]].drop_duplicates()
        mapping["global_cid"] = cudf.Series(
            range(global_cid_counter, global_cid_counter + len(mapping)),
            dtype="int32",
        ).values
        return mapping, global_cid_counter + int(n_local)

    overlap_counts = (
        overlap.groupby(["partition", "global_cid"], as_index=False)
        .agg(overlap_count=("node", "count"))
    )

    overlap_counts = overlap_counts.merge(curr_sizes, on="partition", how="left")
    overlap_counts = overlap_counts.merge(prev_sizes, on="global_cid", how="left")

    overlap_counts["jaccard"] = (
        overlap_counts["overlap_count"].astype("float32")
        / (
            overlap_counts["curr_size"]
            + overlap_counts["prev_size"]
            - overlap_counts["overlap_count"]
            + 1e-9
        ).astype("float32")
    )

    valid = overlap_counts[overlap_counts["jaccard"] >= jaccard_thresh]

    if len(valid) > 0:
        best_matches = (
            valid.sort_values("jaccard", ascending=False)
            .drop_duplicates(subset=["partition"], keep="first")
            [["partition", "global_cid"]]
        )
    else:
        best_matches = cudf.DataFrame(columns=["partition", "global_cid"])

    all_partitions = curr_partitions[["partition"]].drop_duplicates()
    mapping = all_partitions.merge(best_matches, on="partition", how="left")

    unmatched_mask = mapping["global_cid"].isna()
    n_new = int(unmatched_mask.sum())

    if n_new > 0:
        new_ids = cudf.Series(
            range(global_cid_counter, global_cid_counter + n_new),
            dtype="int32",
        )
        unmatched = mapping[unmatched_mask].reset_index(drop=True)
        unmatched["global_cid"] = new_ids.values
        matched = mapping[~unmatched_mask]
        mapping = cudf.concat([matched, unmatched], ignore_index=True)
        global_cid_counter += n_new

    mapping["global_cid"] = mapping["global_cid"].astype("int32")

    del all_prev
    gc.collect()
    return mapping[["partition", "global_cid"]], global_cid_counter


def detect_communities(
    df: cudf.DataFrame,
    weighted_temporal_edges: cudf.DataFrame,
    cfg: Config,
) -> tuple:
    """
    4.4 — Phát hiện cộng đồng:
      - Relay-preserving edge aggregation (giữ nút trung gian)
      - Recursive Leiden (chống mega-community)
      - Layering role classification
      - Multi-window Jaccard tracking

    Returns:
        snapshots (list[dict])
        global_cid_counter (int)
    """
    print("\n" + "=" * 60)
    print("  BƯỚC 4.4: Phát hiện Cộng đồng (Recursive Leiden)")
    print(f"  Window={cfg.WINDOW_SIZE}, Stride={cfg.STRIDE}, S_MAX={cfg.S_MAX}")
    print(f"  Resolution={cfg.RESOLUTION}, Multiplier={cfg.RESOLUTION_MULTIPLIER}")
    print(f"  Tracking memory={cfg.TRACKING_MEMORY} windows")
    print("=" * 60)

    steps = df["step"].unique().to_pandas().sort_values().tolist()
    snapshots = []
    global_cid_counter = 0
    timeline = []
    # Multi-window tracking buffer
    tracking_buffer = []  # List of cudf.DataFrames ['node', 'global_cid']

    starts = range(0, len(steps) - cfg.WINDOW_SIZE + 1, cfg.STRIDE)

    for i in tqdm(starts, desc="Recursive Leiden + Tracking"):
        window_steps = steps[i : i + cfg.WINDOW_SIZE]

        # ── Lọc cạnh thời gian thuộc window ──
        if len(weighted_temporal_edges) > 0:
            mask = (
                weighted_temporal_edges["step_1"].isin(window_steps) |
                weighted_temporal_edges["step_2"].isin(window_steps)
            )
            window_temporal = weighted_temporal_edges[mask]
        else:
            window_temporal = cudf.DataFrame()

        if len(window_temporal) == 0:
            continue

        # ── Relay-preserving edge list (MỚI) ──
        edge_agg = build_relay_preserving_edge_list(window_temporal)

        if len(edge_agg) == 0:
            continue

        # ── Node features with Layering role (MỚI) ──
        mask_raw = df["step"].isin(window_steps)
        df_window = df[mask_raw]
        node_features = compute_refined_node_roles(df_window, cfg)

        # ── Recursive Leiden (MỚI) ──
        partitions_df = run_recursive_leiden(edge_agg, cfg, depth=0)

        if len(partitions_df) == 0:
            continue

        # ── Multi-window Jaccard tracking (MỚI) ──
        snap_idx = i // cfg.STRIDE

        if len(tracking_buffer) == 0:
            local_map = partitions_df[["partition"]].drop_duplicates()
            n_local = len(local_map)
            local_map["global_cid"] = cudf.Series(
                range(global_cid_counter, global_cid_counter + n_local),
                dtype="int32",
            ).values
            global_cid_counter += n_local
            partitions_df = partitions_df.merge(local_map, on="partition", how="left")
        else:
            mapping, global_cid_counter = _vectorized_jaccard_tracking_multi(
                partitions_df, tracking_buffer, cfg.JACCARD_THRESH, global_cid_counter
            )
            partitions_df = partitions_df.merge(mapping, on="partition", how="left")

        partitions_df = partitions_df[["node", "global_cid"]].copy()
        partitions_df["global_cid"] = partitions_df["global_cid"].astype("int32")

        # Cập nhật tracking buffer (giữ N windows gần nhất)
        tracking_buffer.append(partitions_df[["node", "global_cid"]].copy())
        if len(tracking_buffer) > cfg.TRACKING_MEMORY:
            tracking_buffer.pop(0)

        snapshots.append({
            "window_id" : snap_idx,
            "steps"     : window_steps,
            "step_start": window_steps[0],
            "step_end"  : window_steps[-1],
            "edge_feat" : edge_agg,
            "node_feat" : node_features,
            "partition_df": partitions_df,
            "n_edges"   : len(edge_agg),
            "n_nodes"   : len(node_features),
        })

        timeline.append((snap_idx, len(partitions_df), partitions_df["global_cid"].nunique()))
        gc.collect()

    # ── Summary ──
    print(f"\n  ✅ Recursive Leiden community detection hoàn tất.")
    print(f"     Tổng snapshots: {len(snapshots)}")
    print(f"     Tổng communities toàn cục: {global_cid_counter}")
    for w_idx, n_nodes, n_comms in timeline[:5]:
        print(f"     Window {w_idx}: {n_nodes:,} nodes, {n_comms} communities")
    if len(timeline) > 5:
        print(f"     ... ({len(timeline) - 5} more)")

    gc.collect()
    return snapshots, global_cid_counter



# ============================================================================
# 4.5  PHÁT HIỆN CỘNG ĐỒNG BẤT THƯỜNG (Soft Suspicion Scoring)
# ============================================================================

def extract_community_features(
    snapshots: list,
    df_raw: cudf.DataFrame,
    cfg: Config,
) -> cudf.DataFrame:
    """
    Trích xuất đặc trưng community (directed features + size-normalized).
    Vectorized trên GPU.
    """
    comm_records_list = []

    for snap in tqdm(snapshots, desc="GPU Community Feature Extraction"):
        edge_df = snap["edge_feat"]
        part_df = snap["partition_df"]
        node_df = snap["node_feat"]

        if len(part_df) == 0:
            continue

        # ── Ánh xạ cạnh vào community ──
        part_src = part_df[["node", "global_cid"]].rename(
            columns={"node": "src", "global_cid": "src_cid"}
        )
        part_dst = part_df[["node", "global_cid"]].rename(
            columns={"node": "dst", "global_cid": "dst_cid"}
        )

        edges_mapped = edge_df.merge(part_src, on="src", how="inner")
        edges_mapped = edges_mapped.merge(part_dst, on="dst", how="inner")

        # Cạnh nội bộ
        internal = edges_mapped[edges_mapped["src_cid"] == edges_mapped["dst_cid"]].copy()
        internal = internal.rename(columns={"src_cid": "global_cid"}).drop(columns=["dst_cid"])

        if len(internal) == 0:
            continue

        # ── Đặc trưng cơ bản ──
        internal["is_alert_edge"] = (internal["n_alert"] > 0).astype("float32")

        comm_basic = internal.groupby("global_cid", as_index=False).agg(
            total_volume=("total_amount", "sum"),
            n_internal_edges=("total_amount", "count"),
            alert_ratio=("is_alert_edge", "mean"),
        )

        # ── Flow Ratio ──
        node_with_cid = part_df[["node", "global_cid"]].merge(
            node_df[["node", "total_volume_out", "total_volume_in", "role",
                      "flow_consistency", "layering_score"]],
            on="node", how="left"
        ).fillna(0)

        comm_flow = node_with_cid.groupby("global_cid", as_index=False).agg(
            comm_total_out=("total_volume_out", "sum"),
            comm_total_in=("total_volume_in", "sum"),
        )

        comm_flow["flow_ratio"] = (
            comm_flow["comm_total_in"] / (comm_flow["comm_total_out"] + 1e-9)
        ).astype("float32")

        comm_basic = comm_basic.merge(
            comm_flow[["global_cid", "flow_ratio", "comm_total_out", "comm_total_in"]],
            on="global_cid", how="left"
        )

        # ── Internal Recirculation ──
        reverse_check = internal[["src", "dst", "global_cid", "total_amount"]].rename(
            columns={"src": "dst_r", "dst": "src_r", "total_amount": "rev_amount"}
        )

        recirc = internal[["src", "dst", "global_cid", "total_amount"]].merge(
            reverse_check,
            left_on=["src", "dst", "global_cid"],
            right_on=["src_r", "dst_r", "global_cid"],
            how="inner"
        )

        if len(recirc) > 0:
            recirc_vol = recirc.groupby("global_cid", as_index=False).agg(
                recirc_volume=("total_amount", "sum")
            )
        else:
            recirc_vol = cudf.DataFrame({
                "global_cid": cudf.Series(dtype="int32"),
                "recirc_volume": cudf.Series(dtype="float32"),
            })

        comm_basic = comm_basic.merge(recirc_vol, on="global_cid", how="left")
        comm_basic["recirc_volume"] = comm_basic["recirc_volume"].fillna(0)
        comm_basic["internal_recirc"] = (
            comm_basic["recirc_volume"] / (comm_basic["total_volume"] + 1e-9)
        ).astype("float32")
        comm_basic = comm_basic.drop(columns=["recirc_volume"])

        # ── Sink Concentration ──
        node_in_vol = internal.groupby(["global_cid", "dst"], as_index=False).agg(
            in_vol=("total_amount", "sum")
        )
        max_in = node_in_vol.groupby("global_cid", as_index=False).agg(
            max_sink_in=("in_vol", "max")
        )
        total_in = node_in_vol.groupby("global_cid", as_index=False).agg(
            total_in_vol=("in_vol", "sum")
        )
        sink_conc = max_in.merge(total_in, on="global_cid")
        sink_conc["sink_concentration"] = (
            sink_conc["max_sink_in"] / (sink_conc["total_in_vol"] + 1e-9)
        ).astype("float32")

        comm_basic = comm_basic.merge(
            sink_conc[["global_cid", "sink_concentration"]], on="global_cid", how="left"
        )

        # ── Source Concentration ──
        node_out_vol = internal.groupby(["global_cid", "src"], as_index=False).agg(
            out_vol=("total_amount", "sum")
        )
        max_out = node_out_vol.groupby("global_cid", as_index=False).agg(
            max_source_out=("out_vol", "max")
        )
        total_out = node_out_vol.groupby("global_cid", as_index=False).agg(
            total_out_vol=("out_vol", "sum")
        )
        source_conc = max_out.merge(total_out, on="global_cid")
        source_conc["source_concentration"] = (
            source_conc["max_source_out"] / (source_conc["total_out_vol"] + 1e-9)
        ).astype("float32")

        comm_basic = comm_basic.merge(
            source_conc[["global_cid", "source_concentration"]], on="global_cid", how="left"
        )

        # ── Max Flow ──
        max_flow_per_comm = internal.groupby("global_cid", as_index=False).agg(
            max_single_flow=("total_amount", "max")
        )
        comm_basic = comm_basic.merge(max_flow_per_comm, on="global_cid", how="left")

        # ── Kích thước + Lọc ──
        sizes = part_df.groupby("global_cid", as_index=False).agg(
            size=("node", "count")
        )
        comm_basic = comm_basic.merge(sizes, on="global_cid")
        comm_basic = comm_basic[comm_basic["size"] >= cfg.MIN_COMM_SIZE]

        # ── Role counts (bao gồm Layering - MỚI) ──
        node_with_cid["is_source"]   = (node_with_cid["role"] == 1).astype("int32")
        node_with_cid["is_sink"]     = (node_with_cid["role"] == 2).astype("int32")
        node_with_cid["is_layering"] = (node_with_cid["role"] == 3).astype("int32")

        role_counts = node_with_cid.groupby("global_cid", as_index=False).agg(
            n_sources=("is_source", "sum"),
            n_sinks=("is_sink", "sum"),
            n_layering=("is_layering", "sum"),
            avg_layering_score=("layering_score", "mean"),
        )
        comm_basic = comm_basic.merge(role_counts, on="global_cid", how="left")

        # ── Size-Normalized Features (MỚI — update.md) ──
        comm_basic["vol_density"] = (
            comm_basic["total_volume"] / (comm_basic["size"] + 1e-9)
        ).astype("float32")

        comm_basic["edge_density"] = (
            comm_basic["n_internal_edges"].astype("float32") /
            (comm_basic["size"] * (comm_basic["size"] - 1) + 1e-9)
        ).astype("float32")

        comm_basic["max_flow_norm"] = (
            comm_basic["max_single_flow"] / (comm_basic["size"] + 1e-9)
        ).astype("float32")

        # ── Metadata ──
        comm_basic["window_id"]  = snap["window_id"]
        comm_basic["step_start"] = snap["step_start"]
        comm_basic["step_end"]   = snap["step_end"]

        # Fill NaN
        for col in ["flow_ratio", "internal_recirc", "sink_concentration",
                     "source_concentration", "max_single_flow",
                     "n_sinks", "n_sources", "n_layering", "avg_layering_score",
                     "vol_density", "edge_density", "max_flow_norm",
                     "comm_total_out", "comm_total_in"]:
            if col in comm_basic.columns:
                comm_basic[col] = comm_basic[col].fillna(0)

        comm_records_list.append(comm_basic)

    if comm_records_list:
        final_comm_df = cudf.concat(comm_records_list, ignore_index=True)
    else:
        final_comm_df = cudf.DataFrame()

    return final_comm_df


def detect_anomalous_communities(
    comm_df: cudf.DataFrame,
    cfg: Config,
    n_alert: int,
    n_rows: int,
) -> cudf.DataFrame:
    """
    4.5 — Phát hiện cộng đồng bất thường bằng SOFT SCORING (MỚI — update.md).

    Thay vì binary (C1 & C2 & C3), dùng hệ thống chấm điểm tổng hợp:
    S = w1·f(C2) + w2·f(C3) + w3·Velocity + w4·alert_ratio + w5·structural

    Mỗi thành phần trả về [0, 1]. Cho phép bắt community "mấp mé" rủi ro.

    Returns:
        comm_df (cudf.DataFrame): + suspicion_score, is_suspicious, velocity, ...
    """
    print("\n" + "=" * 60)
    print("  BƯỚC 4.5: Phát hiện Cộng đồng Bất thường (Soft Scoring)")
    print(f"  Weights: C2={cfg.W_C2}, C3={cfg.W_C3}, Velocity={cfg.W_VELOCITY}, "
          f"Alert={cfg.W_ALERT}, Structure={cfg.W_STRUCTURE}")
    print(f"  Global threshold: {cfg.GLOBAL_SUSP_THRESHOLD}")
    print("=" * 60)

    if len(comm_df) == 0:
        print("  ⚠️ Không có community nào để đánh giá.")
        return comm_df

    # ══════════════════════════════════════════════════════════
    # PHẦN 1: Velocity Score (MỚI — Temporal Burstiness)
    # ══════════════════════════════════════════════════════════
    duration = (comm_df["step_end"] - comm_df["step_start"] + 1).astype("float32")
    comm_df["velocity"] = comm_df["total_volume"] / (duration + 1e-9)

    # Normalize velocity: clip to [0, 1] relative to system mean
    mean_velocity = float(comm_df["velocity"].mean())
    comm_df["velocity_score"] = (
        comm_df["velocity"] / (mean_velocity + 1e-9)
    ).clip(upper=1.0).astype("float32")

    # ══════════════════════════════════════════════════════════
    # PHẦN 2: C2 Score (Soft — distance from optimal range center)
    # ══════════════════════════════════════════════════════════
    mid_c2 = (cfg.C2_ALLOC_MIN + cfg.C2_ALLOC_MAX) / 2
    comm_df["c2_score"] = (
        1 - (comm_df["sink_concentration"] - mid_c2).abs()
    ).clip(lower=0.0).astype("float32")

    # ══════════════════════════════════════════════════════════
    # PHẦN 3: C3 Score (Soft — normalized max flow)
    # ══════════════════════════════════════════════════════════
    comm_df["c3_score"] = (
        comm_df["max_single_flow"] / (cfg.C3_MIN_FLOW + 1e-9)
    ).clip(upper=1.0).astype("float32")

    # ══════════════════════════════════════════════════════════
    # PHẦN 4: Structural Score (C1 + layering bonus)
    # ══════════════════════════════════════════════════════════
    # Has source AND sink
    has_roles = ((comm_df["n_sources"] >= 1) & (comm_df["n_sinks"] >= 1)).astype("float32")
    # Layering bonus
    layering_bonus = (comm_df["n_layering"] >= 1).astype("float32") * 0.5
    comm_df["structural_score"] = (has_roles * 0.5 + layering_bonus).clip(upper=1.0).astype("float32")

    # ══════════════════════════════════════════════════════════
    # PHẦN 5: Suspicion Score tổng hợp
    # ══════════════════════════════════════════════════════════
    comm_df["suspicion_score"] = (
        cfg.W_C2 * comm_df["c2_score"] +
        cfg.W_C3 * comm_df["c3_score"] +
        cfg.W_VELOCITY * comm_df["velocity_score"] +
        cfg.W_ALERT * comm_df["alert_ratio"] +
        cfg.W_STRUCTURE * comm_df["structural_score"]
    ).astype("float32")

    # ── Size-penalty (MỚI — 08_oversegmentation.md) ──
    # Penalized: Risk_adj = Risk_raw - γ·log(|C|)
    # Triệt tiêu ưu thế của community lớn
    GAMMA = 0.05  # Hệ số penalty nhẹ
    size_penalty = GAMMA * cudf.Series(cp.log(cp.asarray(comm_df["size"].astype("float32").values) + 1))
    comm_df["suspicion_score"] = (comm_df["suspicion_score"] - size_penalty).clip(lower=0.0)

    # ── Binary flag ──
    comm_df["is_suspicious"] = (
        comm_df["suspicion_score"] >= cfg.GLOBAL_SUSP_THRESHOLD
    ).astype("int8")

    # Giữ lại component flags cho explainability
    comm_df["c1_flag"] = has_roles.astype("int8")
    comm_df["c2_flag"] = (comm_df["c2_score"] >= 0.3).astype("int8")
    comm_df["c3_flag"] = (comm_df["c3_score"] >= 0.5).astype("int8")

    # ── Statistics ──
    total = len(comm_df)
    n_susp = int(comm_df["is_suspicious"].sum())

    global_alert_rate = float(n_alert) / float(n_rows) if n_rows > 0 else 0
    susp_alert_rate = float(
        comm_df[comm_df["is_suspicious"] == 1]["alert_ratio"].mean()
    ) if n_susp > 0 else 0
    norm_alert_rate = float(
        comm_df[comm_df["is_suspicious"] == 0]["alert_ratio"].mean()
    ) if total - n_susp > 0 else 0

    AER = susp_alert_rate / (global_alert_rate + 1e-9)

    # Score statistics
    mean_score = float(comm_df["suspicion_score"].mean())
    p90_score = float(comm_df["suspicion_score"].quantile(0.9))

    print(f"\n  Kết quả Soft Scoring:")
    print(f"    Tổng communities: {total:,}")
    print(f"    Suspicious (score ≥ {cfg.GLOBAL_SUSP_THRESHOLD}): {n_susp:,} ({n_susp/total*100:.1f}%)")
    print(f"    Mean suspicion score: {mean_score:.4f}")
    print(f"    P90 suspicion score:  {p90_score:.4f}")
    print(f"\n  Alert Enrichment Ratio (AER): {AER:.2f}×")
    print(f"    Global alert rate:  {global_alert_rate*100:.2f}%")
    print(f"    Susp alert rate:    {susp_alert_rate*100:.2f}%")
    print(f"    Normal alert rate:  {norm_alert_rate*100:.2f}%")

    # Unknown unknowns — communities with high score but low alert_ratio
    if n_susp > 0:
        unknown = comm_df[
            (comm_df["is_suspicious"] == 1) &
            (comm_df["alert_ratio"] < cfg.ALERT_THRESH)
        ]
        n_unknown = len(unknown)
        print(f"\n  🔍 Unknown Unknowns (suspicious but alert_ratio < {cfg.ALERT_THRESH}): {n_unknown}")
        if n_unknown > 0:
            print(f"     Đây có thể là vụ rửa tiền CHƯA BỊ PHÁT HIỆN.")

    print(f"  ✅ Phát hiện cộng đồng bất thường hoàn tất.")

    gc.collect()
    return comm_df
