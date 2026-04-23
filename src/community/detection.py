# src/community/detection.py
# Bước 4.4: Community detection.
#
# Ba phần chính:
#   1. build_relay_edges: tạo edge list BẢO TOÀN relay structure
#      (không nối tắt src_1→dst_2, giữ nút trung gian cho Leiden)
#   2. run_recursive_leiden: Leiden đệ quy chống mega-community
#   3. compute_node_roles: phân loại source/sink/layering/neutral

import gc

from .config import CommunityConfig


def _cudf():
    import cudf  # noqa: PLC0415
    return cudf


def _cugraph():
    import cugraph  # noqa: PLC0415
    return cugraph


def _cp():
    import cupy as cp  # noqa: PLC0415
    return cp


# ── Relay-preserving edge list ─────────────────────────────────────────────

def build_relay_edges(weighted_temporal_edges):
    """
    Tạo edge list cho Leiden, BẢO TOÀN relay structure.

    Thay vì nối tắt src_1→dst_2 (mất nút trung gian),
    giữ cả 2 cạnh thực tế:
        Cạnh 1: src_1 → dst_1  (Nguồn → Trung gian)
        Cạnh 2: src_2 → dst_2  (Trung gian → Đích)

    Lý do: Leiden cần thấy nút trung gian để nhận diện
    "hub điều phối dòng tiền" (layering node).

    Args:
        weighted_temporal_edges: cuDF DataFrame với weight column
            (output của compute_weights)

    Returns:
        relay_edges (cudf.DataFrame):
            [src, dst, weighted_amount, total_amount, n_tx, n_alert, alert_ratio]
    """
    cudf = _cudf()

    if len(weighted_temporal_edges) == 0:
        return cudf.DataFrame(columns=[
            "src", "dst", "weighted_amount",
            "total_amount", "n_tx", "n_alert", "alert_ratio",
        ])

    # Cạnh 1: Nguồn → Trung gian
    e1 = weighted_temporal_edges[
        ["src_1", "dst_1", "weight", "amount_1", "alert_1"]
    ].rename(columns={"src_1": "src", "dst_1": "dst", "weight": "w",
                       "amount_1": "amt", "alert_1": "alert"})

    # Cạnh 2: Trung gian → Đích
    e2 = weighted_temporal_edges[
        ["src_2", "dst_2", "weight", "amount_2", "alert_2"]
    ].rename(columns={"src_2": "src", "dst_2": "dst", "weight": "w",
                       "amount_2": "amt", "alert_2": "alert"})

    full = cudf.concat([e1, e2], ignore_index=True)
    del e1, e2

    relay_edges = full.groupby(["src", "dst"], as_index=False).agg(
        weighted_amount=("w",     "max"),   # max weight = strongest flow signal
        total_amount   =("amt",   "sum"),
        n_tx           =("amt",   "count"),
        n_alert        =("alert", "sum"),
    )

    # W_final = max_weight × total_amount
    relay_edges["weighted_amount"] = (
        relay_edges["weighted_amount"].astype("float32")
        * relay_edges["total_amount"].astype("float32")
    )
    relay_edges["total_amount"] = relay_edges["total_amount"].astype("float32")
    relay_edges["n_tx"]         = relay_edges["n_tx"].astype("int32")
    relay_edges["n_alert"]      = relay_edges["n_alert"].astype("int32")
    relay_edges["alert_ratio"]  = (
        relay_edges["n_alert"].astype("float32")
        / (relay_edges["n_tx"].astype("float32") + 1e-9)
    )

    del full
    gc.collect()
    return relay_edges


# ── Recursive Leiden ───────────────────────────────────────────────────────

def _run_leiden_once(edge_df, resolution: float):
    """
    Chạy một lần Leiden undirected trên GPU.

    cuGraph chỉ support undirected → canonical pair (min, max).
    Thêm reciprocity bonus: cặp (A,B) có cả A→B và B→A → weight × 2.

    Returns:
        partitions_df (cudf.DataFrame): ['node', 'partition']
        modularity (float)
    """
    cudf    = _cudf()
    cugraph = _cugraph()

    if len(edge_df) == 0:
        return cudf.DataFrame(columns=["node", "partition"]), 0.0

    edges = edge_df[["src", "dst", "weighted_amount"]].copy()

    # Reciprocity bonus
    reverse = edges[["dst", "src", "weighted_amount"]].rename(
        columns={"dst": "src", "src": "dst", "weighted_amount": "rev_w"}
    )
    merged = edges.merge(reverse, on=["src", "dst"], how="left")
    has_reciprocal = merged["rev_w"].notna()
    edges["weighted_amount"] = (
        edges["weighted_amount"] * (1 + has_reciprocal.astype("float32"))
    )
    del merged, reverse

    # Canonical pair cho undirected
    edges["u"] = edges[["src", "dst"]].min(axis=1)
    edges["v"] = edges[["src", "dst"]].max(axis=1)

    undirected = edges.groupby(["u", "v"], as_index=False).agg(
        weight=("weighted_amount", "sum")
    )
    undirected["weight"] = undirected["weight"].astype("float32")

    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(
        undirected, source="u", destination="v",
        edge_attr="weight", renumber=True,
    )

    parts, modularity = cugraph.leiden(G, resolution=resolution)
    parts = parts.rename(columns={"vertex": "node"})

    del G, undirected, edges
    gc.collect()

    return parts[["node", "partition"]], float(modularity)


def run_recursive_leiden(
    edge_df,
    cfg: CommunityConfig,
    depth: int = 0,
):
    """
    Bước 4.4 — Recursive Leiden chống mega-community.

    Thuật toán:
        1. Chạy Leiden một lần
        2. Tìm community có size > S_MAX
        3. Với mỗi oversized community: chạy lại Leiden
           trên subgraph với resolution cao hơn
        4. Dừng khi tất cả community ≤ S_MAX hoặc đạt MAX_RECURSION_DEPTH

    Args:
        edge_df: cuDF DataFrame ['src', 'dst', 'weighted_amount', ...]
        cfg:     CommunityConfig
        depth:   độ sâu đệ quy hiện tại (khởi đầu = 0)

    Returns:
        partitions_df (cudf.DataFrame): ['node', 'partition']
            partition IDs là globally unique integers.
    """
    cudf = _cudf()

    resolution = cfg.RESOLUTION * (cfg.RESOLUTION_MULTIPLIER ** depth)
    parts, modularity = _run_leiden_once(edge_df, resolution)

    if len(parts) == 0 or depth >= cfg.MAX_RECURSION_DEPTH:
        return parts

    print(f"    Leiden depth={depth}: modularity={modularity:.4f}, resolution={resolution:.2f}")

    # Tìm community quá lớn
    comm_sizes = parts.groupby("partition", as_index=False).agg(size=("node", "count"))
    oversized  = comm_sizes[comm_sizes["size"] > cfg.S_MAX]["partition"]

    if len(oversized) == 0:
        return parts

    print(f"    ⚠️  {len(oversized)} community > {cfg.S_MAX} nodes → recursive split")

    max_partition = int(parts["partition"].max()) + 1
    result_parts  = []

    # Giữ nguyên community đã đủ nhỏ
    normal = parts[~parts["partition"].isin(oversized)]
    if len(normal) > 0:
        result_parts.append(normal)

    # Đệ quy cho từng oversized community
    for cid in oversized.to_pandas().tolist():
        sub_nodes = parts[parts["partition"] == cid]["node"]
        sub_edges = edge_df[
            edge_df["src"].isin(sub_nodes) & edge_df["dst"].isin(sub_nodes)
        ]

        if len(sub_edges) == 0:
            result_parts.append(parts[parts["partition"] == cid])
            continue

        sub_parts = run_recursive_leiden(sub_edges, cfg, depth=depth + 1)

        if len(sub_parts) > 0:
            sub_parts = sub_parts.copy()
            sub_parts["partition"] = sub_parts["partition"] + max_partition
            max_partition = int(sub_parts["partition"].max()) + 1
            result_parts.append(sub_parts)

    if result_parts:
        return cudf.concat(result_parts, ignore_index=True)
    return parts


# ── Node role classification ───────────────────────────────────────────────

def compute_node_roles(df_window, cfg: CommunityConfig):
    """
    Phân loại role cho mỗi node trong một window.

    Role:
        0 = Neutral (intermediary)
        1 = Source  (gửi >> nhận)
        2 = Sink    (nhận >> gửi)
        3 = Layering (volume_in ≈ volume_out, volume cao)

    Layering Score = flow_consistency × (1 - |net_flow_ratio|)

    Args:
        df_window: cuDF DataFrame window đã encode
            columns: [src, dst, amount, is_laundering]
        cfg: CommunityConfig

    Returns:
        node_features (cudf.DataFrame):
            [node, total_volume, total_volume_out, total_volume_in,
             n_tx, n_alert_tx, alert_rate, net_flow_ratio,
             flow_consistency, layering_score, role]
    """
    src_stats = df_window.groupby("src", as_index=False).agg(
        total_volume_out =("amount",        "sum"),
        n_tx_out         =("amount",        "count"),
        n_alert_tx_out   =("is_laundering", "sum"),
    ).rename(columns={"src": "node"})

    dst_stats = df_window.groupby("dst", as_index=False).agg(
        total_volume_in  =("amount",        "sum"),
        n_tx_in          =("amount",        "count"),
        n_alert_tx_in    =("is_laundering", "sum"),
    ).rename(columns={"dst": "node"})

    nf = src_stats.merge(dst_stats, on="node", how="outer").fillna(0)

    nf["total_volume"] = nf["total_volume_out"] + nf["total_volume_in"]
    nf["n_tx"]         = nf["n_tx_out"]       + nf["n_tx_in"]
    nf["n_alert_tx"]   = nf["n_alert_tx_out"] + nf["n_alert_tx_in"]
    nf["alert_rate"]   = nf["n_alert_tx"] / (nf["n_tx"] + 1e-9)

    nf["net_flow_ratio"] = (
        (nf["total_volume_out"] - nf["total_volume_in"])
        / (nf["total_volume_out"] + nf["total_volume_in"] + 1e-9)
    ).astype("float32")

    # Flow consistency = 2·min(in, out) / (in + out + ε)
    vol_min = nf[["total_volume_in", "total_volume_out"]].min(axis=1)
    nf["flow_consistency"] = (
        2 * vol_min / (nf["total_volume_in"] + nf["total_volume_out"] + 1e-9)
    ).astype("float32")

    # Layering score = flow_consistency × (1 - |net_flow_ratio|)
    nf["layering_score"] = (
        nf["flow_consistency"] * (1 - nf["net_flow_ratio"].abs())
    ).astype("float32")

    # Role assignment (vectorized — không dùng .loc)
    is_source   = nf["net_flow_ratio"] >  cfg.ROLE_THRESHOLD
    is_sink     = nf["net_flow_ratio"] < -cfg.ROLE_THRESHOLD
    is_layering = (
        (nf["flow_consistency"] >  cfg.LAYERING_CONSISTENCY)
        & (nf["net_flow_ratio"].abs() < cfg.ROLE_THRESHOLD * 0.67)
    )

    nf["role"] = (
        is_layering.astype("int8") * 3
        + (~is_layering & is_source).astype("int8") * 1
        + (~is_layering & is_sink).astype("int8")   * 2
    )

    return nf[[
        "node", "total_volume", "total_volume_out", "total_volume_in",
        "n_tx", "n_alert_tx", "alert_rate", "net_flow_ratio",
        "flow_consistency", "layering_score", "role",
    ]]


__all__ = ["build_relay_edges", "run_recursive_leiden", "compute_node_roles"]