# src/community/pipeline.py
# Orchestrator: chạy toàn bộ community pipeline qua các windows.
#
# Thứ tự mỗi window:
#   1. Lọc weighted_temporal_edges theo window steps
#   2. build_relay_edges
#   3. compute_node_roles
#   4. run_recursive_leiden
#   5. match_communities_jaccard (tracking)
#   6. Lưu snapshot
#
# Sau khi xong tất cả windows:
#   7. extract_community_features
#   8. score_communities

import gc

from tqdm.auto import tqdm

from .config import CommunityConfig
from .detection import build_relay_edges, run_recursive_leiden, compute_node_roles
from .tracking import match_communities_jaccard
from .scoring import extract_community_features, score_communities


def _cudf():
    import cudf  # noqa: PLC0415
    return cudf


def run_community_pipeline(
    df,
    weighted_temporal_edges,
    cfg: CommunityConfig,
    n_alert: int,
    n_rows: int,
):
    """
    Chạy toàn bộ community detection pipeline.

    Args:
        df: cuDF DataFrame đã encode — columns: [step, src, dst, amount, is_laundering]
        weighted_temporal_edges: output của compute_weights()
        cfg:     CommunityConfig
        n_alert: số alert rows trong dataset (từ load_raw summary)
        n_rows:  tổng số rows trong dataset

    Returns:
        snapshots (list[dict]): mỗi dict là một window snapshot với keys:
            window_id, steps, step_start, step_end,
            edge_feat, node_feat, partition_df, n_edges, n_nodes
        comm_df (cudf.DataFrame): community features + suspicion scores
        global_cid_counter (int): tổng số global community IDs đã tạo
    """
    cudf = _cudf()

    print("\n" + "=" * 60)
    print("  BƯỚC 4.4: Community Detection Pipeline")
    print(f"  Window={cfg.WINDOW_SIZE}, Stride={cfg.STRIDE}, S_MAX={cfg.S_MAX}")
    print(f"  JACCARD_THRESH={cfg.JACCARD_THRESH}, TRACKING_MEMORY={cfg.TRACKING_MEMORY}")
    print("=" * 60)

    all_steps = df["step"].unique().to_pandas().sort_values().tolist()
    snapshots            = []
    global_cid_counter   = 0
    tracking_buffer      = []  # list of cuDF ['node', 'global_cid']

    window_starts = list(range(0, len(all_steps) - cfg.WINDOW_SIZE + 1, cfg.STRIDE))

    for i in tqdm(window_starts, desc="Community detection (windows)"):
        window_steps = all_steps[i : i + cfg.WINDOW_SIZE]
        window_id    = i // cfg.STRIDE

        # ── Lọc weighted temporal edges thuộc window ──────────────────────
        if len(weighted_temporal_edges) > 0:
            mask = (
                weighted_temporal_edges["step_1"].isin(window_steps)
                | weighted_temporal_edges["step_2"].isin(window_steps)
            )
            w_temp = weighted_temporal_edges[mask]
        else:
            w_temp = cudf.DataFrame()

        if len(w_temp) == 0:
            continue

        # ── Relay-preserving edge aggregation ─────────────────────────────
        relay_edges = build_relay_edges(w_temp)
        if len(relay_edges) == 0:
            continue

        # ── Node roles ────────────────────────────────────────────────────
        mask_raw  = df["step"].isin(window_steps)
        df_window = df[mask_raw]
        node_feat = compute_node_roles(df_window, cfg)

        # ── Recursive Leiden ──────────────────────────────────────────────
        partitions = run_recursive_leiden(relay_edges, cfg, depth=0)
        if len(partitions) == 0:
            continue

        # ── Cross-window tracking ─────────────────────────────────────────
        partitions_with_gcid, global_cid_counter = match_communities_jaccard(
            partitions, tracking_buffer, cfg, global_cid_counter
        )

        # Cập nhật buffer
        tracking_buffer.append(partitions_with_gcid[["node", "global_cid"]].copy())
        if len(tracking_buffer) > cfg.TRACKING_MEMORY:
            tracking_buffer.pop(0)

        snapshots.append({
            "window_id":    window_id,
            "steps":        window_steps,
            "step_start":   window_steps[0],
            "step_end":     window_steps[-1],
            "edge_feat":    relay_edges,
            "node_feat":    node_feat,
            "partition_df": partitions_with_gcid,
            "n_edges":      len(relay_edges),
            "n_nodes":      len(node_feat),
        })

        gc.collect()

    print(f"\n  ✅ Detection xong: {len(snapshots)} snapshots, {global_cid_counter} global CIDs")

    # ── Feature extraction + scoring ──────────────────────────────────────
    comm_df = extract_community_features(snapshots, cfg)

    if len(comm_df) > 0:
        comm_df = score_communities(comm_df, cfg, n_alert=n_alert, n_rows=n_rows)

    gc.collect()
    return snapshots, comm_df, global_cid_counter


__all__ = ["run_community_pipeline"]