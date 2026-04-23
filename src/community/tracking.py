# src/community/tracking.py
# Cross-window community tracking bằng Jaccard similarity.
#
# Thay vì chỉ so sánh với window T-1, so sánh với buffer N windows gần nhất.
# → Community "tắt đèn" 1-2 windows mà không mất global ID.
#
# Output: partition_df với cột 'global_cid' thay cho 'partition'.
# global_cid là ID duy nhất toàn cục, ổn định qua nhiều window.

import gc

from .config import CommunityConfig


def _cudf():
    import cudf  # noqa: PLC0415
    return cudf


def match_communities_jaccard(
    curr_partitions,
    tracking_buffer: list,
    cfg: CommunityConfig,
    global_cid_counter: int,
) -> tuple:
    """
    Gắn global_cid cho các community trong window hiện tại bằng
    Jaccard similarity so với buffer N windows gần nhất.

    Jaccard(C_curr, C_prev) = |C_curr ∩ C_prev| / |C_curr ∪ C_prev|

    Nếu Jaccard >= JACCARD_THRESH → giữ nguyên global_cid của community cũ.
    Nếu không match → tạo global_cid mới.

    Args:
        curr_partitions:  cuDF DataFrame ['node', 'partition']
        tracking_buffer:  list of cuDF DataFrame ['node', 'global_cid']
                          (N windows gần nhất, oldest first)
        cfg:              CommunityConfig
        global_cid_counter: bộ đếm ID toàn cục hiện tại

    Returns:
        (partitions_with_gcid, global_cid_counter)
            partitions_with_gcid: cuDF DataFrame ['node', 'global_cid']
            global_cid_counter: updated counter
    """
    cudf = _cudf()

    # Không có history → gán ID mới cho tất cả
    if not tracking_buffer:
        local = curr_partitions[["partition"]].drop_duplicates()
        n     = len(local)
        local["global_cid"] = cudf.Series(
            range(global_cid_counter, global_cid_counter + n), dtype="int32"
        ).values
        result = curr_partitions.merge(local, on="partition", how="left")
        result = result[["node", "global_cid"]].copy()
        result["global_cid"] = result["global_cid"].astype("int32")
        return result, global_cid_counter + n

    # Gộp tất cả windows trong buffer → de-dup theo node (giữ window gần nhất)
    all_prev = cudf.concat(tracking_buffer, ignore_index=True)
    all_prev = all_prev.drop_duplicates(subset=["node"], keep="last")

    # Kích thước community hiện tại và trước
    curr_sizes = (
        curr_partitions.groupby("partition", as_index=False)
        .agg(curr_size=("node", "count"))
    )
    prev_sizes = (
        all_prev.groupby("global_cid", as_index=False)
        .agg(prev_size=("node", "count"))
    )

    # Overlap matrix: (partition, global_cid) → overlap_count
    overlap = curr_partitions[["node", "partition"]].merge(
        all_prev[["node", "global_cid"]], on="node", how="inner"
    )

    if len(overlap) == 0:
        # Không có overlap → tất cả là community mới
        local = curr_partitions[["partition"]].drop_duplicates()
        n     = len(local)
        local["global_cid"] = cudf.Series(
            range(global_cid_counter, global_cid_counter + n), dtype="int32"
        ).values
        result = curr_partitions.merge(local, on="partition", how="left")
        result = result[["node", "global_cid"]].copy()
        result["global_cid"] = result["global_cid"].astype("int32")
        del all_prev
        return result, global_cid_counter + n

    overlap_counts = (
        overlap.groupby(["partition", "global_cid"], as_index=False)
        .agg(overlap_count=("node", "count"))
    )

    overlap_counts = overlap_counts.merge(curr_sizes, on="partition",    how="left")
    overlap_counts = overlap_counts.merge(prev_sizes, on="global_cid",   how="left")

    overlap_counts["jaccard"] = (
        overlap_counts["overlap_count"].astype("float32")
        / (
            overlap_counts["curr_size"]
            + overlap_counts["prev_size"]
            - overlap_counts["overlap_count"]
            + 1e-9
        ).astype("float32")
    )

    # Lấy best match per partition (highest Jaccard)
    valid = overlap_counts[overlap_counts["jaccard"] >= cfg.JACCARD_THRESH]

    if len(valid) > 0:
        best = (
            valid.sort_values("jaccard", ascending=False)
            .drop_duplicates(subset=["partition"], keep="first")
            [["partition", "global_cid"]]
        )
    else:
        best = cudf.DataFrame(columns=["partition", "global_cid"])

    # Merge best match vào tất cả partitions
    all_parts = curr_partitions[["partition"]].drop_duplicates()
    mapping   = all_parts.merge(best, on="partition", how="left")

    # Unmatched → tạo global_cid mới
    unmatched_mask = mapping["global_cid"].isna()
    n_new          = int(unmatched_mask.sum())

    if n_new > 0:
        new_ids   = cudf.Series(
            range(global_cid_counter, global_cid_counter + n_new), dtype="int32"
        )
        unmatched = mapping[unmatched_mask].reset_index(drop=True)
        unmatched["global_cid"] = new_ids.values
        matched   = mapping[~unmatched_mask]
        mapping   = cudf.concat([matched, unmatched], ignore_index=True)
        global_cid_counter += n_new

    mapping["global_cid"] = mapping["global_cid"].astype("int32")

    result = curr_partitions.merge(mapping, on="partition", how="left")
    result = result[["node", "global_cid"]].copy()
    result["global_cid"] = result["global_cid"].astype("int32")

    del all_prev
    gc.collect()
    return result, global_cid_counter


__all__ = ["match_communities_jaccard"]