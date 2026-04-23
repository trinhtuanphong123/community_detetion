# src/motif/index.py
# Xây dựng index cần thiết cho motif matching.
#
# Motif matching cần lookup nhanh theo:
#   - node gửi (out_index)   → tìm tx tiếp theo từ v
#   - node nhận (in_index)   → tìm tx đến x
#   - step                   → tìm tx trong một ngày
#
# Dùng pandas để build index (không iterrows, dùng groupby + itertuples).
# Trả về dict thuần Python để matchers không phụ thuộc pandas trong vòng lặp.

import gc
from collections import defaultdict

import pandas as pd


def build_event_indexes(event_df: pd.DataFrame) -> tuple[dict, dict, dict]:
    """
    Xây dựng 3 index từ bảng giao dịch để motif matchers search nhanh.

    Args:
        event_df: pandas DataFrame với columns:
            [event_id, step, src, dst, amount, is_laundering]
            (src/dst là int32 từ encoder; hoặc nameOrig/nameDest nếu chưa encode)

    Returns:
        out_index  (dict): {src_node: [event_dict, ...]} sorted by step
        in_index   (dict): {dst_node: [event_dict, ...]} sorted by step
        step_index (dict): {step: [event_dict, ...]}

    event_dict keys: event_id, step, src, dst, amount, is_laundering
    """
    # Chuẩn hóa tên cột: hỗ trợ cả tên encode (src/dst) và tên gốc
    col_map = {}
    cols = set(event_df.columns)
    if "nameOrig" in cols and "src" not in cols:
        col_map["nameOrig"] = "src"
    if "nameDest" in cols and "dst" not in cols:
        col_map["nameDest"] = "dst"
    if col_map:
        event_df = event_df.rename(columns=col_map)

    required = {"event_id", "step", "src", "dst", "amount"}
    missing = required - set(event_df.columns)
    if missing:
        raise ValueError(f"event_df thiếu columns: {sorted(missing)}")

    has_label = "is_laundering" in event_df.columns

    # Sort một lần, dùng cho tất cả index
    event_df = event_df.sort_values(["step", "event_id"]).reset_index(drop=True)

    out_index:  dict = defaultdict(list)
    in_index:   dict = defaultdict(list)
    step_index: dict = defaultdict(list)

    # itertuples nhanh hơn iterrows ~10x, không tạo object overhead
    for row in event_df.itertuples(index=False):
        e = {
            "event_id":      int(row.event_id),
            "step":          int(row.step),
            "src":           int(row.src),
            "dst":           int(row.dst),
            "amount":        float(row.amount),
            "is_laundering": int(row.is_laundering) if has_label else 0,
        }
        out_index[e["src"]].append(e)
        in_index[e["dst"]].append(e)
        step_index[e["step"]].append(e)

    # Đã sort khi build → không cần sort lại từng bucket
    # Chuyển defaultdict → dict thường để tránh auto-create key khi lookup
    out_index  = dict(out_index)
    in_index   = dict(in_index)
    step_index = dict(step_index)

    n_nodes = len(out_index)
    n_steps = len(step_index)
    n_events = len(event_df)
    print(f"  Event indexes built: {n_events:,} events, {n_nodes:,} unique src nodes, {n_steps} steps")

    gc.collect()
    return out_index, in_index, step_index


def filter_events_by_window(
    event_df: pd.DataFrame,
    step_start: int,
    step_end: int,
) -> pd.DataFrame:
    """
    Lọc event_df theo cửa sổ thời gian [step_start, step_end].
    Dùng trước build_event_indexes khi chạy windowed search.

    Args:
        event_df:   pandas DataFrame đầy đủ
        step_start: step bắt đầu (inclusive)
        step_end:   step kết thúc (inclusive)

    Returns:
        pandas DataFrame đã lọc, reset_index.
    """
    mask = (event_df["step"] >= step_start) & (event_df["step"] <= step_end)
    return event_df[mask].reset_index(drop=True)


__all__ = ["build_event_indexes", "filter_events_by_window"]