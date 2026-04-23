# src/motif/scoring.py
# Support counting và null model z-score cho motif instances.
#
# Null model: shuffle timestamps trong cùng step-bucket,
# giữ nguyên degree và amount distribution.
# Tính z-score: (observed - mean_null) / std_null

import gc
import random
from collections import Counter

import numpy as np
import pandas as pd

from .config import MotifConfig
from .index import build_event_indexes
from .matchers import find_fanin, find_fanout, find_cycle3, find_relay4, find_split_merge


def count_support(motif_instances: list[dict]) -> dict[str, int]:
    """
    Đếm số instance theo motif type.

    Args:
        motif_instances: list[dict] — output gộp từ các matchers

    Returns:
        dict {motif_type: count}
    """
    c: Counter = Counter()
    for inst in motif_instances:
        c[inst["motif_type"]] += 1
    return dict(c)


def _run_all_matchers(out_index: dict, in_index: dict, cfg: MotifConfig) -> dict[str, int]:
    """Chạy tất cả matchers và trả về support counts. Dùng nội bộ cho null model."""
    instances = (
        find_fanin(in_index, cfg)
        + find_fanout(out_index, cfg)
        + find_cycle3(out_index, cfg)
        + find_relay4(out_index, cfg)
        + find_split_merge(out_index, in_index, cfg)
    )
    return count_support(instances)


def _shuffle_timestamps(event_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Tạo một bản null model bằng cách shuffle timestamps.

    Chiến lược:
    - Shuffle cột 'step' trong toàn bộ DataFrame.
    - Giữ nguyên (src, dst, amount) — chỉ ngẫu nhiên hóa thời gian.
    - Đây là null model đơn giản nhưng hiệu quả để phá vỡ temporal structure.

    Args:
        event_df: pandas DataFrame gốc
        seed: random seed để reproducible

    Returns:
        DataFrame với cột 'step' đã shuffle
    """
    rng = random.Random(seed)
    steps = event_df["step"].tolist()
    rng.shuffle(steps)
    df_null = event_df.copy()
    df_null["step"] = steps
    df_null = df_null.sort_values(["step", "event_id"]).reset_index(drop=True)
    return df_null


def compute_null_zscore(
    observed_counts: dict[str, int],
    event_df: pd.DataFrame,
    cfg: MotifConfig,
    verbose: bool = False,
) -> dict[str, dict]:
    """
    Tính z-score cho mỗi motif type so với null model.

    z(M) = (C_obs(M) - mean_null(M)) / (std_null(M) + ε)

    Null model: shuffle timestamps N_PERMUTATIONS lần,
    mỗi lần chạy lại tất cả matchers.

    Args:
        observed_counts: dict từ count_support() trên data thực
        event_df:        pandas DataFrame gốc (chưa shuffle)
        cfg:             MotifConfig
        verbose:         in tiến độ mỗi permutation

    Returns:
        dict {motif_type: {"observed": int, "mean_null": float,
                           "std_null": float, "zscore": float}}
    """
    null_counts: dict[str, list[int]] = {mt: [] for mt in observed_counts}

    for i in range(cfg.N_PERMUTATIONS):
        if verbose:
            print(f"  Null permutation {i + 1}/{cfg.N_PERMUTATIONS}...")

        df_null = _shuffle_timestamps(event_df, seed=i)
        out_idx, in_idx, _ = build_event_indexes(df_null)
        perm_counts = _run_all_matchers(out_idx, in_idx, cfg)

        for mt in observed_counts:
            null_counts[mt].append(perm_counts.get(mt, 0))

        del df_null, out_idx, in_idx
        gc.collect()

    results = {}
    for mt, c_obs in observed_counts.items():
        arr        = np.array(null_counts[mt], dtype=float)
        mean_null  = float(arr.mean())
        std_null   = float(arr.std())
        zscore     = (c_obs - mean_null) / (std_null + 1e-9)
        results[mt] = {
            "observed":  c_obs,
            "mean_null": round(mean_null, 2),
            "std_null":  round(std_null, 2),
            "zscore":    round(zscore, 3),
        }

    return results


__all__ = ["count_support", "compute_null_zscore"]