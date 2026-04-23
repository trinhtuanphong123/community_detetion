# tests/test_community_smoke.py
# Smoke test cho src/community/ — không cần GPU.
# Test Config, logic thuần Python, và schema validation.
#
# Chạy:
#   python -m pytest tests/test_community_smoke.py -v

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.community.config import CommunityConfig


# ── Config tests ──────────────────────────────────────────────────────────

def test_community_config_defaults():
    cfg = CommunityConfig()
    assert cfg.WINDOW_SIZE == 5
    assert cfg.STRIDE == 2
    assert cfg.S_MAX == 50
    assert cfg.MIN_COMM_SIZE == 3
    assert cfg.TRACKING_MEMORY == 3
    assert cfg.JACCARD_THRESH == 0.25
    assert cfg.MAX_RECURSION_DEPTH == 3
    assert cfg.RESOLUTION > 0


def test_community_config_weights_sum():
    """Trọng số scoring phải tổng = 1.0."""
    cfg = CommunityConfig()
    total = cfg.W_C2 + cfg.W_C3 + cfg.W_VELOCITY + cfg.W_ALERT + cfg.W_STRUCTURE
    assert abs(total - 1.0) < 1e-6, f"Weights sum = {total}, expected 1.0"


def test_community_config_override():
    cfg = CommunityConfig()
    cfg.S_MAX = 100
    assert cfg.S_MAX == 100


def test_community_config_thresholds_valid():
    cfg = CommunityConfig()
    assert 0 < cfg.GLOBAL_SUSP_THRESHOLD < 1
    assert 0 < cfg.JACCARD_THRESH < 1
    assert 0 < cfg.ROLE_THRESHOLD < 1
    assert 0 < cfg.LAYERING_CONSISTENCY < 1
    assert cfg.C2_ALLOC_MIN < cfg.C2_ALLOC_MAX
    assert cfg.C3_MIN_FLOW > 0


# ── Import test (không cần GPU) ───────────────────────────────────────────

def test_imports_no_gpu():
    """Tất cả module phải importable mà không cần GPU."""
    from src.community.config import CommunityConfig  # noqa
    # Các module sau chỉ import cuDF lazily bên trong function
    # → import module không crash dù không có GPU
    import importlib
    for mod in [
        "src.community.weighting",
        "src.community.detection",
        "src.community.tracking",
        "src.community.scoring",
        "src.community.pipeline",
    ]:
        m = importlib.import_module(mod)
        assert m is not None, f"Failed to import {mod}"


# ── GPU tests (chỉ chạy khi có cuDF) ─────────────────────────────────────

def _has_cudf():
    try:
        import cudf  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_cudf(), reason="cuDF không có")
def test_build_relay_edges_schema():
    """build_relay_edges trả về đúng columns."""
    import cudf
    from src.community.detection import build_relay_edges

    # Tạo weighted_temporal_edges giả lập
    wt = cudf.DataFrame({
        "src_1":   [0, 1],
        "dst_1":   [1, 2],
        "src_2":   [1, 2],
        "dst_2":   [2, 3],
        "amount_1": [100.0, 200.0],
        "amount_2": [95.0,  190.0],
        "alert_1":  [0, 1],
        "alert_2":  [0, 0],
        "weight":   [0.8, 0.7],
    })

    relay = build_relay_edges(wt)
    expected_cols = {"src", "dst", "weighted_amount", "total_amount",
                     "n_tx", "n_alert", "alert_ratio"}
    assert expected_cols.issubset(set(relay.columns))
    assert len(relay) > 0


@pytest.mark.skipif(not _has_cudf(), reason="cuDF không có")
def test_compute_node_roles_schema():
    """compute_node_roles trả về đúng columns và role values."""
    import cudf
    from src.community.config import CommunityConfig
    from src.community.detection import compute_node_roles

    df_win = cudf.DataFrame({
        "src":           [0, 0, 1, 2],
        "dst":           [1, 2, 2, 3],
        "amount":        [100.0, 200.0, 150.0, 50.0],
        "is_laundering": [0, 0, 1, 0],
    })
    cfg  = CommunityConfig()
    nf   = compute_node_roles(df_win, cfg)

    expected_cols = {
        "node", "total_volume", "n_tx", "alert_rate",
        "net_flow_ratio", "flow_consistency", "layering_score", "role",
    }
    assert expected_cols.issubset(set(nf.columns))
    # Role phải nằm trong {0, 1, 2, 3}
    assert set(nf["role"].to_pandas().unique()).issubset({0, 1, 2, 3})


@pytest.mark.skipif(not _has_cudf(), reason="cuDF không có")
def test_match_communities_jaccard_new_window():
    """Khi không có history → tạo global_cid mới cho tất cả partitions."""
    import cudf
    from src.community.config import CommunityConfig
    from src.community.tracking import match_communities_jaccard

    partitions = cudf.DataFrame({
        "node":      [0, 1, 2, 3],
        "partition": [0, 0, 1, 1],
    })
    cfg = CommunityConfig()

    result, counter = match_communities_jaccard(
        partitions, tracking_buffer=[], cfg=cfg, global_cid_counter=0
    )

    assert "global_cid" in result.columns
    assert counter == 2  # 2 partitions → 2 new global IDs
    assert result["global_cid"].nunique() == 2


@pytest.mark.skipif(not _has_cudf(), reason="cuDF không có")
def test_match_communities_jaccard_with_history():
    """Community với overlap cao phải giữ nguyên global_cid."""
    import cudf
    from src.community.config import CommunityConfig
    from src.community.tracking import match_communities_jaccard

    cfg = CommunityConfig()
    cfg.JACCARD_THRESH = 0.3

    # Window T-1: global_cid=0 có nodes {0,1,2}
    prev = cudf.DataFrame({
        "node":       [0, 1, 2],
        "global_cid": [0, 0, 0],
    })

    # Window T: partition=0 có nodes {0,1,2,3} → overlap với global_cid=0
    curr = cudf.DataFrame({
        "node":      [0, 1, 2, 3],
        "partition": [0, 0, 0,  1],
    })

    result, counter = match_communities_jaccard(
        curr, tracking_buffer=[prev], cfg=cfg, global_cid_counter=1
    )

    # Partition 0 phải match với global_cid=0
    partition0_gcids = (
        result[result["node"].isin([0, 1, 2])]["global_cid"].to_pandas().unique().tolist()
    )
    assert 0 in partition0_gcids


@pytest.mark.skipif(not _has_cudf(), reason="cuDF không có")
def test_score_communities_schema():
    """score_communities thêm đúng columns."""
    import cudf
    from src.community.config import CommunityConfig
    from src.community.scoring import score_communities

    comm_df = cudf.DataFrame({
        "global_cid":          [0, 1, 2],
        "size":                [5, 10, 3],
        "total_volume":        [10000.0, 50000.0, 2000.0],
        "n_internal_edges":    [8, 20, 3],
        "alert_ratio":         [0.6, 0.1, 0.0],
        "flow_ratio":          [1.2, 0.8, 0.5],
        "internal_recirc":     [0.3, 0.1, 0.0],
        "sink_concentration":  [0.7, 0.4, 0.9],
        "source_concentration":[0.5, 0.3, 0.8],
        "max_single_flow":     [8000.0, 15000.0, 500.0],
        "n_sources":           [1, 2, 0],
        "n_sinks":             [1, 1, 1],
        "n_layering":          [1, 0, 0],
        "avg_layering_score":  [0.7, 0.2, 0.0],
        "vol_density":         [2000.0, 5000.0, 667.0],
        "edge_density":        [0.4, 0.2, 0.3],
        "max_flow_norm":       [1600.0, 1500.0, 167.0],
        "comm_total_out":      [10000.0, 50000.0, 2000.0],
        "comm_total_in":       [9500.0,  48000.0, 1800.0],
        "window_id":           [0, 0, 0],
        "step_start":          [0, 0, 0],
        "step_end":            [4, 4, 4],
    })

    cfg = CommunityConfig()
    result = score_communities(comm_df, cfg, n_alert=100, n_rows=10000)

    for col in ("suspicion_score", "is_suspicious", "c1_flag", "c2_flag", "c3_flag"):
        assert col in result.columns, f"Thiếu column: {col}"

    # is_suspicious phải là 0 hoặc 1
    vals = set(result["is_suspicious"].to_pandas().unique().tolist())
    assert vals.issubset({0, 1})

    # suspicion_score phải trong [0, 1]
    assert float(result["suspicion_score"].min()) >= 0.0