# tests/test_motif_smoke.py
# Smoke test cho src/motif/ — chạy được local, không cần GPU.
# Dùng data synthetic nhỏ để verify logic từng matcher.
#
# Chạy:
#   python -m pytest tests/test_motif_smoke.py -v

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.motif.config import MotifConfig
from src.motif.index import build_event_indexes, filter_events_by_window
from src.motif.matchers import (
    find_fanin, find_fanout, find_cycle3, find_relay4, find_split_merge,
)
from src.motif.scoring import count_support
from src.motif.features import (
    build_entity_motif_features,
    build_entity_feature_wide,
    build_window_motif_features,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

def _make_event_df(rows: list[dict]) -> pd.DataFrame:
    """Tạo event_df từ list of dicts."""
    df = pd.DataFrame(rows)
    df["event_id"] = np.arange(len(df), dtype=np.int64)
    df = df.sort_values(["step", "event_id"]).reset_index(drop=True)
    return df


@pytest.fixture
def cfg():
    c = MotifConfig()
    c.DELTA        = 3
    c.RHO_MIN      = 0.5
    c.RHO_MAX      = 2.0
    c.R_MIN_FANIN  = 3
    c.R_MIN_FANOUT = 3
    return c


@pytest.fixture
def fanin_df():
    """3 nguồn khác nhau → node 99, cùng cửa sổ."""
    return _make_event_df([
        {"step": 1, "src": 1, "dst": 99, "amount": 100.0, "is_laundering": 0},
        {"step": 2, "src": 2, "dst": 99, "amount":  90.0, "is_laundering": 0},
        {"step": 3, "src": 3, "dst": 99, "amount": 110.0, "is_laundering": 1},
    ])


@pytest.fixture
def fanout_df():
    """Node 10 gửi đến 3 đích khác nhau."""
    return _make_event_df([
        {"step": 1, "src": 10, "dst": 21, "amount": 200.0, "is_laundering": 0},
        {"step": 2, "src": 10, "dst": 22, "amount": 190.0, "is_laundering": 0},
        {"step": 3, "src": 10, "dst": 23, "amount": 210.0, "is_laundering": 0},
    ])


@pytest.fixture
def cycle3_df():
    """Chu kỳ 3 node: A→B→C→A."""
    return _make_event_df([
        {"step": 1, "src": 1, "dst": 2, "amount": 100.0, "is_laundering": 0},
        {"step": 2, "src": 2, "dst": 3, "amount":  95.0, "is_laundering": 0},
        {"step": 3, "src": 3, "dst": 1, "amount":  90.0, "is_laundering": 1},
    ])


@pytest.fixture
def relay4_df():
    """Relay 4 node: A→B→C→D."""
    return _make_event_df([
        {"step": 1, "src": 1, "dst": 2, "amount": 500.0, "is_laundering": 0},
        {"step": 2, "src": 2, "dst": 3, "amount": 480.0, "is_laundering": 0},
        {"step": 3, "src": 3, "dst": 4, "amount": 460.0, "is_laundering": 0},
    ])


@pytest.fixture
def split_merge_df():
    """Split-merge: U→V1→Z và U→V2→Z."""
    return _make_event_df([
        {"step": 1, "src": 10, "dst": 21, "amount": 200.0, "is_laundering": 0},
        {"step": 1, "src": 10, "dst": 22, "amount": 200.0, "is_laundering": 0},
        {"step": 3, "src": 21, "dst": 99, "amount": 190.0, "is_laundering": 0},
        {"step": 3, "src": 22, "dst": 99, "amount": 185.0, "is_laundering": 0},
    ])


# ── Config tests ──────────────────────────────────────────────────────────

def test_motif_config_defaults():
    cfg = MotifConfig()
    assert cfg.DELTA == 3
    assert cfg.RHO_MIN < cfg.RHO_MAX
    assert cfg.R_MIN_FANIN >= 2
    assert cfg.N_PERMUTATIONS > 0
    assert len(cfg.WINDOW_STEPS) > 0


# ── Index tests ───────────────────────────────────────────────────────────

def test_build_indexes_keys(fanin_df, cfg):
    out_idx, in_idx, step_idx = build_event_indexes(fanin_df)
    # 3 nguồn → out_index có 3 keys
    assert len(out_idx) == 3
    # 1 đích → in_index có 1 key
    assert 99 in in_idx
    assert len(in_idx[99]) == 3
    # 3 steps
    assert len(step_idx) == 3


def test_filter_events_by_window(relay4_df):
    filtered = filter_events_by_window(relay4_df, step_start=1, step_end=2)
    assert filtered["step"].max() <= 2
    assert len(filtered) == 2


# ── Matcher tests ─────────────────────────────────────────────────────────

def test_fanin_finds_instance(fanin_df, cfg):
    _, in_idx, _ = build_event_indexes(fanin_df)
    instances = find_fanin(in_idx, cfg)
    assert len(instances) >= 1
    inst = instances[0]
    assert inst["motif_type"] == "fanin"
    assert 99 in inst["nodes"]
    assert len(inst["edges"]) >= cfg.R_MIN_FANIN


def test_fanin_missing_when_too_few_sources(cfg):
    """Chỉ 2 nguồn, R_MIN_FANIN=3 → không match."""
    df = _make_event_df([
        {"step": 1, "src": 1, "dst": 99, "amount": 100.0, "is_laundering": 0},
        {"step": 2, "src": 2, "dst": 99, "amount":  90.0, "is_laundering": 0},
    ])
    _, in_idx, _ = build_event_indexes(df)
    instances = find_fanin(in_idx, cfg)
    assert len(instances) == 0


def test_fanout_finds_instance(fanout_df, cfg):
    out_idx, _, _ = build_event_indexes(fanout_df)
    instances = find_fanout(out_idx, cfg)
    assert len(instances) >= 1
    assert instances[0]["motif_type"] == "fanout"
    assert 10 in instances[0]["nodes"]


def test_cycle3_finds_instance(cycle3_df, cfg):
    out_idx, _, _ = build_event_indexes(cycle3_df)
    instances = find_cycle3(out_idx, cfg)
    assert len(instances) >= 1
    inst = instances[0]
    assert inst["motif_type"] == "cycle3"
    # Node đầu và cuối giống nhau (vòng tròn)
    assert inst["nodes"][0] == inst["nodes"][-1]
    assert len(inst["edges"]) == 3


def test_cycle3_not_found_when_delta_too_small(cycle3_df):
    """DELTA=0 → không có cycle nào."""
    cfg_strict = MotifConfig()
    cfg_strict.DELTA = 0
    out_idx, _, _ = build_event_indexes(cycle3_df)
    instances = find_cycle3(out_idx, cfg_strict)
    assert len(instances) == 0


def test_relay4_finds_instance(relay4_df, cfg):
    out_idx, _, _ = build_event_indexes(relay4_df)
    instances = find_relay4(out_idx, cfg)
    assert len(instances) >= 1
    inst = instances[0]
    assert inst["motif_type"] == "relay4"
    assert len(set(inst["nodes"])) == 4  # 4 node khác nhau
    assert len(inst["edges"]) == 3


def test_relay4_amount_ratio_pruning(cfg):
    """Amount thay đổi quá lớn → bị prune."""
    df = _make_event_df([
        {"step": 1, "src": 1, "dst": 2, "amount": 1000.0, "is_laundering": 0},
        {"step": 2, "src": 2, "dst": 3, "amount":    1.0, "is_laundering": 0},  # ratio << RHO_MIN
        {"step": 3, "src": 3, "dst": 4, "amount":    1.0, "is_laundering": 0},
    ])
    out_idx, _, _ = build_event_indexes(df)
    instances = find_relay4(out_idx, cfg)
    assert len(instances) == 0


def test_split_merge_finds_instance(split_merge_df, cfg):
    out_idx, in_idx, _ = build_event_indexes(split_merge_df)
    instances = find_split_merge(out_idx, in_idx, cfg)
    assert len(instances) >= 1
    inst = instances[0]
    assert inst["motif_type"] == "split_merge"
    # Phải có 4 node: U, V1, V2, Z
    assert len(set(inst["nodes"])) == 4


# ── Instance schema tests ─────────────────────────────────────────────────

def test_instance_schema(cycle3_df, cfg):
    """Mọi instance phải có đủ keys bắt buộc."""
    out_idx, _, _ = build_event_indexes(cycle3_df)
    instances = find_cycle3(out_idx, cfg)
    assert instances
    inst = instances[0]
    for key in ("motif_type", "nodes", "edges", "steps", "amounts", "lags", "n_alert"):
        assert key in inst, f"Thiếu key: {key}"
    assert len(inst["lags"]) == len(inst["steps"]) - 1


# ── Scoring tests ─────────────────────────────────────────────────────────

def test_count_support(fanin_df, fanout_df, cfg):
    _, in_idx, _ = build_event_indexes(fanin_df)
    fi = find_fanin(in_idx, cfg)
    out_idx, _, _ = build_event_indexes(fanout_df)
    fo = find_fanout(out_idx, cfg)

    support = count_support(fi + fo)
    assert "fanin"  in support
    assert "fanout" in support
    assert support["fanin"]  >= 1
    assert support["fanout"] >= 1


# ── Feature tests ─────────────────────────────────────────────────────────

def test_entity_features_schema(cycle3_df, relay4_df, cfg):
    out_idx1, _, _ = build_event_indexes(cycle3_df)
    out_idx2, _, _ = build_event_indexes(relay4_df)
    instances = find_cycle3(out_idx1, cfg) + find_relay4(out_idx2, cfg)

    feat = build_entity_motif_features(instances)
    assert not feat.empty
    for col in ("node", "motif_type", "count", "avg_amount", "avg_lag"):
        assert col in feat.columns, f"Thiếu column: {col}"
    assert feat["count"].min() >= 1


def test_entity_feature_wide_pivots(cycle3_df, cfg):
    out_idx, _, _ = build_event_indexes(cycle3_df)
    instances = find_cycle3(out_idx, cfg)
    feat = build_entity_motif_features(instances)
    wide = build_entity_feature_wide(feat)
    assert "node" in wide.columns
    # Phải có ít nhất 1 cột feature từ cycle3
    feature_cols = [c for c in wide.columns if c != "node"]
    assert len(feature_cols) >= 1
    # Không có NaN sau fill
    assert not wide.isnull().any().any()


def test_window_features_schema(relay4_df, cfg):
    out_idx, _, _ = build_event_indexes(relay4_df)
    instances = find_relay4(out_idx, cfg)
    wfeat = build_window_motif_features(instances, window_size=7)
    assert not wfeat.empty
    for col in ("window_start", "motif_type", "count", "total_amount"):
        assert col in wfeat.columns


def test_empty_instances_returns_empty_df():
    feat = build_entity_motif_features([])
    assert feat.empty or len(feat) == 0