# tests/test_graph_smoke.py
# Smoke test cho src/graph/ — không cần GPU thật.
# Dùng pandas DataFrame giả lập để test logic cột và schema.
#
# Chạy trên Colab (có GPU):
#   !python -m pytest tests/test_graph_smoke.py -v
#
# Chạy local (không cần GPU — chỉ test import và Config):
#   python -m pytest tests/test_graph_smoke.py -v -k "not gpu"

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Test 1: Config ────────────────────────────────────────────────────────

def test_config_defaults():
    """Config khởi tạo được và giữ đúng default values."""
    from src.graph.config import Config
    cfg = Config()
    assert cfg.DELTA_W == 5
    assert cfg.S_MAX == 50
    assert cfg.RESOLUTION == 1.0
    assert cfg.WINDOW_SIZE == 5
    assert cfg.STRIDE == 2
    assert cfg.TRACKING_MEMORY == 3
    assert abs(cfg.W_C2 + cfg.W_C3 + cfg.W_VELOCITY + cfg.W_ALERT + cfg.W_STRUCTURE - 1.0) < 1e-6


def test_config_override():
    """Config có thể override từng field."""
    from src.graph.config import Config
    cfg = Config()
    cfg.DELTA_W = 10
    assert cfg.DELTA_W == 10


# ── Test 2: validate_schema ───────────────────────────────────────────────

def test_validate_schema_passes():
    """validate_schema không raise khi đủ cột."""
    import pandas as pd
    from src.graph.loader import validate_schema

    class FakeDF:
        columns = ["step", "src", "dst", "amount", "is_laundering", "extra_col"]

    validate_schema(FakeDF())  # không raise


def test_validate_schema_raises_on_missing():
    """validate_schema raise ValueError khi thiếu cột."""
    import pytest
    from src.graph.loader import validate_schema

    class FakeDF:
        columns = ["step", "src", "dst"]  # thiếu amount và is_laundering

    with pytest.raises(ValueError, match="Thiếu cột"):
        validate_schema(FakeDF())


# ── Test 3: GPU tests (chỉ chạy khi có cuDF) ─────────────────────────────

def _has_cudf():
    try:
        import cudf  # noqa: F401
        return True
    except ImportError:
        return False


import pytest

@pytest.mark.skipif(not _has_cudf(), reason="cuDF không có, bỏ qua GPU tests")
def test_build_node_map_gpu():
    """build_node_map tạo đúng số node và kiểu int32."""
    import cudf
    from src.graph.encoder import build_node_map

    df = cudf.DataFrame({
        "src": ["A", "B", "A", "C"],
        "dst": ["B", "C", "C", "A"],
    })
    node_map = build_node_map(df)

    assert "node_name" in node_map.columns
    assert "node_id"   in node_map.columns
    assert len(node_map) == 3  # A, B, C
    assert str(node_map["node_id"].dtype) == "int32"


@pytest.mark.skipif(not _has_cudf(), reason="cuDF không có, bỏ qua GPU tests")
def test_encode_nodes_gpu():
    """encode_nodes thay src/dst bằng int32 node_id."""
    import cudf
    from src.graph.encoder import build_node_map, encode_nodes

    df = cudf.DataFrame({
        "src":           ["A", "B", "C"],
        "dst":           ["B", "C", "A"],
        "step":          [1, 2, 3],
        "amount":        [100.0, 200.0, 300.0],
        "is_laundering": [0, 1, 0],
    })
    node_map  = build_node_map(df)
    df_enc    = encode_nodes(df, node_map)

    assert "src" in df_enc.columns
    assert "dst" in df_enc.columns
    assert str(df_enc["src"].dtype) == "int32"
    assert str(df_enc["dst"].dtype) == "int32"
    # Không còn string trong src/dst
    assert df_enc["src"].min() >= 0
    assert df_enc["dst"].min() >= 0


@pytest.mark.skipif(not _has_cudf(), reason="cuDF không có, bỏ qua GPU tests")
def test_create_temporal_graph_output_schema():
    """create_temporal_graph trả về đúng columns."""
    import cudf
    from src.graph.config import Config
    from src.graph.temporal import create_temporal_graph

    # Tạo dữ liệu nhỏ: A→B rồi B→C trong cùng DELTA_W
    df = cudf.DataFrame({
        "src":           [0, 1, 2],
        "dst":           [1, 2, 3],
        "step":          [1, 2, 3],
        "amount":        [100.0, 200.0, 300.0],
        "is_laundering": [0,     1,     0],
    })
    cfg = Config()
    cfg.DELTA_W = 5

    temporal_edges = create_temporal_graph(df, cfg)

    expected_cols = {
        "src_1", "dst_1", "step_1", "amount_1", "alert_1",
        "src_2", "dst_2", "step_2", "amount_2", "alert_2",
    }
    assert expected_cols.issubset(set(temporal_edges.columns))
    # A→B và B→C tạo 1 cạnh thời gian: (A→B) → (B→C)
    assert len(temporal_edges) >= 1


@pytest.mark.skipif(not _has_cudf(), reason="cuDF không có, bỏ qua GPU tests")
def test_create_second_order_graph_output_schema():
    """create_second_order_graph trả về đúng columns."""
    import cudf
    from src.graph.config import Config
    from src.graph.temporal import create_temporal_graph
    from src.graph.second_order import create_second_order_graph

    df = cudf.DataFrame({
        "src":           [0, 1, 2],
        "dst":           [1, 2, 3],
        "step":          [1, 2, 3],
        "amount":        [100.0, 200.0, 300.0],
        "is_laundering": [0,     1,     0],
    })
    cfg = Config()
    cfg.DELTA_W = 5

    temporal_edges     = create_temporal_graph(df, cfg)
    second_order_edges = create_second_order_graph(temporal_edges, cfg)

    expected_cols = {
        "src_2nd", "dst_2nd", "count",
        "total_amount_src", "total_amount_dst",
        "avg_time_gap", "n_alert", "_max_node_id",
    }
    assert expected_cols.issubset(set(second_order_edges.columns))