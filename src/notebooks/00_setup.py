# notebooks/00_setup.py
# Dán từng cell vào Colab notebook.
# Chạy file này một lần ở đầu mỗi session.

# ============================================================
# CELL 1 — Cài thư viện (chỉ chạy 1 lần, sau đó restart runtime)
# ============================================================
"""
!pip install -q cudf-cu12 cugraph-cu12 --extra-index-url=https://pypi.nvidia.com
!pip install -q cupy-cuda12x
!pip install -q python-igraph tqdm pyarrow networkx matplotlib seaborn

import importlib
for pkg in ["cudf", "cugraph", "cupy"]:
    assert importlib.util.find_spec(pkg), f"Missing: {pkg}. Bật T4 GPU trước."
print("✅ GPU stack sẵn sàng.")
"""

# ============================================================
# CELL 2 — Mount Drive + sys.path (chạy mỗi session)
# ============================================================
"""
from google.colab import drive
drive.mount('/content/drive')

import sys
PROJECT_ROOT = "/content/drive/MyDrive/aml-graph-mining"  # ← đổi nếu cần
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print(f"✅ Project root: {PROJECT_ROOT}")
"""

# ============================================================
# CELL 3 — Smoke test imports (xác nhận src/ hoạt động)
# ============================================================
"""
from src.graph.config import Config
from src.graph.loader import load_raw, validate_schema
from src.graph.encoder import build_node_map, encode_nodes
from src.graph.temporal import create_temporal_graph
from src.graph.second_order import create_second_order_graph

cfg = Config()
print("Config OK")
print(f"  DELTA_W={cfg.DELTA_W}, S_MAX={cfg.S_MAX}, RESOLUTION={cfg.RESOLUTION}")
print("✅ Tất cả imports thành công.")
"""