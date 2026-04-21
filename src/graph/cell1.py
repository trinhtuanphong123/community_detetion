# ============================================================
# CELL 1: Install RAPIDS (cuDF, cuGraph) + igraph for GPU Acceleration
# ============================================================
# Cài đặt bộ thư viện của NVIDIA, bỏ qua các thư viện mạng thuần CPU
!pip install -q cudf-cu12 cugraph-cu12 --extra-index-url=https://pypi.nvidia.com
# cupy cần cho Cell 3 và Cell 5 (import cupy as cp)
!pip install -q cupy-cuda12x
# python-igraph: Infomap community detection tôn trọng directed edges
# networkx, matplotlib, seaborn cần cho Cell 6 (Evaluation + Visualisation)
!pip install -q tqdm pyarrow networkx matplotlib seaborn python-igraph

import importlib
# Kiểm tra xem GPU và thư viện đã sẵn sàng chưa
for pkg in ["cudf", "cugraph"]:
    assert importlib.util.find_spec(pkg), f"Missing: {pkg}. Lỗi: Bạn chưa bật T4 GPU hoặc cài đặt thất bại."

assert importlib.util.find_spec("igraph"), "Missing: python-igraph. Cần cho Infomap directed community detection."
    
print("✅ All dependencies installed (RAPIDS + igraph). GPU is ready.")