# src/community/config.py
# Hyperparameters cho community detection pipeline.
# Tách riêng khỏi src/graph/config.py để community module
# có thể được import và test độc lập.
# Thuần Python — không import cuDF/cuGraph.


class CommunityConfig:
    """
    Tất cả hyperparameters cho community detection.

    Các giá trị default được lấy từ cell2.py (Config class gốc).
    Thay đổi ở đây, không hardcode trong các module.
    """

    # ── Weight Calculation (bước 4.3) ──────────────────────────────────────
    # ALPHA: penalty monetary continuity — exp(-α·|log(a1/a2)|)
    ALPHA: float = 1.0

    # BETA: penalty temporal decay — exp(-β·Δt/Δw)
    BETA: float = 0.5

    # WEIGHT_FILTER_THRESH: loại bỏ cạnh bậc 2 có weight < ngưỡng
    WEIGHT_FILTER_THRESH: float = 0.01

    # TOP_K_NEIGHBORS: sparsification — chỉ giữ K cạnh mạnh nhất per node
    TOP_K_NEIGHBORS: int = 10

    # ── Community Detection (bước 4.4) ─────────────────────────────────────
    # MIN_COMM_SIZE: bỏ community có ít hơn N nodes
    MIN_COMM_SIZE: int = 3

    # RESOLUTION: Leiden resolution — tăng → nhiều community nhỏ hơn
    RESOLUTION: float = 1.0

    # S_MAX: kích thước tối đa trước khi recursive Leiden split
    S_MAX: int = 50

    # RESOLUTION_MULTIPLIER: nhân resolution mỗi lần đệ quy
    RESOLUTION_MULTIPLIER: float = 1.5

    # MAX_RECURSION_DEPTH: giới hạn độ sâu đệ quy
    MAX_RECURSION_DEPTH: int = 3

    # ── Role Classification ─────────────────────────────────────────────────
    # ROLE_THRESHOLD: |net_flow_ratio| > ngưỡng → source (1) hoặc sink (2)
    ROLE_THRESHOLD: float = 0.3

    # LAYERING_CONSISTENCY: flow_consistency > ngưỡng → layering node (3)
    LAYERING_CONSISTENCY: float = 0.8

    # ── Window / Tracking ──────────────────────────────────────────────────
    WINDOW_SIZE: int = 5       # số steps mỗi window
    STRIDE: int = 2            # bước nhảy giữa windows
    JACCARD_THRESH: float = 0.25  # ngưỡng overlap để coi là cùng community
    TRACKING_MEMORY: int = 3   # so sánh với N windows gần nhất

    # ── Suspicion Scoring (bước 4.5) ───────────────────────────────────────
    ALERT_THRESH: float = 0.30
    C2_ALLOC_MIN: float = 0.50
    C2_ALLOC_MAX: float = 0.99
    C3_MIN_FLOW: float = 10000.0
    GLOBAL_SUSP_THRESHOLD: float = 0.45

    # Trọng số scoring — tổng = 1.0
    W_C2: float = 0.20        # sink concentration
    W_C3: float = 0.25        # max single flow
    W_VELOCITY: float = 0.25  # temporal burstiness
    W_ALERT: float = 0.20     # alert ratio
    W_STRUCTURE: float = 0.10 # structural bonus (roles + layering)

    # ── Output ─────────────────────────────────────────────────────────────
    TOP_K_EXPORT: int = 50     # số community xuất ra output


__all__ = ["CommunityConfig"]