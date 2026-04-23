# src/motif/config.py
# Hyperparameters cho motif mining.
# Thuần Python — không import pandas/cuDF.
# Tất cả ngưỡng phải configurable, không hardcode trong matchers.


class MotifConfig:
    """
    Tập trung tất cả hyperparameters cho motif mining.

    Nguyên tắc: không có ngưỡng nào mặc định là "đúng" cho mọi dataset.
    Calibrate lại DELTA, RHO_MIN, RHO_MAX theo phân phối amount thực tế.
    """

    # ── Temporal constraints ────────────────────────────────────────────────
    # DELTA: số steps tối đa giữa hai giao dịch liên tiếp trong một motif.
    # Ví dụ: DELTA=3 nghĩa là tx_j phải xảy ra trong vòng 3 steps sau tx_i.
    DELTA: int = 3

    # ── Amount ratio constraints ────────────────────────────────────────────
    # RHO_MIN / RHO_MAX: tỷ lệ amount giữa hai giao dịch liên tiếp.
    # Layering thường giữ amount gần nhau → tỷ lệ gần 1.0.
    # Smurfing/structuring thường chia nhỏ → tỷ lệ < 1.0.
    # [0.5, 2.0] là ngưỡng ban đầu, nới rộng nếu miss quá nhiều SAR.
    RHO_MIN: float = 0.5
    RHO_MAX: float = 2.0

    # ── Support threshold ───────────────────────────────────────────────────
    # R_MIN: số lần tối thiểu một motif phải xuất hiện để được coi là hợp lệ.
    # Fan-in/out: R_MIN = số nhánh tối thiểu (e.g. 3 nguồn vào 1 đích).
    # Cycle/relay: R_MIN = 1 (một lần xảy ra đã là signal).
    R_MIN_FANIN: int  = 3   # số nguồn tối thiểu cho fan-in
    R_MIN_FANOUT: int = 3   # số đích tối thiểu cho fan-out
    R_MIN_CYCLE: int  = 1
    R_MIN_RELAY: int  = 1
    R_MIN_SPLIT_MERGE: int = 1

    # ── Z-score null model ──────────────────────────────────────────────────
    # N_PERMUTATIONS: số lần shuffle để tính null distribution.
    # 20–50 là đủ cho bước đầu; tăng lên 100+ khi cần confidence cao hơn.
    N_PERMUTATIONS: int = 30

    # Z_MIN: ngưỡng z-score để coi motif là statistically significant.
    Z_MIN: float = 2.0

    # ── Search limits ───────────────────────────────────────────────────────
    # MAX_NODES: số node tối đa trong một motif instance.
    MAX_NODES: int = 4

    # MAX_EDGES: số cạnh tối đa trong một motif instance.
    MAX_EDGES: int = 5

    # ── Window sizes để search ──────────────────────────────────────────────
    # WINDOW_STEPS: các kích thước cửa sổ (steps) dùng khi chạy windowed search.
    # Tương ứng 1d / 3d / 7d / 30d nếu step = 1 ngày.
    WINDOW_STEPS: list = None

    def __init__(self):
        if self.WINDOW_STEPS is None:
            self.WINDOW_STEPS = [1, 3, 7, 30]


__all__ = ["MotifConfig"]