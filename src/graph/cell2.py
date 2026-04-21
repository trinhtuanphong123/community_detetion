#!/usr/bin/env python3
# ============================================================================
# community.py — AML Community Detection Pipeline (GPU-Accelerated)
# ============================================================================
# Triển khai đầy đủ 5 bước theo framework AMLGentex:
#   4.1  Tạo đồ thị thời gian (Temporal Graph Creation)
#   4.2  Tạo đồ thị bậc 2 (2nd Order Graph — Line Graph)
#   4.3  Tính toán trọng số (Monetary Continuity + Co-occurrence + Temporal Decay)
#   4.4  Phát hiện cộng đồng (Recursive Leiden + Relay-preserving graph)
#   4.5  Phát hiện cộng đồng bất thường (Soft Suspicion Scoring)
#
# Cải tiến theo update.md (Advisor):
#   - Trọng số tích hợp monetary continuity + time decay
#   - Giữ nguyên relay structure (không nối tắt src_1→dst_2)
#   - Recursive Leiden chống mega-community
#   - Layering role classification (role=3)
#   - Multi-window Jaccard tracking buffer
#   - Soft scoring thay binary C1/C2/C3
#
# GPU: cuDF / cuGraph (chạy trên Colab T4/A100)
# ============================================================================

import os
import gc
import pickle
import json
import importlib
import warnings

import numpy as np

# ── GPU Libraries ──
import cudf
import cugraph
import cupy as cp

from tqdm.auto import tqdm

# ── Visualization ──
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================================
# 0. CONFIGURATION
# ============================================================================

class Config:
    """Tập trung tất cả hyperparameters."""

    # ── Đường dẫn dữ liệu ──
    DATA_PATH: str = "/content/drive/MyDrive/AML/dataset/tx_log.csv"
    OUTPUT_DIR: str = "/content/drive/MyDrive/AML_outputs"

    # ── Temporal Graph (4.1) ──
    DELTA_W: int = 5         # Khoảng thời gian cho phép Δw (steps)

    # ── Weight Calculation (4.3) ──
    ALPHA: float = 1.0       # Monetary continuity penalty: exp(-α|log(a1/a2)|)
    BETA: float = 0.5        # Time decay penalty: exp(-β·Δt/Δw)
    WEIGHT_FILTER_THRESH: float = 0.01   # Lọc bỏ cạnh weight < ngưỡng
    TOP_K_NEIGHBORS: int = 10  # Layer 1 sparsification: giữ top-K neighbors

    # ── Community Detection (4.4) ──
    MIN_COMM_SIZE: int = 3
    RESOLUTION: float = 1.0
    S_MAX: int = 50               # Recursive Leiden: max community size
    RESOLUTION_MULTIPLIER: float = 1.5  # Hệ số tăng resolution khi recursive split
    MAX_RECURSION_DEPTH: int = 3  # Giới hạn đệ quy

    # ── Role Classification ──
    ROLE_THRESHOLD: float = 0.3   # Net flow ratio threshold cho source/sink
    LAYERING_CONSISTENCY: float = 0.8  # Flow consistency threshold cho layering

    # ── Anomalous Detection (4.5) — Soft Scoring ──
    ALERT_THRESH: float = 0.30
    C1_ENABLED: bool = True
    C2_ALLOC_MIN: float = 0.50
    C2_ALLOC_MAX: float = 0.99
    C3_MIN_FLOW: float = 10000.0
    GLOBAL_SUSP_THRESHOLD: float = 0.45  # Ngưỡng tổng hợp suspicion score
    # Trọng số cho từng thành phần scoring
    W_C2: float = 0.20
    W_C3: float = 0.25
    W_VELOCITY: float = 0.25
    W_ALERT: float = 0.20
    W_STRUCTURE: float = 0.10  # Structural bonus (C1 + layering)

    # ── Cross-window tracking ──
    WINDOW_SIZE: int = 5
    STRIDE: int = 2
    JACCARD_THRESH: float = 0.25
    TRACKING_MEMORY: int = 3     # So sánh với N windows gần nhất (thay vì chỉ T-1)

    # ── Visualization ──
    TOP_N_DISPLAY: int = 10
    TOP_K_EXPORT: int = 50