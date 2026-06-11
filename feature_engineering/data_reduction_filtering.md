import sys
import os
from pathlib import Path

# Setup path
ROOT = Path(os.getcwd()).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print('ROOT:', ROOT)
print('features_nodes:', (ROOT / 'data' / 'features_nodes.csv').exists())
print('reduced_features_nodes:', (ROOT / 'reduced_features_nodes.csv').exists())

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

pd.set_option("display.float_format", "{:,.4f}".format)

RANDOM_STATE = 42
TARGET_FRAC  = 0.12   # dừng khi remaining / original <= 12%
DROP_FRAC    = 0.50   # drop bottom 50% mỗi vòng



df_nodes = pd.read_csv(ROOT / 'data' / 'features_nodes.csv')

print(f"Accounts : {len(df_nodes):,}")
print(f"Features : {df_nodes.shape[1] - 2}")
print(f"SAR=1    : {df_nodes['is_sar'].sum():,}  ({df_nodes['is_sar'].mean()*100:.2f}%)")
df_nodes.head(3)


NON_FEATURE_COLS = ["account", "is_sar"]
FEATURE_COLS = [c for c in df_nodes.columns if c not in NON_FEATURE_COLS]

print(f"Training on {len(FEATURE_COLS)} features.")


XGB_PARAMS = dict(
    n_estimators     = 150,
    max_depth        = 4,
    learning_rate    = 0.1,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    eval_metric      = "logloss",
    random_state     = RANDOM_STATE,
    n_jobs           = -1,
    tree_method      = "hist",
)


current_df = df_nodes.copy()
original_n = len(df_nodes)
iteration  = 0
history    = []

print(f"Target  : ≤ {TARGET_FRAC:.0%} of original  ({int(original_n * TARGET_FRAC):,} accounts)")
print(f"Drop    : bottom {DROP_FRAC:.0%} each round")
print(f"{'Iter':>4}  {'Before':>8}  {'After':>8}  {'SAR before':>10}  {'SAR after':>9}  {'AUC':>6}  {'% of orig':>9}")
print("-" * 70)

while len(current_df) / original_n > TARGET_FRAC:
    iteration += 1
    n_before   = len(current_df)
    sar_before = current_df["is_sar"].sum()

    X = current_df[FEATURE_COLS].fillna(0).values
    y = current_df["is_sar"].values

    # Xử lý class imbalance
    pos = y.sum()
    neg = (y == 0).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    model = XGBClassifier(scale_pos_weight=scale_pos_weight, **XGB_PARAMS)
    model.fit(X, y, verbose=False)

    scores    = model.predict_proba(X)[:, 1]
    threshold = np.percentile(scores, DROP_FRAC * 100)
    keep_mask = scores >= threshold

    current_df = current_df[keep_mask].copy()
    n_after    = len(current_df)
    sar_after  = current_df["is_sar"].sum()
    frac_orig  = n_after / original_n

    try:
        auc = roc_auc_score(y, scores)
    except Exception:
        auc = float("nan")

    history.append({
        "iteration" : iteration,
        "n_before"  : n_before,
        "n_after"   : n_after,
        "sar_before": sar_before,
        "sar_after" : sar_after,
        "auc"       : auc,
        "frac_orig" : frac_orig,
    })

    print(f"{iteration:>4}  {n_before:>8,}  {n_after:>8,}  "
          f"{sar_before:>10,}  {sar_after:>9,}  {auc:>6.4f}  {frac_orig:>8.1%}")

print("-" * 70)
print(f"\n✅ Done after {iteration} iterations.")
print(f"   Remaining : {len(current_df):,} accounts  ({len(current_df)/original_n:.1%} of original)")
print(f"   SAR kept  : {current_df['is_sar'].sum():,} / {df_nodes['is_sar'].sum():,}"
      f"  ({current_df['is_sar'].sum()/df_nodes['is_sar'].sum()*100:.1f}% recall)")


hist_df = pd.DataFrame(history)
hist_df["sar_recall_%"]   = (hist_df["sar_after"]  / df_nodes["is_sar"].sum() * 100).round(1)
hist_df["acct_retained_%"] = (hist_df["frac_orig"] * 100).round(1)
hist_df["auc"] = hist_df["auc"].round(4)

print(hist_df[[
    "iteration", "n_before", "n_after",
    "sar_before", "sar_after", "sar_recall_%",
    "acct_retained_%", "auc"
]].to_string(index=False))


fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Left: accounts remaining
ax = axes[0]
ax.plot(hist_df["iteration"], hist_df["n_after"],
        marker="o", color="#2563eb", linewidth=2, label="Accounts remaining")
ax.axhline(original_n * TARGET_FRAC, color="red",
           linestyle="--", label=f"{TARGET_FRAC:.0%} target")
ax.set_title("Accounts Remaining per Iteration", fontsize=13)
ax.set_xlabel("Iteration")
ax.set_ylabel("# Accounts")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend()
ax.grid(alpha=0.3)

# Right: SAR recall vs account retention
ax2 = axes[1]
ax2.plot(hist_df["iteration"], hist_df["sar_recall_%"],
         marker="s", color="#16a34a", linewidth=2, label="SAR recall %")
ax2.plot(hist_df["iteration"], hist_df["acct_retained_%"],
         marker="^", color="#dc2626", linewidth=2,
         linestyle="--", label="Account retention %")
ax2.set_title("SAR Recall vs Account Retention", fontsize=13)
ax2.set_xlabel("Iteration")
ax2.set_ylabel("% of Original")
ax2.legend()
ax2.grid(alpha=0.3)

plt.suptitle("Recursive XGBoost Reduction (RM-2 / RM-3)", fontsize=14, fontweight="bold")
plt.tight_layout()
# plt.savefig("data/reduction_summary.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved → data/reduction_summary.png")


total_final  = len(current_df)
sar_original = df_nodes["is_sar"].sum()
sar_final    = current_df["is_sar"].sum()

print("=" * 55)
print("  REDUCTION SUMMARY")
print("=" * 55)
print(f"  {'Stage':<22} {'Accounts':>8}   {'SAR=1':>5}   {'SAR%':>6}")
print(f"  {'-'*50}")
print(f"  {'Original':<22} {original_n:>8,}   {sar_original:>5}   {sar_original/original_n*100:.2f}%")
print(f"  {'After RM-2/RM-3':<22} {total_final:>8,}   {sar_final:>5}   {sar_final/total_final*100:.2f}%")
print("=" * 55)
print(f"  Total reduction : {1 - total_final/original_n:.1%}")
print(f"  SAR recall      : {sar_final/sar_original*100:.1f}%")
print("=" * 55)


current_df.to_csv(OUT_PATH, index=False)
print(f"✅  Saved → {OUT_PATH}")
print(f"   Shape  : {current_df.shape}")
print(f"   SAR=1  : {current_df['is_sar'].sum():,}  ({current_df['is_sar'].mean()*100:.2f}%)")


# =====================================================================
# 🚀 TRÍCH XUẤT CẠNH (1-HOP) & ÁNH XẠ RAW ID TUYỆT ĐỐI
# =====================================================================
import polars as pl
import time

print("\n" + "=" * 65)
print("  BẮT ĐẦU TRÍCH XUẤT GIAO DỊCH (1-HOP)")
print("=" * 65)

start_time = time.time()

# 1. Trích xuất danh sách 6,250 Core Nodes từ bước trước
core_accounts = current_df["account"].tolist()
core_series = pl.Series("core", core_accounts)

# 2. Tải dữ liệu GỐC (tx_log.csv)
RAW_PATH = DATA_DIR / 'tx_log.csv' # Use DATA_DIR and .csv extension
print(f"Đang tải dữ liệu từ: {RAW_PATH}")
df_raw = pl.read_csv(RAW_PATH) # Use pl.read_csv directly

# ---------------------------------------------------------------------
# CHÚ Ý: ĐỊNH DANH TRÊN TOÀN BỘ BẢNG RAW CHƯA QUA XỬ LÝ
# ---------------------------------------------------------------------
# Lệnh này đảm bảo edge_id chính là số thứ tự dòng (Index) trong file gốc ban đầu.
# Nếu edge_id = 500,000, anh kéo xuống đúng dòng 500,000 trong file raw là thấy nó.
df_raw = df_raw.with_row_index(name="edge_id", offset=0)

# 3. Áp dụng bộ lọc nghiệp vụ
net = df_raw.filter(
    (pl.col("bankOrig") != "source") &
    (pl.col("bankDest") != "sink") &
    (pl.col("type") == "TRANSFER")
)
original_edges_count = net.height
print(f"Tổng số cạnh ban đầu       : {original_edges_count:,}")

# 4. Quét mạng lưới 1-Hop
print("Đang quét mạng lưới để tìm hàng xóm 1-Hop...")
subgraph_edges = net.filter(
    pl.col("nameOrig").is_in(core_series) | pl.col("nameDest").is_in(core_series)
)
retained_edges_count = subgraph_edges.height

# 5. Toàn bộ Nodes trong Subgraph (re-introducing this calculation)
subgraph_nodes = (
    pl.concat([
        subgraph_edges.select(pl.col("nameOrig").alias("account")),
        subgraph_edges.select(pl.col("nameDest").alias("account"))
    ])
    .unique()
)
retained_nodes_count = subgraph_nodes.height

print("\n" + "=" * 55)
print("  KẾT QUẢ ĐỒ THỊ 1-HOP (INDUCED SUBGRAPH)")
print("=" * 55)
print(f"  Nodes ban đầu (ước tính) : ~99,996")
print(f"  Nodes sau lọc XGBoost    : {len(core_accounts):,} (Core)")
print(f"  Nodes trong 1-Hop Graph  :   {retained_nodes_count:>8,} (Tăng {(retained_nodes_count/len(core_accounts)):.1f} lần so với Core)")
print(f"  {'-'*50}")
print(f"  Edges ban đầu            :  {original_edges_count:>8,}")
print(f"  Edges trong 1-Hop Graph  :  {retained_edges_count:>8,}")
print(f"  Tỷ lệ Edges giữ lại      : {(retained_edges_count / original_edges_count * 100):.2f}%")
print("=" * 55)

if retained_nodes_count > 60000:
    print("⚠️  BÁO ĐỘNG ĐỎ: Đồ thị 1-hop phình lên quá 60% kích thước gốc.")

# 6. Lưu subgraph (saving subgraph_edges and subgraph_nodes for EDA)
out_edges_path = DATA_DIR / 'subgraph_1hop_edges.parquet'
out_nodes_path = DATA_DIR / 'subgraph_1hop_nodes.parquet'

subgraph_edges.write_parquet(out_edges_path)
subgraph_nodes.write_parquet(out_nodes_path)

print(f"\n✅ Đã lưu Edges 1-Hop : {out_edges_path}")
print(f"✅ Đã lưu Nodes 1-Hop : {out_nodes_path}")
print(f"⏱ Hoàn thành trong   : {time.time() - start_time:.2f} giây.")