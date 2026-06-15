from google.colab import drive
drive.mount('/content/drive')


import sys
import os
from pathlib import Path

# Set the data directory to the specified Google Drive path
DATA_DIR = Path('/content/drive/MyDrive/AML/dataset/')

# Removed ROOT and related sys.path modifications as DATA_DIR is now absolute

print('DATA_DIR:', DATA_DIR)
print('features_nodes:', (DATA_DIR / 'features_nodes.csv').exists())
print('reduced_features_nodes:', (DATA_DIR / 'reduced_features_nodes.csv').exists())

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




df_nodes = pd.read_csv(DATA_DIR / 'features_nodes.csv')

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



# =====================================================================
# TRÍCH XUẤT GIAO DỊCH LIÊN QUAN TỚI CORE ACCOUNTS
# =====================================================================
import polars as pl
import time

start_time = time.time()

# ------------------------------------------------------------------
# 1. Danh sách 6,250 core accounts từ RM-2/RM-3
# ------------------------------------------------------------------
core_series = pl.Series("core", current_df["account"].tolist())

print(f"Core accounts : {len(core_series):,}")

# ------------------------------------------------------------------
# 2. Load tx_log và đánh edge_id TRƯỚC — đây là ID tuyệt đối
#    edge_id = số thứ tự dòng trong file gốc, bắt đầu từ 0
# ------------------------------------------------------------------
RAW_PATH = DATA_DIR / 'tx_log.csv'
df_raw = (
    pl.read_csv(RAW_PATH)
    .with_row_index(name="edge_id", offset=0)   # đánh ID toàn bộ trước
)

print(f"Tổng edges (raw)  : {df_raw.height:,}")

# ------------------------------------------------------------------
# 3. Lọc: giữ edge nếu nameOrig HOẶC nameDest thuộc core accounts
#    Không filter type, không filter bank — lấy toàn bộ
# ------------------------------------------------------------------
subgraph_edges = df_raw.filter(
    pl.col("nameOrig").is_in(core_series) | pl.col("nameDest").is_in(core_series)
)

# ------------------------------------------------------------------
# 4. Lấy toàn bộ nodes xuất hiện trong subgraph (cả 2 chiều)
# ------------------------------------------------------------------
subgraph_nodes = (
    pl.concat([
        subgraph_edges.select(pl.col("nameOrig").alias("account")),
        subgraph_edges.select(pl.col("nameDest").alias("account")),
    ])
    .unique()
)

# ------------------------------------------------------------------
# 5. Thống kê
# ------------------------------------------------------------------
sar_total    = df_raw["isSAR"].sum()
sar_retained = subgraph_edges["isSAR"].sum()

print(f"\n{'='*55}")
print(f"  {'Metric':<28} {'Before':>10}  {'After':>10}")
print(f"  {'-'*48}")
print(f"  {'Edges':<28} {df_raw.height:>10,}  {subgraph_edges.height:>10,}")
print(f"  {'isSAR=1 edges':<28} {sar_total:>10,}  {sar_retained:>10,}")
print(f"  {'isSAR rate':<28} {sar_total/df_raw.height*100:>9.2f}%  "
      f"{sar_retained/subgraph_edges.height*100:>9.2f}%")
print(f"  {'SAR recall':<28} {'100.00%':>10}  "
      f"{sar_retained/sar_total*100:>9.1f}%")
print(f"  {'Nodes':<28} {'~99,996':>10}  {subgraph_nodes.height:>10,}")
print(f"{'='*55}")
print(f"\n⏱  Hoàn thành trong: {time.time() - start_time:.2f}s")

# ------------------------------------------------------------------
# 6. Lưu
# ------------------------------------------------------------------
out_edges_path = DATA_DIR / 'subgraph_1hop_edges.parquet'
out_nodes_path = DATA_DIR / 'subgraph_1hop_nodes.parquet'

subgraph_edges.write_parquet(out_edges_path)
subgraph_nodes.write_parquet(out_nodes_path)

print(f"\n✅  Edges → {out_edges_path}  {subgraph_edges.shape}")
print(f"✅  Nodes → {out_nodes_path}  {subgraph_nodes.shape}")


# Load the subgraph edges data
subgraph_edges_path = DATA_DIR / 'subgraph_1hop_edges.parquet'
df_subgraph_edges = pl.read_parquet(subgraph_edges_path)

print(f"Loaded 1-Hop Edges: {df_subgraph_edges.shape[0]:,} rows, {df_subgraph_edges.shape[1]} columns")

print("\n--- Head of the DataFrame ---")
print(df_subgraph_edges.head())

print("\n--- Schema of the DataFrame ---")
print(df_subgraph_edges.schema)



print("\n--- Descriptive Statistics ---")
print(df_subgraph_edges.describe())

print("\n--- Missing Values ---")
# Polars' null_count() method returns a DataFrame with null counts for each column
print(df_subgraph_edges.null_count())


import matplotlib.pyplot as plt
import seaborn as sns

# Convert to pandas for easier plotting with seaborn/matplotlib
df_subgraph_edges_pd = df_subgraph_edges.to_pandas()

# Distribution of 'amount'
plt.figure(figsize=(10, 6))
sns.histplot(df_subgraph_edges_pd['amount'], bins=50, kde=True)
plt.title('Distribution of Transaction Amount in 1-Hop Subgraph')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.yscale('log') # Use log scale for y-axis if distribution is heavily skewed
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# Value counts for 'type' (transaction type)
if 'type' in df_subgraph_edges_pd.columns:
    print("\n--- Value Counts for Transaction Type ---")
    type_counts = df_subgraph_edges_pd['type'].value_counts()
    print(type_counts)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=type_counts.index, y=type_counts.values, palette='viridis')
    plt.title('Distribution of Transaction Types')
    plt.xlabel('Transaction Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
else:
    print("Column 'type' not found for value count analysis.")


# Time series trend for 'step' (if applicable)
if 'step' in df_subgraph_edges_pd.columns:
    plt.figure(figsize=(12, 6))
    df_subgraph_edges_pd['step'].value_counts().sort_index().plot(kind='line')
    plt.title('Number of Transactions per Step')
    plt.xlabel('Step')
    plt.ylabel('Number of Transactions')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
else:
    print("Column 'step' not found for time series trend analysis.")



import polars as pl
import time

print("=" * 55)
print("  PHÂN TÍCH BASELINE: SỐ LƯỢNG CẠNH SAR GỐC")
print("=" * 55)

start_time = time.time()

# 1. Tải dữ liệu gốc từ HuggingFace
print("Đang tải dữ liệu gốc...")
RAW_PATH = DATA_DIR / 'tx_log.csv'
df_raw = pl.read_csv(RAW_PATH) # Changed to pl.read_csv directly

# 2. Đếm SAR trên TOÀN BỘ dữ liệu thô (Bao gồm cả dòng tiền từ hệ thống/Deposit)
total_raw_edges = df_raw.height
sar_raw_edges = df_raw.filter(pl.col("isSAR") == 1).height

print(f"\n[1] TRÊN TẬP DỮ LIỆU THÔ TUYỆT ĐỐI (Toàn bộ loại giao dịch):")
print(f"    - Tổng số cạnh : {total_raw_edges:>10,}")
print(f"    - Số cạnh SAR  : {sar_raw_edges:>10,} ({(sar_raw_edges/total_raw_edges*100):.3f}%)の結果です。")

# 3. Đếm SAR trên TẬP MẠNG LƯỚI GIAO DỊCH (Chỉ lấy TRANSFER, bỏ source/sink)
net = df_raw.filter(
    (pl.col("bankOrig") != "source") &
    (pl.col("bankDest") != "sink") &
    (pl.col("type") == "TRANSFER")
)

total_net_edges = net.height
sar_net_edges = net.filter(pl.col("isSAR") == 1).height

print(f"\n[2] TRÊN TẬP MẠNG LƯỚI LÕI (Chỉ tính TRANSFER giữa người với người):")
print(f"    - Tổng số cạnh : {total_net_edges:>10,}")
print(f"    - Số cạnh SAR  : {sar_net_edges:>10,} ({(sar_net_edges/total_net_edges*100):.3f}%)の結果です。")
print("=" * 55)
print(f"⏱ Hoàn thành trong : {time.time() - start_time:.2f} giây.")