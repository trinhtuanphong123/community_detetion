from google.colab import drive
drive.mount('/content/drive')


import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
from pathlib import Path


df = pl.read_csv('/content/drive/MyDrive/AML/dataset/tx_log.csv')
print(f"Shape: {df.shape}")
print(df.schema)


import polars as pl

# Base: tất cả accounts (loại source/sink)
net = df.filter(
    (pl.col("bankOrig") != "source") &
    (pl.col("bankDest") != "sink") &
    (pl.col("type") == "TRANSFER")
)

# Tất cả unique accounts
all_accounts = (
    pl.concat([
        net.select(pl.col("nameOrig").alias("account")),
        net.select(pl.col("nameDest").alias("account")),
    ])
    .unique()
)

# SAR label: account là SAR nếu tham gia bất kỳ SAR transaction nào
sar_labels = (
    pl.concat([
        net.select(pl.col("nameOrig").alias("account"), pl.col("isSAR")),
        net.select(pl.col("nameDest").alias("account"), pl.col("isSAR")),
    ])
    .group_by("account")
    .agg(pl.col("isSAR").max().alias("is_sar"))
)

base = all_accounts.join(sar_labels, on="account", how="left").fill_null(0)

print(base.shape)
print(base["is_sar"].value_counts())



# ── RM-1 Filter ──────────────────────────────────────────────
# Tính trên raw (bỏ INITIALBALANCE), KHÔNG phải trên net
df_clean = df.filter(pl.col("type") != "INITIALBALANCE")

# F2: tx_count > 1000 (tính trên ALL types)
tx_count_raw = df_clean.group_by("nameOrig").agg(pl.len().alias("tx_count"))
flag_f2 = set(
    tx_count_raw.filter(pl.col("tx_count") > 1000)["nameOrig"].to_list()
)

# F3: unique counterparty > 50 (tính trên ALL types)
unique_cp_raw = df_clean.group_by("nameOrig").agg(
    pl.col("nameDest").n_unique().alias("n_cp")
)
flag_f3 = set(
    unique_cp_raw.filter(pl.col("n_cp") > 50)["nameOrig"].to_list()
)

accounts_to_remove = flag_f2 | flag_f3

print(f"Flagged by F2 (tx > 1000) : {len(flag_f2):,}")
print(f"Flagged by F3 (cp > 50)   : {len(flag_f3):,}")
print(f"Total to remove           : {len(accounts_to_remove):,}")

# Kiểm tra SAR bị loại
sar_accounts = set(
    df_clean.filter(pl.col("isSAR") == 1)["nameOrig"].unique().to_list()
)
print(f"SAR bị loại               : {len(accounts_to_remove & sar_accounts):,}")

# Drop khỏi net
net = net.filter(
    ~pl.col("nameOrig").is_in(list(accounts_to_remove)) &
    ~pl.col("nameDest").is_in(list(accounts_to_remove))
)
print(f"\nTransactions remaining: {len(net):,}")
# ─────────────────────────────────────────────────────────────



# Thêm window bin (0-27 → W0, 28-55 → W1, 56-83 → W2, 84-111 → W3)
net = net.with_columns(
    (pl.col("step") // 28).alias("window")
)

# Outgoing features (từ góc nhìn nameOrig)
out_feats = (
    net.group_by(["nameOrig", "window"])
    .agg([
        pl.col("amount").sum().alias("out_amount_sum"),
        pl.col("amount").mean().alias("out_amount_mean"),
        pl.col("amount").median().alias("out_amount_median"),
        pl.col("amount").std().alias("out_amount_std"),
        pl.col("amount").max().alias("out_amount_max"),
        pl.col("amount").min().alias("out_amount_min"),
        pl.col("amount").count().alias("out_count"),
        pl.col("nameDest").n_unique().alias("out_count_unique"),
    ])
    .rename({"nameOrig": "account"})
)

print(out_feats.shape)
print(out_feats.head())


# Incoming features (từ góc nhìn nameDest)
in_feats = (
    net.group_by(["nameDest", "window"])
    .agg([
        pl.col("amount").sum().alias("in_amount_sum"),
        pl.col("amount").mean().alias("in_amount_mean"),
        pl.col("amount").median().alias("in_amount_median"),
        pl.col("amount").std().alias("in_amount_std"),
        pl.col("amount").max().alias("in_amount_max"),
        pl.col("amount").min().alias("in_amount_min"),
        pl.col("amount").count().alias("in_count"),
        pl.col("nameOrig").n_unique().alias("in_count_unique"),
    ])
    .rename({"nameDest": "account"})
)

print(in_feats.shape)


def pivot_window_feats(feats_df, feature_cols):
    """Pivot long (account, window, feat...) → wide (account, feat_w0, feat_w1...)"""
    windows = [0, 1, 2, 3]
    result = None
    for w in windows:
        w_df = (
            feats_df.filter(pl.col("window") == w)
            .drop("window")
            .rename({c: f"{c}_w{w}" for c in feature_cols})
        )
        if result is None:
            result = w_df
        else:
            result = result.join(w_df, on="account", how="full", coalesce=True)
    return result

out_feature_cols = [c for c in out_feats.columns if c not in ("account", "window")]
in_feature_cols  = [c for c in in_feats.columns  if c not in ("account", "window")]

out_wide = pivot_window_feats(out_feats, out_feature_cols)
in_wide  = pivot_window_feats(in_feats,  in_feature_cols)

print(out_wide.shape)
print(in_wide.shape)


# Left join từ base đảm bảo tất cả 99996 accounts đều có mặt; fill_null(0) cho accounts không có giao dịch trong window đó
features = (
    base
    .join(out_wide, on="account", how="left")
    .join(in_wide,  on="account", how="left")
    .fill_null(0)
)

print(features.shape)
print(features.head(3))


# Timing features cho outgoing
out_timing = (
    net.group_by(["nameOrig", "window"])
    .agg([
        pl.col("step").min().alias("out_first_step"),
        pl.col("step").max().alias("out_last_step"),
        (pl.col("step").max() - pl.col("step").min()).alias("out_time_span"),
        pl.col("step").std().alias("out_time_std"),
    ])
    .rename({"nameOrig": "account"})
)

# Timing features cho incoming
in_timing = (
    net.group_by(["nameDest", "window"])
    .agg([
        pl.col("step").min().alias("in_first_step"),
        pl.col("step").max().alias("in_last_step"),
        (pl.col("step").max() - pl.col("step").min()).alias("in_time_span"),
        pl.col("step").std().alias("in_time_std"),
    ])
    .rename({"nameDest": "account"})
)

out_timing_cols = [c for c in out_timing.columns if c not in ("account", "window")]
in_timing_cols  = [c for c in in_timing.columns  if c not in ("account", "window")]

out_timing_wide = pivot_window_feats(out_timing, out_timing_cols)
in_timing_wide  = pivot_window_feats(in_timing,  in_timing_cols)

features = (
    features
    .join(out_timing_wide, on="account", how="left")
    .join(in_timing_wide,  on="account", how="left")
    .fill_null(0)
)

print(features.shape)


# daysInBank và phoneChanges (lấy max vì các rows của cùng account có thể khác nhau)
kyc_orig = (
    net.group_by("nameOrig")
    .agg([
        pl.col("daysInBankOrig").max().alias("days_in_bank"),
        pl.col("phoneChangesOrig").max().alias("phone_changes"),
    ])
    .rename({"nameOrig": "account"})
)

kyc_dest = (
    net.group_by("nameDest")
    .agg([
        pl.col("daysInBankDest").max().alias("days_in_bank"),
        pl.col("phoneChangesDest").max().alias("phone_changes"),
    ])
    .rename({"nameDest": "account"})
)

kyc = (
    pl.concat([kyc_orig, kyc_dest])
    .group_by("account")
    .agg([
        pl.col("days_in_bank").max(),
        pl.col("phone_changes").max(),
    ])
)

# Global degree
out_deg_global = net.group_by("nameOrig").agg(pl.col("nameDest").n_unique().alias("out_degree")).rename({"nameOrig": "account"})
in_deg_global  = net.group_by("nameDest").agg(pl.col("nameOrig").n_unique().alias("in_degree")).rename({"nameDest": "account"})

# n_banks_interacted
banks = (
    pl.concat([
        net.select(pl.col("nameOrig").alias("account"), pl.col("bankDest").alias("bank")),
        net.select(pl.col("nameDest").alias("account"), pl.col("bankOrig").alias("bank")),
    ])
    .group_by("account")
    .agg(pl.col("bank").n_unique().alias("n_banks_interacted"))
)

features = (
    features
    .join(kyc,            on="account", how="left")
    .join(out_deg_global, on="account", how="left")
    .join(in_deg_global,  on="account", how="left")
    .join(banks,          on="account", how="left")
    .fill_null(0)
)

print(features.shape)



# still fixing


# n_active_windows: account active bao nhiêu windows (có ít nhất 1 giao dịch)
out_count_cols = [f"out_count_w{w}" for w in range(4)]
in_count_cols  = [f"in_count_w{w}"  for w in range(4)]

features = features.with_columns([
    sum([(pl.col(c) > 0).cast(pl.Int32) for c in out_count_cols]).alias("n_active_windows_out"),
    sum([(pl.col(c) > 0).cast(pl.Int32) for c in in_count_cols]).alias("n_active_windows_in"),
    (pl.col("out_amount_sum_w3") - pl.col("out_amount_sum_w0")).alias("volume_trend_out"),
    (pl.col("in_amount_sum_w3")  - pl.col("in_amount_sum_w0")).alias("volume_trend_in"),
])

print(features.shape)


features.write_parquet("features_nodes.parquet")
features.write_csv("features_nodes.csv")
print("Saved:", features.shape)
print("SAR rate:", features["is_sar"].mean())


flow_cols = []
for w in range(4):
    out_s = f"out_amount_sum_w{w}"
    in_s  = f"in_amount_sum_w{w}"
    flow_cols += [
        (pl.col(out_s) / (pl.col(in_s) + 1e-9)).alias(f"flow_through_ratio_w{w}"),
        (pl.col(in_s) - pl.col(out_s)).alias(f"net_flow_w{w}"),
        ((pl.col(in_s) - pl.col(out_s)) / (pl.col(in_s) + 1e-9)).alias(f"balance_retention_ratio_w{w}"),
    ]

features = features.with_columns(flow_cols)



vel_cols = []
for w in range(4):
    oc = f"out_count_w{w}"
    ic = f"in_count_w{w}"
    ts = f"out_time_span_w{w}"
    vel_cols += [
        (pl.col(oc) / (pl.col(ts) + 1)).alias(f"out_tx_rate_w{w}"),
        (pl.col(ic) / (pl.col(ts) + 1)).alias(f"in_tx_rate_w{w}"),
    ]

features = features.with_columns(vel_cols)

# busiest-step count needs a groupby — compute separately then join
out_burst = (
    net.group_by(["nameOrig", "window"])
    .agg(
        pl.col("step")
        .value_counts()
        .struct.field("count")
        .max()
        .alias("max_tx_per_step")
    )
    .rename({"nameOrig": "account"})
)
out_burst_wide = pivot_window_feats(
    out_burst, ["max_tx_per_step"]
)
features = features.join(out_burst_wide, on="account", how="left").fill_null(0)


cv_cols = []
for w in range(4):
    mean_c = f"out_amount_mean_w{w}"
    std_c  = f"out_amount_std_w{w}"
    cv_cols += [
        (pl.col(std_c) / (pl.col(mean_c) + 1e-9)).alias(f"out_amount_cv_w{w}"),
        ((pl.col(std_c) / (pl.col(mean_c) + 1e-9)) < 0.05)
        .cast(pl.Int8)
        .alias(f"is_structured_w{w}"),
    ]

features = features.with_columns(cv_cols)

fanio_cols = []
for w in range(4):
    iu = f"in_count_unique_w{w}"
    ou = f"out_count_unique_w{w}"
    fanio_cols += [
        (pl.col(iu) / (pl.col(ou) + 1)).alias(f"fan_in_ratio_w{w}"),
        (pl.col(ou) / (pl.col(iu) + 1)).alias(f"fan_out_ratio_w{w}"),
    ]

features = features.with_columns(fanio_cols)



sent_to   = net.group_by("nameOrig").agg(pl.col("nameDest").unique().alias("sent_to"))
recv_from = net.group_by("nameDest").agg(pl.col("nameOrig").unique().alias("recv_from"))

reciprocal = (
    sent_to
    .join(recv_from.rename({"nameDest": "nameOrig"}), on="nameOrig", how="left")
    .with_columns(
        pl.struct(["sent_to", "recv_from"])
        .map_elements(
            lambda s: len(
                set(s["sent_to"] or []) & set(s["recv_from"] or [])
            ),
            return_dtype=pl.Int32,
        )
        .alias("reciprocal_count")
    )
    .select([
        pl.col("nameOrig").alias("account"),
        pl.col("reciprocal_count"),
        (pl.col("reciprocal_count") / (pl.col("sent_to").list.len() + 1e-9))
        .alias("reciprocal_ratio"),
    ])
)

features = features.join(reciprocal, on="account", how="left").fill_null(0)


features = features.with_columns([
    # 2nd derivative of out volume
    (
        (pl.col("out_amount_sum_w2") - pl.col("out_amount_sum_w1"))
        - (pl.col("out_amount_sum_w1") - pl.col("out_amount_sum_w0"))
    ).alias("volume_accel_out"),
    (
        (pl.col("in_amount_sum_w2") - pl.col("in_amount_sum_w1"))
        - (pl.col("in_amount_sum_w1") - pl.col("in_amount_sum_w0"))
    ).alias("volume_accel_in"),
    # accounts that go dormant → active (layering then integration)
    (
        (pl.col("out_count_w3") > 0).cast(pl.Int8)
        * (pl.col("out_count_w0") == 0).cast(pl.Int8)
    ).alias("late_activator_out"),
    (
        (pl.col("out_count_w0") > 0).cast(pl.Int8)
        * (pl.col("out_count_w3") == 0).cast(pl.Int8)
    ).alias("early_exit_out"),
])


total_out = sum([pl.col(f"out_amount_sum_w{w}") for w in range(4)])
total_in  = sum([pl.col(f"in_amount_sum_w{w}")  for w in range(4)])

features = features.with_columns([
    total_out.alias("total_out_sum"),
    total_in.alias("total_in_sum"),
])

features = features.with_columns([
    (pl.col("total_out_sum") / (pl.col("days_in_bank") + 1))
    .alias("volume_per_bank_day"),

    (pl.col("phone_changes") / (pl.col("days_in_bank") + 1))
    .alias("phone_change_rate"),

    (pl.col("days_in_bank") < 90).cast(pl.Int8)
    .alias("is_new_account"),

    (pl.col("total_out_sum") * (pl.col("days_in_bank") < 90).cast(pl.Float64))
    .alias("new_acct_volume_interaction"),

    # global amount asymmetry
    (pl.col("total_out_sum") / (pl.col("total_in_sum") + 1e-9))
    .alias("global_amount_asymmetry"),
])


banks_sent = (
    net.group_by("nameOrig")
    .agg(pl.col("bankDest").n_unique().alias("banks_sent_to"))
    .rename({"nameOrig": "account"})
)
banks_recv = (
    net.group_by("nameDest")
    .agg(pl.col("bankOrig").n_unique().alias("banks_recv_from"))
    .rename({"nameDest": "account"})
)
bank_dir = (
    banks_sent
    .join(banks_recv, on="account", how="full", coalesce=True)
    .fill_null(0)
    .with_columns(
        (pl.col("banks_sent_to") / (pl.col("banks_recv_from") + 1))
        .alias("bank_fan_out_ratio"),
        (pl.col("banks_sent_to") > 3).cast(pl.Int8)
        .alias("is_cross_bank_distributor"),
    )
)

features = features.join(bank_dir, on="account", how="left").fill_null(0)


# Thêm cột 'id' đánh số từ 1 đến hết
features = features.with_row_index(name="id", offset=1)

# (Tùy chọn) Đưa cột 'id' lên vị trí đầu tiên cho dễ nhìn
features = features.select([
    pl.col("id"),
    pl.all().exclude("id")
])

print(features.head())

features = features.drop(["total_out_sum", "total_in_sum"])

print(features.shape)
print("SAR rate:", features["is_sar"].mean())

folder_path = '/content/drive/MyDrive/AML/dataset/'


features.write_parquet(folder_path + "features_nodes.parquet")
features.write_csv(folder_path + "features_nodes.csv")
print("Saved:", features.shape)