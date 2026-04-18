import pandas as pd
import numpy as np
import argparse

import os
# ===============================
# 路径 & 命令行参数
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "out")
os.makedirs(OUT_DIR, exist_ok=True)

parser = argparse.ArgumentParser(description="Build rule features from transaction data")
parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to input transaction CSV"
)

args = parser.parse_args()

INPUT_CSV = args.input
OUTPUT_CSV = os.path.join(OUT_DIR, "txn_with_rule_features.csv")

TIME_FMT = "%Y/%m/%d %H:%M"
TARGET_N = 50000
RANDOM_STATE = 42

# ===============================
# 1. 读取 + 抽样（只读一次）
# ===============================
print("[1] Loading raw data...")
df = pd.read_csv(INPUT_CSV)
print(f"[OK] Total rows loaded: {len(df)}")
if len(df) > TARGET_N:
    df = df.sample(n=TARGET_N, random_state=RANDOM_STATE)

print(f"[OK] Using rows: {len(df)}")

# ===============================
# 2. 字段标准化
# ===============================
df.columns = [
    "Timestamp",
    "From Bank",
    "From Account",
    "To Bank",
    "To Account",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering"
]

df["ts"] = pd.to_datetime(df["Timestamp"], format=TIME_FMT, errors="coerce")
df = df.dropna(subset=["ts"])
df = df.sort_values("ts").reset_index(drop=True)

df["txn_date"] = df["ts"].dt.date
df["out_amount"] = df["Amount Paid"].fillna(0.0)
df["in_amount"] = df["Amount Received"].fillna(0.0)

# ===============================
# 3. 开户时间（首次交易）
# ===============================
print("[2] Computing open time...")
open_time = pd.concat([
    df[["From Account", "ts"]].rename(columns={"From Account": "account"}),
    df[["To Account", "ts"]].rename(columns={"To Account": "account"})
]).groupby("account")["ts"].min()

df["open_time_from"] = df["From Account"].map(open_time)
df["open_time_to"] = df["To Account"].map(open_time)

df["days_since_open_from"] = (df["ts"] - df["open_time_from"]).dt.days
df["days_since_open_to"] = (df["ts"] - df["open_time_to"]).dt.days

# ===============================
# 4. S1 新开户即转出
# ===============================
print("[3] Computing S1 features...")
df["is_new_account_from"] = df["days_since_open_from"] == 0

daily_new_cnt = (
    df[df["is_new_account_from"]]
    .groupby("txn_date")["From Account"]
    .nunique()
)

daily_out_sum = (
    df[df["is_new_account_from"]]
    .groupby("txn_date")["out_amount"]
    .sum()
)

df["daily_new_account_cnt"] = df["txn_date"].map(daily_new_cnt).fillna(0).astype(int)
df["daily_out_amount_sum"] = df["txn_date"].map(daily_out_sum).fillna(0.0)

# ===============================
# 5. S3 + S7（To Account，一次 groupby）
# ===============================
print("[4] Computing S3 & S7 features...")

def calc_s3_s7(sub):
    sub = sub.sort_values("ts").copy()
    sub = sub.set_index("ts")

    # S3：10 天内流入笔数
    sub["in_cnt_10d_to"] = sub["in_amount"].rolling("10D").count()

    sub = sub.reset_index()
    sub["last_in_ts_to"] = sub["ts"]
    sub["out_after_last_in_days_to"] = 0

    # S7：相同金额次数
    amt_cnt = sub["in_amount"].value_counts()
    sub["same_amount_txn_cnt"] = sub["in_amount"].map(amt_cnt).fillna(0).astype(int)

    # S7：时间间隔稳定性
    gaps = sub["ts"].diff().dt.days.dropna()
    sub["time_gap_range_days"] = gaps.max() - gaps.min() if len(gaps) >= 2 else np.nan

    # S7：历史大额整数倍转出
    sub["has_prior_large_out"] = (
        (sub["out_amount"] >= 10000) &
        (sub["out_amount"] % 10000 == 0)
    ).any()

    return sub

df = df.groupby("To Account", group_keys=False).apply(calc_s3_s7)

# ===============================
# 6. S5 + S6（From Account，一次 groupby）
# ===============================
print("[5] Computing S5 & S6 s8 features...")
def calc_s5_s6_s8(sub):
    # 保证时间有序
    sub = sub.sort_values("ts").copy()

    # ======================================================
    # -------------------- S5 ------------------------------
    # ======================================================
    d = sub["days_since_open_from"]

    sub["cnt_first_15d_from"] = (d < 15).sum()
    sub["amt_first_15d_from"] = sub.loc[d < 15, "out_amount"].sum()

    sub["cnt_last_15d_from"] = ((d >= 15) & (d < 30)).sum()
    sub["amt_last_15d_from"] = sub.loc[(d >= 15) & (d < 30), "out_amount"].sum()

    sub["total_cnt_30d_from"] = (d < 30).sum()
    sub["total_amt_30d_from"] = sub.loc[d < 30, "out_amount"].sum()

    # ======================================================
    # -------------------- S6 ------------------------------
    # ======================================================
    sub = sub.set_index("ts")

    sub["amt_in_3d"] = sub["in_amount"].rolling("3D").sum()
    sub["amt_out_3d"] = sub["out_amount"].rolling("3D").sum()

    sub["in_out_ratio_3d"] = sub["amt_in_3d"] / (sub["amt_out_3d"] + 1e-6)
    sub["end_balance_3d_est"] = sub["amt_in_3d"] - sub["amt_out_3d"]

    # ======================================================
    # -------------------- S8 ------------------------------
    # ======================================================

    # ---------- 10 天流入 / 流出笔数 ----------
    in_cnt_10d = sub["in_amount"].rolling("10D").count()
    out_cnt_10d = sub["out_amount"].rolling("10D").count()
    sub["in_cnt_div_out_cnt_10d"] = in_cnt_10d / (out_cnt_10d + 1e-6)

    # ---------- 10 天流入 / 流出金额 ----------
    sub["total_in_amt_10d"] = sub["in_amount"].rolling("10D").sum()
    total_out_amt_10d = sub["out_amount"].rolling("10D").sum()
    sub["in_out_amt_ratio_10d"] = sub["total_in_amt_10d"] / (total_out_amt_10d + 1e-6)

    # ---------- 10 天不同转入对手账户数（关键修复点） ----------
    # 把字符串账户映射成 int（rolling 只能吃数值）
    sub["_to_acct_code"] = pd.factorize(sub["To Account"])[0]

    sub["_in_flag"] = (sub["in_amount"] > 0).astype(int)

    sub["_in_acct_code"] = sub["_to_acct_code"].where(sub["_in_flag"] == 1)

    sub["unique_in_counterparty_cnt_10d"] = (
        sub["_in_acct_code"]
        .rolling("10D")
        .apply(lambda x: len(set(x.dropna())), raw=False)
    )

    # ---------- 流入 / 流出时间均值 ----------
    sub["_ts_int"] = sub.index.view("int64")

    sub["_mean_in_time_10d"] = (
        sub["_ts_int"]
        .where(sub["in_amount"] > 0)
        .rolling("10D")
        .mean()
    )

    sub["_mean_out_time_10d"] = (
        sub["_ts_int"]
        .where(sub["out_amount"] > 0)
        .rolling("10D")
        .mean()
    )

    sub["mean_in_time_gt_mean_out_time"] = (
        sub["_mean_in_time_10d"] > sub["_mean_out_time_10d"]
    )

    # ======================================================
    # 清理中间字段 & 还原索引
    # ======================================================
    sub = sub.reset_index()

    sub.drop(
        columns=[
            "_to_acct_code",
            "_in_flag",
            "_in_acct_code",
            "_ts_int",
            "_mean_in_time_10d",
            "_mean_out_time_10d",
        ],
        inplace=True,
        errors="ignore"
    )

    return sub

df = df.groupby("From Account", group_keys=False).apply(calc_s5_s6_s8)


# ===============================
# 7. 保存
# ===============================
df = df.sort_values("ts").reset_index(drop=True)
df.to_csv(OUTPUT_CSV, index=False)

print(f"[OK] Feature generation finished: {OUTPUT_CSV}")
