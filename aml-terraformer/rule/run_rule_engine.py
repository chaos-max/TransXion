import json
import pandas as pd
import numpy as np
import os

# ===============================
# 配置
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "out")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_CSV = os.path.join(OUT_DIR, "txn_with_rule_features.csv")
RULE_JSON = os.path.join(DATA_DIR, "rule.json")

OUTPUT_ALL = os.path.join(OUT_DIR, "txn_with_rule_hits.csv")
OUTPUT_NORMAL = os.path.join(OUT_DIR, "txn_normal.csv")
OUTPUT_ABNORMAL = os.path.join(OUT_DIR, "txn_abnormal.csv")

# ===============================
# 1. 读取数据
# ===============================
print("[1] Loading feature CSV...")
df = pd.read_csv(FEATURE_CSV)

print("[2] Loading rule JSON...")
with open(RULE_JSON, "r", encoding="utf-8") as f:
    rule_cfg = json.load(f)

rules = rule_cfg["rules"]

# 初始化命中字段
for r in rules:
    df[f"hit_{r['rule_id']}"] = 0

# ===============================
# 2. 语义字段 → 物理字段映射
# ===============================
FIELD_ALIAS = {
    "days_since_open": {
        "OUT": "days_since_open_from",
        "IN": "days_since_open_to"
    },

    # ---------- S3 ----------
    "in_cnt_10d": {"IN": "in_cnt_10d_to"},
    "out_after_last_in_days": {"IN": "out_after_last_in_days_to"},

    # ---------- S5 ----------
    "total_cnt_30d": {"OUT": "total_cnt_30d_from"},
    "total_amt_30d": {"OUT": "total_amt_30d_from"},
    "cnt_last_15d_gt_first_15d": {"OUT": "__CNT_15D_GT__"},
    "amt_last_15d_gt_first_15d": {"OUT": "__AMT_15D_GT__"},

    # ---------- S6 ----------
    "in_out_ratio": {
        "OUT": "in_out_ratio_3d",
        "IN": "in_out_ratio_3d"
    },
    "end_balance": {
        "OUT": "end_balance_3d_est",
        "IN": "end_balance_3d_est"
    },
    "max_in_or_out_amt_3d": {
        "OUT": "__MAX_IN_OUT_3D__",
        "IN": "__MAX_IN_OUT_3D__"
    }
}

# ===============================
# 3. 条件判断函数
# ===============================
def check_condition(series, cond):
    op = cond["op"]
    val = cond["value"]

    if op == ">=":
        return series >= val
    if op == "<=":
        return series <= val
    if op == ">":
        return series > val
    if op == "<":
        return series < val
    if op == "==":
        return series == val
    if op == "between":
        return (series >= val[0]) & (series <= val[1])

    raise ValueError(f"Unsupported op: {op}")

# ===============================
# 4. 派生字段解析
# ===============================
def resolve_series(field, direction, rid):
    real_field = field
    if field in FIELD_ALIAS:
        real_field = FIELD_ALIAS[field].get(direction)

    if real_field == "__CNT_15D_GT__":
        return df["cnt_last_15d_from"] > df["cnt_first_15d_from"]

    if real_field == "__AMT_15D_GT__":
        return df["amt_last_15d_from"] > df["amt_first_15d_from"]

    if real_field == "__MAX_IN_OUT_3D__":
        return np.maximum(df["amt_in_3d"], df["amt_out_3d"])

    if real_field not in df.columns:
        raise KeyError(f"[Rule {rid}] Missing feature column: {real_field}")

    return df[real_field]

# ===============================
# 5. 执行规则
# ===============================
print("[3] Running rule engine...")

for rule in rules:
    rid = rule["rule_id"]
    direction = rule.get("direction", "OUT")
    mask = pd.Series(True, index=df.index, dtype=bool)

    if "precondition" in rule:
        for field, cond in rule["precondition"].items():
            mask &= check_condition(resolve_series(field, direction, rid), cond)

    for field, cond in rule["conditions"].items():
        series = resolve_series(field, direction, rid)
        if isinstance(cond, dict):
            mask &= check_condition(series, cond)
        elif cond is True:
            mask &= series.astype(bool)
        elif cond is False:
            mask &= ~series.astype(bool)

    df.loc[mask, f"hit_{rid}"] = 1

# ===============================
# 6. 保存完整命中结果（可选留存）
# ===============================
df.to_csv(OUTPUT_ALL, index=False)

# ===============================
# 7. 拆分正常 / 异常结果（核心新增）
# ===============================
print("[4] Splitting normal / abnormal transactions...")

hit_cols = [c for c in df.columns if c.startswith("hit_")]

# 原始字段（不含特征、不含 hit）
raw_cols = [
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

# ---------- 异常数据 ----------
abnormal_mask = df[hit_cols].sum(axis=1) > 0
df_abnormal = df.loc[abnormal_mask, raw_cols].copy()

df_abnormal["violated_rules"] = (
    df.loc[abnormal_mask, hit_cols]
    .apply(lambda row: ",".join(
        col.replace("hit_", "") for col in hit_cols if row[col] == 1
    ), axis=1)
)

# ---------- 正常数据 ----------
df_normal = df.loc[~abnormal_mask, raw_cols].copy()

# 保存
df_abnormal.to_csv(OUTPUT_ABNORMAL, index=False)
df_normal.to_csv(OUTPUT_NORMAL, index=False)

# ===============================
# 8. 统计信息
# ===============================
hit_counts = df[hit_cols].sum().sort_values(ascending=False)

print("=== 各规则命中数量（交易级） ===")
print(hit_counts)

print("\n=== 输出文件 ===")
print(f"正常数据:   {OUTPUT_NORMAL}")
print(f"异常数据:   {OUTPUT_ABNORMAL}")
print(f"完整命中表: {OUTPUT_ALL}")
print("[OK] Rule engine finished.")
