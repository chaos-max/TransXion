"""分析规则引擎在异常簇上的命中情况（优化版：一次性构建特征 + 逐簇检测）。

这个脚本会：
1. 通过连通分量发现异常簇（基于洗钱交易）
2. 扩展所有簇的1跳邻居并去重（快速）
3. 一次性构建特征（避免重复计算，大幅提速）
4. 逐簇应用规则检测
5. 统计每个规则的总命中数和异常交易命中率

性能优化：
- 不再逐簇构建特征（慢）
- 先收集所有扩展交易去重，一次性构建特征（快）
- 1843个簇的处理时间从数小时降到几分钟

Usage:
    python scripts/analyze_rule_hits.py \
        --accounts data/account-reference.csv \
        --transactions data/trans-reference.csv \
        --output output/rule_analysis.json
"""

import argparse
import sys
import json
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aml_terraformer.io import read_transactions, read_accounts
from aml_terraformer.core.normalize import normalize_data
from aml_terraformer.core.clusters import find_laundering_clusters
from aml_terraformer.monitor import RuleBasedMonitor


def save_intermediate_results(output_dir: Path, df_with_features: pd.DataFrame,
                              clusters: list, cluster_to_edges: dict,
                              total_anomaly_accounts: set, original_anomaly_txn_count: int,
                              all_expanded_edges: set):
    """保存中间结果到磁盘。

    Args:
        output_dir: 输出目录
        df_with_features: 构建好特征的DataFrame
        clusters: 异常簇列表
        cluster_to_edges: 簇ID到edge_id集合的映射
        total_anomaly_accounts: 所有异常账户集合
        original_anomaly_txn_count: 原始异常交易数
        all_expanded_edges: 所有扩展后的edge_id集合
    """
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n保存中间结果到 {cache_dir} ...")

    # 保存DataFrame（使用parquet格式，压缩且快速）
    parquet_path = cache_dir / "features.parquet"
    df_with_features.to_parquet(parquet_path, compression='snappy', index=False)
    print(f"  ✓ 特征数据: {parquet_path} ({parquet_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # 保存其他元数据（使用pickle）
    metadata = {
        'clusters': clusters,
        'cluster_to_edges': cluster_to_edges,
        'total_anomaly_accounts': total_anomaly_accounts,
        'original_anomaly_txn_count': original_anomaly_txn_count,
        'all_expanded_edges': all_expanded_edges,
        'timestamp': datetime.now().isoformat()
    }

    metadata_path = cache_dir / "metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"  ✓ 元数据: {metadata_path} ({metadata_path.stat().st_size / 1024:.1f} KB)")

    print(f"✓ 中间结果保存完成！")


def load_intermediate_results(output_dir: Path):
    """从磁盘加载中间结果。

    Args:
        output_dir: 输出目录

    Returns:
        tuple: (df_with_features, clusters, cluster_to_edges,
                total_anomaly_accounts, original_anomaly_txn_count, all_expanded_edges)
        如果加载失败返回 None
    """
    cache_dir = output_dir / "cache"
    parquet_path = cache_dir / "features.parquet"
    metadata_path = cache_dir / "metadata.pkl"

    if not parquet_path.exists() or not metadata_path.exists():
        return None

    try:
        print(f"\n发现缓存的中间结果: {cache_dir}")

        # 加载元数据
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        cache_time = metadata.get('timestamp', '未知')
        print(f"  缓存时间: {cache_time}")
        print(f"  簇数量: {len(metadata['clusters'])}")
        print(f"  扩展交易数: {len(metadata['all_expanded_edges']):,}")

        response = input("\n是否使用缓存的中间结果？(y/n，默认y): ").strip().lower()
        if response and response != 'y':
            print("跳过缓存，重新计算...")
            return None

        print("\n加载中间结果...")
        # 加载DataFrame
        df_with_features = pd.read_parquet(parquet_path)
        print(f"  ✓ 加载特征数据: {len(df_with_features):,} 笔交易, {len(df_with_features.columns)} 个特征")

        return (
            df_with_features,
            metadata['clusters'],
            metadata['cluster_to_edges'],
            metadata['total_anomaly_accounts'],
            metadata['original_anomaly_txn_count'],
            metadata['all_expanded_edges']
        )

    except Exception as e:
        print(f"⚠ 加载缓存失败: {e}")
        print("将重新计算...")
        return None


def build_features_with_progress(df: pd.DataFrame, monitor: RuleBasedMonitor) -> pd.DataFrame:
    """Build rule-based features with real progress bars.

    Args:
        df: DataFrame with standardized transaction columns
        monitor: RuleBasedMonitor instance (for compatibility)

    Returns:
        DataFrame with added rule features
    """
    # Sort by timestamp
    df = df.sort_values("ts").reset_index(drop=True)
    df["txn_date"] = df["ts"].dt.date
    df["out_amount"] = df["Amount Paid"].fillna(0.0)
    df["in_amount"] = df["Amount Received"].fillna(0.0)

    print("  [1/6] 计算账户开户时间...")
    # Compute account open time (first transaction)
    open_time = pd.concat([
        df[["From Account", "ts"]].rename(columns={"From Account": "account"}),
        df[["To Account", "ts"]].rename(columns={"To Account": "account"})
    ]).groupby("account")["ts"].min()

    df["open_time_from"] = df["From Account"].map(open_time)
    df["open_time_to"] = df["To Account"].map(open_time)

    df["days_since_open_from"] = (df["ts"] - df["open_time_from"]).dt.days
    df["days_since_open_to"] = (df["ts"] - df["open_time_to"]).dt.days

    print("  [2/6] 计算新账户特征 (S1)...")
    # S1: New account outflows
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

    print("  [3/6] 计算收款账户特征 (S3, S7)...")
    # S3 & S7: Features per To Account
    def calc_s3_s7(sub):
        sub = sub.sort_values("ts").copy()
        sub = sub.set_index("ts")

        # S3: 10-day inflow count
        sub["in_cnt_10d_to"] = sub["in_amount"].rolling("10D").count()

        sub = sub.reset_index()
        sub["last_in_ts_to"] = sub["ts"]
        sub["out_after_last_in_days_to"] = 0

        # S7: Same amount transaction count
        amt_cnt = sub["in_amount"].value_counts()
        sub["same_amount_txn_cnt"] = sub["in_amount"].map(amt_cnt).fillna(0).astype(int)

        # S7: Time gap stability
        gaps = sub["ts"].diff().dt.days.dropna()
        sub["time_gap_range_days"] = gaps.max() - gaps.min() if len(gaps) >= 2 else np.nan

        # S7: Prior large outflows
        sub["has_prior_large_out"] = (
            (sub["out_amount"] >= 10000) &
            (sub["out_amount"] % 10000 == 0)
        ).any()

        return sub

    # Group by To Account with progress bar
    grouped_to = df.groupby("To Account", group_keys=False, as_index=False)
    num_to_accounts = len(grouped_to)
    print(f"      处理 {num_to_accounts:,} 个收款账户...")

    # 检查账户交易量分布
    to_account_sizes = df.groupby("To Account").size()
    print(f"      账户交易量: 最小={to_account_sizes.min()}, 最大={to_account_sizes.max()}, 平均={to_account_sizes.mean():.1f}")

    results_to = []
    with tqdm(total=num_to_accounts, desc="      收款账户", unit="账户") as pbar:
        for account, group in grouped_to:
            group_size = len(group)
            if group_size > 500:
                pbar.set_postfix({"当前账户交易数": group_size})

            results_to.append(calc_s3_s7(group))
            pbar.update(1)
    df = pd.concat(results_to, ignore_index=False)

    print("  [4/6] 计算付款账户特征 (S5, S6, S8) - 第1阶段...")
    # S5, S6, S8: Features per From Account
    def calc_s5_s6_s8(sub):
        sub = sub.sort_values("ts").copy()

        # S5: 30-day features
        d = sub["days_since_open_from"]
        sub["cnt_first_15d_from"] = (d < 15).sum()
        sub["amt_first_15d_from"] = sub.loc[d < 15, "out_amount"].sum()
        sub["cnt_last_15d_from"] = ((d >= 15) & (d < 30)).sum()
        sub["amt_last_15d_from"] = sub.loc[(d >= 15) & (d < 30), "out_amount"].sum()
        sub["total_cnt_30d_from"] = (d < 30).sum()
        sub["total_amt_30d_from"] = sub.loc[d < 30, "out_amount"].sum()

        # S6 & S8: Rolling window features
        sub = sub.set_index("ts")

        # S6: 3-day windows
        sub["amt_in_3d"] = sub["in_amount"].rolling("3D").sum()
        sub["amt_out_3d"] = sub["out_amount"].rolling("3D").sum()
        sub["in_out_ratio_3d"] = sub["amt_in_3d"] / (sub["amt_out_3d"] + 1e-6)
        sub["end_balance_3d_est"] = sub["amt_in_3d"] - sub["amt_out_3d"]

        # S8: 10-day features
        in_cnt_10d = sub["in_amount"].rolling("10D").count()
        out_cnt_10d = sub["out_amount"].rolling("10D").count()
        sub["in_cnt_div_out_cnt_10d"] = in_cnt_10d / (out_cnt_10d + 1e-6)

        sub["total_in_amt_10d"] = sub["in_amount"].rolling("10D").sum()
        total_out_amt_10d = sub["out_amount"].rolling("10D").sum()
        sub["in_out_amt_ratio_10d"] = sub["total_in_amt_10d"] / (total_out_amt_10d + 1e-6)

        # S8: Unique counterparties - 高效滑动窗口版本
        sub["_to_acct_code"] = pd.factorize(sub["To Account"])[0]
        sub["_in_flag"] = (sub["in_amount"] > 0).astype(int)
        sub["_in_acct_code"] = sub["_to_acct_code"].where(sub["_in_flag"] == 1)

        # 高效计算：使用numpy和双指针滑动窗口
        ten_days_ns = pd.Timedelta(days=10).value  # 转为纳秒
        timestamps = sub.index.values.astype('int64')  # DatetimeIndex转为int64纳秒
        acct_codes = sub["_in_acct_code"].to_numpy()

        unique_counts = np.zeros(len(sub), dtype=int)

        # 双指针滑动窗口
        left = 0
        for right in range(len(sub)):
            current_time = timestamps[right]
            window_start = current_time - ten_days_ns

            # 移动左指针，移除窗口外的元素
            while left < right and timestamps[left] < window_start:
                left += 1

            # 计算窗口内的唯一账户数
            window_codes = acct_codes[left:right+1]
            # 过滤掉NaN，计算唯一值
            valid_codes = window_codes[~pd.isna(window_codes)]
            unique_counts[right] = len(np.unique(valid_codes)) if len(valid_codes) > 0 else 0

        sub["unique_in_counterparty_cnt_10d"] = unique_counts

        # S8: Mean in/out time comparison
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

        # Clean up intermediate columns
        sub = sub.reset_index()
        sub.drop(
            columns=[
                "_to_acct_code", "_in_flag", "_in_acct_code",
                "_ts_int", "_mean_in_time_10d", "_mean_out_time_10d"
            ],
            inplace=True,
            errors="ignore"
        )

        return sub

    print("  [5/6] 计算付款账户特征 (S5, S6, S8) - 第2阶段...")
    # Group by From Account with progress bar
    grouped_from = df.groupby("From Account", group_keys=False, as_index=False)
    num_from_accounts = len(grouped_from)
    print(f"      处理 {num_from_accounts:,} 个付款账户...")

    # 先检查账户交易量分布
    account_sizes = df.groupby("From Account").size()
    print(f"      账户交易量: 最小={account_sizes.min()}, 最大={account_sizes.max()}, 平均={account_sizes.mean():.1f}")
    if account_sizes.max() > 1000:
        large_accounts = (account_sizes > 1000).sum()
        print(f"      警告: {large_accounts} 个账户有超过1000笔交易，可能较慢")

    results_from = []
    processed = 0
    with tqdm(total=num_from_accounts, desc="      付款账户", unit="账户") as pbar:
        for account, group in grouped_from:
            group_size = len(group)
            if group_size > 500:  # 大账户显示详细信息
                pbar.set_postfix({"当前账户交易数": group_size})

            results_from.append(calc_s5_s6_s8(group))
            processed += 1
            pbar.update(1)
    df = pd.concat(results_from, ignore_index=False)

    print("  [6/6] 整理最终结果...")
    return df.sort_values("ts").reset_index(drop=True)


def analyze_rule_hits(
    transactions_df: pd.DataFrame,
    accounts_df: pd.DataFrame,
    monitor: RuleBasedMonitor,
    output_dir: Path = None
) -> dict:
    """分析规则命中情况（逐个簇检测）。

    Args:
        transactions_df: 交易数据
        accounts_df: 账户数据
        monitor: 规则监控器
        output_dir: 输出目录（用于缓存中间结果）

    Returns:
        包含统计结果的字典
    """
    print("=" * 80)
    print("规则命中分析（异常簇 + 1跳扩展，逐簇检测）")
    print("=" * 80)

    # 获取规则列表
    rules = monitor.rules
    rule_ids = [r['rule_id'] for r in rules]

    print(f"\n加载的规则: {rule_ids}")
    print(f"总交易数: {len(transactions_df):,}")

    # 尝试加载缓存的中间结果
    if output_dir:
        cached = load_intermediate_results(output_dir)
        if cached is not None:
            (df_with_features, clusters, cluster_to_edges,
             total_anomaly_accounts, original_anomaly_txn_count,
             all_expanded_edges) = cached

            print("\n✓ 成功加载缓存，跳过步骤1-3")
            print(f"  簇数量: {len(clusters)}")
            print(f"  特征数据: {len(df_with_features):,} 笔交易")

            # 直接跳转到步骤4：规则检测
            goto_step_4 = True
        else:
            goto_step_4 = False
    else:
        goto_step_4 = False

    if not goto_step_4:
        # 步骤1：发现异常簇
        print("\n" + "="*80)
        print("步骤 1: 发现异常簇（连通分量）")
        print("="*80)

        clusters = find_laundering_clusters(transactions_df)
        print(f"发现 {len(clusters)} 个异常簇")

        # 统计簇的大小分布
        cluster_sizes = [len(c.nodes_in_cluster) for c in clusters]
        if cluster_sizes:
            print(f"  最小簇: {min(cluster_sizes)} 个节点")
            print(f"  最大簇: {max(cluster_sizes)} 个节点")
            print(f"  平均簇: {np.mean(cluster_sizes):.1f} 个节点")

        # 用于聚合所有簇的结果
        total_anomaly_accounts = set()
        original_anomaly_txn_count = 0
        cluster_to_edges = {}  # 记录每个簇包含的交易edge_id

        # 步骤2：扩展所有簇的1跳邻居
        print("\n" + "="*80)
        print("步骤 2: 扩展所有簇的1跳邻居")
        print("="*80)
        print(f"将处理 {len(clusters)} 个簇...")

        all_expanded_edges = set()  # 所有扩展后的edge_id（去重）

        for cluster in tqdm(clusters, desc="扩展簇", unit="簇"):
            # 获取簇中的账户（节点）
            cluster_accounts = cluster.nodes_in_cluster
            total_anomaly_accounts.update(cluster_accounts)

            # 找出簇内的异常交易
            cluster_anomaly_txns = transactions_df[
                (transactions_df["from_node_id"].isin(cluster_accounts)) &
                (transactions_df["to_node_id"].isin(cluster_accounts)) &
                (transactions_df["Is Laundering"] == 1)
            ]
            original_anomaly_txn_count += len(cluster_anomaly_txns)

            # 扩展1跳：找出这些账户参与的所有交易的edge_id
            expanded_mask = (
                transactions_df["from_node_id"].isin(cluster_accounts) |
                transactions_df["to_node_id"].isin(cluster_accounts)
            )
            cluster_edges = set(transactions_df.loc[expanded_mask, 'edge_id'].values)
            cluster_to_edges[cluster.cluster_id] = cluster_edges
            all_expanded_edges.update(cluster_edges)

        print(f"\n✓ 扩展完成")
        print(f"  原始异常交易: {original_anomaly_txn_count:,} 笔")
        print(f"  涉及账户: {len(total_anomaly_accounts):,} 个")
        print(f"  扩展后交易（去重）: {len(all_expanded_edges):,} 笔")

        # 步骤3：一次性构建特征
        print("\n" + "="*80)
        print("步骤 3: 构建特征（一次性，这可能需要几分钟...）")
        print("="*80)

        # 获取所有扩展后的交易
        expanded_txns = transactions_df[transactions_df['edge_id'].isin(all_expanded_edges)].copy()

        # 准备数据列
        df_for_monitor = expanded_txns.copy()
        if 'ts' not in df_for_monitor.columns and 'ts_int' in df_for_monitor.columns:
            df_for_monitor['ts'] = pd.to_datetime(df_for_monitor['ts_int'], unit='s')
        elif 'Timestamp' in df_for_monitor.columns:
            df_for_monitor['ts'] = pd.to_datetime(df_for_monitor['Timestamp'])

        print(f"需要处理 {len(df_for_monitor):,} 笔交易的特征构建")
        print()

        # 构建特征（一次性，带真实进度条）
        import time
        start_time = time.time()

        df_with_features = build_features_with_progress(df_for_monitor, monitor)

        elapsed = time.time() - start_time
        print(f"\n  ✓ 特征构建完成，生成 {len(df_with_features.columns)} 个特征列 (耗时: {elapsed:.1f}秒)")
        print()

        # 保存中间结果
        if output_dir:
            save_intermediate_results(
                output_dir, df_with_features, clusters, cluster_to_edges,
                total_anomaly_accounts, original_anomaly_txn_count, all_expanded_edges
            )

    # 步骤4：逐簇应用规则检测
    print("\n" + "="*80)
    print("步骤 4: 逐簇应用规则检测")
    print("="*80)

    # 初始化命中列（全局，针对所有扩展交易）
    for rule in monitor.rules:
        df_with_features[f"hit_{rule['rule_id']}"] = 0

    # 为每个簇单独检测规则
    for cluster in tqdm(clusters, desc="检测规则", unit="簇"):
        # 获取当前簇的交易（基于edge_id）
        cluster_edges = cluster_to_edges[cluster.cluster_id]
        cluster_mask = df_with_features['edge_id'].isin(cluster_edges)
        cluster_df = df_with_features[cluster_mask]

        if len(cluster_df) == 0:
            continue

        # 对当前簇应用规则
        for rule in monitor.rules:
            rid = rule['rule_id']
            direction = rule.get("direction", "OUT")
            mask = pd.Series(True, index=cluster_df.index, dtype=bool)

            # Apply preconditions
            if "precondition" in rule:
                for field, cond in rule["precondition"].items():
                    mask &= monitor._check_condition(
                        monitor._resolve_series(cluster_df, field, direction, rid),
                        cond
                    )

            # Apply main conditions
            for field, cond in rule["conditions"].items():
                series = monitor._resolve_series(cluster_df, field, direction, rid)
                if isinstance(cond, dict):
                    # 字典格式: {"op": ">=", "value": 10}
                    mask &= monitor._check_condition(series, cond)
                elif isinstance(cond, list):
                    # 列表格式: [{"op": ">=", "value": 10}, ...]
                    sub_mask = pd.Series(False, index=cluster_df.index, dtype=bool)
                    for sub_cond in cond:
                        sub_mask |= monitor._check_condition(series, sub_cond)
                    mask &= sub_mask
                elif isinstance(cond, bool):
                    # 布尔格式: true (表示字段值必须为true)
                    if cond:
                        mask &= (series == True) | (series == 1)
                    else:
                        mask &= (series == False) | (series == 0)
                else:
                    raise ValueError(f"Invalid condition type for rule {rid}, field {field}: {type(cond)}")

            # 更新全局df中的hit列（取最大值，如果任一簇命中就算命中）
            hit_indices = cluster_df.index[mask]
            df_with_features.loc[hit_indices, f"hit_{rid}"] = 1

    df_with_hits = df_with_features
    print(f"\n✓ 所有簇规则检测完成！")

    # 统计异常交易数
    print("\n" + "="*80)
    print("统计交易分布...")
    print("="*80)

    laundering_df = df_with_hits[df_with_hits["Is Laundering"] == 1]
    normal_df = df_with_hits[df_with_hits["Is Laundering"] == 0]

    total_txns = len(df_with_hits)
    laundering_txns = len(laundering_df)
    normal_txns = len(normal_df)

    print(f"\n交易分布（扩展1跳后）:")
    print(f"  总交易数: {total_txns:,}")
    print(f"  异常交易数: {laundering_txns:,} ({laundering_txns/total_txns*100:.2f}%)")
    print(f"  正常交易数: {normal_txns:,} ({normal_txns/total_txns*100:.2f}%)")

    # 统计每个规则的命中情况
    results = {
        'summary': {
            'total_transactions': total_txns,
            'laundering_transactions': laundering_txns,
            'normal_transactions': normal_txns,
            'laundering_ratio': laundering_txns / total_txns if total_txns > 0 else 0,
        },
        'expansion_stats': {
            'num_clusters': len(clusters),
            'original_anomaly_transactions': original_anomaly_txn_count,
            'anomaly_accounts': len(total_anomaly_accounts),
            'expanded_transactions': len(all_expanded_edges),
            'expansion_ratio': len(all_expanded_edges) / original_anomaly_txn_count if original_anomaly_txn_count > 0 else 0,
        },
        'rules': {}
    }

    print("\n" + "=" * 80)
    print("规则命中统计")
    print("=" * 80)

    for rule in tqdm(rules, desc="统计规则", unit="规则"):
        rule_id = rule['rule_id']
        rule_name = rule.get('name', rule_id)
        hit_col = f"hit_{rule_id}"

        # 总命中数
        total_hits = df_with_hits[hit_col].sum()
        total_hit_rate = total_hits / total_txns if total_txns > 0 else 0

        # 异常交易中的命中数
        laundering_hits = laundering_df[hit_col].sum()
        laundering_hit_rate = laundering_hits / laundering_txns if laundering_txns > 0 else 0

        # 正常交易中的命中数
        normal_hits = normal_df[hit_col].sum()
        normal_hit_rate = normal_hits / normal_txns if normal_txns > 0 else 0

        # 命中的交易中有多少是异常的（精确率）
        precision = laundering_hits / total_hits if total_hits > 0 else 0

        # 异常交易中有多少被这个规则命中（召回率）
        recall = laundering_hit_rate

        results['rules'][rule_id] = {
            'rule_name': rule_name,
            'direction': rule.get('direction', 'OUT'),
            'all_transactions': {
                'total_hits': int(total_hits),
                'hit_rate': float(total_hit_rate),
            },
            'laundering_transactions': {
                'total_hits': int(laundering_hits),
                'hit_rate': float(laundering_hit_rate),
                'total_laundering': int(laundering_txns),
            },
            'normal_transactions': {
                'total_hits': int(normal_hits),
                'hit_rate': float(normal_hit_rate),
                'total_normal': int(normal_txns),
            },
            'metrics': {
                'precision': float(precision),  # 命中的有多少是真的异常
                'recall': float(recall),        # 异常中有多少被命中
            }
        }

        print(f"\n【{rule_id}】{rule_name}")
        print(f"  方向: {rule.get('direction', 'OUT')}")
        print(f"  所有交易:")
        print(f"    命中数: {total_hits:,} / {total_txns:,} ({total_hit_rate*100:.2f}%)")
        print(f"  异常交易:")
        print(f"    命中数: {laundering_hits:,} / {laundering_txns:,} ({laundering_hit_rate*100:.2f}%)")
        print(f"  正常交易:")
        print(f"    命中数: {normal_hits:,} / {normal_txns:,} ({normal_hit_rate*100:.2f}%)")
        print(f"  性能指标:")
        print(f"    精确率 (Precision): {precision*100:.2f}% (命中的交易中有多少是真异常)")
        print(f"    召回率 (Recall):    {recall*100:.2f}% (异常交易中有多少被命中)")

    # 计算规则组合统计
    print("\n" + "=" * 80)
    print("规则组合分析")
    print("=" * 80)

    # 统计每笔交易命中了多少个规则
    hit_cols = [f"hit_{r['rule_id']}" for r in rules]
    df_with_hits['total_rule_hits'] = df_with_hits[hit_cols].sum(axis=1)

    # 重新创建 laundering_df 和 normal_df（包含 total_rule_hits 列）
    laundering_df = df_with_hits[df_with_hits["Is Laundering"] == 1]
    normal_df = df_with_hits[df_with_hits["Is Laundering"] == 0]

    # 异常交易的规则命中分布
    laundering_rule_hits = laundering_df['total_rule_hits'].value_counts().sort_index()
    normal_rule_hits = normal_df['total_rule_hits'].value_counts().sort_index()

    print("\n异常交易的规则命中分布:")
    for num_hits, count in laundering_rule_hits.items():
        pct = count / laundering_txns * 100
        print(f"  命中 {int(num_hits)} 个规则: {count:,} 笔 ({pct:.2f}%)")

    print("\n正常交易的规则命中分布:")
    for num_hits, count in normal_rule_hits.items():
        pct = count / normal_txns * 100
        print(f"  命中 {int(num_hits)} 个规则: {count:,} 笔 ({pct:.2f}%)")

    # 添加到结果
    results['rule_combination'] = {
        'laundering': {int(k): int(v) for k, v in laundering_rule_hits.items()},
        'normal': {int(k): int(v) for k, v in normal_rule_hits.items()},
    }

    # 计算平均命中规则数
    avg_laundering_hits = laundering_df['total_rule_hits'].mean()
    avg_normal_hits = normal_df['total_rule_hits'].mean()

    print(f"\n平均命中规则数:")
    print(f"  异常交易: {avg_laundering_hits:.2f}")
    print(f"  正常交易: {avg_normal_hits:.2f}")

    results['average_hits'] = {
        'laundering': float(avg_laundering_hits),
        'normal': float(avg_normal_hits),
    }

    return results, df_with_hits


def main():
    parser = argparse.ArgumentParser(description="分析规则引擎命中情况")

    # 数据路径
    parser.add_argument("--accounts", required=True, help="账户数据 CSV 路径")
    parser.add_argument("--transactions", required=True, help="交易数据 CSV 路径")
    parser.add_argument("--output", required=True, help="输出 JSON 文件路径")

    # 规则配置
    parser.add_argument("--rule-config", default=None, help="规则配置文件路径 (默认: rule/data/rule.json)")
    parser.add_argument("--score-aggregation", default="weighted_average",
                       choices=["weighted_average", "max", "count"],
                       help="分数聚合方法")

    # 可选：保存详细的命中数据
    parser.add_argument("--save-detailed-csv", default=None, help="保存带规则命中标记的完整交易数据到 CSV")

    args = parser.parse_args()

    print("=" * 80)
    print("规则命中分析工具（异常簇 + 1跳扩展，逐簇检测）")
    print("=" * 80)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"账户数据: {args.accounts}")
    print(f"交易数据: {args.transactions}")
    print(f"输出文件: {args.output}")

    # 1. 加载数据
    print("\n" + "="*80)
    print("[1/4] 加载数据")
    print("="*80)

    with tqdm(total=2, desc="读取文件") as pbar:
        transactions_df = read_transactions(args.transactions)
        pbar.set_postfix({"交易": f"{len(transactions_df):,}"})
        pbar.update(1)

        accounts_df = read_accounts(args.accounts)
        pbar.set_postfix({"交易": f"{len(transactions_df):,}", "账户": f"{len(accounts_df):,}"})
        pbar.update(1)

    print(f"✓ 加载了 {len(transactions_df):,} 笔交易, {len(accounts_df):,} 个账户")

    # 2. 标准化数据
    print("\n" + "="*80)
    print("[2/4] 标准化数据")
    print("="*80)

    with tqdm(total=1, desc="数据标准化") as pbar:
        transactions_df, accounts_df = normalize_data(transactions_df, accounts_df)
        pbar.update(1)

    print("✓ 数据标准化完成")

    # 3. 创建规则监控器
    print("\n" + "="*80)
    print("[3/4] 创建规则监控器")
    print("="*80)

    with tqdm(total=1, desc="加载规则配置") as pbar:
        if args.rule_config is None:
            rule_config_path = Path(__file__).parent.parent / "rule" / "data" / "rule.json"
        else:
            rule_config_path = Path(args.rule_config)

        monitor = RuleBasedMonitor(
            rule_config_path=str(rule_config_path),
            score_aggregation=args.score_aggregation,
            save_debug_output=False
        )
        pbar.update(1)

    print(f"✓ 规则配置: {rule_config_path}")
    print(f"✓ 规则数量: {len(monitor.rules)}")

    # 4. 分析规则命中
    print("\n" + "="*80)
    print("[4/4] 发现异常簇并运行规则检测")
    print("="*80)
    print("注意: 将逐个簇进行扩展1跳并检测，进度条显示簇的处理进度...")
    print()

    results, df_with_hits = analyze_rule_hits(
        transactions_df, accounts_df, monitor,
        output_dir=Path(args.output).parent  # 使用输出文件的父目录
    )

    # 添加元数据
    results['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'accounts_file': args.accounts,
        'transactions_file': args.transactions,
        'rule_config': str(rule_config_path),
        'score_aggregation': args.score_aggregation,
    }

    # 5. 保存结果
    print("\n" + "="*80)
    print("[保存结果]")
    print("="*80)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tqdm(total=1, desc="写入 JSON") as pbar:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        pbar.update(1)

    print(f"\n{'='*80}")
    print(f"✓ 分析完成！")
    print(f"{'='*80}")
    print(f"结果已保存到: {output_path}")
    print(f"文件大小: {output_path.stat().st_size / 1024:.1f} KB")

    # 可选：保存详细的 CSV
    if args.save_detailed_csv:
        print("\n保存详细 CSV...")
        csv_path = Path(args.save_detailed_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with tqdm(total=1, desc="写入 CSV", unit="文件") as pbar:
            df_with_hits.to_csv(csv_path, index=False)
            pbar.update(1)

        print(f"✓ 详细数据已保存到: {csv_path}")
        print(f"  文件大小: {csv_path.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"  包含列: 原始交易数据 + hit_S1, hit_S3, ..., total_rule_hits")

    # 打印快速摘要
    print(f"\n{'='*80}")
    print("快速摘要")
    print(f"{'='*80}")
    print(f"\n【异常簇扩展统计】")
    print(f"异常簇数量: {results['expansion_stats']['num_clusters']:,}")
    print(f"原始异常交易数: {results['expansion_stats']['original_anomaly_transactions']:,}")
    print(f"涉及账户数: {results['expansion_stats']['anomaly_accounts']:,}")
    print(f"扩展后交易数: {results['expansion_stats']['expanded_transactions']:,} (扩展比例: {results['expansion_stats']['expansion_ratio']:.2f}x)")
    print(f"\n【交易分布】")
    print(f"总交易数: {results['summary']['total_transactions']:,}")
    print(f"异常交易数: {results['summary']['laundering_transactions']:,} ({results['summary']['laundering_ratio']*100:.2f}%)")
    print(f"\n【规则性能排名】(按异常交易召回率):")

    # 按召回率排序
    rule_stats = [(rid, data['metrics']['recall']) for rid, data in results['rules'].items()]
    rule_stats.sort(key=lambda x: x[1], reverse=True)

    for i, (rule_id, recall) in enumerate(rule_stats, 1):
        rule_data = results['rules'][rule_id]
        print(f"  {i}. {rule_id}: 召回率 {recall*100:.2f}% "
              f"(命中 {rule_data['laundering_transactions']['total_hits']:,} / "
              f"{rule_data['laundering_transactions']['total_laundering']:,})")


if __name__ == "__main__":
    main()
