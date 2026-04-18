#!/usr/bin/env python3
"""
交易数据生成主程序 - 异步并行版本
输出所有文件到 ./transaction_output 目录
"""

import os
import sys
import json
import csv
import glob
import requests
import asyncio
from datetime import datetime, timedelta

os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import yaml
import agentscope
from LLMGraph.environments.transaction import TransactionEnvironment
from LLMGraph.llms.load_configs import load_model_configs


def check_llm_service(base_url):
    """
    检查指定 URL 的 LLM 服务是否可用

    Args:
        base_url: LLM 服务的基础 URL

    Returns:
        bool: 服务是否可用
    """
    try:
        response = requests.get(f"{base_url}/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"  ✓ {base_url} 可用")
            if "data" in models and len(models["data"]) > 0:
                model_id = models["data"][0]["id"]
                print(f"    可用模型: {model_id}")
            return True
        else:
            print(f"  ✗ {base_url} 响应异常: HTTP {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"  ✗ 无法连接到本地 LLM 服务: {base_url}")
        print(f"    请先启动对应服务: ./manage_llm_service.sh start")
        return False
    except requests.exceptions.Timeout:
        print(f"  ✗ {base_url} 连接超时")
        return False
    except Exception as e:
        print(f"  ✗ 检查 {base_url} 时出错: {e}")
        return False


def get_local_service_urls(model_config_path, task_config_path):
    """
    从配置中提取当前角色映射实际用到的本地服务 URL。
    只检查 base_url 包含 localhost 或 127.0.0.1 的配置项。
    """
    with open(model_config_path, 'r', encoding='utf-8') as f:
        model_data = json.load(f)
    with open(task_config_path, 'r', encoding='utf-8') as f:
        task_data = yaml.safe_load(f)

    # 优先级：task_config > model_config（与 load_configs.py 一致）
    role_mappings = model_data.get('role_mappings', {})
    if task_data and 'llm_role_mappings' in task_data:
        role_mappings.update(task_data['llm_role_mappings'])

    used_config_names = set(role_mappings.values())
    local_urls = set()
    for cfg in model_data.get('model_configs', []):
        if cfg.get('config_name') in used_config_names:
            base_url = cfg.get('client_args', {}).get('base_url', '')
            if 'localhost' in base_url or '127.0.0.1' in base_url:
                local_urls.add(base_url)
    return local_urls


async def main_async(max_parallel: int = 20):
    """主函数 - 异步并行版本

    Args:
        max_parallel: 最大并行数（控制同时执行的 geo 数量）
    """
    # 配置路径
    config_path = os.path.join(project_root, "LLMGraph/tasks/transaction/config.yaml")
    output_dir = os.path.join(project_root, "transaction_output")

    print("=" * 70)
    print("交易数据生成程序 (异步并行版本)")
    print("=" * 70)
    print(f"项目根目录: {project_root}")
    print(f"配置文件: {config_path}")
    print(f"输出目录: {output_dir}")
    print(f"最大并行数: {max_parallel}")

    # 检查本地 LLM 服务（仅检查配置中实际用到的本地端点）
    model_config_path = os.path.join(project_root, "LLMGraph/llms/default_model_configs.json")
    local_urls = get_local_service_urls(model_config_path, config_path)
    if local_urls:
        print("\n检查本地 LLM 服务...")
        failed = [url for url in sorted(local_urls) if not check_llm_service(url)]
        if failed:
            print("\n" + "=" * 70)
            print("错误: 以下本地 LLM 服务不可用:")
            for url in failed:
                print(f"  {url}")
            print("请先启动对应服务后重新运行: ./manage_llm_service.sh start")
            print("=" * 70)
            sys.exit(1)
    else:
        print("\n✓ 当前配置仅使用 API 模式，无需本地 LLM 服务")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载配置
    print("\n加载配置文件...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    env_cfg = config["environment"]
    time_cfg = env_cfg["time_configs"]
    txn_cfg = env_cfg["transaction_configs"]
    mgr_cfg = env_cfg["managers"]["transaction"]

    # 解析时间
    start_time = datetime.strptime(time_cfg["start_time"], "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(time_cfg["end_time"], "%Y-%m-%d %H:%M:%S")
    duration = end_time - start_time

    print(f"\n配置摘要:")
    print(f"  - 开始时间: {time_cfg['start_time']}")
    print(f"  - 结束时间: {time_cfg['end_time']}")
    print(f"  - 时间跨度: {duration.days} 天 {duration.seconds//3600} 小时")
    print(f"  - 目标交易数: {txn_cfg['target_txn_count']:,}")

    # 优先显示num_merchants和num_persons,如果没有则使用num_accounts
    if 'num_merchants' in txn_cfg and 'num_persons' in txn_cfg:
        print(f"  - 商户数: {txn_cfg['num_merchants']:,}")
        print(f"  - 个人数: {txn_cfg['num_persons']:,}")
        print(f"  - 总账户数: {txn_cfg['num_merchants'] + txn_cfg['num_persons']:,}")
    else:
        print(f"  - 总账户数: {txn_cfg['num_accounts']:,}")
        if 'merchant_ratio' in txn_cfg:
            print(f"  - 商户比例: {txn_cfg['merchant_ratio']:.2%}")

    # 初始化 AgentScope
    print("\n初始化 AgentScope...")
    # 加载配置（处理 role_mappings）
    model_configs = load_model_configs(
        model_config_path=os.path.join(project_root, "LLMGraph/llms/default_model_configs.json"),
        task_config_path=os.path.join(project_root, "LLMGraph/tasks/transaction/config.yaml")
    )
    agentscope.init(
        project="transaction_generation",
        name="main_txn_async",
        model_configs=model_configs,
        use_monitor=False,
        save_code=False,
        save_api_invoke=False,
    )

    # 创建环境
    print("创建交易生成环境...")

    # 检查是否存在state.json（用于恢复）
    state_file = os.path.join(output_dir, "state.json")
    has_state = os.path.exists(state_file)

    env_config = {
        "time_configs": {
            "start_time": start_time,
            "cur_time": start_time,  # 如果有state.json，这个会被覆盖
            "end_time": end_time,
            "window_size_minutes": time_cfg["window_size_minutes"],
        },
        "transaction_configs": txn_cfg,
        "managers": {
            "transaction": {
                **mgr_cfg,
                "generated_data_dir": output_dir,  # 强制指定输出目录
            }
        },
        "task_path": project_root,
        "config_path": project_root,
    }

    env = TransactionEnvironment(**env_config)
    await env.initialize_async()

    # 主循环 - 使用异步版本
    print("\n开始生成交易数据 (异步并行模式)...")
    total_seconds = (end_time - start_time).total_seconds()
    window_seconds = time_cfg["window_size_minutes"] * 60
    total_windows = int(total_seconds / window_seconds)
    print(f"总窗口数: {total_windows}")
    print("-" * 70)

    window_count = 0
    try:
        while not env.is_done():
            window_count += 1
            # 使用异步版本的 step
            await env.step_async(max_parallel=max_parallel)

            if window_count % 5 == 0 or env.is_done():
                progress = (env.time_configs['cur_time'] - start_time).total_seconds() / total_seconds * 100
                print(f"窗口 {window_count}/{total_windows} | "
                      f"进度: {progress:.1f}% | "
                      f"已生成: {env.manager_agent.total_txn_generated:,} 笔")

    except KeyboardInterrupt:
        print("\n\n检测到中断 (Ctrl+C)")
        print("正在保存...")

    # 最终保存
    print(f"\n{'=' * 70}")
    print("保存数据...")
    env.save(start_time=start_time)

    # 验证输出文件
    print("\n检查输出目录:")
    print(f"目录: {output_dir}")
    output_files = []
    for filename in os.listdir(output_dir):
        filepath = os.path.join(output_dir, filename)
        size = os.path.getsize(filepath)
        output_files.append((filename, size))
        print(f"  ✓ {filename} ({size:,} bytes)")

    # 合并CSV
    csv_files = sorted(glob.glob(os.path.join(output_dir, "transactions_*.csv")))
    if csv_files:
        print(f"\n找到 {len(csv_files)} 个交易CSV文件")

        # 读取所有交易
        all_transactions = []
        header = None

        for csv_file in csv_files:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if header is None:
                    header = reader.fieldnames
                for row in reader:
                    all_transactions.append(row)

        print(f"总交易数: {len(all_transactions):,}")

        # 保存样例
        sample_size = min(1000, len(all_transactions))
        sample_file = os.path.join(output_dir, "transactions_sample.csv")

        with open(sample_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for txn in all_transactions[:sample_size]:
                writer.writerow(txn)

        print(f"✓ 样例文件已保存: transactions_sample.csv ({sample_size} 条)")
    else:
        print("\n警告: 未找到交易CSV文件")

    # 统计摘要
    print(f"\n{'=' * 70}")
    print("生成完成!")
    print(f"{'=' * 70}")
    print(f"总交易数: {env.manager_agent.total_txn_generated:,}")
    print(f"目标交易数: {txn_cfg['target_txn_count']:,}")
    completion = env.manager_agent.total_txn_generated / txn_cfg['target_txn_count'] * 100
    print(f"完成率: {completion:.2f}%")
    print(f"\n输出目录: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    # 支持命令行参数指定并行数
    import argparse
    parser = argparse.ArgumentParser(description="交易数据生成程序 (异步并行版本)")
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=5,
        help="最大并行数（控制同时执行的 geo 数量，默认: 5）"
    )
    args = parser.parse_args()

    # 运行异步主函数
    asyncio.run(main_async(max_parallel=args.max_parallel))
