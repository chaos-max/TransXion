#!/usr/bin/env python3
"""
AML交易图扰动 - 一键运行脚本
调用aml_terraformer CLI，然后实时生成可视化

示例:
  # 设置数据目录
  export AML_DATA_DIR=/path/to/your/data

  # 使用 DeepSeek API
  python3 scripts/run_perturbation.py \
      --accounts $AML_DATA_DIR/accounts.csv \
      --transactions $AML_DATA_DIR/transactions.csv \
      --output output/exp_1 \
      --provider deepseek \
      --api-key YOUR_API_KEY \
      --max-cluster-size 500 \
      --min-cluster-size 2 \
      --max-clusters 1800


  # 使用本地 LoRA 模型
  python3 scripts/run_perturbation.py \
      --accounts $AML_DATA_DIR/accounts.csv \
      --transactions $AML_DATA_DIR/transactions.csv \
      --output output/lora_exp \
      --provider local-lora \
      --model /path/to/Qwen2.5-7B \
      --lora-checkpoint /path/to/lora/checkpoint-final \
      --device cuda:0 \
      --max-cluster-size 500 \
      --min-cluster-size 2 \
      --max-clusters 1800

  # 使用本地模型
  python3 scripts/run_perturbation.py \
      --accounts $AML_DATA_DIR/accounts.csv \
      --transactions $AML_DATA_DIR/transactions.csv \
      --output output/qwen_exp \
      --provider local \
      --model /path/to/Qwen2.5-7B \
      --device cuda \
      --max-cluster-size 200 \
      --min-cluster-size 3
"""

import sys
import os
import json
import subprocess
import argparse
from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def read_transactions_csv(path):
    """读取交易CSV（处理重复的Account列）"""
    with open(path, 'r', encoding='utf-8') as f:
        header_line = f.readline().strip()

    cols = header_line.split(',')
    new_cols = []
    account_count = 0
    for col in cols:
        col = col.strip()
        if col == "Account":
            if account_count == 0:
                new_cols.append("From Account")
                account_count += 1
            elif account_count == 1:
                new_cols.append("To Account")
                account_count += 1
        else:
            new_cols.append(col)

    df = pd.read_csv(path, skiprows=1, names=new_cols)
    return df


def build_graph(df):
    """从交易构建图"""
    G = nx.DiGraph()

    for _, row in df.iterrows():
        from_node = f"{row['From Bank']}|{row['From Account']}"
        to_node = f"{row['To Bank']}|{row['To Account']}"

        G.add_node(from_node)
        G.add_node(to_node)

        try:
            amount = float(row['Amount Received'])
        except:
            amount = 0.0

        is_laundering = str(row.get('Is Laundering', '0')) == '1'

        G.add_edge(from_node, to_node, amount=amount, is_laundering=is_laundering)

    return G


def visualize_comparison(before_path, after_path, output_path, title_prefix=""):
    """生成对比图"""
    print(f"  生成可视化: {output_path.name}")

    try:
        df_before = read_transactions_csv(before_path)
        df_after = read_transactions_csv(after_path)

        G_before = build_graph(df_before)
        G_after = build_graph(df_after)

        # 识别新节点
        new_nodes = set(G_after.nodes()) - set(G_before.nodes())

        # 生成两张图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

        # 扰动前
        plot_single_graph(G_before, ax1, f"{title_prefix}Before Perturbation", set())

        # 扰动后
        plot_single_graph(G_after, ax2, f"{title_prefix}After Perturbation", new_nodes)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"    节点: {G_before.number_of_nodes()} → {G_after.number_of_nodes()}")
        print(f"    边: {G_before.number_of_edges()} → {G_after.number_of_edges()}")

    except Exception as e:
        print(f"    ✗ 可视化失败: {e}")


def plot_single_graph(G, ax, title, highlight_nodes):
    """绘制单个图到指定的axes"""
    pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)

    # 节点颜色
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node in highlight_nodes:
            node_colors.append('#ff6b6b')
            node_sizes.append(1500)
        else:
            node_colors.append('#4ecdc4')
            node_sizes.append(1000)

    # 边分类
    laundering_edges = [(u, v) for u, v, d in G.edges(data=True)
                       if d.get('is_laundering', False)]
    normal_edges = [(u, v) for u, v, d in G.edges(data=True)
                   if not d.get('is_laundering', False)]

    # 绘制
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, ax=ax,
                          edge_color='#95a5a6', width=2, alpha=0.5,
                          arrows=True, arrowsize=20)

    nx.draw_networkx_edges(G, pos, edgelist=laundering_edges, ax=ax,
                          edge_color='#e74c3c', width=4, alpha=0.8,
                          arrows=True, arrowsize=25)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=ax,
                          node_size=node_sizes, alpha=0.9)

    # 标签
    labels = {}
    for node in G.nodes():
        bank, account = node.split('|')
        if 'SPLIT' in account or 'INTERM' in account:
            labels[node] = f"Bank {bank}\n{account.split('_')[0]}"
        else:
            labels[node] = f"Bank {bank}\n{account[:10]}..."

    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax)

    # 图例
    legend = [
        mpatches.Patch(color='#4ecdc4', label='Original'),
        mpatches.Patch(color='#ff6b6b', label='New'),
        mpatches.Patch(color='#e74c3c', label='Laundering'),
        mpatches.Patch(color='#95a5a6', label='Normal')
    ]
    ax.legend(handles=legend, loc='upper right', fontsize=10)

    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.axis('off')


def watch_and_visualize(output_dir, original_trans):
    """监控输出目录，实时生成可视化"""
    output_dir = Path(output_dir)
    original_trans = Path(original_trans)

    print(f"\n监控目录: {output_dir}")
    print(f"原始交易: {original_trans}")

    # 等待perturb_log.jsonl出现并持续监控
    log_path = output_dir / 'perturb_log.jsonl'

    processed_clusters = set()

    while True:
        if not log_path.exists():
            import time
            time.sleep(1)
            continue

        # 读取日志
        try:
            with open(log_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)

                    if entry['type'] == 'cluster_end':
                        cluster_id = entry['cluster_id']

                        if cluster_id not in processed_clusters:
                            # 生成该簇的可视化
                            after_trans = output_dir / 'transactions_perturbed.csv'
                            if after_trans.exists():
                                vis_path = output_dir / f'cluster_{cluster_id}_comparison.png'
                                visualize_comparison(
                                    original_trans,
                                    after_trans,
                                    vis_path,
                                    f"Cluster {cluster_id} - "
                                )
                                processed_clusters.add(cluster_id)
        except:
            pass

        # 检查是否完成
        summary_path = output_dir / 'summary_report.json'
        if summary_path.exists():
            print("\n✓ 处理完成")
            break

        import time
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser(
        description='AML交易图扰动 - 一键运行',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用 DeepSeek API
  python run_perturbation.py \\
      --accounts \$AML_DATA_DIR/accounts.csv \\
      --transactions \$AML_DATA_DIR/transactions.csv \\
      --output /path/to/output \\
      --provider deepseek \\
      --api-key YOUR_KEY

  # 使用本地 LoRA 模型
  python scripts/run_perturbation.py \\
      --accounts \$AML_DATA_DIR/accounts.csv \\
      --transactions \$AML_DATA_DIR/transactions.csv \\
      --output output/lora_output \\
      --provider local-lora \\
      --model \$QWEN_MODEL_PATH \\
      --lora-checkpoint output/test_train_lora_rulebased/grpo_train/checkpoint-50 \\
      --device cuda
  # 使用本地模型
  python scripts/run_perturbation.py \\
      --accounts \$AML_DATA_DIR/accounts.csv \\
      --transactions \$AML_DATA_DIR/transactions.csv \\
      --output output/lora_output \\
      --provider local \\
      --model \$QWEN_MODEL_PATH \\
      --device cuda
        """
    )

    parser.add_argument('--accounts', type=str, required=True,
                       help='账户CSV文件路径')
    parser.add_argument('--transactions', type=str, required=True,
                       help='交易CSV文件路径')
    parser.add_argument('--output', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--provider', type=str, required=True,
                       choices=['openai', 'deepseek', 'local', 'local-lora'],
                       help='LLM提供商')
    parser.add_argument('--api-key', type=str,
                       help='API密钥（API provider需要）')
    parser.add_argument('--model', type=str,
                       help='模型名称或路径（API provider使用模型名，local使用完整模型路径，local-lora使用base模型路径）')
    parser.add_argument('--lora-checkpoint', type=str,
                       help='LoRA checkpoint路径（仅用于local-lora provider）')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备（cuda, cpu等，默认cuda）')
    parser.add_argument('--load-in-8bit', action='store_true',
                       help='使用8-bit精度加载模型')
    parser.add_argument('--load-in-4bit', action='store_true',
                       help='使用4-bit精度加载模型')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--max-steps', type=int, default=1,
                       help='每个簇的最大步数')
    parser.add_argument('--topk', type=int, default=10,
                       help='候选数量')
    parser.add_argument('--max-cluster-size', type=int, default=2000,
                       help='最大簇大小（节点数），超过此大小的簇将被跳过（默认：10）')
    parser.add_argument('--min-cluster-size', type=int, default=0,
                       help='最小簇大小（节点数），小于此大小的簇将被跳过（默认：3）')
    parser.add_argument('--max-clusters', type=int, default=None,
                       help='最大簇数量，如果指定，只处理前N个簇（默认：None，处理所有簇）')
    parser.add_argument('--save-cluster-details', action='store_true',
                       help='保存每个簇的中间CSV文件和可视化图片（默认：False）')

    args = parser.parse_args()

    print("="*60)
    print("AML交易图扰动 - 一键运行")
    print("="*60)
    print(f"输入交易: {args.transactions}")
    print(f"输出目录: {args.output}")
    print(f"LLM: {args.provider}")

    # 构建CLI命令
    # Add src to Python path
    src_path = Path(__file__).parent.parent / 'src'
    env = os.environ.copy()
    env['PYTHONPATH'] = str(src_path) + os.pathsep + env.get('PYTHONPATH', '')

    cmd = [
        sys.executable, '-m', 'aml_terraformer.cli',
        '--accounts', args.accounts,
        '--transactions', args.transactions,
        '--outdir', args.output,
        '--llm-provider', args.provider,
        '--seed', str(args.seed),
        '--topk-candidates', str(args.topk),
        '--max-steps-per-cluster', str(args.max_steps),
        '--timestamp-output-format', 'iso'
    ]

    # Add API key for API providers
    if args.api_key:
        cmd.extend(['--llm-api-key', args.api_key])

    # Add model (required for all providers)
    if args.model:
        cmd.extend(['--llm-model', args.model])

    # Add LoRA checkpoint for local-lora provider
    if args.provider == 'local-lora':
        if not args.lora_checkpoint:
            print("ERROR: --lora-checkpoint is required for local-lora provider")
            sys.exit(1)
        cmd.extend(['--lora-checkpoint', args.lora_checkpoint])

    # Add device for local providers
    if args.provider in ['local', 'local-lora']:
        cmd.extend(['--device', args.device])
        if args.load_in_8bit:
            cmd.append('--load-in-8bit')
        if args.load_in_4bit:
            cmd.append('--load-in-4bit')

    if args.max_cluster_size:
        cmd.extend(['--max-cluster-size', str(args.max_cluster_size)])

    if args.min_cluster_size:
        cmd.extend(['--min-cluster-size', str(args.min_cluster_size)])

    if args.max_clusters:
        cmd.extend(['--max-clusters', str(args.max_clusters)])

    if args.save_cluster_details:
        cmd.append('--save-cluster-details')

    print(f"\n运行命令:")
    print(' '.join(cmd))
    print()

    try:
        # 运行CLI（添加 PYTHONUNBUFFERED 环境变量强制无缓冲输出）
        env['PYTHONUNBUFFERED'] = '1'

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )

        # 实时输出
        for line in process.stdout:
            print(line, end='')
            sys.stdout.flush()  # 强制刷新输出

        process.wait()

        if process.returncode != 0:
            print(f"\n❌ 处理失败，退出码: {process.returncode}")
            sys.exit(1)

        print(f"\n{'='*60}")
        print("✓ 全部完成!")
        print(f"{'='*60}")
        print(f"\n结果保存在: {args.output}/")
        print(f"  - transactions_perturbed.csv")
        print(f"  - accounts_perturbed.csv")
        print(f"  - perturb_log.jsonl")
        print(f"  - summary_report.json")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
