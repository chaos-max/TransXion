"""Command-line interface for AML Terraformer."""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .io import read_accounts, read_transactions, write_accounts, write_transactions
from .core import normalize_data, find_laundering_clusters
from .agent import LLMAgent, DummyLLMClient, OpenAILLMClient, AnthropicLLMClient, DeepSeekLLMClient, create_llm_client
from .pipeline import PerturbationRunner, PerturbationLogger, generate_summary_report


def generate_cluster_visualization(trans_before, trans_after, cluster_nodes, output_path, cluster_id, affected_nodes):
    """Generate before/after comparison visualization for a cluster.

    Args:
        trans_before: Transactions DataFrame before perturbation
        trans_after: Transactions DataFrame after perturbation
        cluster_nodes: Set of node IDs in the cluster
        output_path: Path to save the visualization
        cluster_id: Cluster ID for title
        affected_nodes: Set of node IDs that were directly operated on
    """
    # Build graphs - mark edge as laundering if ANY transaction is laundering
    G_before = nx.DiGraph()
    # Use vectorized operations instead of iterrows
    for from_node, to_node, is_laundering in zip(
        trans_before["from_node_id"].values,
        trans_before["to_node_id"].values,
        (trans_before.get("Is Laundering", 0) == 1).values
    ):
        # If edge exists, update is_laundering to True if any transaction is laundering
        if G_before.has_edge(from_node, to_node):
            existing_laundering = G_before[from_node][to_node].get('is_laundering', False)
            G_before[from_node][to_node]['is_laundering'] = existing_laundering or is_laundering
        else:
            G_before.add_edge(from_node, to_node, is_laundering=is_laundering)

    G_after = nx.DiGraph()
    for from_node, to_node, is_laundering in zip(
        trans_after["from_node_id"].values,
        trans_after["to_node_id"].values,
        (trans_after.get("Is Laundering", 0) == 1).values
    ):
        # If edge exists, update is_laundering to True if any transaction is laundering
        if G_after.has_edge(from_node, to_node):
            existing_laundering = G_after[from_node][to_node].get('is_laundering', False)
            G_after[from_node][to_node]['is_laundering'] = existing_laundering or is_laundering
        else:
            G_after.add_edge(from_node, to_node, is_laundering=is_laundering)

    # Identify modified nodes (use the affected_nodes passed from tools)
    original_nodes = set(cluster_nodes)
    modified_nodes = affected_nodes  # Use the nodes directly operated on by tools

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot before
    _plot_single_graph(G_before, ax1, f"Cluster {cluster_id} - Before", original_nodes, set())

    # Plot after
    _plot_single_graph(G_after, ax2, f"Cluster {cluster_id} - After", original_nodes, modified_nodes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_single_graph(G, ax, title, original_nodes, modified_nodes):
    """Plot a single graph on given axes.

    Args:
        G: NetworkX graph
        ax: Matplotlib axes
        title: Plot title
        original_nodes: Set of original nodes in the cluster
        modified_nodes: Set of nodes that were modified (edges changed)
    """
    if len(G.nodes()) == 0:
        ax.text(0.5, 0.5, 'Empty Graph', ha='center', va='center')
        ax.set_title(title)
        ax.axis('off')
        return

    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

    # Node colors and sizes
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node in modified_nodes:
            node_colors.append('#ff6b6b')  # Red for modified nodes
            node_sizes.append(800)
        else:
            node_colors.append('#4ecdc4')  # Cyan for original nodes
            node_sizes.append(600)

    # Separate laundering and normal edges
    laundering_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('is_laundering', False)]
    normal_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get('is_laundering', False)]

    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, ax=ax,
                          edge_color='#95a5a6', width=1.5, alpha=0.5,
                          arrows=True, arrowsize=15)
    nx.draw_networkx_edges(G, pos, edgelist=laundering_edges, ax=ax,
                          edge_color='#e74c3c', width=3, alpha=0.8,
                          arrows=True, arrowsize=20)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=ax,
                          node_size=node_sizes, alpha=0.9)

    # Draw labels (simplified)
    labels = {}
    for node in G.nodes():
        parts = node.split('|')
        if len(parts) == 2:
            bank, account = parts
            if 'SPLIT' in account or 'INTERM' in account:
                labels[node] = f"{bank}\n{account[:10]}"
            else:
                labels[node] = f"{bank}\n{account[:8]}..."
        else:
            labels[node] = node[:10]

    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)

    # Legend
    legend_elements = [
        mpatches.Patch(color='#4ecdc4', label='Original'),
        mpatches.Patch(color='#ff6b6b', label='Modified'),
        mpatches.Patch(color='#e74c3c', label='Laundering'),
        mpatches.Patch(color='#95a5a6', label='Normal')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')


def main():
    """Main CLI entry point."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="AML Terraformer: LLM-based perturbation for AML transaction graphs"
    )

    # Input/output
    parser.add_argument(
        "--accounts",
        required=True,
        help="Path to accounts.csv"
    )
    parser.add_argument(
        "--transactions",
        required=True,
        help="Path to transactions.csv"
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for perturbed CSVs and logs"
    )

    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic behavior (default: 42)"
    )

    # Candidates
    parser.add_argument(
        "--topk-candidates",
        type=int,
        default=30,
        help="Number of top candidates to include (default: 30)"
    )

    # Budget parameters
    parser.add_argument(
        "--max-steps-per-cluster",
        type=int,
        default=3,
        help="Maximum steps per cluster (default: 3)"
    )
    parser.add_argument(
        "--max-attempts-per-step",
        type=int,
        default=2,
        help="Maximum attempts per step (default: 2)"
    )
    parser.add_argument(
        "--fail-limit",
        type=int,
        default=2,
        help="Maximum consecutive failures before hard stop (default: 2)"
    )
    parser.add_argument(
        "--max-merges-per-cluster",
        type=int,
        default=1,
        help="Maximum merge operations per cluster (default: 1)"
    )
    parser.add_argument(
        "--max-splits-per-cluster",
        type=int,
        default=1,
        help="Maximum split operations per cluster (default: 1)"
    )
    parser.add_argument(
        "--max-new-nodes-ratio",
        type=float,
        default=0.2,
        help="Maximum new nodes ratio relative to cluster size (default: 0.2)"
    )
    parser.add_argument(
        "--max-new-edges-ratio",
        type=float,
        default=0.3,
        help="Maximum new edges ratio relative to cluster edges (default: 0.3)"
    )
    parser.add_argument(
        "--max-cluster-size",
        type=int,
        default=None,
        help="Maximum cluster size (nodes) to process. Clusters larger than this will be skipped (default: None, process all)"
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=None,
        help="Minimum cluster size (nodes) to process. Clusters smaller than this will be skipped (default: None, process all)"
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=None,
        help="Maximum number of clusters to process. If specified, only the first N clusters will be processed (default: None, process all)"
    )

    # Timestamp format
    parser.add_argument(
        "--timestamp-output-format",
        choices=["iso", "unix", "original"],
        default="iso",
        help="Output timestamp format (default: iso)"
    )

    # Debug options
    parser.add_argument(
        "--save-cluster-details",
        action="store_true",
        help="Save intermediate CSV files and visualizations for each cluster (default: False)"
    )

    # LLM configuration
    parser.add_argument(
        "--llm-provider",
        choices=["dummy", "openai", "anthropic", "deepseek", "local", "local-lora"],
        default="dummy",
        help="LLM provider (default: dummy)"
    )
    parser.add_argument(
        "--llm-model",
        help="LLM model name or path. For API providers: model name. For 'local': full model path. For 'local-lora': base model path"
    )
    parser.add_argument(
        "--llm-api-key",
        help="LLM API key (or set via environment variable, not needed for local providers)"
    )
    parser.add_argument(
        "--lora-checkpoint",
        help="Path to LoRA checkpoint directory (only for local-lora provider)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for local models (cuda, cpu, etc., default: cuda)"
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load local model in 8-bit precision"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load local model in 4-bit precision"
    )

    args = parser.parse_args()

    # Setup paths
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    transactions_out = outdir / "transactions_perturbed.csv"
    accounts_out = outdir / "accounts_perturbed.csv"
    log_out = outdir / "perturb_log.jsonl"
    report_out = outdir / "summary_report.json"

    print("=" * 60)
    print("AML Terraformer")
    print("=" * 60)
    print(f"Accounts: {args.accounts}")
    print(f"Transactions: {args.transactions}")
    print(f"Output directory: {args.outdir}")
    print(f"Seed: {args.seed}")
    print(f"LLM provider: {args.llm_provider}")
    print("=" * 60)

    # Read data
    print("\n[1/6] Reading data...")
    try:
        accounts_df = read_accounts(args.accounts)
        transactions_df = read_transactions(args.transactions)
        print(f"  Accounts: {len(accounts_df)} rows")
        print(f"  Transactions: {len(transactions_df)} rows")
    except Exception as e:
        print(f"ERROR: Failed to read data: {e}", file=sys.stderr)
        sys.exit(1)

    # Normalize data
    print("\n[2/6] Normalizing data...")
    try:
        transactions_df, accounts_df = normalize_data(transactions_df, accounts_df)
        print(f"  Normalized {len(transactions_df)} transactions")
        print(f"  Accounts after normalization: {len(accounts_df)}")
    except Exception as e:
        print(f"ERROR: Failed to normalize data: {e}", file=sys.stderr)
        sys.exit(1)

    # Find laundering clusters
    print("\n[3/6] Finding laundering clusters...")
    try:
        clusters = find_laundering_clusters(transactions_df)
        print(f"  Found {len(clusters)} laundering clusters")
        for cluster in clusters[:5]:  # Show first 5
            print(f"    Cluster {cluster.cluster_id}: {len(cluster.nodes_in_cluster)} nodes, "
                  f"{len(cluster.laundering_edges_internal)} edges")
        if len(clusters) > 5:
            print(f"    ... and {len(clusters) - 5} more")
    except Exception as e:
        print(f"ERROR: Failed to find clusters: {e}", file=sys.stderr)
        sys.exit(1)

    # Setup LLM client
    print("\n[4/6] Setting up LLM client...")
    try:
        # Prepare kwargs for local models
        local_kwargs = {}
        if args.llm_provider in ["local", "local-lora"]:
            local_kwargs["device"] = args.device
            local_kwargs["load_in_8bit"] = args.load_in_8bit
            local_kwargs["load_in_4bit"] = args.load_in_4bit
            local_kwargs["max_new_tokens"] = 512
            local_kwargs["temperature"] = 0.7

        # Create client using factory function
        client = create_llm_client(
            provider=args.llm_provider,
            api_key=args.llm_api_key,
            model=args.llm_model,
            lora_path=args.lora_checkpoint if args.llm_provider == "local-lora" else None,
            **local_kwargs
        )

        # Print status
        if args.llm_provider == "dummy":
            print("  Using dummy LLM client (always returns stop)")
        elif args.llm_provider == "openai":
            print(f"  Using OpenAI client with model: {args.llm_model or 'gpt-4'}")
        elif args.llm_provider == "anthropic":
            print(f"  Using Anthropic client with model: {args.llm_model or 'claude-3-sonnet-20240229'}")
        elif args.llm_provider == "deepseek":
            print(f"  Using DeepSeek client with model: {args.llm_model or 'deepseek-chat'}")
        elif args.llm_provider == "local":
            print(f"  Using local Qwen model: {args.llm_model}")
        elif args.llm_provider == "local-lora":
            print(f"  Using local Qwen model with LoRA adapter")
            print(f"    Base model: {args.llm_model}")
            print(f"    LoRA checkpoint: {args.lora_checkpoint}")
    except Exception as e:
        print(f"ERROR: Failed to setup LLM client: {e}", file=sys.stderr)
        print("\nHINT: For OpenAI, set OPENAI_API_KEY environment variable or use --llm-api-key")
        print("      For Anthropic, set ANTHROPIC_API_KEY environment variable or use --llm-api-key")
        print("      For DeepSeek, set DEEPSEEK_API_KEY environment variable or use --llm-api-key")
        print("      For local, use --llm-model to specify model path")
        print("      For local-lora, use --llm-model for base model and --lora-checkpoint for adapter")
        sys.exit(1)

    agent = LLMAgent(client)

    # Setup logger and runner
    logger = PerturbationLogger(str(log_out))

    config = {
        "seed": args.seed,
        "topk_candidates": args.topk_candidates,
        "max_steps_per_cluster": args.max_steps_per_cluster,
        "max_attempts_per_step": args.max_attempts_per_step,
        "fail_limit": args.fail_limit,
        "max_merges_per_cluster": args.max_merges_per_cluster,
        "max_splits_per_cluster": args.max_splits_per_cluster,
        "max_new_nodes_ratio": args.max_new_nodes_ratio,
        "max_new_edges_ratio": args.max_new_edges_ratio,
        "timestamp_output_format": args.timestamp_output_format,
    }

    runner = PerturbationRunner(agent, logger, config)

    # Run perturbation
    print("\n[5/6] Running perturbation...")

    # Filter clusters by size
    filtered_clusters = []
    skipped_count = 0
    for cluster in clusters:
        cluster_size = len(cluster.nodes_in_cluster)

        # Check max size
        if args.max_cluster_size is not None and cluster_size > args.max_cluster_size:
            skipped_count += 1
            continue

        # Check min size
        if args.min_cluster_size is not None and cluster_size < args.min_cluster_size:
            skipped_count += 1
            continue

        filtered_clusters.append(cluster)

    # Limit number of clusters if specified
    if args.max_clusters is not None and len(filtered_clusters) > args.max_clusters:
        filtered_clusters = filtered_clusters[:args.max_clusters]
        print(f"  Total clusters: {len(clusters)}")
        print(f"  Filtered by size: {len(filtered_clusters) + skipped_count}")
        print(f"  Limited to: {len(filtered_clusters)} clusters (max_clusters={args.max_clusters})")
        if skipped_count > 0:
            print(f"  Skipped by size: {skipped_count}")
    else:
        print(f"  Total clusters: {len(clusters)}")
        print(f"  Filtered clusters: {len(filtered_clusters)}")
        if skipped_count > 0:
            print(f"  Skipped clusters: {skipped_count}")
    print()

    # Save original state for each cluster BEFORE any processing (only if needed)
    cluster_original_states = {}
    if args.save_cluster_details:
        print("  Saving original cluster states...")
        for cluster in filtered_clusters:
            original_nodes = cluster.nodes_in_cluster.copy()
            # Only save internal transactions (both endpoints in cluster)
            cluster_trans_original = transactions_df[
                transactions_df["from_node_id"].isin(original_nodes) &
                transactions_df["to_node_id"].isin(original_nodes)
            ].copy()
            cluster_original_states[cluster.cluster_id] = {
                'nodes': original_nodes,
                'transactions': cluster_trans_original
            }
        print(f"  Saved {len(cluster_original_states)} cluster states")
        print()

    for i, cluster in enumerate(filtered_clusters):
        print(f"  [{i+1}/{len(filtered_clusters)}] Cluster {cluster.cluster_id} "
              f"({len(cluster.nodes_in_cluster)} nodes, "
              f"{len(cluster.laundering_edges_internal)} edges)...", end="")

        try:
            # Run perturbation
            transactions_df, accounts_df, cluster_affected_nodes = runner.run_cluster(
                cluster, transactions_df, accounts_df
            )
            print(" Done")

            # Save intermediate results and visualization (only if requested)
            if args.save_cluster_details:
                # Get original cluster state
                original_state = cluster_original_states[cluster.cluster_id]
                original_cluster_nodes = original_state['nodes']
                cluster_trans_before = original_state['transactions']

                # Get all nodes in the cluster (including external connections)
                cluster_trans_full = transactions_df[
                    transactions_df["from_node_id"].isin(cluster.nodes_in_cluster) |
                    transactions_df["to_node_id"].isin(cluster.nodes_in_cluster)
                ]

                # Get all unique nodes from these transactions (includes new INTERM/SPLIT nodes)
                all_cluster_nodes = set(cluster_trans_full["from_node_id"].unique()) | \
                                   set(cluster_trans_full["to_node_id"].unique())

                # Get new nodes created during perturbation
                new_nodes_created = set()
                for node in all_cluster_nodes:
                    if 'INTERM' in node or 'SPLIT' in node or 'MERGED' in node:
                        new_nodes_created.add(node)

                # Core nodes = original cluster nodes + new nodes
                core_nodes = cluster.nodes_in_cluster | new_nodes_created

                # Filter transactions to only those between core nodes
                cluster_trans_core = transactions_df[
                    transactions_df["from_node_id"].isin(core_nodes) &
                    transactions_df["to_node_id"].isin(core_nodes)
                ]

                # Filter accounts in cluster (including new accounts)
                cluster_accts = accounts_df[
                    (accounts_df["Bank ID"].astype(str).str.strip() + "|" +
                     accounts_df["Account Number"].astype(str).str.strip()).isin(core_nodes)
                ]

                # Save CSV files (core nodes only)
                intermediate_trans = outdir / f"transactions_cluster_{cluster.cluster_id}.csv"
                intermediate_acct = outdir / f"accounts_cluster_{cluster.cluster_id}.csv"
                write_transactions(cluster_trans_core, str(intermediate_trans))
                write_accounts(cluster_accts, str(intermediate_acct))
                print(f"    Saved: {len(cluster_trans_core)} transactions, {len(cluster_accts)} accounts")

                # Generate visualization (before vs after)
                try:
                    # Filter before data: only transactions between original cluster nodes
                    cluster_trans_before_core = cluster_trans_before[
                        cluster_trans_before["from_node_id"].isin(original_cluster_nodes) &
                        cluster_trans_before["to_node_id"].isin(original_cluster_nodes)
                    ]

                    vis_path = outdir / f"cluster_{cluster.cluster_id}_comparison.png"
                    generate_cluster_visualization(cluster_trans_before_core, cluster_trans_core,
                                                  original_cluster_nodes, vis_path, cluster.cluster_id,
                                                  cluster_affected_nodes)
                    print(f"    Visualization: {vis_path.name}")
                except Exception as vis_error:
                    print(f"    Warning: Visualization failed: {vis_error}")

        except Exception as e:
            print(f" ERROR: {e}")
            print(f"ERROR: Failed to process cluster {cluster.cluster_id}: {e}", file=sys.stderr)
            # Continue with next cluster
            continue

    # Write output
    print("\n[6/6] Writing output...")
    try:
        write_transactions(transactions_df, str(transactions_out))
        write_accounts(accounts_df, str(accounts_out))
        print(f"  Transactions: {transactions_out}")
        print(f"  Accounts: {accounts_out}")
        print(f"  Log: {log_out}")
    except Exception as e:
        print(f"ERROR: Failed to write output: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate summary report
    print("\nGenerating summary report...")
    try:
        cli_args = vars(args)
        generate_summary_report(str(log_out), str(report_out), cli_args)
        print(f"  Report: {report_out}")
    except Exception as e:
        print(f"ERROR: Failed to generate report: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Perturbation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
