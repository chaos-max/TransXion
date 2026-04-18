"""Train GRPO agent on AML perturbation task.

This script demonstrates how to use the GRPO trainer with a random monitor
to collect training data or train a local model. It supports two modes:

1. Data Collection Mode (--train-mode=collect):
   - Collect trajectories and compute advantages
   - Save data for future fine-tuning (no parameter updates)
   - Works with API-based models (DeepSeek/OpenAI) and local models

2. Training Mode (--train-mode=train):
   - Perform gradient updates on local model
   - Full GRPO training with policy gradient
   - Supports LoRA for efficient fine-tuning
   - Only works with local models

Usage:
    # Set data directory
    export AML_DATA_DIR=/path/to/your/data

    # Data collection (API-based)
    python scripts/train_grpo.py \\
        --accounts $AML_DATA_DIR/accounts.csv \\
        --transactions $AML_DATA_DIR/transactions.csv \\
        --output output/grpo_experiment \\
        --provider deepseek \\
        --api-key YOUR_API_KEY \\
        --num-clusters 5 \\
        --num-samples 4

    # Data collection (Local model, no training)
    python scripts/train_grpo.py \\
        --accounts $AML_DATA_DIR/accounts.csv \\
        --transactions $AML_DATA_DIR/transactions.csv \\
        --output output/grpo_local \\
        --provider local \\
        --model /path/to/Qwen3-1.7B \\
        --device cuda \\
        --num-clusters 2 \\
        --num-samples 2 \\
        --train-mode collect

    # Training (Local model with gradient updates - Rule-based Monitor)
    python3 scripts/train_grpo.py \\
      --accounts $AML_DATA_DIR/accounts.csv \\
      --transactions $AML_DATA_DIR/transactions.csv \\
      --output output/test \\
      --provider local \\
      --model /path/to/Qwen2.5-7B \\
      --device cuda:0 \\
      --num-clusters 50 \\
      --max-cluster-size 50 \\
      --min-cluster-size 10 \\
      --num-samples 4 \\
      --max-steps 2 \\
      --train-mode train \\
      --learning-rate 1e-4 \\
      --batch-size 1 \\
      --gradient-accumulation-steps 4 \\
      --use-lora \\
      --lora-r 8 \\
      --monitor-type rule_based

    # Training (Local model with GNN Monitor)
    python3 scripts/train_grpo.py \\
      --accounts $AML_DATA_DIR/accounts.csv \\
      --transactions $AML_DATA_DIR/transactions.csv \\
      --output output/gnn_experiment \\
      --provider local \\
      --model /path/to/Qwen2.5-7B \\
      --device cuda \\
      --num-clusters 1000\\
      --max-cluster-size 200 \\
      --min-cluster-size 3 \\
      --num-samples 4 \\
      --max-steps 2 \\
      --train-mode train \\
      --learning-rate 1e-4 \\
      --batch-size 1 \\
      --gradient-accumulation-steps 4 \\
      --use-lora \\
      --lora-r 8 \\
      --monitor-type gnn \\
      --gnn-project-path /path/to/multignn \\
      --gnn-score-metric weighted \\
      --gnn-model-name gin
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aml_terraformer.io import read_transactions, read_accounts
from aml_terraformer.core.normalize import normalize_data
from aml_terraformer.core.clusters import find_laundering_clusters
from aml_terraformer.agent.client_interface import create_llm_client
from aml_terraformer.agent.llm_agent import LLMAgent
from aml_terraformer.monitor import RandomMonitor, RuleBasedMonitor, GNNMonitor, GBTMonitor
from aml_terraformer.rl import GRPOReward, GRPOTrainer, GRPOLocalTrainer


def main():
    parser = argparse.ArgumentParser(description="Train GRPO agent for AML perturbation")

    # Data
    parser.add_argument("--accounts", required=True, help="Path to accounts CSV")
    parser.add_argument("--transactions", required=True, help="Path to transactions CSV")
    parser.add_argument("--output", required=True, help="Output directory")

    # LLM
    parser.add_argument("--provider", required=True, choices=["deepseek", "openai", "local", "dummy"],
                       help="LLM provider (use 'local' for local Qwen model)")
    parser.add_argument("--api-key", default=None, help="API key (not needed for local)")
    parser.add_argument("--model", default=None,
                       help="Model name or path (for local: /path/to/Qwen3-1.7B)")

    # Local model options
    parser.add_argument("--device", default="cuda", help="Device for local model (cuda, cpu, cuda:0, etc.)")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max tokens for local model")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load local model in 8-bit")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load local model in 4-bit")

    # GRPO
    parser.add_argument("--num-clusters", type=int, default=5, help="Number of clusters to train on")
    parser.add_argument("--min-cluster-size", type=int, default=3, help="Min cluster size")
    parser.add_argument("--max-cluster-size", type=int, default=100, help="Max cluster size")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of trajectories per cluster (K)")
    parser.add_argument("--save-interval", type=int, default=5, help="Save checkpoint every N clusters")

    # Training mode
    parser.add_argument(
        "--train-mode",
        default="collect",
        choices=["collect", "train"],
        help="Training mode: 'collect' for data collection only, 'train' for actual gradient updates (local model only)"
    )

    # Training hyperparameters (for train mode)
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of epochs per cluster")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm for clipping")

    # LoRA options (for train mode)
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")

    # Monitor
    parser.add_argument(
        "--monitor-type",
        default="rule_based",
        choices=["rule_based", "random", "gnn", "gbt"],
        help="Monitor type: 'rule_based' (default), 'random' (testing), 'gnn' (GNN-based), 'gbt' (LightGBM-based)"
    )
    parser.add_argument(
        "--monitor-mode",
        default="decreasing",
        choices=["pure_random", "graph_size_based", "fixed", "decreasing"],
        help="Random monitor mode (only used when --monitor-type=random)"
    )
    parser.add_argument("--monitor-seed", type=int, default=42, help="Random monitor seed")
    parser.add_argument("--rule-config", default=None, help="Path to rule.json (default: rule/data/rule.json)")
    parser.add_argument(
        "--score-aggregation",
        default="weighted_average",
        choices=["weighted_average", "max", "count"],
        help="Score aggregation method for rule-based monitor"
    )

    # GNN Monitor options
    parser.add_argument(
        "--gnn-project-path",
        default=os.environ.get("MULTIGNN_PATH", ""),
        help="Path to multignn project directory (only used when --monitor-type=gnn)"
    )
    parser.add_argument(
        "--gnn-model-name",
        default="gin",
        help="GNN model name to use (default: gin, only used when --monitor-type=gnn)"
    )
    parser.add_argument(
        "--gnn-score-metric",
        default="f1",
        choices=["f1", "auc", "ap", "weighted"],
        help="Metric to use for GNN scoring (only used when --monitor-type=gnn)"
    )

    # GBT Monitor options
    parser.add_argument(
        "--gbt-model-path",
        default=None,
        help="Path to pre-trained LightGBM model .pkl (required when --monitor-type=gbt). "
             "Train with: python scripts/train_gbt_monitor.py"
    )
    parser.add_argument(
        "--gbt-score-aggregation",
        default="mean",
        choices=["mean", "max", "top_k_mean"],
        help="How to aggregate per-transaction GBT probs into cluster score (default: mean)"
    )

    # Pipeline config
    parser.add_argument("--max-steps", type=int, default=2, help="Max steps per cluster")
    parser.add_argument("--topk", type=int, default=4, help="Top-K candidates")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Reward config
    parser.add_argument("--w-detection", type=float, default=10.0, help="Detection reward weight")
    parser.add_argument("--w-validity", type=float, default=1.0, help="Validity reward weight")
    parser.add_argument("--w-budget", type=float, default=0.1, help="Budget penalty weight")
    parser.add_argument("--w-final-bonus", type=float, default=20.0, help="Final bonus weight")

    args = parser.parse_args()

    # Validate args
    if args.provider not in ["local", "dummy"] and args.api_key is None:
        parser.error("--api-key is required for non-local/non-dummy providers")

    if args.provider == "local" and args.model is None:
        parser.error("--model (path) is required for local provider")

    # Validate train mode
    if args.train_mode == "train" and args.provider != "local":
        parser.error("--train-mode=train is only supported with --provider=local")

    # Validate cluster size
    if args.min_cluster_size > args.max_cluster_size:
        parser.error(f"--min-cluster-size ({args.min_cluster_size}) must be <= --max-cluster-size ({args.max_cluster_size})")

    if args.min_cluster_size < 1:
        parser.error("--min-cluster-size must be >= 1")

    if args.train_mode == "train":
        print("\n" + "="*80)
        print("WARNING: Training mode enabled - model parameters will be updated!")
        print("="*80)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GRPO Training for AML Perturbation")
    print("=" * 80)

    # 1. Load data
    print("\n[1/6] Loading data...")
    transactions_df = read_transactions(args.transactions)
    accounts_df = read_accounts(args.accounts)
    print(f"  Loaded {len(transactions_df)} transactions, {len(accounts_df)} accounts")

    # 2. Normalize data
    print("\n[2/6] Normalizing data...")
    transactions_df, accounts_df = normalize_data(transactions_df, accounts_df)

    # 3. Detect clusters
    print("\n[3/6] Detecting laundering clusters...")
    all_clusters = find_laundering_clusters(transactions_df)

    # Filter by size
    clusters = [c for c in all_clusters
                if args.min_cluster_size <= len(c.nodes_in_cluster) <= args.max_cluster_size]
    print(f"  Detected {len(all_clusters)} clusters, using {len(clusters)} clusters "
          f"(size: {args.min_cluster_size} ~ {args.max_cluster_size})")

    # Limit number of clusters for training
    if len(clusters) > args.num_clusters:
        print(f"  Using first {args.num_clusters} clusters for training")
        clusters = clusters[:args.num_clusters]

    # 4. Setup LLM client
    print("\n[4/6] Setting up LLM client...")

    # Build kwargs for local model
    llm_kwargs = {}
    if args.provider == "local":
        llm_kwargs = {
            "device": args.device,
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.7,
            "load_in_8bit": args.load_in_8bit,
            "load_in_4bit": args.load_in_4bit,
        }

    client = create_llm_client(
        provider=args.provider,
        api_key=args.api_key,
        model=args.model,
        **llm_kwargs
    )
    agent = LLMAgent(client)

    if args.provider == "local":
        print(f"  Provider: {args.provider}")
        print(f"  Model path: {args.model}")
        print(f"  Device: {args.device}")
    else:
        print(f"  Provider: {args.provider}")
        print(f"  Model: {args.model or 'default'}")

    # 5. Setup monitor and reward
    print("\n[5/6] Setting up monitor and reward...")

    # Determine trainer output directory (will be created by trainer)
    trainer_output_dir = output_dir / ("grpo_train" if args.train_mode == "train" else "grpo_data")

    if args.monitor_type == "rule_based":
        # Use RuleBasedMonitor (default)
        if args.rule_config is None:
            rule_config_path = Path(__file__).parent.parent / "rule" / "data" / "rule.json"
        else:
            rule_config_path = Path(args.rule_config)

        monitor = RuleBasedMonitor(
            rule_config_path=str(rule_config_path),
            score_aggregation=args.score_aggregation,
            save_debug_output=True,  # Enable debug output
            debug_output_dir=str(trainer_output_dir)  # Save to trainer output directory
        )
        print(f"  Monitor type: Rule-based")
        print(f"  Rule config: {rule_config_path}")
        print(f"  Score aggregation: {args.score_aggregation}")
        print(f"  Debug output: {trainer_output_dir}/rule_debug/")
    elif args.monitor_type == "gnn":
        # Use GNNMonitor
        monitor = GNNMonitor(
            gnn_project_path=args.gnn_project_path,
            model_name=args.gnn_model_name,
            score_metric=args.gnn_score_metric,
            save_debug_output=True,  # Enable debug output
            debug_output_dir=str(trainer_output_dir)  # Save to trainer output directory
        )
        print(f"  Monitor type: GNN-based")
        print(f"  GNN project path: {args.gnn_project_path}")
        print(f"  GNN model name: {args.gnn_model_name}")
        print(f"  Score metric: {args.gnn_score_metric}")
        print(f"  Debug output: {trainer_output_dir}/gnn_debug/")
    elif args.monitor_type == "gbt":
        # Use GBTMonitor (LightGBM-based)
        if args.gbt_model_path is None:
            raise ValueError(
                "--gbt-model-path is required when --monitor-type=gbt.\n"
                "Train the model first:\n"
                "  export AML_DATA_DIR=/path/to/your/data\n"
                "  python scripts/train_gbt_monitor.py "
                "--accounts \$AML_DATA_DIR/accounts.csv "
                "--transactions \$AML_DATA_DIR/transactions.csv "
                "--output models/gbt_model.pkl"
            )
        monitor = GBTMonitor(
            model_path=args.gbt_model_path,
            score_aggregation=args.gbt_score_aggregation,
            save_debug_output=True,
            debug_output_dir=str(trainer_output_dir),
        )
        print(f"  Monitor type: GBT (LightGBM)")
        print(f"  Model path: {args.gbt_model_path}")
        print(f"  Score aggregation: {args.gbt_score_aggregation}")
        print(f"  Debug output: {trainer_output_dir}/gbt_debug/")
    else:
        # Use RandomMonitor (for testing)
        monitor = RandomMonitor(
            mode=args.monitor_mode,
            seed=args.monitor_seed
        )
        print(f"  Monitor type: Random")
        print(f"  Monitor mode: {args.monitor_mode}")

    reward_config = {
        "w_detection": args.w_detection,
        "w_validity": args.w_validity,
        "w_budget": args.w_budget,
        "w_final_bonus": args.w_final_bonus,
        "reward_mode": "hit_rate",  # Use hit_rate mode instead of score
    }
    reward_fn = GRPOReward(monitor, reward_config)
    print(f"  Reward weights: detection={args.w_detection}, validity={args.w_validity}, "
          f"budget={args.w_budget}, final_bonus={args.w_final_bonus}")
    print(f"  Reward mode: hit_rate (encourages complex laundering networks)")

    # 6. Setup GRPO trainer
    print("\n[6/6] Setting up GRPO trainer...")
    trainer_config = {
        # GRPO config
        "num_samples_per_cluster": args.num_samples,
        "output_dir": output_dir / ("grpo_train" if args.train_mode == "train" else "grpo_data"),
        "save_interval": args.save_interval,

        # Pipeline config (from runner)
        "seed": args.seed,
        "topk_candidates": args.topk,
        "max_steps_per_cluster": args.max_steps,
        "max_attempts_per_step": 2,
        "fail_limit": 3,
        "max_merges_per_cluster": 2,
        "max_splits_per_cluster": 2,
        "max_new_nodes_ratio": 0.3,
        "max_new_edges_ratio": 0.5,
        "timestamp_output_format": "original",

        # Parallel sampling config (disabled due to CPU bottleneck)
        "use_parallel_sampling": False,
        "num_sampling_workers": 2,
    }

    # Add training-specific config if in train mode
    if args.train_mode == "train":
        trainer_config.update({
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_grad_norm": args.max_grad_norm,
            "use_lora": args.use_lora,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
        })
        trainer = GRPOLocalTrainer(agent, reward_fn, trainer_config)
        print(f"  Mode: Training (gradient updates enabled)")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Use LoRA: {args.use_lora}")
    else:
        trainer = GRPOTrainer(agent, reward_fn, trainer_config)
        print(f"  Mode: Data collection (no gradient updates)")

    print(f"  Samples per cluster: {args.num_samples}")
    print(f"  Max steps per cluster: {args.max_steps}")
    print(f"  Output directory: {trainer.output_dir}")

    # 7. Train
    print("\n" + "=" * 80)
    print("Starting GRPO Training")
    print("=" * 80)

    stats = trainer.train(clusters, transactions_df, accounts_df)

    # 8. Print summary
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"Total trajectories: {stats.get('total_trajectories', 'N/A')}")
    print(f"Total steps: {stats.get('total_steps', 'N/A')}")
    print(f"Mean return: {stats.get('mean_return', 0):.2f} ± {stats.get('std_return', 0):.2f}")
    print(f"Best return: {stats.get('best_return', 0):.2f}")
    print(f"Worst return: {stats.get('worst_return', 0):.2f}")
    if 'mean_advantage' in stats:
        print(f"Mean advantage: {stats['mean_advantage']:.2f}")
    if 'mean_step_reward' in stats:
        print(f"Mean step reward: {stats['mean_step_reward']:.2f}")
    if 'successful_evasions' in stats:
        print(f"Successful evasions: {stats['successful_evasions']}/{stats['total_clusters']}")
    print(f"\nOutput saved to: {trainer.output_dir}")

    if args.train_mode == "train":
        print("\nTraining complete!")
        print(f"  Policy loss: {stats.get('mean_policy_loss', 'N/A')}")
        print("\nNext steps:")
        print("  1. Test the trained model on new clusters")
        print("  2. Continue training with more clusters if needed")
        print(f"  3. Model checkpoint saved to: {trainer.output_dir}")
    else:
        print("\nData collection complete!")
        print("\nNext steps:")
        print("  1. Inspect training_examples_final.jsonl for collected data")
        print("  2. Use data for fine-tuning via API or train local model")
        print("  3. To train local model with gradient updates, use --train-mode=train")


if __name__ == "__main__":
    main()
