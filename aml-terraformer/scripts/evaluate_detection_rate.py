"""Evaluate detection rate comparison between base model and trained LoRA model.

This script compares the detection evasion performance between:
1. Base model (Qwen2.5-7B without fine-tuning)
2. Trained model (Qwen2.5-7B + LoRA adapter from GRPO training)

The evaluation measures how well each model can modify transactions to evade
rule-based AML detection while maintaining data validity.

Usage:
    export AML_DATA_DIR=/path/to/your/data
    export QWEN_MODEL_PATH=/path/to/Qwen2.5-7B

    python scripts/evaluate_detection_rate.py \
        --accounts $AML_DATA_DIR/accounts.csv \
        --transactions $AML_DATA_DIR/transactions.csv \
        --base-model $QWEN_MODEL_PATH \
        --lora-checkpoint output/train_lora/grpo_train/checkpoint-50 \
        --output output/evaluation \
        --num-clusters 100 \
        --num-samples 3 \
        --device cuda
"""

import argparse
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aml_terraformer.io import read_transactions, read_accounts
from aml_terraformer.core.normalize import normalize_data
from aml_terraformer.core.clusters import find_laundering_clusters
from aml_terraformer.agent.client_interface import LocalQwenLLMClient
from aml_terraformer.agent.llm_agent import LLMAgent
from aml_terraformer.monitor import RuleBasedMonitor
from aml_terraformer.rl.sampler import GRPOSampler
from aml_terraformer.rl.reward import GRPOReward


def load_base_model(model_path: str, device: str = "cuda", **kwargs) -> LocalQwenLLMClient:
    """Load base model without LoRA adapters.

    Args:
        model_path: Path to base model
        device: Device to use
        **kwargs: Additional arguments for model loading

    Returns:
        LocalQwenLLMClient instance
    """
    print(f"Loading base model from {model_path}...")
    client = LocalQwenLLMClient(
        model_path=model_path,
        device=device,
        **kwargs
    )
    print(f"Base model loaded successfully")
    return client


def load_lora_model(base_model_path: str, lora_checkpoint: str, device: str = "cuda", **kwargs) -> LocalQwenLLMClient:
    """Load model with LoRA adapters.

    Args:
        base_model_path: Path to base model
        lora_checkpoint: Path to LoRA checkpoint directory
        device: Device to use
        **kwargs: Additional arguments for model loading

    Returns:
        LocalQwenLLMClient instance with LoRA adapters loaded
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch

    print(f"Loading LoRA model...")
    print(f"  Base model: {base_model_path}")
    print(f"  LoRA checkpoint: {lora_checkpoint}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )

    # Load base model
    load_in_8bit = kwargs.pop('load_in_8bit', False)
    load_in_4bit = kwargs.pop('load_in_4bit', False)

    load_kwargs = {
        "pretrained_model_name_or_path": base_model_path,
        "trust_remote_code": True,
    }

    if load_in_8bit:
        print("  Loading in 8-bit precision...")
        load_kwargs["load_in_8bit"] = True
        load_kwargs["device_map"] = "auto"
    elif load_in_4bit:
        print("  Loading in 4-bit precision...")
        load_kwargs["load_in_4bit"] = True
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["dtype"] = torch.float16 if "cuda" in device else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, lora_checkpoint)

    # Move to device if not using device_map
    if not (load_in_8bit or load_in_4bit):
        model = model.to(device)

    model.eval()
    print(f"  LoRA model loaded successfully")

    # Create a custom client object that wraps the LoRA model
    client = LocalQwenLLMClient.__new__(LocalQwenLLMClient)
    client.model = model
    client.tokenizer = tokenizer
    client.device = device
    client.model_path = base_model_path
    client.max_new_tokens = kwargs.get('max_new_tokens', 512)
    client.temperature = kwargs.get('temperature', 0.7)
    client.top_p = kwargs.get('top_p', 0.9)
    client.do_sample = kwargs.get('do_sample', True)

    return client


def evaluate_model_on_clusters(
    agent: LLMAgent,
    monitor: RuleBasedMonitor,
    clusters: List,
    transactions_df: pd.DataFrame,
    accounts_df: pd.DataFrame,
    num_samples: int,
    config: Dict[str, Any],
    model_name: str
) -> Dict[str, Any]:
    """Evaluate a model on test clusters.

    Args:
        agent: LLM agent to evaluate
        monitor: Rule-based monitor for detection
        clusters: List of test clusters
        transactions_df: Transactions DataFrame
        accounts_df: Accounts DataFrame
        num_samples: Number of samples per cluster
        config: Pipeline configuration
        model_name: Name for logging

    Returns:
        Dict with evaluation results
    """
    print(f"\n{'='*80}")
    print(f"Evaluating {model_name}")
    print(f"{'='*80}")

    results = []

    # Create reward function for sampling
    reward_config = {
        "w_detection": 10.0,
        "w_validity": 1.0,
        "w_budget": 0.1,
        "w_final_bonus": 0.0,
        "reward_mode": "hit_rate",
    }
    reward_fn = GRPOReward(monitor, reward_config)

    # Create sampler
    sampler = GRPOSampler(agent, reward_fn, config)

    for cluster_idx, cluster in enumerate(clusters):
        print(f"\n[Cluster {cluster_idx+1}/{len(clusters)}] ID={cluster.cluster_id}")
        print(f"  Nodes: {len(cluster.nodes_in_cluster)}")

        # Get initial detection score
        initial_score = monitor.predict_proba(
            transactions_df,
            accounts_df,
            cluster.nodes_in_cluster
        )
        print(f"  Initial detection score: {initial_score:.4f}")

        # Sample trajectories for this cluster
        print(f"  Sampling {num_samples} trajectories...")
        trajectories = sampler.sample_trajectories(
            cluster,
            transactions_df,
            accounts_df,
            num_samples=num_samples
        )

        # Extract results from trajectories
        cluster_results = []
        for sample_idx, traj in enumerate(trajectories):
            final_score = traj.final_detection_score if traj.final_detection_score is not None else initial_score
            score_reduction = initial_score - final_score
            relative_reduction = score_reduction / initial_score if initial_score > 0 else 0
            # Success criteria: either reduced by > 5% relatively OR absolute reduction > 0.05
            success = (relative_reduction > 0.05) or (score_reduction > 0.05)
            num_steps = traj.get_length()

            cluster_results.append({
                'cluster_id': cluster.cluster_id,
                'sample_idx': sample_idx,
                'initial_score': initial_score,
                'final_score': final_score,
                'score_reduction': score_reduction,
                'relative_reduction': score_reduction / initial_score if initial_score > 0 else 0,
                'success': success,
                'num_steps': num_steps,
                'total_return': traj.get_return(),
            })

            print(f"    Sample {sample_idx+1}: final={final_score:.4f}, reduction={score_reduction:.4f}, return={traj.get_return():.2f}")

        # Average across samples
        avg_final_score = np.mean([r['final_score'] for r in cluster_results])
        avg_reduction = np.mean([r['score_reduction'] for r in cluster_results])
        success_rate = np.mean([r['success'] for r in cluster_results])

        results.append({
            'cluster_id': cluster.cluster_id,
            'initial_score': initial_score,
            'avg_final_score': avg_final_score,
            'avg_reduction': avg_reduction,
            'avg_relative_reduction': avg_reduction / initial_score if initial_score > 0 else 0,
            'success_rate': success_rate,
            'samples': cluster_results,
        })

        print(f"  Average final score: {avg_final_score:.4f}")
        print(f"  Average reduction: {avg_reduction:.4f}")
        print(f"  Success rate: {success_rate:.2%}")

    # Compute overall statistics
    overall_stats = {
        'model_name': model_name,
        'num_clusters': len(clusters),
        'num_samples_per_cluster': num_samples,
        'avg_initial_score': np.mean([r['initial_score'] for r in results]),
        'avg_final_score': np.mean([r['avg_final_score'] for r in results]),
        'avg_score_reduction': np.mean([r['avg_reduction'] for r in results]),
        'avg_relative_reduction': np.mean([r['avg_relative_reduction'] for r in results]),
        'avg_success_rate': np.mean([r['success_rate'] for r in results]),
        'cluster_results': results,
    }

    return overall_stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate detection rate: base vs LoRA model")

    # Data
    parser.add_argument("--accounts", required=True, help="Path to accounts CSV")
    parser.add_argument("--transactions", required=True, help="Path to transactions CSV")
    parser.add_argument("--output", required=True, help="Output directory for evaluation results")

    # Models
    parser.add_argument("--base-model", required=True, help="Path to base model (e.g., Qwen2.5-7B)")
    parser.add_argument("--lora-checkpoint", required=True, help="Path to LoRA checkpoint directory")

    # Model config
    parser.add_argument("--device", default="cuda", help="Device for model (cuda, cpu, etc.)")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max tokens for generation")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit")

    # Evaluation config
    parser.add_argument("--num-clusters", type=int, default=10, help="Number of clusters to evaluate")
    parser.add_argument("--min-cluster-size", type=int, default=3, help="Min cluster size")
    parser.add_argument("--max-cluster-size", type=int, default=100, help="Max cluster size")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples per cluster")

    # Monitor config
    parser.add_argument("--rule-config", default=None, help="Path to rule.json")
    parser.add_argument("--score-aggregation", default="weighted_average",
                       choices=["weighted_average", "max", "count"],
                       help="Score aggregation method")

    # Pipeline config
    parser.add_argument("--max-steps", type=int, default=2, help="Max steps per cluster")
    parser.add_argument("--topk", type=int, default=4, help="Top-K candidates")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Comparison mode
    parser.add_argument("--skip-base", action="store_true", help="Skip base model evaluation")
    parser.add_argument("--skip-lora", action="store_true", help="Skip LoRA model evaluation")

    args = parser.parse_args()

    # Validate
    if args.skip_base and args.skip_lora:
        parser.error("Cannot skip both base and LoRA evaluation")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Detection Rate Evaluation: Base vs LoRA Model")
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

    # Limit number of clusters
    if len(clusters) > args.num_clusters:
        print(f"  Using first {args.num_clusters} clusters for evaluation")
        clusters = clusters[:args.num_clusters]

    # 4. Setup monitor
    print("\n[4/6] Setting up monitor...")
    if args.rule_config is None:
        rule_config_path = Path(__file__).parent.parent / "rule" / "data" / "rule.json"
    else:
        rule_config_path = Path(args.rule_config)

    monitor = RuleBasedMonitor(
        rule_config_path=str(rule_config_path),
        score_aggregation=args.score_aggregation,
        save_debug_output=False
    )
    print(f"  Rule config: {rule_config_path}")
    print(f"  Score aggregation: {args.score_aggregation}")

    # Pipeline config
    pipeline_config = {
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
    }

    # Model kwargs
    model_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.7,
        "load_in_8bit": args.load_in_8bit,
        "load_in_4bit": args.load_in_4bit,
    }

    # Results
    evaluation_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'base_model': args.base_model,
            'lora_checkpoint': args.lora_checkpoint,
            'num_clusters': len(clusters),
            'num_samples_per_cluster': args.num_samples,
            'pipeline_config': pipeline_config,
        },
        'results': {}
    }

    # 5. Evaluate base model
    if not args.skip_base:
        print("\n[5a/6] Evaluating base model...")
        base_client = load_base_model(args.base_model, args.device, **model_kwargs)
        base_agent = LLMAgent(base_client)

        base_results = evaluate_model_on_clusters(
            agent=base_agent,
            monitor=monitor,
            clusters=clusters,
            transactions_df=transactions_df,
            accounts_df=accounts_df,
            num_samples=args.num_samples,
            config=pipeline_config,
            model_name="Base Model"
        )

        evaluation_results['results']['base'] = base_results

        # Clean up
        del base_client, base_agent
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 6. Evaluate LoRA model
    if not args.skip_lora:
        print("\n[5b/6] Evaluating LoRA model...")
        lora_client = load_lora_model(args.base_model, args.lora_checkpoint, args.device, **model_kwargs)
        lora_agent = LLMAgent(lora_client)

        lora_results = evaluate_model_on_clusters(
            agent=lora_agent,
            monitor=monitor,
            clusters=clusters,
            transactions_df=transactions_df,
            accounts_df=accounts_df,
            num_samples=args.num_samples,
            config=pipeline_config,
            model_name="LoRA Model"
        )

        evaluation_results['results']['lora'] = lora_results

        # Clean up
        del lora_client, lora_agent
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 7. Compare results
    print("\n[6/6] Comparison Summary")
    print("=" * 80)

    if 'base' in evaluation_results['results'] and 'lora' in evaluation_results['results']:
        base_stats = evaluation_results['results']['base']
        lora_stats = evaluation_results['results']['lora']

        print(f"\nBase Model:")
        print(f"  Average initial detection score: {base_stats['avg_initial_score']:.4f}")
        print(f"  Average final detection score:   {base_stats['avg_final_score']:.4f}")
        print(f"  Average score reduction:         {base_stats['avg_score_reduction']:.4f} ({base_stats['avg_relative_reduction']:.2%})")
        print(f"  Average success rate:            {base_stats['avg_success_rate']:.2%}")

        print(f"\nLoRA Model (Trained):")
        print(f"  Average initial detection score: {lora_stats['avg_initial_score']:.4f}")
        print(f"  Average final detection score:   {lora_stats['avg_final_score']:.4f}")
        print(f"  Average score reduction:         {lora_stats['avg_score_reduction']:.4f} ({lora_stats['avg_relative_reduction']:.2%})")
        print(f"  Average success rate:            {lora_stats['avg_success_rate']:.2%}")

        print(f"\nImprovement (LoRA vs Base):")
        improvement_reduction = lora_stats['avg_score_reduction'] - base_stats['avg_score_reduction']
        improvement_final = base_stats['avg_final_score'] - lora_stats['avg_final_score']
        improvement_success = lora_stats['avg_success_rate'] - base_stats['avg_success_rate']

        print(f"  Score reduction improvement:     {improvement_reduction:+.4f}")
        print(f"  Final score improvement:         {improvement_final:+.4f} (lower is better)")
        print(f"  Success rate improvement:        {improvement_success:+.2%}")

        if improvement_final > 0:
            print(f"\n✓ LoRA model achieves LOWER detection rates (better evasion)")
        else:
            print(f"\n✗ LoRA model does NOT achieve lower detection rates")

        # Add comparison stats
        evaluation_results['comparison'] = {
            'score_reduction_improvement': improvement_reduction,
            'final_score_improvement': improvement_final,
            'success_rate_improvement': improvement_success,
            'is_better': improvement_final > 0,
        }

    # Save results
    output_file = output_dir / "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
