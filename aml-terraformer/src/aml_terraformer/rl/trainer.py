"""GRPO trainer for AML perturbation."""

import json
import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Dict, Any, List
from ..core.clusters import LaunderingCluster
from ..agent import LLMAgent
from .sampler import GRPOSampler, Trajectory
from .reward import GRPOReward


class GRPOTrainer:
    """GRPO (Group Relative Policy Optimization) trainer.

    This trainer implements the GRPO algorithm for training an LLM agent
    to perform AML transaction graph perturbations. Since the LLM is accessed
    via API (DeepSeek/OpenAI), this trainer operates in "data collection mode":

    1. Sample K trajectories per cluster (group)
    2. Compute group-relative advantages
    3. Save training data for future fine-tuning

    The saved data can be used for:
    - Fine-tuning via OpenAI/DeepSeek fine-tuning API
    - Training a local model with supervised learning
    - Offline RL training

    Args:
        agent: LLM agent for decision making
        reward_fn: Reward function
        config: Configuration dict with keys:
            - num_samples_per_cluster: Number of trajectories per cluster (K)
            - output_dir: Directory to save training data
            - save_interval: Save data every N clusters

    Example:
        >>> from aml_terraformer.monitor import RandomMonitor
        >>> from aml_terraformer.agent import LLMAgent
        >>>
        >>> monitor = RandomMonitor(mode='decreasing')
        >>> reward_fn = GRPOReward(monitor, config={})
        >>> trainer = GRPOTrainer(agent, reward_fn, config={})
        >>>
        >>> trainer.train(clusters, transactions_df, accounts_df)
    """

    def __init__(
        self,
        agent: LLMAgent,
        reward_fn: GRPOReward,
        config: Dict[str, Any]
    ):
        """Initialize GRPO trainer."""
        self.agent = agent
        self.reward_fn = reward_fn
        self.config = config

        # GRPO hyperparameters
        self.num_samples_per_cluster = config.get("num_samples_per_cluster", 4)
        self.advantage_normalization = config.get("advantage_normalization", "group")  # 'group' or 'global'

        # Data collection
        self.output_dir = Path(config.get("output_dir", "output/grpo_data"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = config.get("save_interval", 10)

        # Statistics
        self.all_trajectories = []
        self.cluster_statistics = []

        # Create sampler
        self.sampler = GRPOSampler(agent, reward_fn, config)

    def train(
        self,
        clusters: List[LaunderingCluster],
        transactions_df: pd.DataFrame,
        accounts_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Train on multiple clusters.

        Args:
            clusters: List of LaunderingCluster objects
            transactions_df: Full transactions DataFrame
            accounts_df: Full accounts DataFrame

        Returns:
            Dict with training statistics
        """
        print(f"[GRPO] Starting training on {len(clusters)} clusters")
        print(f"[GRPO] Sampling {self.num_samples_per_cluster} trajectories per cluster")
        print(f"[GRPO] Output directory: {self.output_dir}")

        # Track overall timing
        training_start_time = time.time()
        total_sampling_time = 0.0
        total_advantage_time = 0.0
        total_prepare_time = 0.0
        total_stats_time = 0.0
        total_save_time = 0.0

        for i, cluster in enumerate(clusters):
            cluster_start_time = time.time()
            print(f"\n[GRPO] Cluster {i+1}/{len(clusters)}: ID={cluster.cluster_id}")

            # Sample trajectories for this cluster
            t_sampling_start = time.time()
            trajectories = self.sampler.sample_trajectories(
                cluster,
                transactions_df,
                accounts_df,
                num_samples=self.num_samples_per_cluster
            )
            t_sampling_end = time.time()
            sampling_time = t_sampling_end - t_sampling_start
            total_sampling_time += sampling_time

            # Compute advantages
            t_advantage_start = time.time()
            advantages = self._compute_advantages(trajectories)
            t_advantage_end = time.time()
            advantage_time = t_advantage_end - t_advantage_start
            total_advantage_time += advantage_time

            # Prepare training examples
            t_prepare_start = time.time()
            training_examples = self._prepare_training_examples(
                trajectories, advantages
            )
            t_prepare_end = time.time()
            prepare_time = t_prepare_end - t_prepare_start
            total_prepare_time += prepare_time

            # Save data
            self.all_trajectories.extend(training_examples)

            # Compute statistics
            t_stats_start = time.time()
            stats = self._compute_cluster_statistics(trajectories, advantages)
            self.cluster_statistics.append(stats)
            t_stats_end = time.time()
            stats_time = t_stats_end - t_stats_start
            total_stats_time += stats_time

            # Print stats
            print(f"  Returns: {[f'{t.get_return():.2f}' for t in trajectories]}")
            print(f"  Advantages: {[f'{a:.2f}' for a in advantages]}")
            print(f"  Best return: {stats['best_return']:.2f}")
            print(f"  Mean length: {stats['mean_length']:.1f}")
            print(f"  Final scores: {[f'{t.final_detection_score:.3f}' for t in trajectories]}")

            # Save intermediate results
            if (i + 1) % self.save_interval == 0:
                t_save_start = time.time()
                self._save_data(checkpoint=True, cluster_idx=i+1)
                t_save_end = time.time()
                save_time = t_save_end - t_save_start
                total_save_time += save_time
            else:
                save_time = 0.0

            # Print timing breakdown for this cluster
            cluster_total_time = time.time() - cluster_start_time
            print(f"\n[GRPO] ⏱️  Cluster {i+1} timing breakdown:")
            print(f"  Total time: {cluster_total_time:.2f}s")
            print(f"    - Sampling: {sampling_time:.2f}s ({sampling_time/cluster_total_time*100:.1f}%)")
            print(f"    - Advantages: {advantage_time:.3f}s ({advantage_time/cluster_total_time*100:.1f}%)")
            print(f"    - Prepare examples: {prepare_time:.3f}s ({prepare_time/cluster_total_time*100:.1f}%)")
            print(f"    - Statistics: {stats_time:.3f}s ({stats_time/cluster_total_time*100:.1f}%)")
            if save_time > 0:
                print(f"    - Save checkpoint: {save_time:.3f}s ({save_time/cluster_total_time*100:.1f}%)")

        # Final save
        t_final_save_start = time.time()
        self._save_data(checkpoint=False)
        t_final_save_end = time.time()
        final_save_time = t_final_save_end - t_final_save_start
        total_save_time += final_save_time

        # Compute overall statistics
        overall_stats = self._compute_overall_statistics()

        # Print overall timing summary
        training_total_time = time.time() - training_start_time
        print(f"\n{'='*80}")
        print(f"[GRPO] ⏱️  === OVERALL TIMING SUMMARY ===")
        print(f"{'='*80}")
        print(f"Total training time: {training_total_time:.2f}s ({training_total_time/60:.1f} min)")
        print(f"\nTime breakdown:")
        print(f"  - Sampling trajectories: {total_sampling_time:.2f}s ({total_sampling_time/training_total_time*100:.1f}%)")
        print(f"  - Computing advantages: {total_advantage_time:.2f}s ({total_advantage_time/training_total_time*100:.1f}%)")
        print(f"  - Preparing examples: {total_prepare_time:.2f}s ({total_prepare_time/training_total_time*100:.1f}%)")
        print(f"  - Computing statistics: {total_stats_time:.2f}s ({total_stats_time/training_total_time*100:.1f}%)")
        print(f"  - Saving data: {total_save_time:.2f}s ({total_save_time/training_total_time*100:.1f}%)")

        avg_cluster_time = training_total_time / len(clusters) if len(clusters) > 0 else 0
        print(f"\nAverage time per cluster: {avg_cluster_time:.2f}s")
        print(f"{'='*80}\n")

        print(f"[GRPO] Training complete!")
        print(f"  Total trajectories: {overall_stats['total_trajectories']}")
        print(f"  Mean return: {overall_stats['mean_return']:.2f}")
        print(f"  Best return: {overall_stats['best_return']:.2f}")
        print(f"  Data saved to: {self.output_dir}")

        return overall_stats

    def _compute_advantages(self, trajectories: List[Trajectory]) -> List[float]:
        """Compute group-relative advantages for trajectories.

        In GRPO, advantages are computed by normalizing returns within each group:
            advantage_i = (return_i - mean(returns)) / std(returns)

        Args:
            trajectories: List of Trajectory objects (same cluster)

        Returns:
            List of advantage values (one per trajectory)
        """
        returns = [traj.get_return() for traj in trajectories]

        if len(returns) == 1:
            # Single trajectory: advantage = 0
            return [0.0]

        # Group normalization
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return < 1e-8:
            # All returns are the same: all advantages = 0
            advantages = [0.0] * len(returns)
        else:
            advantages = [(r - mean_return) / std_return for r in returns]

        return advantages

    def _prepare_training_examples(
        self,
        trajectories: List[Trajectory],
        advantages: List[float]
    ) -> List[Dict[str, Any]]:
        """Prepare training examples from trajectories.

        Each training example contains:
        - prompt: Input to LLM
        - action: Action taken (tool + args)
        - advantage: Advantage value for weighting
        - reward: Step reward
        - metadata: Additional info

        Args:
            trajectories: List of Trajectory objects
            advantages: List of advantage values

        Returns:
            List of training example dicts
        """
        training_examples = []

        for traj, advantage in zip(trajectories, advantages):
            for step in traj.steps:
                example = {
                    "cluster_id": traj.cluster_id,
                    "step_idx": step["step_idx"],
                    "prompt": step["prompt"],
                    "action": step["action"],
                    "action_result": step.get("action_result"),  # Add for debugging
                    "reward": step["reward"],
                    "advantage": advantage,  # Trajectory-level advantage
                    "trajectory_return": traj.get_return(),
                    "is_terminal": step["is_terminal"],
                    "decision": step["decision"],
                    "info": step["info"],
                }
                training_examples.append(example)

        return training_examples

    def _compute_cluster_statistics(
        self,
        trajectories: List[Trajectory],
        advantages: List[float]
    ) -> Dict[str, Any]:
        """Compute statistics for a cluster."""
        returns = [traj.get_return() for traj in trajectories]
        lengths = [traj.get_length() for traj in trajectories]
        final_scores = [traj.final_detection_score for traj in trajectories if traj.final_detection_score is not None]

        stats = {
            "cluster_id": trajectories[0].cluster_id,
            "num_trajectories": len(trajectories),
            "mean_return": np.mean(returns),
            "std_return": np.std(returns),
            "best_return": np.max(returns),
            "worst_return": np.min(returns),
            "mean_length": np.mean(lengths),
            "mean_advantage": np.mean(advantages),
            "std_advantage": np.std(advantages),
        }

        if final_scores:
            stats["mean_final_score"] = np.mean(final_scores)
            stats["best_final_score"] = np.min(final_scores)  # Lower is better

        return stats

    def _compute_overall_statistics(self) -> Dict[str, Any]:
        """Compute overall training statistics."""
        all_returns = [ex["trajectory_return"] for ex in self.all_trajectories]
        all_advantages = [ex["advantage"] for ex in self.all_trajectories]
        all_rewards = [ex["reward"] for ex in self.all_trajectories]

        # Count successful evasions
        successful_evasions = sum(
            1 for stats in self.cluster_statistics
            if stats.get("best_final_score", 1.0) < 0.3
        )

        return {
            "total_trajectories": len(set((ex["cluster_id"], ex["trajectory_return"]) for ex in self.all_trajectories)),
            "total_steps": len(self.all_trajectories),
            "mean_return": np.mean(all_returns),
            "std_return": np.std(all_returns),
            "best_return": np.max(all_returns),
            "worst_return": np.min(all_returns),
            "mean_advantage": np.mean(all_advantages),
            "std_advantage": np.std(all_advantages),
            "mean_step_reward": np.mean(all_rewards),
            "successful_evasions": successful_evasions,
            "total_clusters": len(self.cluster_statistics),
        }

    def _save_data(self, checkpoint: bool = False, cluster_idx: int = None):
        """Save training data to disk.

        Args:
            checkpoint: Whether this is a checkpoint save
            cluster_idx: Cluster index (for checkpoint naming)
        """
        if checkpoint:
            suffix = f"_checkpoint_cluster{cluster_idx}"
        else:
            suffix = "_final"

        # Save training examples (JSONL format)
        examples_path = self.output_dir / f"training_examples{suffix}.jsonl"
        with open(examples_path, "w") as f:
            for ex in self.all_trajectories:
                f.write(json.dumps(ex) + "\n")

        # Save cluster statistics (JSON format)
        stats_path = self.output_dir / f"cluster_statistics{suffix}.json"
        with open(stats_path, "w") as f:
            json.dump(self.cluster_statistics, f, indent=2)

        # Save overall statistics
        if not checkpoint:
            overall_stats = self._compute_overall_statistics()
            overall_path = self.output_dir / "overall_statistics.json"
            with open(overall_path, "w") as f:
                json.dump(overall_stats, f, indent=2)

        # Save reward statistics
        reward_stats = self.reward_fn.get_statistics()
        reward_path = self.output_dir / f"reward_statistics{suffix}.json"
        with open(reward_path, "w") as f:
            json.dump(reward_stats, f, indent=2)

        print(f"  [GRPO] Saved data to {self.output_dir}")

    def load_training_data(self, path: str) -> List[Dict[str, Any]]:
        """Load training data from JSONL file.

        Args:
            path: Path to JSONL file

        Returns:
            List of training examples
        """
        examples = []
        with open(path, "r") as f:
            for line in f:
                examples.append(json.loads(line))
        return examples
