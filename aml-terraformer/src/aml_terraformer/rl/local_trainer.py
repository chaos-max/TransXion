"""GRPO Local Trainer for training local models with gradient updates.

This trainer extends the base GRPO trainer to support actual gradient-based
training of local transformer models (e.g., Qwen3-1.7B).

Key differences from base GRPOTrainer:
1. Computes log probabilities during trajectory collection
2. Performs gradient updates using policy gradient with group-relative advantages
3. Supports LoRA/full fine-tuning
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import logging
from datetime import datetime

from ..core.clusters import LaunderingCluster
from ..agent import LLMAgent
from .sampler import GRPOSampler, Trajectory
from .reward import GRPOReward


class GRPOLocalTrainer:
    """GRPO trainer with gradient updates for local models.

    This trainer implements full GRPO training loop:
    1. Sample K trajectories per cluster (group)
    2. Compute group-relative advantages
    3. Compute policy gradient loss
    4. Update model parameters via backpropagation

    Args:
        agent: LLM agent (must use LocalQwenLLMClient)
        reward_fn: Reward function
        config: Configuration dict with keys:
            - num_samples_per_cluster: Number of trajectories per cluster (K)
            - learning_rate: Learning rate for optimizer
            - num_epochs: Number of training epochs
            - batch_size: Batch size for training
            - gradient_accumulation_steps: Steps to accumulate gradients
            - max_grad_norm: Max gradient norm for clipping
            - output_dir: Directory to save checkpoints
            - save_interval: Save checkpoint every N clusters
            - use_lora: Whether to use LoRA (default: False)
            - lora_r: LoRA rank (default: 8)
            - lora_alpha: LoRA alpha (default: 16)
    """

    def __init__(
        self,
        agent: LLMAgent,
        reward_fn: GRPOReward,
        config: Dict[str, Any]
    ):
        """Initialize GRPO local trainer."""
        self.agent = agent
        self.reward_fn = reward_fn
        self.config = config

        # Check that agent uses local client
        if not hasattr(agent.client, 'model'):
            raise ValueError(
                "GRPOLocalTrainer requires agent with LocalQwenLLMClient. "
                "Current client does not have 'model' attribute."
            )

        self.model = agent.client.model
        self.tokenizer = agent.client.tokenizer
        self.device = agent.client.device

        # GRPO hyperparameters
        self.num_samples_per_cluster = config.get("num_samples_per_cluster", 4)
        self.learning_rate = config.get("learning_rate", 1e-5)
        self.num_epochs = config.get("num_epochs", 1)
        self.batch_size = config.get("batch_size", 1)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 4)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)

        # LoRA configuration
        self.use_lora = config.get("use_lora", False)
        if self.use_lora:
            self._setup_lora(
                lora_r=config.get("lora_r", 8),
                lora_alpha=config.get("lora_alpha", 16),
                lora_dropout=config.get("lora_dropout", 0.05)
            )

        # Setup optimizer
        self._setup_optimizer()

        # Output directory
        self.output_dir = Path(config.get("output_dir", "output/grpo_local"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = config.get("save_interval", 5)

        # Statistics
        self.training_stats = []
        self.cluster_statistics = []

        # Setup logging
        self._setup_logging()

        # Create sampler
        self.sampler = GRPOSampler(agent, reward_fn, config)

        print(f"[GRPO Local] Initialized trainer")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"  Use LoRA: {self.use_lora}")
        print(f"  Detailed log: {self.log_file}")

    def _setup_lora(self, lora_r: int, lora_alpha: int, lora_dropout: float):
        """Setup LoRA for efficient fine-tuning."""
        try:
            from peft import get_peft_model, LoraConfig, TaskType
        except ImportError:
            raise ImportError(
                "peft package required for LoRA. Install with: pip install peft"
            )

        print(f"[GRPO Local] Setting up LoRA (r={lora_r}, alpha={lora_alpha})")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen modules
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        # Only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=0.01
        )

        print(f"[GRPO Local] Optimizer initialized with {len(trainable_params)} parameter groups")

    def _setup_logging(self):
        """Setup detailed logging for training."""
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"training_detailed_{timestamp}.log"

        # Print log file path for debugging
        print(f"[GRPO Local] Creating log file: {self.log_file}")

        # Setup file logger
        self.logger = logging.getLogger('GRPO_Local')
        self.logger.setLevel(logging.INFO)

        # IMPORTANT: Prevent propagation to root logger to avoid console output
        self.logger.propagate = False

        # Clear existing handlers
        self.logger.handlers.clear()

        # File handler
        fh = logging.FileHandler(str(self.log_file), mode='w')
        fh.setLevel(logging.INFO)

        # Format: timestamp - message
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)

        self.logger.addHandler(fh)

        # Log header
        self.logger.info("="*80)
        self.logger.info("GRPO Local Training - Detailed Log")
        self.logger.info("="*80)
        self.logger.info(f"Learning rate: {self.learning_rate}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        self.logger.info(f"Use LoRA: {self.use_lora}")
        self.logger.info(f"Samples per cluster: {self.num_samples_per_cluster}")
        self.logger.info("="*80)

        print(f"[GRPO Local] Log file created successfully")

    def train(
        self,
        clusters: List[LaunderingCluster],
        transactions_df: pd.DataFrame,
        accounts_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Train on multiple clusters with gradient updates.

        Args:
            clusters: List of LaunderingCluster objects
            transactions_df: Full transactions DataFrame
            accounts_df: Full accounts DataFrame

        Returns:
            Dict with training statistics
        """
        print(f"\n[GRPO Local] Starting training on {len(clusters)} clusters")
        print(f"[GRPO Local] Samples per cluster: {self.num_samples_per_cluster}")
        print(f"[GRPO Local] Output directory: {self.output_dir}")

        self.model.train()
        global_step = 0

        for cluster_idx, cluster in enumerate(clusters):
            print(f"\n{'='*80}")
            print(f"[GRPO Local] Cluster {cluster_idx+1}/{len(clusters)}: ID={cluster.cluster_id}")
            print(f"{'='*80}")

            self.logger.info("")
            self.logger.info("="*80)
            self.logger.info(f"Cluster {cluster_idx+1}/{len(clusters)}: ID={cluster.cluster_id}")
            self.logger.info(f"Nodes: {len(cluster.nodes_in_cluster)}")
            self.logger.info("="*80)

            # 1. Sample trajectories for this cluster
            print(f"[1/3] Sampling {self.num_samples_per_cluster} trajectories...")
            trajectories = self.sampler.sample_trajectories(
                cluster,
                transactions_df,
                accounts_df,
                num_samples=self.num_samples_per_cluster
            )

            # Log detailed trajectory information
            self._log_trajectories(trajectories, cluster_idx)

            # 2. Compute advantages
            print(f"[2/3] Computing advantages...")
            advantages = self._compute_advantages(trajectories)

            # Print trajectory info
            returns = [t.get_return() for t in trajectories]
            print(f"  Returns: {[f'{r:.2f}' for r in returns]}")
            print(f"  Advantages: {[f'{a:.2f}' for a in advantages]}")
            print(f"  Mean return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")

            # 3. Update model using policy gradient
            print(f"[3/3] Updating model parameters...")
            loss_info = self._update_model(trajectories, advantages)
            global_step += loss_info['num_updates']

            # Log training info
            self.logger.info(f"\nTraining results:")
            self.logger.info(f"  Returns: {[f'{r:.4f}' for r in returns]}")
            self.logger.info(f"  Advantages: {[f'{a:.4f}' for a in advantages]}")
            self.logger.info(f"  Mean return: {np.mean(returns):.4f} ± {np.std(returns):.4f}")
            self.logger.info(f"  Policy loss: {loss_info['mean_loss']:.6f}")
            self.logger.info(f"  Gradient updates: {loss_info['num_updates']}")
            self.logger.info(f"  Global step: {global_step}")

            # Record statistics
            stats = {
                "cluster_id": cluster.cluster_id,
                "cluster_idx": cluster_idx,
                "global_step": global_step,
                "mean_return": np.mean(returns),
                "std_return": np.std(returns),
                "best_return": np.max(returns),
                "mean_advantage": np.mean(advantages),
                "policy_loss": loss_info['mean_loss'],
                "num_updates": loss_info['num_updates'],
            }
            self.training_stats.append(stats)
            self.cluster_statistics.append(stats)

            print(f"  Policy loss: {loss_info['mean_loss']:.4f}")
            print(f"  Updates: {loss_info['num_updates']}")

            # Save checkpoint
            if (cluster_idx + 1) % self.save_interval == 0:
                self._save_checkpoint(cluster_idx + 1, global_step)

        # Final save
        self._save_checkpoint("final", global_step)

        # Compute overall statistics
        overall_stats = self._compute_overall_statistics()

        print(f"\n{'='*80}")
        print(f"[GRPO Local] Training complete!")
        print(f"{'='*80}")
        print(f"  Total clusters: {len(clusters)}")
        print(f"  Total steps: {global_step}")
        print(f"  Mean return: {overall_stats['mean_return']:.2f}")
        print(f"  Mean policy loss: {overall_stats['mean_policy_loss']:.4f}")
        print(f"  Model saved to: {self.output_dir}")

        # Log final summary
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info("Training Complete - Summary")
        self.logger.info("="*80)
        self.logger.info(f"Total clusters: {len(clusters)}")
        self.logger.info(f"Total gradient updates: {global_step}")
        self.logger.info(f"Mean return: {overall_stats['mean_return']:.4f}")
        self.logger.info(f"Std return: {overall_stats['std_return']:.4f}")
        self.logger.info(f"Best return: {overall_stats['best_return']:.4f}")
        self.logger.info(f"Mean policy loss: {overall_stats['mean_policy_loss']:.6f}")
        self.logger.info(f"Final policy loss: {overall_stats['final_policy_loss']:.6f}")
        self.logger.info(f"Model saved to: {self.output_dir}")
        self.logger.info("="*80)

        return overall_stats

    def _compute_advantages(self, trajectories: List[Trajectory]) -> List[float]:
        """Compute group-relative advantages.

        Same as base GRPO: advantage_i = (return_i - mean) / std
        """
        returns = [traj.get_return() for traj in trajectories]

        if len(returns) == 1:
            return [0.0]

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return < 1e-8:
            advantages = [0.0] * len(returns)
        else:
            advantages = [(r - mean_return) / std_return for r in returns]

        return advantages

    def _log_trajectories(
        self,
        trajectories: List[Trajectory],
        cluster_idx: int
    ):
        """Log detailed trajectory information including agent decisions and rewards.

        Args:
            trajectories: List of trajectory objects
            cluster_idx: Current cluster index
        """
        self.logger.info(f"\nSampled {len(trajectories)} trajectories:")
        self.logger.info("-" * 80)

        for traj_idx, traj in enumerate(trajectories):
            self.logger.info(f"\nTrajectory {traj_idx + 1}:")
            self.logger.info(f"  Total return: {traj.get_return():.4f}")
            self.logger.info(f"  Length: {traj.get_length()} steps")
            self.logger.info(f"  Final detection score: {traj.final_detection_score:.4f}")
            self.logger.info("")

            # Log each step in detail
            for step_idx, step in enumerate(traj.steps):
                self.logger.info(f"  Step {step_idx + 1}:")

                # Input prompt to agent
                prompt = step.get("prompt", "N/A")
                self.logger.info(f"    Input prompt:")
                # Log prompt with indentation for readability
                for line in prompt.split('\n'):
                    self.logger.info(f"      {line}")
                self.logger.info("")

                # Agent decision
                decision = step.get("decision", {})
                if isinstance(decision, dict):
                    decision_raw = decision.get("decision_raw", "N/A")
                    decision_parsed = decision.get("decision_parsed", {})
                    is_valid = decision.get("is_valid", False)

                    self.logger.info(f"    Agent output (raw): {decision_raw}")
                    if decision_parsed:
                        self.logger.info(f"    Parsed decision:")
                        self.logger.info(f"      Tool: {decision_parsed.get('tool', 'N/A')}")
                        self.logger.info(f"      Args: {decision_parsed.get('args', {})}")
                        self.logger.info(f"      Rationale: {decision_parsed.get('rationale', 'N/A')}")
                    self.logger.info(f"    Valid: {is_valid}")
                    # If not valid or action failed, show error
                    if not is_valid:
                        self.logger.info(f"    Invalid reason: {decision.get('invalid_reason', 'N/A')}")
                    action_result = step.get("action_result", {})
                    if not action_result.get("ok", True):
                        self.logger.info(f"    Tool execution FAILED: {action_result.get('error', 'Unknown error')}")
                else:
                    self.logger.info(f"    Agent output: {decision}")

                # Reward breakdown
                info = step.get("info", {})
                self.logger.info(f"    Rewards:")
                self.logger.info(f"      Validity:  {info.get('validity_reward', 0):+.4f}")
                self.logger.info(f"      Detection: {info.get('detection_reward', 0):+.4f}")
                self.logger.info(f"      Final bonus: {info.get('final_bonus', 0):+.4f}")
                self.logger.info(f"      Total:     {step.get('reward', 0):+.4f}")

                # Detection scores
                self.logger.info(f"    Detection scores:")
                self.logger.info(f"      Before: {info.get('score_before', 'N/A'):.4f}" if info.get('score_before') is not None else "      Before: N/A")
                self.logger.info(f"      After:  {info.get('score_after', 'N/A'):.4f}" if info.get('score_after') is not None else "      After: N/A")

                # Is terminal
                if step.get("is_terminal", False):
                    self.logger.info(f"    Terminal: Yes")
                self.logger.info("")

            self.logger.info("-" * 80)

    def _update_model(
        self,
        trajectories: List[Trajectory],
        advantages: List[float]
    ) -> Dict[str, Any]:
        """Update model using policy gradient.

        Loss = -E[advantage * log π(action|state)]

        For LLM: we compute log probability of generated response tokens
        given the prompt, and weight by advantage.
        """
        total_loss = 0.0
        num_updates = 0

        self.optimizer.zero_grad()

        # Create training examples
        training_data = []
        for traj, advantage in zip(trajectories, advantages):
            for step in traj.steps:
                # Only train on valid actions
                if step.get("reward", 0) > -5.0:  # Skip invalid actions
                    # Extract raw LLM response from decision dict
                    decision = step["decision"]
                    if isinstance(decision, dict):
                        response = decision.get("decision_raw", "")
                    else:
                        response = str(decision)

                    training_data.append({
                        "prompt": step["prompt"],
                        "response": response,
                        "advantage": advantage,
                    })

        # Train for multiple epochs
        for epoch in range(self.num_epochs):
            # Shuffle data
            import random
            random.shuffle(training_data)

            # Mini-batch training
            for i in range(0, len(training_data), self.batch_size):
                batch = training_data[i:i+self.batch_size]

                batch_loss = 0.0
                for example in batch:
                    # Compute loss for this example
                    loss = self._compute_policy_loss(
                        example["prompt"],
                        example["response"],
                        example["advantage"]
                    )

                    if loss is not None:
                        batch_loss += loss

                # Average over batch
                if len(batch) > 0:
                    batch_loss = batch_loss / len(batch)

                    # Accumulate gradients
                    batch_loss.backward()

                    # Update parameters
                    if (i // self.batch_size + 1) % self.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        num_updates += 1

                    total_loss += batch_loss.item()

        # Final update if gradients remain
        if num_updates % self.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            num_updates += 1

        mean_loss = total_loss / max(num_updates, 1)

        return {
            "mean_loss": mean_loss,
            "total_loss": total_loss,
            "num_updates": num_updates,
        }

    def _compute_policy_loss(
        self,
        prompt: str,
        response: str,
        advantage: float
    ) -> Optional[torch.Tensor]:
        """Compute policy gradient loss for a single example.

        Loss = -advantage * log π(response|prompt)
        """
        # Format prompt (same as in LocalQwenLLMClient.complete)
        messages = [
            {"role": "system", "content": "You are an AI assistant that outputs only valid JSON."},
            {"role": "user", "content": prompt}
        ]

        # Apply chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                formatted_prompt = f"<|im_start|>system\nYou are an AI assistant that outputs only valid JSON.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            formatted_prompt = f"<|im_start|>system\nYou are an AI assistant that outputs only valid JSON.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        # Tokenize prompt + response
        full_text = formatted_prompt + response
        inputs = self.tokenizer(full_text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get prompt length to mask prompt tokens
        prompt_inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        prompt_length = prompt_inputs['input_ids'].shape[1]

        # Forward pass
        outputs = self.model(**inputs, labels=inputs['input_ids'])

        # Get logits and compute log probabilities
        logits = outputs.logits  # [1, seq_len, vocab_size]

        # Shift logits and labels for causal LM
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inputs['input_ids'][:, 1:].contiguous()

        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

        # Gather log probs of actual tokens
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # [1, seq_len-1]

        # Mask out prompt tokens (only compute loss on response)
        mask = torch.zeros_like(token_log_probs, dtype=torch.bool)
        mask[:, prompt_length-1:] = True  # -1 because of shifting

        # Compute mean log prob of response tokens
        response_log_probs = token_log_probs[mask]

        if response_log_probs.numel() == 0:
            return None

        mean_log_prob = response_log_probs.mean()

        # Policy gradient loss: -advantage * log π(a|s)
        loss = -advantage * mean_log_prob

        return loss

    def _save_checkpoint(self, checkpoint_name, global_step):
        """Save model checkpoint and training stats."""
        checkpoint_dir = self.output_dir / f"checkpoint-{checkpoint_name}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        if self.use_lora:
            # Save LoRA adapters only
            self.model.save_pretrained(checkpoint_dir)
        else:
            # Save full model
            self.model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)

        # Save training stats
        stats_path = checkpoint_dir / "training_stats.json"
        with open(stats_path, "w") as f:
            json.dump({
                "global_step": global_step,
                "training_stats": self.training_stats,
                "cluster_statistics": self.cluster_statistics,
            }, f, indent=2)

        print(f"  [GRPO Local] Checkpoint saved to {checkpoint_dir}")

    def _compute_overall_statistics(self) -> Dict[str, Any]:
        """Compute overall training statistics."""
        if not self.training_stats:
            return {}

        mean_returns = [s['mean_return'] for s in self.training_stats]
        policy_losses = [s['policy_loss'] for s in self.training_stats]

        return {
            "total_clusters": len(self.training_stats),
            "total_steps": self.training_stats[-1]['global_step'],
            "mean_return": np.mean(mean_returns),
            "std_return": np.std(mean_returns),
            "best_return": np.max([s['best_return'] for s in self.training_stats]),
            "mean_policy_loss": np.mean(policy_losses),
            "final_policy_loss": policy_losses[-1] if policy_losses else 0.0,
        }
