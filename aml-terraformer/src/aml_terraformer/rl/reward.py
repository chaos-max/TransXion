"""Reward computation for GRPO training."""

import pandas as pd
from typing import Dict, Any, List
from ..monitor.base import MonitorModel


class GRPOReward:
    """Reward function for GRPO training.

    Computes rewards based on:
    1. Detection probability change (main signal)
    2. Operation validity
    3. Terminal bonus

    Args:
        monitor: Monitor model instance
        config: Reward configuration dict with keys:
            - w_detection: Weight for detection reward (default: 10.0)
            - w_validity: Weight for validity reward (default: 1.0)
            - w_final_bonus: Terminal bonus for successful evasion (default: 20.0)
            - success_threshold: Detection threshold for success (default: 0.3)

    Example:
        >>> from aml_terraformer.monitor import RandomMonitor
        >>> monitor = RandomMonitor(mode='decreasing')
        >>> reward_fn = GRPOReward(monitor, config={})
        >>> reward, info = reward_fn.compute_step_reward(
        ...     state_before, state_after, action_result, is_terminal=False
        ... )
    """

    def __init__(self, monitor: MonitorModel, config: Dict[str, Any] = None):
        """Initialize reward function."""
        self.monitor = monitor
        config = config or {}

        # Reward weights
        self.w_detection = config.get("w_detection", 10.0)
        self.w_validity = config.get("w_validity", 1.0)
        self.w_final_bonus = 0
        # self.w_final_bonus = config.get("w_final_bonus", 20.0)

        # Success threshold
        self.success_threshold = config.get("success_threshold", 0.3)

        # Reward mode: 'score' (old) or 'hit_rate' (new, recommended)
        self.reward_mode = config.get("reward_mode", "hit_rate")

        # Statistics
        self.total_rewards = []
        self.detection_scores = []

    def compute_step_reward(
        self,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        action_result: Dict[str, Any],
        is_terminal: bool,
        trajectory_id: int = None,
        step_idx: int = None
    ) -> tuple[float, Dict[str, Any]]:
        """Compute reward for a single step.

        Args:
            state_before: State before action (dict with transactions_df, accounts_df, cluster_nodes)
            state_after: State after action
            action_result: Result from tool execution with keys:
                - ok: Whether operation succeeded
                - tool: Tool name
                - budget_used: Budget consumed
            is_terminal: Whether this is the last step

        Returns:
            Tuple of (total_reward, info_dict) where info_dict contains:
                - validity_reward: Reward for valid operation
                - detection_reward: Reward for detection change
                - final_bonus: Terminal bonus (if applicable)
                - score_before: Detection score before action
                - score_after: Detection score after action
        """
        # 1. Validity reward
        if not action_result.get("ok", False):
            # Invalid action: immediate penalty
            # No need to compute score_after since state didn't change
            return -5.0, {
                "validity_reward": -5.0,
                "detection_reward": 0.0,
                "final_bonus": 0.0,
                "score_before": None,
                "score_after": None,  # State unchanged, use None
                "total_reward": -5.0
            }

        validity_reward = self.w_validity

        # 2. Detection reward
        # Check if monitor supports debug_label (for RuleBasedMonitor)
        import inspect
        monitor_signature = inspect.signature(self.monitor.predict_proba)
        supports_debug_label = 'debug_label' in monitor_signature.parameters

        if supports_debug_label:
            score_before = self.monitor.predict_proba(
                state_before["transactions_df"],
                state_before["accounts_df"],
                state_before["cluster_nodes"],
                debug_label="before",
                trajectory_id=trajectory_id,
                step_idx=step_idx
            )

            score_after = self.monitor.predict_proba(
                state_after["transactions_df"],
                state_after["accounts_df"],
                state_after["cluster_nodes"],
                debug_label="after",
                trajectory_id=trajectory_id,
                step_idx=step_idx
            )
        else:
            score_before = self.monitor.predict_proba(
                state_before["transactions_df"],
                state_before["accounts_df"],
                state_before["cluster_nodes"]
            )

            score_after = self.monitor.predict_proba(
                state_after["transactions_df"],
                state_after["accounts_df"],
                state_after["cluster_nodes"]
            )

        # Reward = reduction in detection probability
        # Note: score_before and score_after are already computed by Monitor
        # Monitor uses normalized_mean_hit_rate internally, so we can use them directly
        detection_delta = score_before - score_after
        detection_reward = detection_delta * self.w_detection

        # Bonus for using adjust_transaction (encourage exploration)
        tool_bonus = 0.0
        if action_result.get("tool") == "adjust_transaction" and action_result.get("ok"):
            tool_bonus = 0.3  # Small bonus to encourage using adjust tool

        # 3. Terminal bonus
        final_bonus = 0.0
        # if is_terminal:
        #     if action_result.get("tool") == "stop":
        #         # Successfully stopped: check if we evaded detection
        #         if score_after < self.success_threshold:
        #             final_bonus = self.w_final_bonus
        #         else:
        #             # Stopped but still detected: small penalty
        #             final_bonus = -5.0
        #     else:
        #         # Hit max steps without stopping: neutral
        #         final_bonus = 0.0

        # Total reward (including tool bonus)
        total_reward = validity_reward + detection_reward + final_bonus

        # Record statistics
        self.total_rewards.append(total_reward)
        self.detection_scores.append((score_before, score_after))

        info = {
            "validity_reward": validity_reward,
            "detection_reward": detection_reward,
            "tool_bonus": tool_bonus,
            "final_bonus": final_bonus,
            "score_before": score_before,
            "score_after": score_after,
            "total_reward": total_reward,
        }

        return total_reward, info

    def _get_rule_hits_stats(self, transactions_df: pd.DataFrame) -> Dict[str, Any]:
        """Get rule hit statistics from transactions.

        Args:
            transactions_df: Transactions DataFrame with hit_* columns

        Returns:
            Dict with mean_hits_per_rule, num_laundering_txns, and normalized_mean_hit_rate
        """
        # Filter to laundering transactions only
        laundering_df = transactions_df[transactions_df["Is Laundering"] == 1]

        if len(laundering_df) == 0:
            return {
                "mean_hits_per_rule": 0.0,
                "num_laundering_txns": 0,
                "normalized_mean_hit_rate": 0.0
            }

        # Count hits for each rule
        hit_cols = [col for col in laundering_df.columns if col.startswith("hit_")]
        if not hit_cols:
            return {
                "mean_hits_per_rule": 0.0,
                "num_laundering_txns": len(laundering_df),
                "normalized_mean_hit_rate": 0.0
            }

        # Get the average hits across all rules
        rule_hits = laundering_df[hit_cols].sum()
        mean_hits = float(rule_hits.mean())

        num_laundering = len(laundering_df)
        # Normalize by number of laundering transactions
        normalized_mean_hit_rate = mean_hits / num_laundering if num_laundering > 0 else 0.0

        return {
            "mean_hits_per_rule": mean_hits,
            "num_laundering_txns": num_laundering,
            "normalized_mean_hit_rate": normalized_mean_hit_rate
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get reward statistics.

        Returns:
            Dict with statistics about rewards and detection scores
        """
        if not self.total_rewards:
            return {
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "total_episodes": 0,
            }

        import numpy as np

        return {
            "mean_reward": np.mean(self.total_rewards),
            "std_reward": np.std(self.total_rewards),
            "min_reward": np.min(self.total_rewards),
            "max_reward": np.max(self.total_rewards),
            "total_steps": len(self.total_rewards),
            "mean_score_before": np.mean([s[0] for s in self.detection_scores]),
            "mean_score_after": np.mean([s[1] for s in self.detection_scores]),
            "mean_score_reduction": np.mean([s[0] - s[1] for s in self.detection_scores]),
        }

    def reset_statistics(self):
        """Reset statistics."""
        self.total_rewards = []
        self.detection_scores = []
