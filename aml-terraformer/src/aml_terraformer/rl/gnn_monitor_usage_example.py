"""
Example: Using GNN Monitor in GRPO Reward Function

This example shows how to replace the existing monitor with GNN Monitor
in the reward computation for reinforcement learning training.
"""

import os
from aml_terraformer.monitor import GNNMonitor
from aml_terraformer.rl.reward import GRPOReward

# ============================================================
# Example 1: Basic Usage with GNN Monitor
# ============================================================

def example_basic_usage():
    """Basic example of using GNN Monitor with GRPO Reward."""

    # Initialize GNN Monitor
    # The monitor will use the Multi-GNN model at the specified path
    monitor = GNNMonitor(
        gnn_project_path=os.environ.get("MULTIGNN_PATH", ""),
        threshold=0.7,  # Classification threshold
        cache_predictions=True  # Cache predictions to avoid redundant inference
    )

    # Initialize reward function with GNN Monitor
    reward_fn = GRPOReward(
        monitor=monitor,
        config={
            "w_detection": 10.0,      # Weight for detection reward
            "w_validity": 1.0,         # Weight for validity reward
            "w_final_bonus": 20.0,     # Terminal bonus
            "success_threshold": 0.3,  # Detection threshold for success
            "reward_mode": "hit_rate"  # Use hit_rate mode
        }
    )

    return reward_fn


# ============================================================
# Example 2: Using GNN Monitor in Training Loop
# ============================================================

def example_training_loop():
    """Example of using GNN Monitor in a training loop."""

    # Initialize GNN Monitor
    monitor = GNNMonitor(
        gnn_project_path=os.environ.get("MULTIGNN_PATH", "/path/to/multignn")
    )

    # Initialize reward function
    reward_fn = GRPOReward(monitor=monitor)

    # Example state (would come from your environment)
    state_before = {
        "transactions_df": None,  # Your transactions DataFrame
        "accounts_df": None,      # Your accounts DataFrame
        "cluster_nodes": []       # List of node IDs in cluster
    }

    state_after = {
        "transactions_df": None,  # Modified transactions DataFrame
        "accounts_df": None,      # Modified accounts DataFrame
        "cluster_nodes": []       # List of node IDs in cluster
    }

    action_result = {
        "ok": True,
        "tool": "adjust_transaction",
        "budget_used": 10
    }

    # Compute reward
    reward, info = reward_fn.compute_step_reward(
        state_before=state_before,
        state_after=state_after,
        action_result=action_result,
        is_terminal=False
    )

    print(f"Reward: {reward}")
    print(f"Detection score before: {info['score_before']}")
    print(f"Detection score after: {info['score_after']}")

    return reward, info
