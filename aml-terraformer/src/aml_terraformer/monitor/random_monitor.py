"""Random monitor for testing purposes."""

import numpy as np
import pandas as pd
from typing import List
from .base import MonitorModel


class RandomMonitor(MonitorModel):
    """Random monitor that returns random probabilities.

    This is a dummy monitor for testing the RL pipeline without a real
    detection model. It can operate in different modes:

    - 'pure_random': Returns uniform random values in [0, 1]
    - 'graph_size_based': Returns probabilities based on graph size
        (larger graphs = higher detection probability)
    - 'fixed': Returns a fixed probability value
    - 'decreasing': Returns decreasing probabilities (simulates successful evasion)

    Args:
        mode: One of ['pure_random', 'graph_size_based', 'fixed', 'decreasing']
        seed: Random seed for reproducibility
        fixed_value: Fixed probability value (used when mode='fixed')
        noise_level: Amount of noise to add (0-1)

    Example:
        >>> monitor = RandomMonitor(mode='graph_size_based', seed=42)
        >>> prob = monitor.predict_proba(txns, accounts, cluster_nodes)
    """

    def __init__(
        self,
        mode: str = "pure_random",
        seed: int = 42,
        fixed_value: float = 0.8,
        noise_level: float = 0.1
    ):
        """Initialize random monitor."""
        self.mode = mode
        self.rng = np.random.RandomState(seed)
        self.fixed_value = fixed_value
        self.noise_level = noise_level
        self.call_count = 0  # Track number of calls for 'decreasing' mode

    def predict_proba(
        self,
        transactions_df: pd.DataFrame,
        accounts_df: pd.DataFrame,
        cluster_nodes: List[str]
    ) -> float:
        """Predict probability (randomly generated).

        Args:
            transactions_df: Transactions DataFrame
            accounts_df: Accounts DataFrame
            cluster_nodes: List of node IDs in cluster

        Returns:
            float: Random probability between 0 and 1
        """
        self.call_count += 1

        if self.mode == "pure_random":
            # Completely random
            return float(self.rng.uniform(0, 1))

        elif self.mode == "graph_size_based":
            # Based on graph size (more nodes/edges = higher probability)
            n_nodes = len(cluster_nodes)

            # Filter transactions in cluster
            cluster_mask = (
                transactions_df["from_node_id"].isin(cluster_nodes) &
                transactions_df["to_node_id"].isin(cluster_nodes)
            )
            n_edges = cluster_mask.sum()

            # Base probability from graph size
            # Normalize by expected cluster size (assume max ~100 nodes, ~200 edges)
            base_prob = min(0.9, (n_nodes / 100.0) * 0.5 + (n_edges / 200.0) * 0.4)

            # Add noise
            noise = self.rng.normal(0, self.noise_level)
            prob = np.clip(base_prob + noise, 0.0, 1.0)

            return float(prob)

        elif self.mode == "fixed":
            # Fixed value with small noise
            noise = self.rng.normal(0, self.noise_level)
            prob = np.clip(self.fixed_value + noise, 0.0, 1.0)
            return float(prob)

        elif self.mode == "decreasing":
            # Simulates successful evasion: probability decreases with each call
            # Starts at ~0.9 and decreases to ~0.3
            initial_prob = 0.9
            final_prob = 0.3
            decay_rate = 0.1

            base_prob = final_prob + (initial_prob - final_prob) * np.exp(-decay_rate * self.call_count)

            # Add noise
            noise = self.rng.normal(0, self.noise_level)
            prob = np.clip(base_prob + noise, 0.0, 1.0)

            return float(prob)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def reset(self):
        """Reset internal state (useful for 'decreasing' mode)."""
        self.call_count = 0
