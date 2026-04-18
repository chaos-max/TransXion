"""Abstract base class for monitor models."""

from abc import ABC, abstractmethod
import pandas as pd
from typing import List


class MonitorModel(ABC):
    """Abstract base class for money laundering detection models.

    This class defines the interface that all monitor models must implement.
    The monitor model takes a transaction graph and predicts the probability
    that it represents money laundering activity.
    """

    @abstractmethod
    def predict_proba(
        self,
        transactions_df: pd.DataFrame,
        accounts_df: pd.DataFrame,
        cluster_nodes: List[str]
    ) -> float:
        """Predict the probability that the given graph is money laundering.

        Args:
            transactions_df: Full transactions DataFrame with columns:
                - from_node_id, to_node_id, edge_id, Amount Paid, etc.
            accounts_df: Full accounts DataFrame with columns:
                - Bank ID, Account Number, node_id, etc.
            cluster_nodes: List of node IDs in the cluster to evaluate

        Returns:
            float: Probability score between 0 and 1, where:
                - 0 = definitely not money laundering (clean)
                - 1 = definitely money laundering
                - Higher values indicate higher confidence of detection

        Example:
            >>> monitor = YourMonitorModel()
            >>> prob = monitor.predict_proba(txns, accounts, cluster_nodes)
            >>> print(f"Detection probability: {prob:.3f}")
            Detection probability: 0.872
        """
        pass

    def __call__(
        self,
        transactions_df: pd.DataFrame,
        accounts_df: pd.DataFrame,
        cluster_nodes: List[str]
    ) -> float:
        """Convenience method to call predict_proba."""
        return self.predict_proba(transactions_df, accounts_df, cluster_nodes)
