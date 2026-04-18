"""Wrapper for user's pre-trained monitor model.

This module provides adapters to integrate pre-trained monitor models
that return a single detection score.
"""

import pandas as pd
from typing import List, Any, Callable
from .base import MonitorModel


class WrappedMonitorModel(MonitorModel):
    """Adapter for pre-trained monitor models that return a single score.

    This wrapper allows you to integrate any monitor model that:
    1. Takes graph data as input
    2. Returns a single detection score

    The wrapper handles:
    - Converting pandas DataFrames to your model's input format
    - Normalizing scores to [0, 1] probability range
    - Reversing score semantics if needed (e.g., if higher score = safer)

    Args:
        predict_fn: Your model's prediction function
        input_format: How to prepare input for your model:
            - 'dataframes': Pass (transactions_df, accounts_df, cluster_nodes)
            - 'cluster_subgraph': Pass only cluster transactions/accounts
            - 'features_dict': Extract features and pass as dict
            - 'custom': Use custom_input_fn
        score_range: Expected score range, e.g., (0, 1) or (0, 100)
        higher_is_laundering: Whether higher score = higher laundering probability
            - True: score 0.9 means 90% likely to be laundering
            - False: score 0.9 means 90% likely to be clean (will invert)
        custom_input_fn: Optional custom function to prepare input
        feature_extractor_fn: Optional function to extract features

    Example 1: Simple model that takes transactions DataFrame
        >>> def my_predict(transactions_df):
        ...     # Your model logic
        ...     return score  # e.g., 0.85
        >>>
        >>> monitor = WrappedMonitorModel(
        ...     predict_fn=my_predict,
        ...     input_format='cluster_subgraph',
        ...     score_range=(0, 1),
        ...     higher_is_laundering=True
        ... )

    Example 2: Model that expects feature dict
        >>> def extract_features(transactions_df, accounts_df, cluster_nodes):
        ...     return {
        ...         'n_nodes': len(cluster_nodes),
        ...         'total_amount': transactions_df['Amount Paid'].sum(),
        ...         # ... more features
        ...     }
        >>>
        >>> def my_predict(features):
        ...     return model.predict([features])[0]
        >>>
        >>> monitor = WrappedMonitorModel(
        ...     predict_fn=my_predict,
        ...     input_format='features_dict',
        ...     score_range=(0, 100),
        ...     higher_is_laundering=True,
        ...     feature_extractor_fn=extract_features
        ... )

    Example 3: Model with complex preprocessing
        >>> def prepare_input(transactions_df, accounts_df, cluster_nodes):
        ...     # Your custom preprocessing
        ...     graph = build_networkx_graph(...)
        ...     features = compute_graph_features(graph)
        ...     return features
        >>>
        >>> monitor = WrappedMonitorModel(
        ...     predict_fn=my_model.predict,
        ...     input_format='custom',
        ...     custom_input_fn=prepare_input,
        ...     score_range=(0, 1),
        ...     higher_is_laundering=True
        ... )
    """

    def __init__(
        self,
        predict_fn: Callable,
        input_format: str = 'dataframes',
        score_range: tuple = (0, 1),
        higher_is_laundering: bool = True,
        custom_input_fn: Callable = None,
        feature_extractor_fn: Callable = None
    ):
        """Initialize wrapped monitor model."""
        self.predict_fn = predict_fn
        self.input_format = input_format
        self.score_min, self.score_max = score_range
        self.higher_is_laundering = higher_is_laundering
        self.custom_input_fn = custom_input_fn
        self.feature_extractor_fn = feature_extractor_fn

        # Validate configuration
        if input_format == 'custom' and custom_input_fn is None:
            raise ValueError("Must provide custom_input_fn when input_format='custom'")
        if input_format == 'features_dict' and feature_extractor_fn is None:
            raise ValueError("Must provide feature_extractor_fn when input_format='features_dict'")

    def predict_proba(
        self,
        transactions_df: pd.DataFrame,
        accounts_df: pd.DataFrame,
        cluster_nodes: List[str]
    ) -> float:
        """Predict laundering probability for the given cluster.

        Args:
            transactions_df: Full transactions DataFrame
            accounts_df: Full accounts DataFrame
            cluster_nodes: List of node IDs in the cluster

        Returns:
            float: Probability between 0 and 1 (0=clean, 1=laundering)
        """
        # Prepare input based on format
        if self.input_format == 'dataframes':
            # Pass all three arguments
            model_input = (transactions_df, accounts_df, cluster_nodes)
            raw_score = self.predict_fn(*model_input)

        elif self.input_format == 'cluster_subgraph':
            # Extract only cluster data
            cluster_txns = self._extract_cluster_transactions(transactions_df, cluster_nodes)
            cluster_accts = self._extract_cluster_accounts(accounts_df, cluster_nodes)

            model_input = (cluster_txns, cluster_accts, cluster_nodes)
            raw_score = self.predict_fn(*model_input)

        elif self.input_format == 'features_dict':
            # Extract features and pass as dict
            features = self.feature_extractor_fn(transactions_df, accounts_df, cluster_nodes)
            raw_score = self.predict_fn(features)

        elif self.input_format == 'custom':
            # Use custom input preparation
            model_input = self.custom_input_fn(transactions_df, accounts_df, cluster_nodes)
            raw_score = self.predict_fn(model_input)

        else:
            raise ValueError(f"Unknown input_format: {self.input_format}")

        # Normalize score to [0, 1]
        normalized_score = self._normalize_score(raw_score)

        return normalized_score

    def _extract_cluster_transactions(
        self,
        transactions_df: pd.DataFrame,
        cluster_nodes: List[str]
    ) -> pd.DataFrame:
        """Extract transactions within the cluster.

        Args:
            transactions_df: Full transactions DataFrame
            cluster_nodes: List of node IDs

        Returns:
            DataFrame containing only cluster transactions
        """
        cluster_nodes_set = set(cluster_nodes)
        mask = (
            transactions_df['from_node_id'].isin(cluster_nodes_set) &
            transactions_df['to_node_id'].isin(cluster_nodes_set)
        )
        return transactions_df[mask].copy()

    def _extract_cluster_accounts(
        self,
        accounts_df: pd.DataFrame,
        cluster_nodes: List[str]
    ) -> pd.DataFrame:
        """Extract accounts in the cluster.

        Args:
            accounts_df: Full accounts DataFrame
            cluster_nodes: List of node IDs

        Returns:
            DataFrame containing only cluster accounts
        """
        if 'node_id' not in accounts_df.columns:
            # Add node_id if not present
            from ..core.identifiers import make_node_id
            accounts_df = accounts_df.copy()
            accounts_df['node_id'] = accounts_df.apply(
                lambda row: make_node_id(
                    str(row['Bank ID']),
                    str(row['Account Number'])
                ),
                axis=1
            )

        cluster_nodes_set = set(cluster_nodes)
        mask = accounts_df['node_id'].isin(cluster_nodes_set)
        return accounts_df[mask].copy()

    def _normalize_score(self, raw_score: float) -> float:
        """Normalize raw score to [0, 1] probability.

        Args:
            raw_score: Raw score from model

        Returns:
            float: Normalized probability in [0, 1]
        """
        # Clip to expected range
        clipped = max(self.score_min, min(self.score_max, raw_score))

        # Normalize to [0, 1]
        if self.score_max == self.score_min:
            normalized = 0.5
        else:
            normalized = (clipped - self.score_min) / (self.score_max - self.score_min)

        # Invert if higher score means cleaner
        if not self.higher_is_laundering:
            normalized = 1.0 - normalized

        return float(normalized)


# Convenience function for quick setup
def wrap_monitor_model(
    model: Any,
    method_name: str = 'predict',
    **kwargs
) -> WrappedMonitorModel:
    """Convenience function to wrap a model object.

    Args:
        model: Your model object (e.g., sklearn model, custom class)
        method_name: Name of the prediction method (default: 'predict')
        **kwargs: Additional arguments for WrappedMonitorModel

    Returns:
        WrappedMonitorModel instance

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> rf_model = load_model('monitor.pkl')
        >>>
        >>> monitor = wrap_monitor_model(
        ...     model=rf_model,
        ...     method_name='predict_proba',
        ...     input_format='features_dict',
        ...     feature_extractor_fn=extract_features
        ... )
    """
    predict_fn = getattr(model, method_name)
    return WrappedMonitorModel(predict_fn=predict_fn, **kwargs)
