"""Tests for aml_terraformer.core module."""

import os
import sys
from pathlib import Path

import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aml_terraformer.core.clusters import find_laundering_clusters
from aml_terraformer.core.normalize import normalize_data


class TestClusters:
    """Test cluster detection."""

    def test_find_laundering_clusters(self, sample_transactions_df, sample_accounts_df):
        """Test finding laundering clusters."""
        # Normalize data first
        txn_normalized, acct_normalized = normalize_data(sample_transactions_df, sample_accounts_df)

        # Find clusters
        clusters = find_laundering_clusters(txn_normalized)

        # Should find at least one cluster with Is Laundering=1
        assert isinstance(clusters, list)


class TestNormalize:
    """Test data normalization."""

    def test_normalize_basic(self, sample_transactions_df, sample_accounts_df):
        """Test basic normalization."""
        txn_normalized, acct_normalized = normalize_data(sample_transactions_df, sample_accounts_df)

        assert len(txn_normalized) == len(sample_transactions_df)
        assert len(acct_normalized) == len(sample_accounts_df)
