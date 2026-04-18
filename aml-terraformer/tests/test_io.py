"""Tests for aml_terraformer.io module."""

import os
import sys
from pathlib import Path

import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aml_terraformer.io import read_accounts, read_transactions


class TestReadAccounts:
    """Test read_accounts function."""

    def test_read_accounts_basic(self, temp_data_dir):
        """Test basic account reading."""
        accounts_path = temp_data_dir["accounts"]
        df = read_accounts(accounts_path)

        assert len(df) == 4
        assert "Bank Name" in df.columns
        assert "Bank ID" in df.columns
        assert "Account Number" in df.columns


class TestReadTransactions:
    """Test read_transactions function."""

    def test_read_transactions_basic(self, temp_data_dir):
        """Test basic transaction reading."""
        transactions_path = temp_data_dir["transactions"]
        df = read_transactions(transactions_path)

        assert len(df) == 3
        assert "From Bank" in df.columns
        assert "From Account" in df.columns
        assert "To Bank" in df.columns
        assert "To Account" in df.columns
        assert "Is Laundering" in df.columns

    def test_read_transactions_duplicate_columns(self, temp_data_dir, sample_transactions_df):
        """Test that transactions with duplicate Account columns are read correctly."""
        transactions_path = temp_data_dir["transactions"]
        df = read_transactions(transactions_path)

        # Verify From Account and To Account are correctly identified
        assert "From Account" in df.columns
        assert "To Account" in df.columns
        assert df["From Account"].iloc[0] == sample_transactions_df["From Account"].iloc[0]
