"""Pytest configuration and shared fixtures for tests."""

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Add src to path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_accounts_df():
    """Create a sample accounts DataFrame for testing."""
    return pd.DataFrame({
        "Bank Name": ["Bank A", "Bank A", "Bank B", "Bank B"],
        "Bank ID": [1, 1, 2, 2],
        "Account Number": ["A001", "A002", "B001", "B002"],
        "Entity ID": ["E001", "E002", "E003", "E004"],
        "Entity Name": ["Entity 1", "Entity 2", "Entity 3", "Entity 4"],
    })


@pytest.fixture
def sample_transactions_df():
    """Create a sample transactions DataFrame for testing."""
    return pd.DataFrame({
        "Timestamp": ["2024-01-01 10:00", "2024-01-01 11:00", "2024-01-01 12:00"],
        "From Bank": [1, 1, 2],
        "From Account": ["A001", "A002", "B001"],
        "To Bank": [1, 2, 1],
        "To Account": ["A002", "B001", "A001"],
        "Amount Received": [1000.0, 500.0, 300.0],
        "Receiving Currency": ["USD", "USD", "USD"],
        "Amount Paid": [1000.0, 500.0, 300.0],
        "Payment Currency": ["USD", "USD", "USD"],
        "Payment Format": ["SWIFT", "SWIFT", "SWIFT"],
        "Is Laundering": [0, 1, 0],
    })


@pytest.fixture
def temp_data_dir(sample_accounts_df, sample_transactions_df):
    """Create a temporary directory with sample data files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write accounts CSV
        accounts_path = os.path.join(tmpdir, "accounts.csv")
        sample_accounts_df.to_csv(accounts_path, index=False)

        # Write transactions CSV with correct format (two Account columns)
        txn_path = os.path.join(tmpdir, "transactions.csv")
        with open(txn_path, "w") as f:
            header = "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering"
            f.write(header + "\n")
            for _, row in sample_transactions_df.iterrows():
                line = f"{row['Timestamp']},{row['From Bank']},{row['From Account']},{row['To Bank']},{row['To Account']},{row['Amount Received']},{row['Receiving Currency']},{row['Amount Paid']},{row['Payment Currency']},{row['Payment Format']},{row['Is Laundering']}"
                f.write(line + "\n")

        yield {"accounts": accounts_path, "transactions": txn_path, "dir": tmpdir}
