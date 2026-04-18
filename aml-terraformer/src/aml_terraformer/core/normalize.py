"""Data normalization and account補全."""

import pandas as pd
from typing import Dict, Set
from .identifiers import make_node_id
from ..io.timestamp_handler import parse_timestamp_to_int_seconds



def normalize_data(transactions_df: pd.DataFrame, accounts_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize transactions and accounts data.

    Simplified version: generates essential fields needed by downstream code.
    Generated fields:
    - from_bank_id, to_bank_id: bank identifiers (directly from From Bank/To Bank)
    - from_account_number, to_account_number: account numbers
    - from_node_id, to_node_id: node identifiers (bank_id|account_number)
    - ts_int: timestamp in integer seconds (for time-based operations)
    - edge_id: unique edge identifier
    - original_row_index: original row position

    Assumes data is already clean and doesn't need bank mapping or account補全.

    Args:
        transactions_df: Transactions DataFrame (with From Account and To Account)
        accounts_df: Accounts DataFrame

    Returns:
        Tuple of (normalized transactions_df, accounts_df)
    """
    # Copy and add original_row_index
    transactions_df = transactions_df.copy()
    transactions_df["original_row_index"] = range(len(transactions_df))

    # Vectorized processing: strip and convert to string
    from_bank_raw = transactions_df["From Bank"].astype(str).str.strip()
    to_bank_raw = transactions_df["To Bank"].astype(str).str.strip()
    from_account_raw = transactions_df["From Account"].astype(str).str.strip()
    to_account_raw = transactions_df["To Account"].astype(str).str.strip()

    # Create node IDs directly (no bank mapping needed)
    from_node_id = from_bank_raw + "|" + from_account_raw
    to_node_id = to_bank_raw + "|" + to_account_raw

    # Parse timestamps (vectorized using pandas)
    try:
        ts_parsed = pd.to_datetime(transactions_df["Timestamp"], format='%Y/%m/%d %H:%M', errors='coerce')
        # If some failed, try without format
        if ts_parsed.isna().any():
            ts_parsed = pd.to_datetime(transactions_df["Timestamp"], errors='coerce')
        ts_int = ts_parsed.astype('int64') // 10**9  # Convert to seconds
    except Exception:
        # Fallback to apply if pandas parsing fails
        ts_int = transactions_df["Timestamp"].apply(parse_timestamp_to_int_seconds)

    # Create edge IDs (vectorized)
    edge_id = "row" + transactions_df["original_row_index"].astype(str)

    # Assign required columns
    transactions_df["from_bank_id"] = from_bank_raw
    transactions_df["to_bank_id"] = to_bank_raw
    transactions_df["from_account_number"] = from_account_raw
    transactions_df["to_account_number"] = to_account_raw
    transactions_df["from_node_id"] = from_node_id
    transactions_df["to_node_id"] = to_node_id
    transactions_df["ts_int"] = ts_int
    transactions_df["edge_id"] = edge_id

    return transactions_df, accounts_df


def ensure_all_accounts_exist(transactions_df: pd.DataFrame, accounts_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all node IDs referenced in transactions exist in accounts.

    If missing, create new account rows.

    Args:
        transactions_df: Transactions DataFrame with node IDs
        accounts_df: Accounts DataFrame

    Returns:
        Updated accounts DataFrame
    """
    # Get existing node IDs (vectorized)
    existing_bank_ids = accounts_df["Bank ID"].astype(str).str.strip()
    existing_account_numbers = accounts_df["Account Number"].astype(str).str.strip()
    existing_nodes = set((existing_bank_ids + "|" + existing_account_numbers).unique())

    # Get referenced node IDs
    referenced_nodes = set()
    referenced_nodes.update(transactions_df["from_node_id"].unique())
    referenced_nodes.update(transactions_df["to_node_id"].unique())

    # Find missing nodes
    missing_nodes = referenced_nodes - existing_nodes

    if not missing_nodes:
        return accounts_df

    # Create bank_id -> bank_name mapping for missing nodes (vectorized)
    bank_name_mapping = pd.Series(
        accounts_df["Bank Name"].astype(str).str.strip().values,
        index=accounts_df["Bank ID"].astype(str).str.strip().values
    ).to_dict()

    # Create new account rows for missing nodes
    accounts_df = accounts_df.copy()
    new_rows = []

    for node_id in missing_nodes:
        bank_id, account_number = node_id.split("|", 1)
        bank_name = bank_name_mapping.get(bank_id, "")

        new_row = {
            "Bank Name": bank_name,
            "Bank ID": bank_id,
            "Account Number": account_number,
            "Entity ID": "",
            "Entity Name": "",
        }
        new_rows.append(new_row)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        accounts_df = pd.concat([accounts_df, new_df], ignore_index=True)

    return accounts_df


def ensure_account_exists(
    accounts_df: pd.DataFrame,
    bank_id: str,
    account_number: str,
    entity_id: str = "",
    entity_name: str = ""
) -> pd.DataFrame:
    """Ensure a single account exists in accounts DataFrame.

    If it doesn't exist, add it.

    Args:
        accounts_df: Accounts DataFrame
        bank_id: Bank ID
        account_number: Account number
        entity_id: Entity ID for new account (default: "")
        entity_name: Entity Name for new account (default: "")

    Returns:
        Updated accounts DataFrame
    """
    node_id = make_node_id(bank_id, account_number)

    # Check if exists (vectorized)
    existing_bank_ids = accounts_df["Bank ID"].astype(str).str.strip()
    existing_account_numbers = accounts_df["Account Number"].astype(str).str.strip()
    existing_node_ids = existing_bank_ids + "|" + existing_account_numbers

    if (existing_node_ids == node_id).any():
        return accounts_df

    # Doesn't exist, add it
    # Try to find bank name (vectorized)
    bank_name_series = accounts_df.loc[existing_bank_ids == bank_id, "Bank Name"]
    bank_name = bank_name_series.iloc[0] if len(bank_name_series) > 0 else ""

    new_row = pd.DataFrame([{
        "Bank Name": bank_name,
        "Bank ID": bank_id,
        "Account Number": account_number,
        "Entity ID": entity_id,
        "Entity Name": entity_name,
    }])

    return pd.concat([accounts_df, new_row], ignore_index=True)
