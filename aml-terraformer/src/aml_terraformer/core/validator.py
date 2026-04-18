"""Validation and rollback utilities."""

import pandas as pd
from typing import Dict, List, Tuple
from .identifiers import make_node_id


def create_snapshot(transactions_df: pd.DataFrame, accounts_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create deep copy snapshot of current state.

    Args:
        transactions_df: Transactions DataFrame
        accounts_df: Accounts DataFrame

    Returns:
        Tuple of (transactions_copy, accounts_copy)
    """
    return transactions_df.copy(deep=True), accounts_df.copy(deep=True)


def rollback_to_snapshot(snapshot: Tuple[pd.DataFrame, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Rollback to snapshot.

    Args:
        snapshot: Tuple of (transactions_df, accounts_df) from create_snapshot

    Returns:
        Tuple of (transactions_df, accounts_df)
    """
    return snapshot[0].copy(deep=True), snapshot[1].copy(deep=True)


def validate_state(transactions_df: pd.DataFrame, accounts_df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate current state after operation.

    Checks:
    1. transactions has all required output columns
    2. All endpoints exist in accounts
    3. Bank fields are consistent with node_id bank_id
    4. No invalid data

    Args:
        transactions_df: Transactions DataFrame
        accounts_df: Accounts DataFrame

    Returns:
        Tuple of (is_valid, list_of_violations)
    """
    violations = []

    # Check 1: Required columns
    required_txn_cols = [
        "Timestamp", "From Bank", "From Account", "To Bank", "To Account",
        "Amount Received", "Receiving Currency", "Amount Paid",
        "Payment Currency", "Payment Format", "Is Laundering"
    ]
    for col in required_txn_cols:
        if col not in transactions_df.columns:
            violations.append(f"Missing required transactions column: {col}")

    required_acc_cols = ["Bank Name", "Bank ID", "Account Number", "Entity ID", "Entity Name"]
    for col in required_acc_cols:
        if col not in accounts_df.columns:
            violations.append(f"Missing required accounts column: {col}")

    if violations:
        return False, violations

    # Build set of existing accounts - VECTORIZED
    bank_ids = accounts_df["Bank ID"].astype(str).str.strip()
    account_numbers = accounts_df["Account Number"].astype(str).str.strip()
    existing_accounts = set(bank_ids + "|" + account_numbers)

    # Check 2 & 3: Endpoint consistency - VECTORIZED
    # Build node IDs from transactions
    from_banks = transactions_df["From Bank"].astype(str).str.strip()
    from_accounts = transactions_df["From Account"].astype(str).str.strip()
    from_node_ids = from_banks + "|" + from_accounts

    to_banks = transactions_df["To Bank"].astype(str).str.strip()
    to_accounts = transactions_df["To Account"].astype(str).str.strip()
    to_node_ids = to_banks + "|" + to_accounts

    # Check from endpoints
    from_missing = ~from_node_ids.isin(existing_accounts)
    if from_missing.any():
        missing_indices = transactions_df.index[from_missing].tolist()[:10]  # Limit to first 10
        for idx in missing_indices:
            violations.append(f"Row {idx}: From endpoint {from_node_ids.iloc[idx]} not found in accounts")

    # Check to endpoints
    to_missing = ~to_node_ids.isin(existing_accounts)
    if to_missing.any():
        missing_indices = transactions_df.index[to_missing].tolist()[:10]  # Limit to first 10
        for idx in missing_indices:
            violations.append(f"Row {idx}: To endpoint {to_node_ids.iloc[idx]} not found in accounts")

    # Check bank consistency if node_id columns exist
    if "from_node_id" in transactions_df.columns:
        from_mismatch = transactions_df["from_node_id"] != from_node_ids
        if from_mismatch.any():
            mismatch_indices = transactions_df.index[from_mismatch].tolist()[:10]
            for idx in mismatch_indices:
                violations.append(f"Row {idx}: From Bank/Account mismatch with from_node_id")

    if "to_node_id" in transactions_df.columns:
        to_mismatch = transactions_df["to_node_id"] != to_node_ids
        if to_mismatch.any():
            mismatch_indices = transactions_df.index[to_mismatch].tolist()[:10]
            for idx in mismatch_indices:
                violations.append(f"Row {idx}: To Bank/Account mismatch with to_node_id")

    is_valid = len(violations) == 0
    return is_valid, violations


def validate_inject_chain(
    transactions_df: pd.DataFrame,
    new_edge_ids: List[str]
) -> Tuple[bool, List[str]]:
    """Validate inject operation: timestamps should be non-decreasing.

    Args:
        transactions_df: Transactions DataFrame
        new_edge_ids: List of newly created edge IDs

    Returns:
        Tuple of (is_valid, list_of_violations)
    """
    violations = []

    # Get rows for new edges, preserving the order in new_edge_ids
    # new_edge_ids is already in chain order (from inject_intermediary)
    prev_ts = None
    for edge_id in new_edge_ids:
        rows = transactions_df[transactions_df["edge_id"] == edge_id]
        if len(rows) == 0:
            violations.append(f"Inject chain: edge {edge_id} not found in transactions")
            continue

        ts = rows.iloc[0]["ts_int"]
        if prev_ts is not None and ts < prev_ts:
            violations.append(f"Inject chain: timestamp not non-decreasing at edge {edge_id} (ts={ts}, prev={prev_ts})")
        prev_ts = ts

    is_valid = len(violations) == 0
    return is_valid, violations


def validate_split_result(
    transactions_df: pd.DataFrame,
    original_node: str,
    new_node: str,
    move_direction: str,
    moved_edge_ids: List[str]
) -> Tuple[bool, List[str]]:
    """Validate split operation: both nodes should have at least 1 edge.

    Args:
        transactions_df: Transactions DataFrame
        original_node: Original node ID
        new_node: New node ID
        move_direction: Direction of move (out/in/both)
        moved_edge_ids: List of moved edge IDs

    Returns:
        Tuple of (is_valid, list_of_violations)
    """
    violations = []

    # Count edges for original node
    orig_out = len(transactions_df[transactions_df["from_node_id"] == original_node])
    orig_in = len(transactions_df[transactions_df["to_node_id"] == original_node])

    # Count edges for new node
    new_out = len(transactions_df[transactions_df["from_node_id"] == new_node])
    new_in = len(transactions_df[transactions_df["to_node_id"] == new_node])

    # Check based on move_direction
    if move_direction in ["out", "both"]:
        if orig_out == 0:
            violations.append(f"Split: original node {original_node} has no outgoing edges left")
    if move_direction in ["in", "both"]:
        if orig_in == 0:
            violations.append(f"Split: original node {original_node} has no incoming edges left")

    # New node should have at least 1 edge
    if new_out + new_in == 0:
        violations.append(f"Split: new node {new_node} has no edges")

    is_valid = len(violations) == 0
    return is_valid, violations


def validate_merge_bank_constraint(
    node_a: str,
    node_b: str
) -> Tuple[bool, List[str]]:
    """Validate merge constraints (bank restriction removed).

    Args:
        node_a: Node ID
        node_b: Node ID

    Returns:
        Tuple of (is_valid, list_of_violations)
    """
    # Bank restriction removed - allow cross-bank merges
    violations = []
    is_valid = True
    return is_valid, violations
