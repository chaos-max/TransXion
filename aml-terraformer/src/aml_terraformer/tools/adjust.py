"""Adjust transaction tool: modify timestamps and amounts of transactions."""

import pandas as pd
from typing import List, Dict, Any


def adjust_transaction(
    transactions_df: pd.DataFrame,
    accounts_df: pd.DataFrame,
    edge_ids: List[str],
    time_offset_seconds: int,
    amount_multiplier: float,
    timestamp_format: str = "iso"
) -> Dict[str, Any]:
    """Adjust timestamps and amounts of specified transactions.

    This tool modifies existing transactions to help evade detection rules
    by changing their timing and amounts while preserving the transaction structure.

    Args:
        transactions_df: Transactions DataFrame
        accounts_df: Accounts DataFrame
        edge_ids: List of edge IDs to adjust
        time_offset_seconds: Time offset in seconds (can be positive or negative)
        amount_multiplier: Amount adjustment multiplier (e.g., 0.95 = reduce by 5%)
        timestamp_format: Format for output timestamps

    Returns:
        Dictionary with:
        - affected_edge_ids: List of adjusted edge IDs
        - adjustments: List of adjustment details
        - counts: Summary counts
    """
    from ..io.timestamp_handler import format_timestamp

    affected_edge_ids = []
    adjustments = []
    affected_nodes = []  # Nodes that were directly operated on

    # OPTIMIZATION: Pre-build edge_id index for fast lookup
    # Store actual DataFrame index labels (not positional indices) for use with .loc and .at
    edge_id_to_idx = {edge_id: transactions_df.index[idx] for idx, edge_id in enumerate(transactions_df["edge_id"].values)}

    # OPTIMIZATION: Collect updates to batch them
    updates_to_apply = []  # List of (idx, field, value) tuples

    for edge_id in edge_ids:
        # OPTIMIZATION: Use index lookup instead of DataFrame filtering
        if edge_id not in edge_id_to_idx:
            continue

        row_idx = edge_id_to_idx[edge_id]
        row = transactions_df.loc[row_idx]

        # Store original values
        original_ts_int = row["ts_int"]
        original_amount_paid = row.get("Amount Paid", 0)
        original_amount_received = row.get("Amount Received", 0)

        # Calculate new values
        new_ts_int = original_ts_int + time_offset_seconds
        new_amount_paid = original_amount_paid * amount_multiplier
        new_amount_received = original_amount_received * amount_multiplier
        new_timestamp = format_timestamp(new_ts_int, timestamp_format, original_value=row["Timestamp"])

        # OPTIMIZATION: Collect updates instead of applying immediately
        updates_to_apply.append((row_idx, "ts_int", new_ts_int))
        updates_to_apply.append((row_idx, "Timestamp", new_timestamp))
        updates_to_apply.append((row_idx, "Amount Paid", new_amount_paid))
        updates_to_apply.append((row_idx, "Amount Received", new_amount_received))

        affected_edge_ids.append(edge_id)
        adjustments.append({
            "edge_id": edge_id,
            "time_offset_seconds": time_offset_seconds,
            "amount_multiplier": amount_multiplier,
            "original_timestamp": row["Timestamp"],
            "new_timestamp": new_timestamp,
            "original_amount": original_amount_paid,
            "new_amount": new_amount_paid
        })

        # Record the nodes involved in this transaction as affected
        from_node = row["from_node_id"]
        to_node = row["to_node_id"]
        if from_node not in affected_nodes:
            affected_nodes.append(from_node)
        if to_node not in affected_nodes:
            affected_nodes.append(to_node)

    # OPTIMIZATION: Batch apply all updates
    for row_idx, field, value in updates_to_apply:
        transactions_df.at[row_idx, field] = value

    return {
        "affected_edge_ids": affected_edge_ids,
        "adjustments": adjustments,
        "affected_nodes": affected_nodes,
        "counts": {
            "transactions_adjusted": len(affected_edge_ids)
        },
        "transactions_df": transactions_df,
        "accounts_df": accounts_df,
    }
