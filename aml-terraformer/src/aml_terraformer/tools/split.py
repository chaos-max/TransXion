"""Split account tool: split one account into two."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from ..core.identifiers import make_node_id, parse_node_id
from ..core.normalize import ensure_account_exists


def split_account(
    transactions_df: pd.DataFrame,
    accounts_df: pd.DataFrame,
    node_ids: List[str],
    split_ratio: float,
    move_direction: str,
    edge_sampling: str,
    seed: int = 42
) -> Dict[str, Any]:
    """Split accounts into two parts.

    For each node U:
    - Create new account U2 (same bank_id)
    - Collect movable edges (out/in/both based on move_direction)
    - Select edges to move based on edge_sampling
    - Retarget selected edges to U2
    - Ensure both U and U2 have at least 1 edge

    Args:
        transactions_df: Transactions DataFrame
        accounts_df: Accounts DataFrame
        node_ids: List of node IDs to split
        split_ratio: Ratio of edges to move (0.2, 0.3, 0.4)
        move_direction: Direction to move edges from ('out', 'in', 'both')
        edge_sampling: Sampling strategy ('time_stratified', 'random_within_currency', 'random')
        seed: Random seed

    Returns:
        Dictionary with:
        - moved_edge_ids: List of edge IDs that were moved
        - created_accounts: List of created node IDs
        - counts: Summary counts
    """
    np.random.seed(seed)

    moved_edge_ids = []
    created_accounts = []
    affected_nodes = []  # Nodes that were directly operated on

    # OPTIMIZATION: Pre-build accounts index for fast lookup (vectorized)
    bank_ids = accounts_df["Bank ID"].astype(str).str.strip().values
    account_nums = accounts_df["Account Number"].astype(str).str.strip().values
    entity_ids = accounts_df["Entity ID"].astype(str).values if "Entity ID" in accounts_df.columns else [""] * len(accounts_df)
    entity_names = accounts_df["Entity Name"].astype(str).values if "Entity Name" in accounts_df.columns else [""] * len(accounts_df)

    accounts_index = {}
    for bank_id, account_num, entity_id, entity_name in zip(bank_ids, account_nums, entity_ids, entity_names):
        key = f"{bank_id}|{account_num}"
        accounts_index[key] = {
            "entity_id": entity_id,
            "entity_name": entity_name
        }

    # OPTIMIZATION: Collect edge updates to batch them
    edges_to_update = []  # List of (idx, field, value) tuples

    # OPTIMIZATION: Collect new accounts to add them in batch
    new_accounts_to_add = []  # List of account info dicts

    for node_idx, node_id in enumerate(node_ids):
        bank_id, account_number = parse_node_id(node_id)

        # Get original node's Entity ID and Entity Name
        # OPTIMIZATION: Use index lookup instead of DataFrame filtering
        original_entity_id = ""
        original_entity_name = ""
        account_key = f"{bank_id}|{account_number}"
        if account_key in accounts_index:
            original_entity_id = accounts_index[account_key]["entity_id"]
            original_entity_name = accounts_index[account_key]["entity_name"]

        # Create new account U2 with inherited entity info
        new_account_number = f"SPLIT_{node_idx}_{account_number}"
        u2_node_id = make_node_id(bank_id, new_account_number)

        # OPTIMIZATION: Collect account info instead of adding immediately
        new_accounts_to_add.append({
            "bank_id": bank_id,
            "account_number": new_account_number,
            "entity_id": original_entity_id,
            "entity_name": original_entity_name
        })
        created_accounts.append(u2_node_id)

        # Update accounts_index for future lookups
        new_key = f"{bank_id}|{new_account_number}"
        accounts_index[new_key] = {
            "entity_id": original_entity_id,
            "entity_name": original_entity_name
        }

        # Record both the original and new account as affected
        affected_nodes.append(node_id)
        affected_nodes.append(u2_node_id)

        # Collect movable edges - VECTORIZED
        out_mask = transactions_df["from_node_id"] == node_id
        in_mask = transactions_df["to_node_id"] == node_id

        # Use index and to_dict('records') instead of iterrows for better performance
        out_df = transactions_df[out_mask]
        in_df = transactions_df[in_mask]
        out_edges = list(zip(out_df.index, out_df.to_dict('records')))
        in_edges = list(zip(in_df.index, in_df.to_dict('records')))

        # Determine how many to move from each direction
        out_count = len(out_edges)
        in_count = len(in_edges)

        # Calculate move counts with constraints
        if move_direction == "out":
            out_to_move = calculate_move_count(out_count, split_ratio)
            in_to_move = 0
        elif move_direction == "in":
            out_to_move = 0
            in_to_move = calculate_move_count(in_count, split_ratio)
        else:  # both
            out_to_move = calculate_move_count(out_count, split_ratio)
            in_to_move = calculate_move_count(in_count, split_ratio)

            # If one direction doesn't have enough edges, transfer to other
            if out_count < 2 and in_count >= 2:
                extra = out_to_move
                out_to_move = 0
                in_to_move = min(in_to_move + extra, calculate_move_count(in_count, split_ratio * 2))
                in_to_move = calculate_move_count(in_count, split_ratio * 2)
            elif in_count < 2 and out_count >= 2:
                extra = in_to_move
                in_to_move = 0
                out_to_move = min(out_to_move + extra, calculate_move_count(out_count, split_ratio * 2))
                out_to_move = calculate_move_count(out_count, split_ratio * 2)

        # Sample edges to move
        out_to_move_indices = sample_edges(out_edges, out_to_move, edge_sampling, seed + node_idx)
        in_to_move_indices = sample_edges(in_edges, in_to_move, edge_sampling, seed + node_idx + 1000)

        # OPTIMIZATION: Collect out edge updates instead of applying immediately
        for idx in out_to_move_indices:
            row = transactions_df.loc[idx]
            moved_edge_ids.append(row["edge_id"])

            # Collect updates for batch processing
            edges_to_update.append((idx, "from_node_id", u2_node_id))
            edges_to_update.append((idx, "from_bank_id", bank_id))
            edges_to_update.append((idx, "from_account_number", new_account_number))
            edges_to_update.append((idx, "From Bank", int(bank_id) if bank_id.isdigit() else bank_id))
            edges_to_update.append((idx, "From Account", new_account_number))

        # OPTIMIZATION: Collect in edge updates instead of applying immediately
        for idx in in_to_move_indices:
            row = transactions_df.loc[idx]
            moved_edge_ids.append(row["edge_id"])

            # Collect updates for batch processing
            edges_to_update.append((idx, "to_node_id", u2_node_id))
            edges_to_update.append((idx, "to_bank_id", bank_id))
            edges_to_update.append((idx, "to_account_number", new_account_number))
            edges_to_update.append((idx, "To Bank", int(bank_id) if bank_id.isdigit() else bank_id))
            edges_to_update.append((idx, "To Account", new_account_number))

    # OPTIMIZATION: Batch add new accounts
    if new_accounts_to_add:
        # Get bank name mapping for new accounts
        bank_name_mapping = {}
        for idx in accounts_df.index:
            bank_id = str(accounts_df.at[idx, "Bank ID"]).strip()
            bank_name = str(accounts_df.at[idx, "Bank Name"]) if "Bank Name" in accounts_df.columns else ""
            if bank_id not in bank_name_mapping:
                bank_name_mapping[bank_id] = bank_name

        # Create new account rows
        new_account_rows = []
        for acc_info in new_accounts_to_add:
            bank_name = bank_name_mapping.get(acc_info["bank_id"], "")
            new_account_rows.append({
                "Bank Name": bank_name,
                "Bank ID": acc_info["bank_id"],
                "Account Number": acc_info["account_number"],
                "Entity ID": acc_info["entity_id"],
                "Entity Name": acc_info["entity_name"]
            })

        # Add all new accounts at once
        accounts_df = pd.concat([accounts_df, pd.DataFrame(new_account_rows)], ignore_index=True)

    # OPTIMIZATION: Batch apply all edge updates
    for idx, field, value in edges_to_update:
        transactions_df.at[idx, field] = value

    return {
        "moved_edge_ids": moved_edge_ids,
        "created_accounts": created_accounts,
        "affected_nodes": affected_nodes,
        "counts": {
            "nodes_split": len(node_ids),
            "edges_moved": len(moved_edge_ids),
            "new_accounts_created": len(created_accounts),
        },
        "transactions_df": transactions_df,
        "accounts_df": accounts_df,
    }


def calculate_move_count(total: int, ratio: float) -> int:
    """Calculate number of edges to move with constraints.

    Args:
        total: Total number of edges
        ratio: Split ratio

    Returns:
        Number of edges to move (at least 1, at most total-1)
    """
    if total < 2:
        return 0

    m = round(total * ratio)
    m = max(1, min(m, total - 1))
    return m


def sample_edges(edges: List[tuple], count: int, strategy: str, seed: int) -> List[int]:
    """Sample edges based on strategy.

    Args:
        edges: List of (index, row) tuples
        count: Number to sample
        strategy: Sampling strategy
        seed: Random seed

    Returns:
        List of DataFrame indices
    """
    np.random.seed(seed)

    if count == 0 or len(edges) == 0:
        return []

    count = min(count, len(edges))

    if strategy == "random":
        indices = [e[0] for e in edges]
        return list(np.random.choice(indices, count, replace=False))

    elif strategy == "random_within_currency":
        # Group by currency
        from collections import defaultdict
        currency_groups = defaultdict(list)
        for idx, row in edges:
            currency = row["Payment Currency"]
            currency_groups[currency].append(idx)

        # Sample proportionally from each group
        selected = []
        currencies = list(currency_groups.keys())
        per_currency = max(1, count // len(currencies))

        for currency in currencies:
            group = currency_groups[currency]
            take = min(per_currency, len(group))
            selected.extend(np.random.choice(group, take, replace=False))

        # If we need more, sample from remaining
        if len(selected) < count:
            remaining = [e[0] for e in edges if e[0] not in selected]
            additional = count - len(selected)
            if len(remaining) > 0:
                selected.extend(np.random.choice(remaining, min(additional, len(remaining)), replace=False))

        return selected[:count]

    elif strategy == "time_stratified":
        # Sort by timestamp and stratify
        sorted_edges = sorted(edges, key=lambda e: e[1]["ts_int"])
        indices = [e[0] for e in sorted_edges]

        # Stratified sampling: divide into buckets
        n_buckets = min(5, len(indices))
        bucket_size = len(indices) // n_buckets
        selected = []

        per_bucket = max(1, count // n_buckets)
        for i in range(n_buckets):
            start = i * bucket_size
            end = (i + 1) * bucket_size if i < n_buckets - 1 else len(indices)
            bucket = indices[start:end]
            take = min(per_bucket, len(bucket))
            selected.extend(np.random.choice(bucket, take, replace=False))

        return selected[:count]

    else:
        raise ValueError(f"Unknown edge_sampling strategy: {strategy}")
