"""Merge accounts tool: merge multiple accounts into one."""

import pandas as pd
from typing import List, Dict, Any, Tuple
from ..core.identifiers import make_node_id, parse_node_id


def merge_accounts(
    transactions_df: pd.DataFrame,
    accounts_df: pd.DataFrame,
    pairs: List[Tuple[str, str]],
    drop_self_loops: bool,
    seed: int = 42
) -> Dict[str, Any]:
    """Merge account pairs into new unified accounts.

    For each pair (a, b):
    - Create new account ab_new (same bank_id)
    - Retarget all transactions: endpoints in {a, b} -> ab_new
    - Update From Bank/Account and To Bank/Account accordingly
    - Optionally drop self-loops (ab_new -> ab_new)
    - Mark original accounts as merged in accounts_df

    Args:
        transactions_df: Transactions DataFrame
        accounts_df: Accounts DataFrame
        pairs: List of (a, b) tuples where a and b are node IDs
        drop_self_loops: Whether to drop self-loops after merging
        seed: Random seed

    Returns:
        Dictionary with:
        - retarget_map: Dict mapping old node_id to new node_id
        - affected_edge_ids: List of edge IDs with modified endpoints
        - created_accounts: List of created node IDs
        - self_loop_dropped_count: Number of self-loops dropped
        - counts: Summary counts
    """
    import numpy as np

    np.random.seed(seed)

    retarget_map = {}
    created_accounts = []
    affected_edge_ids = []
    affected_nodes = []  # Nodes that were directly operated on
    self_loop_dropped_count = 0

    # Initialize Status and MergedInto columns if not exists
    if "Status" not in accounts_df.columns:
        accounts_df["Status"] = ""
    if "MergedInto" not in accounts_df.columns:
        accounts_df["MergedInto"] = ""

    # OPTIMIZATION: Pre-build accounts index for fast lookup (vectorized)
    bank_ids = accounts_df["Bank ID"].astype(str).str.strip().values
    account_nums = accounts_df["Account Number"].astype(str).str.strip().values
    indices = accounts_df.index.values

    accounts_index = {}
    for idx, bank_id, account_num in zip(indices, bank_ids, account_nums):
        key = f"{bank_id}|{account_num}"
        accounts_index[key] = idx

    # OPTIMIZATION: Collect account updates to batch them
    accounts_to_mark_merged = []  # List of (index, ab_new) tuples

    for pair_idx, (a, b) in enumerate(pairs):
        # Parse bank IDs
        bank_a, account_a = parse_node_id(a)
        bank_b, account_b = parse_node_id(b)

        # Bank restriction removed - allow cross-bank merges
        # (Jaccard similarity threshold is enforced in candidates.py)

        # Randomly choose account_a or account_b as the merge target
        # Use pair_idx as part of seed for deterministic randomness
        merge_seed = seed + pair_idx
        np.random.seed(merge_seed)

        if np.random.random() < 0.5:
            # Merge into account_a
            ab_new = a
            merged_account = b
        else:
            # Merge into account_b
            ab_new = b
            merged_account = a

        retarget_map[a] = ab_new
        retarget_map[b] = ab_new

        # Record both accounts as affected (the merged one and the target)
        affected_nodes.append(merged_account)
        affected_nodes.append(ab_new)

        # No need to create new account since we're using existing account
        # The target account (ab_new) already exists in accounts_df

        # OPTIMIZATION: Use index lookup instead of DataFrame filtering
        merged_bank, merged_account_num = parse_node_id(merged_account)
        merged_key = f"{merged_bank}|{merged_account_num}"
        if merged_key in accounts_index:
            merged_idx = accounts_index[merged_key]
            accounts_to_mark_merged.append((merged_idx, ab_new))

    # OPTIMIZATION: Batch update account status
    for merged_idx, ab_new in accounts_to_mark_merged:
        accounts_df.at[merged_idx, "Status"] = "merged"
        accounts_df.at[merged_idx, "MergedInto"] = ab_new

    # Retarget all transaction endpoints - VECTORIZED
    # Find affected rows
    from_affected = transactions_df["from_node_id"].isin(retarget_map.keys())
    to_affected = transactions_df["to_node_id"].isin(retarget_map.keys())
    affected_mask = from_affected | to_affected

    if affected_mask.any():
        # Get affected edge IDs
        affected_edge_ids.extend(transactions_df.loc[affected_mask, "edge_id"].tolist())

        # Update from nodes
        transactions_df.loc[from_affected, "from_node_id"] = transactions_df.loc[from_affected, "from_node_id"].map(retarget_map)

        # Update to nodes
        transactions_df.loc[to_affected, "to_node_id"] = transactions_df.loc[to_affected, "to_node_id"].map(retarget_map)

        # OPTIMIZATION: Vectorize bank and account field updates
        # Parse all affected node IDs at once using list comprehension
        affected_subset = transactions_df[affected_mask]

        # Parse from nodes
        from_parsed = [parse_node_id(nid) for nid in affected_subset["from_node_id"]]
        from_banks = [bank for bank, _ in from_parsed]
        from_accounts = [acc for _, acc in from_parsed]

        # Parse to nodes
        to_parsed = [parse_node_id(nid) for nid in affected_subset["to_node_id"]]
        to_banks = [bank for bank, _ in to_parsed]
        to_accounts = [acc for _, acc in to_parsed]

        # Batch update all fields using .loc
        transactions_df.loc[affected_mask, "from_bank_id"] = from_banks
        transactions_df.loc[affected_mask, "to_bank_id"] = to_banks
        transactions_df.loc[affected_mask, "from_account_number"] = from_accounts
        transactions_df.loc[affected_mask, "to_account_number"] = to_accounts

        # Convert bank_id to int for display columns
        from_banks_display = [int(b) if b.isdigit() else b for b in from_banks]
        to_banks_display = [int(b) if b.isdigit() else b for b in to_banks]

        transactions_df.loc[affected_mask, "From Bank"] = from_banks_display
        transactions_df.loc[affected_mask, "To Bank"] = to_banks_display
        transactions_df.loc[affected_mask, "From Account"] = from_accounts
        transactions_df.loc[affected_mask, "To Account"] = to_accounts

    # Drop self-loops if requested
    if drop_self_loops:
        before_count = len(transactions_df)
        transactions_df = transactions_df[transactions_df["from_node_id"] != transactions_df["to_node_id"]]
        after_count = len(transactions_df)
        self_loop_dropped_count = before_count - after_count

    return {
        "retarget_map": retarget_map,
        "affected_edge_ids": affected_edge_ids,
        "created_accounts": created_accounts,
        "affected_nodes": affected_nodes,
        "self_loop_dropped_count": self_loop_dropped_count,
        "counts": {
            "pairs_merged": len(pairs),
            "affected_edges": len(affected_edge_ids),
            "new_accounts_created": len(created_accounts),
            "self_loops_dropped": self_loop_dropped_count,
        },
        "transactions_df": transactions_df,
        "accounts_df": accounts_df,
    }
