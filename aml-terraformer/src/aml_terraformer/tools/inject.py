"""Inject intermediary tool: insert intermediary nodes into transaction chains."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict
from ..core.identifiers import make_node_id, make_edge_id, parse_node_id
from ..core.normalize import ensure_account_exists


def build_neighbor_graph(transactions_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Build account neighbor relationship graph.

    Args:
        transactions_df: Transactions DataFrame

    Returns:
        Dictionary mapping node_id to set of neighbor node_ids
    """
    neighbors = defaultdict(set)

    # Vectorized approach: extract columns as arrays
    from_nodes = transactions_df["from_node_id"].values
    to_nodes = transactions_df["to_node_id"].values

    # Build neighbor relationships
    for from_node, to_node in zip(from_nodes, to_nodes):
        if from_node != to_node:
            neighbors[from_node].add(to_node)
            neighbors[to_node].add(from_node)

    return neighbors


def select_common_neighbor(
    from_node: str,
    to_node: str,
    neighbors: Dict[str, Set[str]],
    cluster_nodes: Set[str],
    seed: int
) -> str:
    """Select a common neighbor as intermediary account.

    Strategy:
    1. Find common neighbors of from_node and to_node (nodes connected to both)
    2. If no common neighbor, fall back to neighbors of from_node or to_node
    3. Exclude: from_node, to_node, MERGED/INTERM/SPLIT accounts
    4. Prefer accounts within the same cluster

    Args:
        from_node: Source node
        to_node: Destination node
        neighbors: Neighbor graph
        cluster_nodes: Set of nodes in the current cluster
        seed: Random seed

    Returns:
        Selected intermediary node_id, or None if no suitable candidate
    """
    # Ensure seed is within valid range for NumPy (0 to 2^32 - 1)
    seed = seed % (2**32)
    np.random.seed(seed)

    # Get neighbors of from_node and to_node
    from_neighbors = neighbors.get(from_node, set())
    to_neighbors = neighbors.get(to_node, set())

    # Find common neighbors (nodes connected to both from_node and to_node)
    common_neighbors = from_neighbors & to_neighbors

    # Exclude endpoints and synthetic accounts
    def is_valid_node(node):
        if node == from_node or node == to_node:
            return False
        account_num = node.split('|')[1]
        if (account_num.startswith('MERGED_') or
            account_num.startswith('INTERM_') or
            account_num.startswith('SPLIT_')):
            return False
        return True

    valid_common = {node for node in common_neighbors if is_valid_node(node)}

    # Priority 1: Common neighbors within cluster
    cluster_common = valid_common & cluster_nodes
    if len(cluster_common) > 0:
        return np.random.choice(list(cluster_common))

    # Priority 2: Common neighbors outside cluster
    if len(valid_common) > 0:
        return np.random.choice(list(valid_common))

    # Fall back: Any neighbor of from_node or to_node
    all_neighbors = from_neighbors | to_neighbors
    valid_neighbors = {node for node in all_neighbors if is_valid_node(node)}

    # Priority 3: Neighbors within cluster
    cluster_neighbors = valid_neighbors & cluster_nodes
    if len(cluster_neighbors) > 0:
        return np.random.choice(list(cluster_neighbors))

    # Priority 4: Neighbors outside cluster
    if len(valid_neighbors) > 0:
        return np.random.choice(list(valid_neighbors))

    return None


def inject_intermediary(
    transactions_df: pd.DataFrame,
    accounts_df: pd.DataFrame,
    edge_ids: List[str],
    depth: int,
    time_delta_seconds: int,
    timestamp_format: str = "iso",
    seed: int = 42,
    cluster_nodes: Set[str] = None
) -> Dict[str, Any]:
    """Inject intermediary nodes into transaction chains.

    For each edge u->v:
    - Delete original edge
    - Select depth existing accounts as intermediaries (common neighbors preferred)
    - Insert depth+1 new edges: u->x1, x1->x2, ..., xdepth->v
    - Timestamps: t, t+delta, t+2*delta, ...
    - All other fields copied from original

    Intermediary selection strategy:
    1. Common neighbors (nodes connected to both u and v) - preferred
    2. Neighbors of u or v (if no common neighbor)
    3. Create new INTERM account (fallback)

    Args:
        transactions_df: Transactions DataFrame
        accounts_df: Accounts DataFrame
        edge_ids: List of edge IDs to inject into
        depth: Number of intermediary nodes (1 or 2)
        time_delta_seconds: Time delta between hops
        timestamp_format: Format for output timestamps
        seed: Random seed
        cluster_nodes: Set of node IDs in current cluster (optional)

    Returns:
        Dictionary with:
        - affected_original_edge_ids: List of original edge IDs
        - new_edge_ids: List of created edge IDs
        - created_accounts: List of created node IDs (empty if using existing accounts)
        - counts: Summary counts
    """
    from ..io.timestamp_handler import format_timestamp

    # Ensure seed is within valid range for NumPy (0 to 2^32 - 1)
    seed = seed % (2**32)
    np.random.seed(seed)

    affected_original_edge_ids = []
    new_edge_ids = []
    created_accounts = []  # Empty for compatibility
    affected_nodes = []  # Nodes that were directly operated on

    # Track next insertion index per original row
    insertion_counters = {}

    # Build neighbor graph for selecting intermediaries
    neighbors = build_neighbor_graph(transactions_df)

    # Use empty set if cluster_nodes not provided
    if cluster_nodes is None:
        cluster_nodes = set()

    # OPTIMIZATION: Build edge_id index for fast lookup
    edge_id_to_idx = {edge_id: idx for idx, edge_id in enumerate(transactions_df["edge_id"].values)}

    # OPTIMIZATION: Build accounts index for fast lookup (vectorized)
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

    # OPTIMIZATION: Collect rows to delete and rows to add
    rows_to_delete_indices = []
    new_rows = []

    # OPTIMIZATION: Collect new accounts to add them in batch
    new_accounts_to_add = []  # List of account info dicts

    for edge_id in edge_ids:
        # OPTIMIZATION: Use index for fast lookup
        if edge_id not in edge_id_to_idx:
            continue

        idx = edge_id_to_idx[edge_id]
        row = transactions_df.iloc[idx]
        original_row_index = row["original_row_index"]

        # Track insertion counter
        if original_row_index not in insertion_counters:
            insertion_counters[original_row_index] = 0

        # Get original endpoints
        from_node = row["from_node_id"]
        to_node = row["to_node_id"]
        from_bank_id, from_account = parse_node_id(from_node)
        to_bank_id, to_account = parse_node_id(to_node)

        # Get original timestamp
        original_ts_int = row["ts_int"]

        # OPTIMIZATION: Mark for deletion instead of deleting immediately
        rows_to_delete_indices.append(idx)
        affected_original_edge_ids.append(edge_id)

        # Select existing accounts as intermediaries (second-degree neighbors)
        intermediaries = []
        for i in range(depth):
            # Use different seed for each intermediary
            intermediary_seed = seed + original_row_index * 1000 + insertion_counters[original_row_index] * 10 + i

            # Select common neighbor as intermediary
            intermediary = select_common_neighbor(
                from_node=from_node,
                to_node=to_node,
                neighbors=neighbors,
                cluster_nodes=cluster_nodes,
                seed=intermediary_seed
            )

            # If no suitable neighbor found, fall back to creating new account
            if intermediary is None:
                # Fallback: create new INTERM account
                new_account_number = f"INTERM_{original_row_index}_{insertion_counters[original_row_index]}_{i}"
                intermediary = make_node_id(from_bank_id, new_account_number)

                # OPTIMIZATION: Use accounts_index for fast lookup
                source_key = f"{from_bank_id}|{from_account}"
                source_entity_id = ""
                source_entity_name = ""
                if source_key in accounts_index:
                    source_entity_id = accounts_index[source_key]["entity_id"]
                    source_entity_name = accounts_index[source_key]["entity_name"]

                # OPTIMIZATION: Collect account info instead of adding immediately
                new_accounts_to_add.append({
                    "bank_id": from_bank_id,
                    "account_number": new_account_number,
                    "entity_id": source_entity_id,
                    "entity_name": source_entity_name
                })
                created_accounts.append(intermediary)

                # Update accounts_index for future lookups
                new_key = f"{from_bank_id}|{new_account_number}"
                accounts_index[new_key] = {
                    "entity_id": source_entity_id,
                    "entity_name": source_entity_name
                }

            intermediaries.append(intermediary)
            # Record the intermediary node as affected
            affected_nodes.append(intermediary)

        # Build chain: u -> x1 -> x2 -> ... -> v
        chain = [from_node] + intermediaries + [to_node]

        # Create new transaction rows
        for hop_idx in range(len(chain) - 1):
            hop_from = chain[hop_idx]
            hop_to = chain[hop_idx + 1]

            hop_from_bank_id, hop_from_account = parse_node_id(hop_from)
            hop_to_bank_id, hop_to_account = parse_node_id(hop_to)

            # Create new edge_id
            new_edge_id = make_edge_id(original_row_index, insertion_counters[original_row_index])
            insertion_counters[original_row_index] += 1
            new_edge_ids.append(new_edge_id)

            # Calculate timestamp
            hop_ts_int = original_ts_int + hop_idx * time_delta_seconds
            hop_timestamp = format_timestamp(hop_ts_int, timestamp_format, original_value=row["Timestamp"])

            # Create new row (copy from original)
            new_row = row.copy()
            new_row["edge_id"] = new_edge_id
            new_row["from_node_id"] = hop_from
            new_row["to_node_id"] = hop_to
            new_row["from_bank_id"] = hop_from_bank_id
            new_row["to_bank_id"] = hop_to_bank_id
            new_row["from_account_number"] = hop_from_account
            new_row["to_account_number"] = hop_to_account
            new_row["From Bank"] = hop_from_bank_id
            new_row["To Bank"] = hop_to_bank_id
            new_row["From Account"] = hop_from_account
            new_row["To Account"] = hop_to_account
            new_row["ts_int"] = hop_ts_int
            new_row["Timestamp"] = hop_timestamp

            # OPTIMIZATION: Collect new row instead of appending immediately
            new_rows.append(new_row)

    # OPTIMIZATION: Batch add new accounts
    if new_accounts_to_add:
        # Get bank name mapping for new accounts (vectorized)
        bank_ids_for_mapping = accounts_df["Bank ID"].astype(str).str.strip().values
        bank_names_for_mapping = accounts_df["Bank Name"].astype(str).values if "Bank Name" in accounts_df.columns else [""] * len(accounts_df)

        bank_name_mapping = {}
        for bank_id, bank_name in zip(bank_ids_for_mapping, bank_names_for_mapping):
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

    # OPTIMIZATION: Batch delete and add operations
    # Delete rows marked for deletion
    if rows_to_delete_indices:
        transactions_df = transactions_df.drop(transactions_df.index[rows_to_delete_indices])

    # Add all new rows at once
    if new_rows:
        transactions_df = pd.concat([transactions_df, pd.DataFrame(new_rows)], ignore_index=True)

    return {
        "affected_original_edge_ids": affected_original_edge_ids,
        "new_edge_ids": new_edge_ids,
        "created_accounts": created_accounts,
        "affected_nodes": affected_nodes,
        "counts": {
            "original_edges_deleted": len(affected_original_edge_ids),
            "new_edges_created": len(new_edge_ids),
            "new_accounts_created": len(created_accounts),
        },
        "transactions_df": transactions_df,
        "accounts_df": accounts_df,
    }
