"""Pre-validation for tool decisions before execution."""

import networkx as nx
import pandas as pd
from typing import Dict, Any, Tuple, Optional


def pre_validate_decision(
    decision: Dict[str, Any],
    transactions_df: pd.DataFrame,
    accounts_df: pd.DataFrame,
    cluster_nodes: set
) -> Tuple[bool, Optional[str]]:
    """Pre-validate a tool decision before execution.

    This function checks if a decision is likely to succeed before
    actually executing it, preventing common validation failures.

    Args:
        decision: Tool decision dictionary with 'tool' and 'args'
        transactions_df: Current transactions DataFrame
        accounts_df: Current accounts DataFrame
        cluster_nodes: Set of nodes in the cluster

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if decision is likely to succeed
        - error_message: Description of why validation failed (if invalid)
    """
    tool = decision.get("tool")
    args = decision.get("args", {})

    if tool == "split_account":
        return _pre_validate_split(args, transactions_df, accounts_df)
    elif tool == "inject_intermediary":
        return _pre_validate_inject(args, transactions_df)
    elif tool == "merge_accounts":
        return _pre_validate_merge(args, accounts_df)
    elif tool == "adjust_transaction":
        return _pre_validate_adjust(args, transactions_df)
    else:
        # Unknown tool or stop - allow
        return True, None


def _pre_validate_split(
    args: Dict[str, Any],
    transactions_df: pd.DataFrame,
    accounts_df: pd.DataFrame
) -> Tuple[bool, Optional[str]]:
    """Pre-validate split_account operation."""
    node_ids = args.get("node_ids", [])
    split_ratio = args.get("split_ratio", 0.2)
    move_direction = args.get("move_direction", "out")

    if not node_ids:
        return False, "No node_ids specified"

    node_id = node_ids[0]

    # Check if node exists (node_id format: "bank_id|account_number")
    node_id_set = set(
        (accounts_df["Bank ID"].astype(str).str.strip() + "|" +
         accounts_df["Account Number"].astype(str).str.strip()).values
    )
    if node_id not in node_id_set:
        return False, f"Node {node_id} not found in accounts"

    # Build graph to check edge counts
    G = _build_graph_from_transactions(transactions_df)

    if node_id not in G:
        return False, f"Node {node_id} has no transactions"

    # Check outgoing edges
    if move_direction in ["out", "both"]:
        out_degree = G.out_degree(node_id)
        if out_degree == 0:
            return False, f"Node {node_id} has no outgoing edges"

        # Estimate how many edges will be moved
        edges_to_move = int(out_degree * split_ratio)
        edges_remaining = out_degree - edges_to_move

        # Need at least 1 edge remaining
        if edges_remaining < 1:
            return False, f"Split would leave node {node_id} with no outgoing edges (out_degree={out_degree}, ratio={split_ratio})"

    # Check incoming edges
    if move_direction in ["in", "both"]:
        in_degree = G.in_degree(node_id)
        if in_degree == 0:
            return False, f"Node {node_id} has no incoming edges"

        edges_to_move = int(in_degree * split_ratio)
        edges_remaining = in_degree - edges_to_move

        if edges_remaining < 1:
            return False, f"Split would leave node {node_id} with no incoming edges (in_degree={in_degree}, ratio={split_ratio})"

    return True, None


def _pre_validate_inject(
    args: Dict[str, Any],
    transactions_df: pd.DataFrame
) -> Tuple[bool, Optional[str]]:
    """Pre-validate inject_intermediary operation."""
    edge_ids = args.get("edge_ids", [])

    if not edge_ids:
        return False, "No edge_ids specified"

    # Check if edges exist (edge_id is in the "edge_id" column, not the index)
    edge_id_set = set(transactions_df["edge_id"].values)
    for edge_id in edge_ids:
        if edge_id not in edge_id_set:
            return False, f"Edge {edge_id} not found in transactions"

    return True, None


def _pre_validate_merge(
    args: Dict[str, Any],
    accounts_df: pd.DataFrame
) -> Tuple[bool, Optional[str]]:
    """Pre-validate merge_accounts operation."""
    pairs = args.get("pairs", [])

    if not pairs:
        return False, "No pairs specified"

    # Check if all nodes exist (node_id format: "bank_id|account_number")
    for pair in pairs:
        node_a, node_b = pair

        # Parse node IDs
        try:
            bank_a, acc_a = node_a.split('|', 1)
            bank_b, acc_b = node_b.split('|', 1)
        except ValueError:
            return False, f"Invalid node ID format: {node_a} or {node_b}"

        # Check if nodes exist in accounts_df
        mask_a = (accounts_df['Bank ID'].astype(str) == bank_a) & \
                 (accounts_df['Account Number'].astype(str) == acc_a)
        mask_b = (accounts_df['Bank ID'].astype(str) == bank_b) & \
                 (accounts_df['Account Number'].astype(str) == acc_b)

        if not mask_a.any():
            return False, f"Node {node_a} not found in accounts"
        if not mask_b.any():
            return False, f"Node {node_b} not found in accounts"

    return True, None


def _pre_validate_adjust(
    args: Dict[str, Any],
    transactions_df: pd.DataFrame
) -> Tuple[bool, Optional[str]]:
    """Pre-validate adjust_transaction operation."""
    # 支持两种参数名：edge_id (单数) 和 edge_ids (复数)
    edge_ids = args.get("edge_ids") or [args.get("edge_id")]

    if not edge_ids or edge_ids[0] is None:
        return False, "No edge_id specified"

    # Check if all edge_ids exist in the "edge_id" column (not DataFrame index)
    edge_id_set = set(transactions_df["edge_id"].values)
    for edge_id in edge_ids:
        if edge_id not in edge_id_set:
            return False, f"Edge {edge_id} not found in transactions"

    return True, None


def _build_graph_from_transactions(transactions_df: pd.DataFrame) -> nx.DiGraph:
    """Build a directed graph from transactions DataFrame."""
    G = nx.DiGraph()

    for _, row in transactions_df.iterrows():
        from_node = f"{row['From Bank']}|{row['From Account']}"
        to_node = f"{row['To Bank']}|{row['To Account']}"
        G.add_edge(from_node, to_node)

    return G
