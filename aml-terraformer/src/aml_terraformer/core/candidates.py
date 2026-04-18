"""Candidate generation for inject, merge, and split operations."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Set
from collections import defaultdict


def generate_candidates(
    transactions_df: pd.DataFrame,
    accounts_df: pd.DataFrame,
    cluster_nodes: Set[str],
    cluster_internal_edges: List[str],
    topk: int,
    seed: int
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate candidates for inject, merge, and split operations.

    Args:
        transactions_df: Full transactions DataFrame
        accounts_df: Full accounts DataFrame
        cluster_nodes: Set of node IDs in current cluster
        cluster_internal_edges: List of edge IDs that are internal laundering edges
        topk: Number of top candidates to return for each category
        seed: Random seed for deterministic behavior

    Returns:
        Dictionary with keys: inject_candidates, merge_candidates, split_candidates
    """
    np.random.seed(seed)

    inject_cands = generate_inject_candidates(transactions_df, cluster_nodes, cluster_internal_edges, topk)
    merge_cands = generate_merge_candidates(transactions_df, cluster_nodes, topk)
    split_cands = generate_split_candidates(transactions_df, cluster_nodes, topk)
    adjust_cands = generate_adjust_candidates(transactions_df, cluster_nodes, cluster_internal_edges, topk)

    # Shuffle candidates to avoid position bias
    np.random.shuffle(inject_cands)
    np.random.shuffle(merge_cands)
    np.random.shuffle(split_cands)
    np.random.shuffle(adjust_cands)

    return {
        "inject_candidates": inject_cands,
        "merge_candidates": merge_cands,
        "split_candidates": split_cands,
        "adjust_candidates": adjust_cands,
    }


def generate_inject_candidates(
    transactions_df: pd.DataFrame,
    cluster_nodes: Set[str],
    cluster_internal_edges: List[str],
    topk: int
) -> List[Dict[str, Any]]:
    """Generate inject edge candidates.

    Candidates: edges with both endpoints in cluster_nodes and Is Laundering == 1

    Each candidate includes:
    - edge_id
    - from (node_id), to (node_id)
    - ts_int
    - payment_currency
    - amount_paid
    - score_bridge: deg(from) + deg(to) (using全表 incident edges)
    - score_amount_rank: percentile within same Payment Currency
    """
    # Filter to internal laundering edges
    internal_df = transactions_df[
        (transactions_df["edge_id"].isin(cluster_internal_edges)) &
        (transactions_df["Is Laundering"] == 1)
    ].copy()

    if len(internal_df) == 0:
        return []

    # Compute node degrees (using全表) - VECTORIZED
    from_counts = transactions_df["from_node_id"].value_counts()
    to_counts = transactions_df["to_node_id"].value_counts()
    node_degrees = defaultdict(int)
    for node, count in from_counts.items():
        node_degrees[node] += count
    for node, count in to_counts.items():
        node_degrees[node] += count

    # Build candidates - VECTORIZED
    internal_df["score_bridge"] = internal_df["from_node_id"].map(node_degrees) + internal_df["to_node_id"].map(node_degrees)

    candidates = internal_df[[
        "edge_id", "from_node_id", "to_node_id", "ts_int",
        "Payment Currency", "Amount Paid", "score_bridge"
    ]].rename(columns={
        "from_node_id": "from",
        "to_node_id": "to",
        "Payment Currency": "payment_currency",
        "Amount Paid": "amount_paid"
    }).to_dict('records')

    # Compute score_amount_rank (percentile within same currency)
    currency_groups = defaultdict(list)
    for cand in candidates:
        currency_groups[cand["payment_currency"]].append(cand)

    for currency, cands in currency_groups.items():
        amounts = [c["amount_paid"] for c in cands]
        for cand in cands:
            rank = sum(1 for a in amounts if a <= cand["amount_paid"]) / len(amounts)
            cand["score_amount_rank"] = rank

    # Sort by score_bridge (desc), then edge_id (deterministic)
    candidates.sort(key=lambda x: (-x["score_bridge"], x["edge_id"]))

    return candidates[:topk]


def generate_merge_candidates(
    transactions_df: pd.DataFrame,
    cluster_nodes: Set[str],
    topk: int
) -> List[Dict[str, Any]]:
    """Generate merge node pair candidates.

    Candidates: pairs (a, b) where both are in cluster_nodes with Jaccard >= threshold

    Each candidate includes:
    - a, b (node_ids)
    - bank_id_a, bank_id_b (bank IDs of the two nodes)
    - pair_id: "{a}__{b}" (sorted)
    - score_jaccard: Jaccard similarity of cluster-internal neighbors (ignoring direction)
    """
    # Get cluster-internal edges
    cluster_internal_df = transactions_df[
        (transactions_df["from_node_id"].isin(cluster_nodes)) &
        (transactions_df["to_node_id"].isin(cluster_nodes))
    ].copy()

    # Build neighbor sets for each node (within cluster) - VECTORIZED
    neighbors = defaultdict(set)
    for from_node, to_node in zip(cluster_internal_df["from_node_id"], cluster_internal_df["to_node_id"]):
        neighbors[from_node].add(to_node)
        neighbors[to_node].add(from_node)

    # Generate pairs from all cluster nodes (no bank restriction)
    # Filter by Jaccard similarity threshold
    from .identifiers import parse_node_id
    JACCARD_THRESHOLD = 0.05  # Minimum similarity threshold

    candidates = []
    seen_pairs = set()
    cluster_nodes_list = list(cluster_nodes)

    for i, a in enumerate(cluster_nodes_list):
        for b in cluster_nodes_list[i+1:]:
                # Create deterministic pair_id
                pair = tuple(sorted([a, b]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                # Compute Jaccard
                neighbors_a = neighbors[a]
                neighbors_b = neighbors[b]
                if len(neighbors_a) == 0 and len(neighbors_b) == 0:
                    jaccard = 0.0
                else:
                    intersection = len(neighbors_a & neighbors_b)
                    union = len(neighbors_a | neighbors_b)
                    jaccard = intersection / union if union > 0 else 0.0

                # Filter by Jaccard threshold
                if jaccard < JACCARD_THRESHOLD:
                    continue

                pair_id = f"{pair[0]}__{pair[1]}"

                # Get bank IDs for both nodes
                bank_id_a, _ = parse_node_id(a)
                bank_id_b, _ = parse_node_id(b)

                candidates.append({
                    "a": pair[0],
                    "b": pair[1],
                    "bank_id_a": bank_id_a,
                    "bank_id_b": bank_id_b,
                    "pair_id": pair_id,
                    "score_jaccard": jaccard,
                })

    # Sort by score_jaccard (desc), then pair_id (deterministic)
    candidates.sort(key=lambda x: (-x["score_jaccard"], x["pair_id"]))

    return candidates[:topk]


def generate_split_candidates(
    transactions_df: pd.DataFrame,
    cluster_nodes: Set[str],
    topk: int
) -> List[Dict[str, Any]]:
    """Generate split node candidates.

    Candidates: nodes in cluster_nodes with >= 2 incident edges

    Each candidate includes:
    - node_id
    - bank_id
    - incident_edges (count)
    - out_edges (count)
    - in_edges (count)
    - currency_top3: top 3 Payment Currency values
    """
    from .identifiers import parse_node_id
    from collections import Counter

    # Count incident edges per node - VECTORIZED
    node_stats = defaultdict(lambda: {"out": 0, "in": 0, "currencies": []})

    # Filter to edges involving cluster nodes
    cluster_edges_df = transactions_df[
        transactions_df["from_node_id"].isin(cluster_nodes) |
        transactions_df["to_node_id"].isin(cluster_nodes)
    ]

    # Process outgoing edges
    out_edges = cluster_edges_df[cluster_edges_df["from_node_id"].isin(cluster_nodes)]
    for node in cluster_nodes:
        node_edges = out_edges[out_edges["from_node_id"] == node]
        node_stats[node]["out"] = len(node_edges)
        node_stats[node]["currencies"].extend(node_edges["Payment Currency"].tolist())

    # Process incoming edges
    in_edges = cluster_edges_df[cluster_edges_df["to_node_id"].isin(cluster_nodes)]
    for node in cluster_nodes:
        node_edges = in_edges[in_edges["to_node_id"] == node]
        node_stats[node]["in"] = len(node_edges)
        node_stats[node]["currencies"].extend(node_edges["Payment Currency"].tolist())

    # Build candidates
    candidates = []
    for node in cluster_nodes:
        stats = node_stats[node]
        incident = stats["out"] + stats["in"]

        if incident < 2:
            continue

        bank_id, _ = parse_node_id(node)

        # Get top 3 currencies
        currency_counter = Counter(stats["currencies"])
        currency_top3 = [c for c, _ in currency_counter.most_common(3)]

        candidates.append({
            "node_id": node,
            "bank_id": bank_id,
            "incident_edges": incident,
            "out_edges": stats["out"],
            "in_edges": stats["in"],
            "currency_top3": currency_top3,
        })

    # Sort by incident_edges (desc), then node_id (deterministic)
    candidates.sort(key=lambda x: (-x["incident_edges"], x["node_id"]))

    return candidates[:topk]


def generate_adjust_candidates(
    transactions_df: pd.DataFrame,
    cluster_nodes: Set[str],
    cluster_internal_edges: List[str],
    topk: int
) -> List[Dict[str, Any]]:
    """Generate candidates for adjust_transaction operation.
    
    Select laundering transactions that may trigger S6 rule (fast in-fast out).
    
    Returns list of dicts with:
    - edge_id
    - from (node_id), to (node_id)
    - ts_int
    - amount_paid
    - score_s6_risk: likelihood of triggering S6 rule
    """
    # Filter to internal laundering edges
    internal_df = transactions_df[
        (transactions_df["edge_id"].isin(cluster_internal_edges)) &
        (transactions_df["Is Laundering"] == 1)
    ].copy()
    
    if len(internal_df) == 0:
        return []
    
    # Compute S6 risk score for each transaction
    # S6 checks: in_out_ratio in [0.9, 1.1], max_in_or_out_amt_3d >= 200000, end_balance <= 100
    # Higher risk = more likely to trigger S6
    internal_df["score_s6_risk"] = 0.0
    
    # Risk factor 1: High amount (more likely to exceed 200000 threshold)
    internal_df["score_s6_risk"] += (internal_df["Amount Paid"] / 1000000.0).clip(0, 1)
    
    # Risk factor 2: Recent transactions (more likely to be in same 3-day window)
    max_ts = internal_df["ts_int"].max()
    internal_df["score_s6_risk"] += ((max_ts - internal_df["ts_int"]) / (3 * 86400)).clip(0, 1)
    
    candidates = internal_df[[
        "edge_id", "from_node_id", "to_node_id", "ts_int",
        "Amount Paid", "score_s6_risk"
    ]].rename(columns={
        "from_node_id": "from",
        "to_node_id": "to",
        "Amount Paid": "amount_paid"
    }).to_dict('records')
    
    # Sort by score_s6_risk (desc), then edge_id (deterministic)
    candidates.sort(key=lambda x: (-x["score_s6_risk"], x["edge_id"]))
    
    return candidates[:topk]
