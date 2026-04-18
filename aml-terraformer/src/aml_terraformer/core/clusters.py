"""Laundering cluster detection via connected components."""

import pandas as pd
from typing import List, Set
from collections import defaultdict


class LaunderingCluster:
    """Represents a laundering cluster."""

    def __init__(self, cluster_id: int, nodes: Set[str], internal_edges: List[str]):
        self.cluster_id = cluster_id
        self.nodes_in_cluster = nodes
        self.laundering_edges_internal = internal_edges


def find_laundering_clusters(transactions_df: pd.DataFrame) -> List[LaunderingCluster]:
    """Find laundering clusters using connected components (Union-Find / DSU).

    Args:
        transactions_df: Transactions DataFrame with node IDs and Is Laundering column

    Returns:
        List of LaunderingCluster objects
    """
    # Filter to laundering transactions only
    laundering_df = transactions_df[transactions_df["Is Laundering"] == 1].copy()
    if laundering_df.empty:
        return []

    # Extract columns as arrays for faster access
    from_nodes = laundering_df["from_node_id"].to_numpy()
    to_nodes = laundering_df["to_node_id"].to_numpy()
    edge_ids = laundering_df["edge_id"].to_numpy()

    # Union-Find (Disjoint Set Union) with path compression + union by rank
    parent = {}
    rank = {}

    def find(x):
        """Find root with iterative path compression."""
        if x not in parent:
            parent[x] = x
            rank[x] = 0
            return x

        # Find root
        root = x
        while parent[root] != root:
            root = parent[root]

        # Path compression
        while parent[x] != x:
            nxt = parent[x]
            parent[x] = root
            x = nxt

        return root

    def union(x, y):
        """Union by rank."""
        rx = find(x)
        ry = find(y)
        if rx == ry:
            return

        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1

    # 1) Union all laundering edges
    n = len(edge_ids)
    for i in range(n):
        union(from_nodes[i], to_nodes[i])

    # 2) Aggregate nodes and internal edges by component root (single pass over edges)
    components_nodes = defaultdict(set)   # root -> set(nodes)
    components_edges = defaultdict(list)  # root -> list(edge_ids)

    for i in range(n):
        u = from_nodes[i]
        v = to_nodes[i]
        eid = edge_ids[i]
        r = find(u)  # same component as v

        components_nodes[r].add(u)
        components_nodes[r].add(v)
        components_edges[r].append(eid)

    # 3) Build LaunderingCluster objects (preserve interface/fields)
    clusters: List[LaunderingCluster] = []
    for cluster_id, (root, nodes) in enumerate(components_nodes.items()):
        clusters.append(
            LaunderingCluster(
                cluster_id=cluster_id,
                nodes=nodes,
                internal_edges=components_edges[root],
            )
        )

    return clusters
