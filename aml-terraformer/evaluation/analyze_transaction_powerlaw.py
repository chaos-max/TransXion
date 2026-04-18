"""
Power Law Analysis for Transaction Network

This script loads transaction data from CSV, constructs a directed graph,
and performs power law distribution analysis on the degree distribution.
"""

import os
import sys
import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path

# Add the evaluation directory to path to import powerlaw_utils module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import powerlaw_utils as pl


def load_transaction_data(csv_path):
    """
    Load transaction data from CSV file.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing transaction data

    Returns:
    --------
    pd.DataFrame
        Transaction data
    """
    print(f"Loading transaction data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} transactions")
    print(f"Columns: {df.columns.tolist()}")
    return df


def build_transaction_graph(df, weight_column='Amount Received'):
    """
    Build a directed graph from transaction data.

    Parameters:
    -----------
    df : pd.DataFrame
        Transaction data with columns for source and target accounts
    weight_column : str
        Column name to use as edge weight (default: 'Amount Received')

    Returns:
    --------
    nx.DiGraph
        Directed graph where nodes are accounts and edges are transactions
    """
    print("\nBuilding transaction graph...")
    G = nx.DiGraph()

    # Create unique account identifiers by combining bank and account
    df['from_account'] = df['From Bank'].astype(str) + '_' + df['Account'].astype(str)
    df['to_account'] = df['To Bank'].astype(str) + '_' + df['Account.1'].astype(str)

    # Add edges with weights
    for idx, row in df.iterrows():
        from_acc = row['from_account']
        to_acc = row['to_account']
        weight = row[weight_column]

        # Add edge (if edge exists, sum the weights)
        if G.has_edge(from_acc, to_acc):
            G[from_acc][to_acc]['weight'] += weight
            G[from_acc][to_acc]['count'] += 1
        else:
            G.add_edge(from_acc, to_acc, weight=weight, count=1)

    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Print basic statistics
    print(f"\nGraph Statistics:")
    print(f"  Number of nodes: {G.number_of_nodes()}")
    print(f"  Number of edges: {G.number_of_edges()}")
    print(f"  Average in-degree: {sum(dict(G.in_degree()).values()) / G.number_of_nodes():.2f}")
    print(f"  Average out-degree: {sum(dict(G.out_degree()).values()) / G.number_of_nodes():.2f}")

    return G


def analyze_degree_distribution(G, output_dir='./results', graph_name='transaction_network'):
    """
    Analyze the degree distribution of the graph using power law fitting.

    Parameters:
    -----------
    G : nx.DiGraph
        Transaction graph
    output_dir : str
        Directory to save output plots and results
    graph_name : str
        Name for the graph (used in output filenames)

    Returns:
    --------
    dict
        Dictionary containing analysis results (DataFrames with power law statistics)
    """
    print("\n" + "="*60)
    print("POWER LAW ANALYSIS")
    print("="*60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Print basic degree statistics
    print("\n--- DEGREE STATISTICS ---")
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]

    print(f"In-degree: min={min(in_degrees)}, max={max(in_degrees)}, "
          f"mean={np.mean(in_degrees):.2f}, median={np.median(in_degrees):.2f}")
    print(f"Out-degree: min={min(out_degrees)}, max={max(out_degrees)}, "
          f"mean={np.mean(out_degrees):.2f}, median={np.median(out_degrees):.2f}")

    # Perform power law analysis using the calculate_power_law function
    print("\n--- FITTING POWER LAW DISTRIBUTION ---")
    results = pl.calculate_power_law(
        G=G,
        save_dir=output_dir,
        graph_name=graph_name,
        plt_flag=True,
        xmin=3
    )

    return results


def save_results_to_file(results, output_path):
    """
    Save analysis results to a text file.

    Parameters:
    -----------
    results : dict
        Dictionary containing DataFrames with power law analysis results
    output_path : str
        Path to save the results file
    """
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("POWER LAW ANALYSIS RESULTS\n")
        f.write("="*60 + "\n\n")

        for degree_type, df in results.items():
            f.write(f"\n{degree_type.upper().replace('_', ' ')} DEGREE:\n")
            f.write("-"*60 + "\n")
            if df is not None and not df.empty:
                f.write(df.to_string())
                f.write("\n")
            else:
                f.write("No results available\n")

    print(f"\nResults saved to: {output_path}")


def main():
    """
    Main function to run the power law analysis on transaction data.
    """
    # Configuration
    data_dir = os.environ.get('AML_DATA_DIR', 'data')
    csv_path = os.path.join(data_dir, 'transactions.csv')
    output_dir = './results'

    # Load transaction data
    df = load_transaction_data(csv_path)

    # Build transaction graph
    G = build_transaction_graph(df)

    # Analyze degree distribution
    results = analyze_degree_distribution(G, output_dir=output_dir)

    # Save results to file
    results_file = os.path.join(output_dir, 'powerlaw_analysis_results.txt')
    save_results_to_file(results, results_file)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"  - Plots: in_degree_powerlaw.png, out_degree_powerlaw.png, total_degree_powerlaw.png")
    print(f"  - Summary: powerlaw_analysis_results.txt")


if __name__ == '__main__':
    main()
