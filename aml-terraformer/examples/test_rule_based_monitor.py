"""Example script demonstrating RuleBasedMonitor usage."""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from aml_terraformer.monitor import RuleBasedMonitor


def create_sample_data():
    """Create sample transaction and account data for testing."""
    # Sample accounts
    accounts_data = {
        'node_id': ['n1', 'n2', 'n3', 'n4', 'n5'],
        'Bank ID': ['BankA', 'BankA', 'BankB', 'BankB', 'BankC'],
        'Account Number': ['ACC001', 'ACC002', 'ACC003', 'ACC004', 'ACC005']
    }
    accounts_df = pd.DataFrame(accounts_data)

    # Sample transactions (suspicious pattern: quick in-out)
    transactions_data = {
        'Timestamp': [
            '2024/01/01 10:00',
            '2024/01/01 11:00',
            '2024/01/01 12:00',
            '2024/01/02 10:00',
            '2024/01/02 11:00',
            '2024/01/03 10:00',
        ],
        'from_node_id': ['n1', 'n1', 'n1', 'n2', 'n2', 'n3'],
        'to_node_id': ['n2', 'n2', 'n2', 'n3', 'n3', 'n4'],
        'Amount Paid': [100000, 150000, 200000, 250000, 300000, 400000],
        'Amount Received': [100000, 150000, 200000, 250000, 300000, 400000],
        'Receiving Currency': ['USD'] * 6,
        'Payment Currency': ['USD'] * 6,
        'Payment Format': ['Wire'] * 6
    }
    transactions_df = pd.DataFrame(transactions_data)

    return transactions_df, accounts_df


def main():
    """Main function to test RuleBasedMonitor."""
    print("=" * 60)
    print("RuleBasedMonitor Test Script")
    print("=" * 60)

    # Create sample data
    print("\n[1] Creating sample data...")
    transactions_df, accounts_df = create_sample_data()

    print(f"   - Transactions: {len(transactions_df)} rows")
    print(f"   - Accounts: {len(accounts_df)} rows")

    # Initialize monitor
    print("\n[2] Initializing RuleBasedMonitor...")
    rule_config_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        'rule',
        'data',
        'rule.json'
    )

    try:
        monitor = RuleBasedMonitor(
            rule_config_path=rule_config_path,
            score_aggregation='weighted_average'
        )
        print(f"   - Loaded {len(monitor.rules)} rules")
        print(f"   - Rules: {', '.join([r['rule_id'] for r in monitor.rules])}")
    except Exception as e:
        print(f"   ERROR: Failed to initialize monitor: {e}")
        return

    # Test prediction on a cluster
    print("\n[3] Testing prediction on cluster...")
    cluster_nodes = ['n1', 'n2', 'n3']  # Test cluster

    try:
        prob = monitor.predict_proba(
            transactions_df,
            accounts_df,
            cluster_nodes
        )
        print(f"   - Cluster nodes: {cluster_nodes}")
        print(f"   - Detection probability: {prob:.4f}")

        if prob > 0.7:
            print("   - Result: HIGH RISK - Strong suspicious pattern detected")
        elif prob > 0.4:
            print("   - Result: MEDIUM RISK - Moderate suspicious pattern detected")
        elif prob > 0.1:
            print("   - Result: LOW RISK - Weak suspicious pattern detected")
        else:
            print("   - Result: CLEAN - No suspicious pattern detected")

    except Exception as e:
        print(f"   ERROR: Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test with different score aggregation methods
    print("\n[4] Testing different aggregation methods...")
    aggregation_methods = ['weighted_average', 'max', 'count']

    for method in aggregation_methods:
        try:
            monitor = RuleBasedMonitor(
                rule_config_path=rule_config_path,
                score_aggregation=method
            )
            prob = monitor.predict_proba(
                transactions_df,
                accounts_df,
                cluster_nodes
            )
            print(f"   - {method:20s}: {prob:.4f}")
        except Exception as e:
            print(f"   - {method:20s}: ERROR - {e}")

    # Test with custom rule weights
    print("\n[5] Testing with custom rule weights...")
    custom_weights = {
        'S1': 2.0,  # New account outflows - higher weight
        'S3': 1.5,  # Quick in-out - higher weight
        'S5': 1.0,
        'S6': 1.5,  # Fast in-fast out - higher weight
        'S7': 1.0,
        'S8': 2.0   # Fund aggregation - higher weight
    }

    try:
        monitor = RuleBasedMonitor(
            rule_config_path=rule_config_path,
            rule_weights=custom_weights,
            score_aggregation='weighted_average'
        )
        prob = monitor.predict_proba(
            transactions_df,
            accounts_df,
            cluster_nodes
        )
        print(f"   - With custom weights: {prob:.4f}")
    except Exception as e:
        print(f"   ERROR: {e}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
