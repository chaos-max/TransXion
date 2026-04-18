"""Script to pre-train the GBT monitor on transaction data.

Usage:
    # Train on a dataset:
    export AML_DATA_DIR=/path/to/your/data
    python scripts/train_gbt_monitor.py \
        --accounts $AML_DATA_DIR/accounts.csv \
        --transactions $AML_DATA_DIR/transactions.csv \
        --output models/gbt_model.pkl

    # Train on multiple datasets combined:
    python scripts/train_gbt_monitor.py \
        --accounts $AML_DATA_DIR/accounts_1.csv $AML_DATA_DIR/accounts_2.csv \
        --transactions $AML_DATA_DIR/transactions_1.csv $AML_DATA_DIR/transactions_2.csv \
        --output models/gbt_model.pkl

    # Quick smoke-test with --dry-run (no training):
    python scripts/train_gbt_monitor.py \
        --accounts $AML_DATA_DIR/accounts.csv \
        --transactions $AML_DATA_DIR/transactions.csv \
        --output models/gbt_model.pkl \
        --dry-run
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train GBTMonitor on transaction data"
    )
    parser.add_argument(
        "--accounts",
        nargs="+",
        required=True,
        help="Path(s) to accounts CSV(s). Accepts multiple files.",
    )
    parser.add_argument(
        "--transactions",
        nargs="+",
        required=True,
        help="Path(s) to transactions CSV(s) with 'Is Laundering' column.",
    )
    parser.add_argument(
        "--output",
        default="models/gbt_model.pkl",
        help="Output path for the trained model (default: models/gbt_model.pkl)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Max LightGBM boosting rounds (default: 300, early stopping applies)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="LightGBM learning rate (default: 0.05)",
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=63,
        help="Max leaves per tree (default: 63)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="Max tree depth (-1 = no limit, default: 8)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Validation fraction (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and show feature stats only, skip training",
    )
    args = parser.parse_args()

    import pandas as pd
    from aml_terraformer.monitor import GBTMonitor

    # Combine multiple transaction files if provided
    txn_dfs = []
    for p in args.transactions:
        print(f"Loading: {p}")
        txn_dfs.append(pd.read_csv(p))
    txn_df = pd.concat(txn_dfs, ignore_index=True)
    print(f"Total transactions: {len(txn_df)}, laundering: {int(txn_df['Is Laundering'].sum())}")

    if args.dry_run:
        print("\n[dry-run] Extracting features ...")
        feat_df = GBTMonitor.extract_features(txn_df)
        print(f"[dry-run] Feature matrix shape: {feat_df[GBTMonitor.FEATURE_COLS].shape}")
        print(f"[dry-run] Feature stats:\n{feat_df[GBTMonitor.FEATURE_COLS].describe().T}")
        print("[dry-run] OK — skipping training.")
        return

    # Use first accounts file for the call (unused by GBT, but required for API)
    GBTMonitor.train_and_save(
        accounts_path=args.accounts[0],
        transactions_path=args.transactions[0],  # Ignored; we pass the combined df below
        model_path=args.output,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        val_fraction=args.val_fraction,
        random_state=args.seed,
    ) if len(args.transactions) == 1 else _train_combined(
        txn_df, args.output, args
    )


def _train_combined(txn_df, model_path, args):
    """Train on a pre-loaded combined DataFrame (multi-file case)."""
    import os, pickle
    import numpy as np
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import average_precision_score, roc_auc_score
    from aml_terraformer.monitor import GBTMonitor

    print(f"[GBTMonitor.train] Extracting features from combined data ...")
    feat_df = GBTMonitor.extract_features(txn_df)

    X = feat_df[GBTMonitor.FEATURE_COLS].values.astype(np.float32)
    y = txn_df["Is Laundering"].values.astype(int)

    valid_mask = ~np.isnan(X).any(axis=1)
    X, y = X[valid_mask], y[valid_mask]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos_weight = max(1.0, n_neg / max(n_pos, 1))
    print(f"[GBTMonitor.train] Valid rows: {len(y)}, scale_pos_weight={scale_pos_weight:.1f}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_fraction, random_state=args.seed, stratify=y
    )

    train_ds = lgb.Dataset(X_train, label=y_train)
    val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "average_precision"],
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "min_child_samples": 20,
        "scale_pos_weight": scale_pos_weight,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": args.seed,
        "verbosity": -1,
        "n_jobs": -1,
    }

    print("[GBTMonitor.train] Training LightGBM ...")
    model = lgb.train(
        params,
        train_ds,
        num_boost_round=args.n_estimators,
        valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(30, verbose=True), lgb.log_evaluation(50)],
    )

    val_probs = model.predict(X_val)
    ap = average_precision_score(y_val, val_probs)
    auc = roc_auc_score(y_val, val_probs)
    print(f"[GBTMonitor.train] Val AP={ap:.4f}, AUC={auc:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"[GBTMonitor.train] Model saved to: {model_path}")


if __name__ == "__main__":
    main()
