"""GBT (Gradient Boosted Trees) monitor for money laundering detection.

Uses a pre-trained LightGBM binary classifier to evaluate whether
transactions in a cluster are likely to be flagged as money laundering.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from .base import MonitorModel


class GBTMonitor(MonitorModel):
    """LightGBM-based monitor for money laundering detection.

    Predicts the probability that each transaction in a cluster is
    money laundering using a pre-trained LightGBM model.
    The cluster-level score is the mean probability across all
    transactions whose endpoints are in cluster_nodes.

    Args:
        model_path: Path to a saved LightGBM model (.pkl).
                    If None, looks for gbt_model.pkl in the same
                    directory as this file.
        threshold: Classification threshold (unused for scoring,
                   kept for API compatibility).
        score_aggregation: How to aggregate per-transaction probs:
            - 'mean': mean probability (default)
            - 'max': maximum probability
            - 'top_k_mean': mean of the top-5 highest probs
        save_debug_output: Whether to save debug CSV files.
        debug_output_dir: Directory for debug output.

    Example:
        # Pre-train a model:
        GBTMonitor.train_and_save(
            accounts_path=os.path.join(os.environ.get('AML_DATA_DIR', 'data'), 'accounts.csv'),
            transactions_path=os.path.join(os.environ.get('AML_DATA_DIR', 'data'), 'transactions.csv'),
            model_path="models/gbt_model.pkl",
        )
        # Then use as monitor:
        monitor = GBTMonitor(model_path="models/gbt_model.pkl")
        score = monitor.predict_proba(txns_df, accts_df, cluster_nodes)
    """

    # Feature columns expected by the LightGBM model (order matters)
    FEATURE_COLS = [
        "log_amount_paid",
        "log_amount_received",
        "amount_ratio",
        "amount_fraction_sender",
        "is_cross_bank",
        "is_self_loop",
        "payment_format_enc",
        "hour_of_day",
        "day_of_week",
        "log_fan_out",
        "log_fan_in",
        "log_sender_txn_cnt",
        "log_recv_txn_cnt",
        "log_sender_total_out",
        "log_recv_total_in",
    ]

    # Payment format encoding (consistent between train and inference)
    PAYMENT_FORMATS = [
        "Cheque", "Credit Card", "ACH", "Wire", "Reinvestment", "Cash"
    ]

    def __init__(
        self,
        model_path: str = None,
        threshold: float = 0.5,
        score_aggregation: str = "mean",
        save_debug_output: bool = False,
        debug_output_dir: str = None,
    ):
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "gbt_model.pkl",
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"GBT model not found at: {model_path}\n"
                "Run GBTMonitor.train_and_save(...) first, or pass a valid model_path."
            )

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        self.model_path = model_path
        self.threshold = threshold
        self.score_aggregation = score_aggregation
        self.save_debug_output = save_debug_output
        self.debug_output_dir = debug_output_dir
        self.call_count = 0

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    @classmethod
    def extract_features(
        cls,
        df: pd.DataFrame,
        context_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Extract per-transaction features for LightGBM.

        Works on both Transaction-format data (for training) and on
        normalized data produced by aml_terraformer.core.normalize
        (for inference during GRPO).

        Args:
            df: DataFrame with transaction records to compute features for.
            context_df: Optional full-graph DataFrame used to compute
                        global aggregate statistics (fan-out, fan-in, etc.).
                        If None, uses df itself (i.e. local-only stats).
                        During training pass None (df IS the full data).
                        During GRPO inference pass the full transactions_df
                        so that global stats are realistic.

        Supported column name variants:
            - Timestamps: 'Timestamp', 'ts', 'timestamp'
            - From account: 'from_node_id', 'From Account', 'Account',
                            'from_account_number'
            - To account: 'to_node_id', 'To Account', 'Account.1',
                          'to_account_number'
            - From bank: 'From Bank', 'from_bank_id'
            - To bank: 'To Bank', 'to_bank_id'
            - Amount: 'Amount Paid', 'amount_paid', 'amount'
            - Payment format: 'Payment Format', 'payment_format'

        Returns:
            Copy of df with extra feature columns added.
        """
        df = df.copy()

        # Context for global aggregate stats
        ctx = context_df.copy() if context_df is not None else df.copy()

        # ---- Helper: column resolution ----
        def _get_col(d, *names, default=0.0):
            for n in names:
                if n in d.columns:
                    return pd.to_numeric(d[n], errors="coerce").fillna(default)
            return pd.Series(default, index=d.index, dtype=float)

        def _get_str_col(d, *names, default="unknown"):
            for n in names:
                if n in d.columns:
                    return d[n].astype(str).str.strip()
            return pd.Series(default, index=d.index, dtype=str)

        # ---- Parse timestamp ----
        for _d in [df, ctx]:
            if "ts" not in _d.columns:
                raw_ts = _get_str_col(_d, "Timestamp", "timestamp", default="")
                _d["ts"] = pd.to_datetime(raw_ts, errors="coerce")
            elif not pd.api.types.is_datetime64_any_dtype(_d["ts"]):
                _d["ts"] = pd.to_datetime(_d["ts"], errors="coerce")

        # ---- Amount columns ----
        df["amount_paid"] = _get_col(df, "Amount Paid", "amount_paid", "amount")
        df["amount_received"] = _get_col(df, "Amount Received", "amount_received", "amount")

        df["log_amount_paid"] = np.log1p(df["amount_paid"])
        df["log_amount_received"] = np.log1p(df["amount_received"])

        df["amount_ratio"] = (
            df["amount_received"] / (df["amount_paid"] + 1e-6)
        ).clip(0.0, 10.0)

        # ---- Bank / cross-bank flag ----
        from_bank = _get_str_col(df, "From Bank", "from_bank_id")
        to_bank = _get_str_col(df, "To Bank", "to_bank_id")
        df["is_cross_bank"] = (from_bank != to_bank).astype(int)

        # ---- Account identifiers ----
        # Transaction raw: "Account" (from) and "Account.1" (to)
        # Normalized:   "from_node_id" / "to_node_id"
        # Also handle: "From Account" / "To Account"
        def _resolve_acct(d, *from_names, from_bank_col=None, acct_col=None):
            """Return unique account identifier series."""
            for n in from_names:
                if n in d.columns:
                    return d[n].astype(str).str.strip()
            # Transaction fallback: build node_id from bank + account
            if from_bank_col and acct_col and from_bank_col in d.columns and acct_col in d.columns:
                return (
                    d[from_bank_col].astype(str).str.strip()
                    + "|"
                    + d[acct_col].astype(str).str.strip()
                )
            return pd.Series("unknown", index=d.index, dtype=str)

        for _d in [df, ctx]:
            _d["_from_acct"] = _resolve_acct(
                _d,
                "from_node_id", "From Account", "from_account_number",
                from_bank_col="From Bank", acct_col="Account",
            )
            _d["_to_acct"] = _resolve_acct(
                _d,
                "to_node_id", "To Account", "to_account_number",
                from_bank_col="To Bank", acct_col="Account.1",
            )

        # Self-loop flag (computed on df)
        df["is_self_loop"] = (df["_from_acct"] == df["_to_acct"]).astype(int)

        # ---- Payment format encoding ----
        # Full format list: union of Transaction and TransXion formats
        fmt_raw = _get_str_col(df, "Payment Format", "payment_format")
        df["payment_format_enc"] = pd.Categorical(
            fmt_raw, categories=cls.PAYMENT_FORMATS
        ).codes
        # Map unknown categories (-1) to a dedicated "other" bucket = last index + 1
        df["payment_format_enc"] = df["payment_format_enc"].clip(0, len(cls.PAYMENT_FORMATS) - 1)

        # ---- Temporal features ----
        if pd.api.types.is_datetime64_any_dtype(df["ts"]):
            df["hour_of_day"] = df["ts"].dt.hour.fillna(12).astype(int)
            df["day_of_week"] = df["ts"].dt.dayofweek.fillna(0).astype(int)
        else:
            df["hour_of_day"] = 12
            df["day_of_week"] = 0

        # ---- Global aggregate features (computed on ctx, applied to df) ----

        # Fan-out: unique recipients per sender
        fan_out = ctx.groupby("_from_acct")["_to_acct"].nunique().rename("_fan_out")
        df = df.merge(fan_out, on="_from_acct", how="left")
        df["log_fan_out"] = np.log1p(df["_fan_out"].fillna(1))

        # Fan-in: unique senders per receiver
        fan_in = ctx.groupby("_to_acct")["_from_acct"].nunique().rename("_fan_in")
        df = df.merge(fan_in, on="_to_acct", how="left")
        df["log_fan_in"] = np.log1p(df["_fan_in"].fillna(1))

        # Sender stats
        ctx["_ctx_amount_paid"] = _get_col(ctx, "Amount Paid", "amount_paid", "amount")
        ctx["_ctx_amount_received"] = _get_col(ctx, "Amount Received", "amount_received", "amount")

        sender_stats = ctx.groupby("_from_acct").agg(
            _sender_txn_cnt=("_ctx_amount_paid", "count"),
            _sender_total_out=("_ctx_amount_paid", "sum"),
        )
        df = df.merge(sender_stats, on="_from_acct", how="left")
        df["log_sender_txn_cnt"] = np.log1p(df["_sender_txn_cnt"].fillna(1))
        df["log_sender_total_out"] = np.log1p(df["_sender_total_out"].fillna(0))

        # Receiver stats
        recv_stats = ctx.groupby("_to_acct").agg(
            _recv_txn_cnt=("_ctx_amount_received", "count"),
            _recv_total_in=("_ctx_amount_received", "sum"),
        )
        df = df.merge(recv_stats, on="_to_acct", how="left")
        df["log_recv_txn_cnt"] = np.log1p(df["_recv_txn_cnt"].fillna(1))
        df["log_recv_total_in"] = np.log1p(df["_recv_total_in"].fillna(0))

        # Amount as fraction of sender's total outflow (local signal)
        df["amount_fraction_sender"] = (
            df["amount_paid"] / (df["_sender_total_out"].fillna(0) + 1e-6)
        ).clip(0.0, 1.0)

        return df

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------


    @classmethod
    def train_and_save(
        cls,
        accounts_path: str,
        transactions_path: str,
        model_path: str,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        max_depth: int = 8,
        min_child_samples: int = 20,
        scale_pos_weight: Optional[float] = None,
        val_fraction: float = 0.1,
        random_state: int = 42,
        verbose_eval: int = 50,
    ) -> "GBTMonitor":
        """Train a LightGBM binary classifier on Transaction-format data and save it.

        Args:
            accounts_path: Path to accounts CSV (not used for features, but
                           kept for API symmetry with other monitors).
            transactions_path: Path to transactions CSV with 'Is Laundering'.
            model_path: Output path for the saved model (.pkl).
            n_estimators: Maximum number of boosting rounds.
            learning_rate: LightGBM learning rate.
            num_leaves: Max number of leaves in one tree.
            max_depth: Max tree depth (-1 = no limit).
            min_child_samples: Min data in one leaf.
            scale_pos_weight: Class imbalance weight for positive class.
                              If None, computed automatically.
            val_fraction: Fraction of data used for validation.
            random_state: Random seed.
            verbose_eval: Print evaluation every N rounds.

        Returns:
            Trained GBTMonitor instance ready for use.
        """
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split

        print(f"[GBTMonitor.train] Loading data from:\n  {transactions_path}")
        txn_df = pd.read_csv(transactions_path)
        n_total = len(txn_df)
        n_pos = int(txn_df["Is Laundering"].sum())
        n_neg = n_total - n_pos
        print(f"[GBTMonitor.train] {n_total} transactions — "
              f"{n_pos} laundering ({100*n_pos/n_total:.3f}%), "
              f"{n_neg} normal")

        # Feature extraction
        print("[GBTMonitor.train] Extracting features ...")
        feat_df = cls.extract_features(txn_df)

        X = feat_df[cls.FEATURE_COLS].values.astype(np.float32)
        y = txn_df["Is Laundering"].values.astype(int)

        # Drop NaN rows
        valid_mask = ~np.isnan(X).any(axis=1)
        X, y = X[valid_mask], y[valid_mask]
        print(f"[GBTMonitor.train] {valid_mask.sum()} valid rows after NaN filter.")

        # Auto scale_pos_weight
        if scale_pos_weight is None:
            scale_pos_weight = max(1.0, n_neg / max(n_pos, 1))
            print(f"[GBTMonitor.train] scale_pos_weight = {scale_pos_weight:.1f}")

        # Train / validation split (stratified)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=val_fraction,
            random_state=random_state,
            stratify=y,
        )
        print(f"[GBTMonitor.train] Train: {len(y_train)}, Val: {len(y_val)}")

        train_ds = lgb.Dataset(X_train, label=y_train)
        val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

        params = {
            "objective": "binary",
            "metric": ["binary_logloss", "average_precision"],
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_child_samples": min_child_samples,
            "scale_pos_weight": scale_pos_weight,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "seed": random_state,
            "verbosity": -1,
            "n_jobs": -1,
        }

        print("[GBTMonitor.train] Training LightGBM ...")
        callbacks = [
            lgb.early_stopping(stopping_rounds=30, verbose=True),
            lgb.log_evaluation(period=verbose_eval),
        ]
        model = lgb.train(
            params,
            train_ds,
            num_boost_round=n_estimators,
            valid_sets=[val_ds],
            callbacks=callbacks,
        )

        # Evaluate on validation set
        val_probs = model.predict(X_val)
        from sklearn.metrics import average_precision_score, roc_auc_score
        ap = average_precision_score(y_val, val_probs)
        auc = roc_auc_score(y_val, val_probs)
        print(f"[GBTMonitor.train] Val AP={ap:.4f}, AUC={auc:.4f}")

        # Save
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"[GBTMonitor.train] Model saved to: {model_path}")

        return cls(model_path=model_path)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        transactions_df: pd.DataFrame,
        accounts_df: pd.DataFrame,
        cluster_nodes: List[str],
    ) -> float:
        """Predict money laundering probability for a transaction cluster.

        Filters to transactions with at least one endpoint in cluster_nodes,
        extracts features, runs LightGBM inference, and aggregates into a
        single cluster-level score.

        Args:
            transactions_df: Full normalized transactions DataFrame.
            accounts_df: Full accounts DataFrame (unused, kept for interface).
            cluster_nodes: Node IDs (from_node_id / to_node_id) of the cluster.

        Returns:
            float: Detection probability in [0, 1].
                   Higher = cluster is more detectable.
                   Lower = cluster successfully evades detection.
        """
        self.call_count += 1

        # Filter to cluster transactions (any endpoint in cluster)
        # Support both normalized (from_node_id) and raw (From Account) column names
        from_col = "from_node_id" if "from_node_id" in transactions_df.columns else "From Account"
        to_col = "to_node_id" if "to_node_id" in transactions_df.columns else "To Account"

        cluster_set = set(cluster_nodes)
        cluster_mask = (
            transactions_df[from_col].isin(cluster_set) |
            transactions_df[to_col].isin(cluster_set)
        )
        cluster_txns = transactions_df[cluster_mask].copy()

        if len(cluster_txns) == 0:
            return 0.0

        # Extract features — use full transactions_df as context for realistic global stats
        try:
            feat_df = self.extract_features(cluster_txns, context_df=transactions_df)
        except Exception as e:
            print(f"[GBTMonitor] Feature extraction failed: {e}")
            return 0.1

        # If 'Is Laundering' labels are available, score only the laundering transactions.
        # This mirrors the rule-based monitor's _compute_score() behaviour and ensures
        # the reward signal reflects detection of known laundering activity, not all
        # cluster-adjacent transactions.
        if "Is Laundering" in feat_df.columns:
            scoring_mask = feat_df["Is Laundering"].astype(int) == 1
            scoring_feat = feat_df[scoring_mask]
        else:
            scoring_feat = feat_df  # no labels → score all cluster transactions

        if len(scoring_feat) == 0:
            return 0.0

        # Build feature matrix
        X = scoring_feat[self.FEATURE_COLS].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

        # Inference
        try:
            probs = self.model.predict(X)  # shape: (n_txns,)
        except Exception as e:
            print(f"[GBTMonitor] Inference failed: {e}")
            return 0.1

        # Aggregate
        if self.score_aggregation == "max":
            score = float(np.max(probs))
        elif self.score_aggregation == "top_k_mean":
            k = min(5, len(probs))
            score = float(np.mean(np.partition(probs, -k)[-k:]))
        else:  # "mean"
            score = float(np.mean(probs))

        # Optional debug output
        if self.save_debug_output and self.debug_output_dir:
            debug_dir = os.path.join(self.debug_output_dir, "gbt_debug")
            os.makedirs(debug_dir, exist_ok=True)
            out_path = os.path.join(debug_dir, f"gbt_call_{self.call_count:05d}.csv")
            out_df = scoring_feat.copy()
            out_df["gbt_prob"] = probs
            out_df.to_csv(out_path, index=False)

        return float(np.clip(score, 0.0, 1.0))
