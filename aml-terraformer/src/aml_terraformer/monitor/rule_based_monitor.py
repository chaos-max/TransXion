"""Rule-based monitor for money laundering detection."""

import json
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .base import MonitorModel


class RuleBasedMonitor(MonitorModel):
    """Rule-based monitor using predefined AML detection rules.

    This monitor implements a rule-based approach to detect money laundering
    by evaluating transactions against a set of predefined rules.

    Currently active rules:
    - S1: New accounts with immediate large outflows
    - S3: Short-term inflows followed by quick outflows
    - S6: Fast in-fast out (money passing through)

    Args:
        rule_config_path: Path to rule.json configuration file
        rule_weights: Optional dict mapping rule_id to weight (default: all 1.0)
        score_aggregation: How to aggregate rule hits into probability
            - 'weighted_average': Weighted average of hit rules
            - 'max': Maximum weight of hit rules
            - 'count': Normalized count of hit rules

    Example:
        >>> monitor = RuleBasedMonitor(
        ...     rule_config_path="/path/to/rule.json"
        ... )
        >>> prob = monitor.predict_proba(txns, accounts, cluster_nodes)
    """

    def __init__(
        self,
        rule_config_path: str = None,
        rule_weights: Dict[str, float] = None,
        score_aggregation: str = 'weighted_average',
        save_debug_output: bool = False,
        debug_output_dir: str = None
    ):
        """Initialize rule-based monitor."""
        if rule_config_path is None:
            # Default to the rule.json in the rule directory
            rule_config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "rule", "data", "rule.json"
            )

        # Load rule configuration
        with open(rule_config_path, "r", encoding="utf-8") as f:
            self.rule_cfg = json.load(f)

        self.rules = self.rule_cfg["rules"]

        # Set rule weights (default: all rules weighted equally)
        if rule_weights is None:
            self.rule_weights = {r["rule_id"]: 1.0 for r in self.rules}
        else:
            self.rule_weights = rule_weights

        self.score_aggregation = score_aggregation
        self.save_debug_output = save_debug_output
        self.debug_output_dir = debug_output_dir
        self.call_count = 0  # Track number of calls for debug file naming

        # Field mapping: semantic field -> physical field (only for S1, S3, S6)
        self.field_alias = {
            # S1 fields
            "days_since_open": {
                "OUT": "days_since_open_from",
                "IN": "days_since_open_to"
            },
            # S3 fields
            "in_cnt_10d": {"IN": "in_cnt_10d_to"},
            "out_after_last_in_days": {"IN": "out_after_last_in_days_to"},
            # S6 fields
            "in_out_ratio": {
                "OUT": "in_out_ratio_3d",
                "IN": "in_out_ratio_3d"
            },
            "end_balance": {
                "OUT": "end_balance_3d_est",
                "IN": "end_balance_3d_est"
            },
            "max_in_or_out_amt_3d": {
                "OUT": "__MAX_IN_OUT_3D__",
                "IN": "__MAX_IN_OUT_3D__"
            }
        }

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build rule-based features from transaction data.

        Only computes features needed for active rules: S1, S3, S6

        Args:
            df: DataFrame with standardized transaction columns

        Returns:
            DataFrame with added rule features
        """
        # Sort by timestamp
        df = df.sort_values("ts").reset_index(drop=True)
        df["txn_date"] = df["ts"].dt.date
        df["out_amount"] = df["Amount Paid"].fillna(0.0)
        df["in_amount"] = df["Amount Received"].fillna(0.0)

        # Compute account open time (first transaction)
        open_time = pd.concat([
            df[["From Account", "ts"]].rename(columns={"From Account": "account"}),
            df[["To Account", "ts"]].rename(columns={"To Account": "account"})
        ]).groupby("account")["ts"].min()

        df["open_time_from"] = df["From Account"].map(open_time)
        df["open_time_to"] = df["To Account"].map(open_time)

        df["days_since_open_from"] = (df["ts"] - df["open_time_from"]).dt.days
        df["days_since_open_to"] = (df["ts"] - df["open_time_to"]).dt.days

        # S1: New account outflows
        df["is_new_account_from"] = df["days_since_open_from"] == 0

        daily_new_cnt = (
            df[df["is_new_account_from"]]
            .groupby("txn_date")["From Account"]
            .nunique()
        )
        daily_out_sum = (
            df[df["is_new_account_from"]]
            .groupby("txn_date")["out_amount"]
            .sum()
        )

        df["daily_new_account_cnt"] = df["txn_date"].map(daily_new_cnt).fillna(0).astype(int)
        df["daily_out_amount_sum"] = df["txn_date"].map(daily_out_sum).fillna(0.0)

        # S3: Features per To Account
        def calc_s3(sub):
            sub = sub.sort_values("ts").copy()
            sub = sub.set_index("ts")

            # S3: 10-day inflow count
            sub["in_cnt_10d_to"] = sub["in_amount"].rolling("10D").count()

            sub = sub.reset_index()
            sub["last_in_ts_to"] = sub["ts"]
            sub["out_after_last_in_days_to"] = 0

            return sub

        # Group by To Account and apply
        grouped = df.groupby("To Account", group_keys=False, as_index=False)
        df = pd.concat([calc_s3(group) for _, group in grouped], ignore_index=False)

        # S6: Features per From Account
        def calc_s6(sub):
            sub = sub.sort_values("ts").copy()
            sub = sub.set_index("ts")

            # S6: 3-day windows
            sub["amt_in_3d"] = sub["in_amount"].rolling("3D").sum()
            sub["amt_out_3d"] = sub["out_amount"].rolling("3D").sum()
            sub["in_out_ratio_3d"] = sub["amt_in_3d"] / (sub["amt_out_3d"] + 1e-6)
            sub["end_balance_3d_est"] = sub["amt_in_3d"] - sub["amt_out_3d"]

            sub = sub.reset_index()
            return sub

        # Group by From Account and apply
        grouped = df.groupby("From Account", group_keys=False, as_index=False)
        df = pd.concat([calc_s6(group) for _, group in grouped], ignore_index=False)

        return df.sort_values("ts").reset_index(drop=True)

    def _check_condition(self, series: pd.Series, cond: Dict[str, Any]) -> pd.Series:
        """Check a condition against a series.

        Args:
            series: Data series to check
            cond: Condition dict with 'op' and 'value'

        Returns:
            Boolean series indicating which rows satisfy the condition
        """
        op = cond["op"]
        val = cond["value"]

        if op == ">=":
            return series >= val
        if op == "<=":
            return series <= val
        if op == ">":
            return series > val
        if op == "<":
            return series < val
        if op == "==":
            return series == val
        if op == "between":
            return (series >= val[0]) & (series <= val[1])

        raise ValueError(f"Unsupported op: {op}")

    def _resolve_series(
        self,
        df: pd.DataFrame,
        field: str,
        direction: str,
        rule_id: str
    ) -> pd.Series:
        """Resolve semantic field name to physical series.

        Args:
            df: Transaction DataFrame
            field: Semantic field name
            direction: Rule direction ('IN' or 'OUT')
            rule_id: Rule ID for error messages

        Returns:
            Resolved data series
        """
        real_field = field

        if field in self.field_alias:
            real_field = self.field_alias[field].get(direction)

        # Handle derived fields
        if real_field == "__CNT_15D_GT__":
            return df["cnt_last_15d_from"] > df["cnt_first_15d_from"]

        if real_field == "__AMT_15D_GT__":
            return df["amt_last_15d_from"] > df["amt_first_15d_from"]

        if real_field == "__MAX_IN_OUT_3D__":
            return np.maximum(df["amt_in_3d"], df["amt_out_3d"])

        if real_field not in df.columns:
            raise KeyError(f"[Rule {rule_id}] Missing feature column: {real_field}")

        return df[real_field]

    def _run_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all rules on the feature DataFrame.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with hit_<rule_id> columns added
        """
        # Initialize hit columns
        for rule in self.rules:
            df[f"hit_{rule['rule_id']}"] = 0

        # Execute each rule
        for rule in self.rules:
            rid = rule["rule_id"]
            direction = rule.get("direction", "OUT")
            mask = pd.Series(True, index=df.index, dtype=bool)

            # Apply preconditions
            if "precondition" in rule:
                for field, cond in rule["precondition"].items():
                    mask &= self._check_condition(
                        self._resolve_series(df, field, direction, rid),
                        cond
                    )

            # Apply main conditions
            for field, cond in rule["conditions"].items():
                series = self._resolve_series(df, field, direction, rid)
                if isinstance(cond, dict):
                    mask &= self._check_condition(series, cond)
                elif cond is True:
                    mask &= series.astype(bool)
                elif cond is False:
                    mask &= ~series.astype(bool)

            df.loc[mask, f"hit_{rid}"] = 1

        return df

    def predict_proba(
        self,
        transactions_df: pd.DataFrame,
        accounts_df: pd.DataFrame,
        cluster_nodes: List[str],
        debug_label: str = None,
        trajectory_id: int = None,
        step_idx: int = None
    ) -> float:
        """Predict probability of money laundering using rule-based detection.

        Args:
            transactions_df: Full transactions DataFrame
            accounts_df: Full accounts DataFrame
            cluster_nodes: List of node IDs in the cluster to evaluate
            debug_label: Optional label for debug output (e.g., "before", "after")
            trajectory_id: Optional trajectory ID for grouping debug output
            step_idx: Optional step index within trajectory

        Returns:
            float: Probability score between 0 and 1
        """
        # Only increment counter for "before" or no label to keep before/after/summary aligned
        if debug_label != "after":
            self.call_count += 1

        # Filter transactions: include all transactions involving cluster nodes
        # This includes:
        # 1. Transactions within the cluster (both endpoints in cluster)
        # 2. Transactions with one endpoint in cluster (one-hop expansion)
        # No longer filtering by "Is Laundering" to capture full transaction context
        cluster_mask = (
            transactions_df["from_node_id"].isin(cluster_nodes) |
            transactions_df["to_node_id"].isin(cluster_nodes)
        )
        cluster_txns = transactions_df[cluster_mask].copy()

        if len(cluster_txns) == 0:
            return 0.0

        # Extract bank and account from node_id (format: "bank_id|account_number")
        # Use the existing columns from transactions_df
        df = pd.DataFrame({
            "Timestamp": cluster_txns["Timestamp"],
            "From Bank": cluster_txns.get("from_bank_id", cluster_txns.get("From Bank", "")),
            "From Account": cluster_txns.get("from_account_number", cluster_txns.get("From Account", "")),
            "To Bank": cluster_txns.get("to_bank_id", cluster_txns.get("To Bank", "")),
            "To Account": cluster_txns.get("to_account_number", cluster_txns.get("To Account", "")),
            "Amount Received": cluster_txns.get("Amount Received", cluster_txns.get("amount", 0)),
            "Amount Paid": cluster_txns.get("Amount Paid", cluster_txns.get("amount", 0)),
            "Receiving Currency": cluster_txns.get("Receiving Currency", "USD"),
            "Payment Currency": cluster_txns.get("Payment Currency", "USD"),
            "Payment Format": cluster_txns.get("Payment Format", ""),
        })

        # Preserve "Is Laundering" field if it exists in the source data
        if "Is Laundering" in cluster_txns.columns:
            df["Is Laundering"] = cluster_txns["Is Laundering"].values

        # Parse timestamp
        df["ts"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["ts"])

        if len(df) == 0:
            return 0.0

        # Build features
        try:
            df = self._build_features(df)
        except Exception as e:
            # If feature building fails, return low probability
            print(f"Warning: Feature building failed: {e}")
            return 0.1

        # Run rules
        try:
            df = self._run_rules(df)
        except Exception as e:
            # If rule execution fails, return low probability
            print(f"Warning: Rule execution failed: {e}")
            return 0.1

        # Save debug output if enabled
        if self.save_debug_output and self.debug_output_dir:
            debug_dir = os.path.join(self.debug_output_dir, "rule_debug")
            os.makedirs(debug_dir, exist_ok=True)

            # Create filename with label
            if debug_label:
                filename = f"txn_with_rule_hits_{debug_label}_{self.call_count}.csv"
            else:
                filename = f"txn_with_rule_hits_{self.call_count}.csv"

            output_path = os.path.join(debug_dir, filename)
            df.to_csv(output_path, index=False)

            # Also save a summary
            hit_cols = [f"hit_{r['rule_id']}" for r in self.rules]
            hit_counts = df[hit_cols].sum()

            # Merge before/after summaries into one file per trajectory
            if debug_label in ["before", "after"] and trajectory_id is not None:
                summary_file = os.path.join(debug_dir, f"summary_traj_{trajectory_id}.txt")

                # Logic: Only save BEFORE for step 0, always save AFTER
                should_write = False
                if debug_label == "before" and step_idx == 0:
                    # First step: save BEFORE
                    mode = 'w'
                    should_write = True
                elif debug_label == "after":
                    # Always save AFTER (append mode)
                    mode = 'a'
                    should_write = True

                if should_write:
                    # Compute score and get laundering hit counts
                    score, laundering_hit_counts = self._compute_score(df)

                    with open(summary_file, mode) as f:
                        if debug_label == "before":
                            f.write(f"=== Step {step_idx + 1} BEFORE ===\n")
                        else:
                            f.write(f"\n=== Step {step_idx + 1} AFTER ===\n")

                        f.write(f"Total transactions: {len(df)}\n")

                        # Count laundering transactions
                        num_laundering = df["Is Laundering"].sum() if "Is Laundering" in df.columns else 0
                        f.write(f"Laundering transactions: {num_laundering}\n")
                        f.write(f"Cluster nodes: {len(cluster_nodes)}\n\n")

                        f.write(f"Rule hits (all transactions):\n")
                        for rid in [r['rule_id'] for r in self.rules]:
                            f.write(f"  {rid}: {hit_counts[f'hit_{rid}']}\n")

                        f.write(f"\nRule hits (laundering only):\n")
                        for rid in [r['rule_id'] for r in self.rules]:
                            f.write(f"  {rid}: {laundering_hit_counts.get(rid, 0)}\n")

                        f.write(f"\nScore: {score:.4f}\n")
            else:
                # Fallback: old behavior for backward compatibility
                summary_file = os.path.join(debug_dir, f"summary_{self.call_count}.txt")
                score, laundering_hit_counts = self._compute_score(df)

                with open(summary_file, 'w') as f:
                    f.write(f"=== Rule Hits Summary ===\n")
                    f.write(f"Total transactions: {len(df)}\n")

                    # Count laundering transactions
                    num_laundering = df["Is Laundering"].sum() if "Is Laundering" in df.columns else 0
                    f.write(f"Laundering transactions: {num_laundering}\n")
                    f.write(f"Cluster nodes: {len(cluster_nodes)}\n\n")

                    f.write(f"Rule hits (all transactions):\n")
                    for rid in [r['rule_id'] for r in self.rules]:
                        f.write(f"  {rid}: {hit_counts[f'hit_{rid}']}\n")

                    f.write(f"\nRule hits (laundering only):\n")
                    for rid in [r['rule_id'] for r in self.rules]:
                        f.write(f"  {rid}: {laundering_hit_counts.get(rid, 0)}\n")

                    f.write(f"\nFinal score: {score:.4f}\n")

        # Aggregate rule hits into probability score
        score, _ = self._compute_score(df)

        return float(np.clip(score, 0.0, 1.0))

    def _compute_score(self, df: pd.DataFrame) -> tuple:
        """Compute aggregated score from rule hits.

        Args:
            df: DataFrame with hit_<rule_id> columns

        Returns:
            Tuple of (score, laundering_hit_counts_dict):
                - score: Normalized average hit rate: mean(rule_hits) / num_laundering_txns
                - laundering_hit_counts_dict: Dict mapping rule_id to hit count for laundering txns only
        """
        # Filter to only laundering transactions for score computation
        if "Is Laundering" in df.columns:
            laundering_df = df[df["Is Laundering"] == 1].copy()
        else:
            # If no "Is Laundering" column, use all transactions
            laundering_df = df.copy()

        if len(laundering_df) == 0:
            return 0.0, {}

        hit_cols = [f"hit_{r['rule_id']}" for r in self.rules]
        hit_counts = laundering_df[hit_cols].sum()

        # Get the average hits across all rules
        mean_hits = hit_counts.mean()
        num_laundering = len(laundering_df)

        # Normalize by number of laundering transactions
        score = mean_hits / num_laundering if num_laundering > 0 else 0.0

        # Build dict for laundering hit counts
        laundering_hit_counts_dict = {r['rule_id']: int(hit_counts[f"hit_{r['rule_id']}"]) for r in self.rules}

        return score, laundering_hit_counts_dict
