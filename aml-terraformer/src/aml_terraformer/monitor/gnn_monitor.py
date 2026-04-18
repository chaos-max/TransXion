"""GNN-based monitor for money laundering detection."""

import os
import sys
import json
import subprocess
import tempfile
import hashlib
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from .base import MonitorModel


# Default GNN project path from environment variable
# Must be set via MULTIGNN_PATH env var or passed explicitly
DEFAULT_GNN_PROJECT_PATH = os.environ.get("MULTIGNN_PATH", "")


class GNNMonitor(MonitorModel):
    """GNN-based monitor using Multi-GNN model for detection.

    This monitor integrates the Multi-GNN model to detect money laundering
    by running inference on transaction data and returning aggregate metrics.

    The GNN model is called via subprocess execution:
    1. Transaction data is saved to the data directory
    2. format_kaggle_files.py is run to format the data
    3. main.py is executed with inference arguments
    4. Aggregate metrics (f1, auc, ap) are read from JSON output

    Performance Optimizations:
    - Result caching: Avoids redundant inference on identical cluster data
    - Formatted file retention: Reduces data formatting overhead
    - Data fingerprinting: Fast cache lookup using MD5 hashing

    Args:
        gnn_project_path: Path to multignn project directory (required: set MULTIGNN_PATH env var or pass explicitly)
        data_config_path: Path to data_config.json (optional, will use default if not provided)
        model_name: GNN model to use (default: "gin")
        score_metric: Metric to use for scoring (default: "f1")
            - "f1": F1 score
            - "auc": AUC score
            - "ap": Average Precision score
            - "weighted": Weighted combination of metrics
        metric_weights: Custom weights for weighted metric (optional)
        save_debug_output: Whether to save debug output (default: False)
        debug_output_dir: Directory for debug output (optional)
        enable_cache: Enable result caching to avoid redundant inference (default: True)
        keep_formatted_files: Keep formatted data files to reduce overhead (default: True)

    Example:
        >>> monitor = GNNMonitor(
        ...     gnn_project_path=os.environ.get("MULTIGNN_PATH", ""),
        ...     score_metric="f1",
        ...     enable_cache=True
        ... )
        >>> prob = monitor.predict_proba(txns, accounts, cluster_nodes)
    """

    def __init__(
        self,
        gnn_project_path: str = DEFAULT_GNN_PROJECT_PATH,
        data_config_path: Optional[str] = None,
        model_name: str = "gin",
        score_metric: str = "f1",
        metric_weights: Optional[Dict[str, float]] = None,
        save_debug_output: bool = False,
        debug_output_dir: Optional[str] = None,
        enable_cache: bool = True,
        keep_formatted_files: bool = True
    ):
        """Initialize GNN monitor."""
        self.gnn_project_path = Path(gnn_project_path)

        # Validate GNN project path
        if not self.gnn_project_path.exists():
            raise ValueError(f"GNN project path does not exist: {gnn_project_path}")

        # Load data config
        if data_config_path is None:
            data_config_path = self.gnn_project_path / "data_config.json"

        with open(data_config_path, "r") as f:
            self.data_config = json.load(f)

        # Model configuration
        self.model_name = model_name

        # Metric configuration
        self.score_metric = score_metric
        self.metric_weights = metric_weights or {
            "f1": 0.3,
            "auc": 0.3,
            "ap": 0.4
        }

        # Debug output configuration
        self.save_debug_output = save_debug_output
        self.debug_output_dir = debug_output_dir

        # Performance optimization
        self.enable_cache = enable_cache
        self.keep_formatted_files = keep_formatted_files
        self._result_cache = {}  # Cache for inference results
        self._data_fingerprints = {}  # Cache for data fingerprints

        # Call counter for unique file naming
        self.call_count = 0

    def _prepare_transaction_data(
        self,
        transactions_df: pd.DataFrame,
        accounts_df: pd.DataFrame,
        cluster_nodes: List[str]
    ) -> pd.DataFrame:
        """Prepare transaction data for GNN inference.

        Args:
            transactions_df: Full transactions DataFrame
            accounts_df: Full accounts DataFrame
            cluster_nodes: List of node IDs in the cluster

        Returns:
            DataFrame formatted for GNN inference
        """
        # Filter transactions involving cluster nodes
        cluster_mask = (
            transactions_df["from_node_id"].isin(cluster_nodes) |
            transactions_df["to_node_id"].isin(cluster_nodes)
        )
        cluster_txns = transactions_df[cluster_mask].copy()

        if len(cluster_txns) == 0:
            return pd.DataFrame()

        # Map to GNN expected column names
        # GNN expects: Account, Account.1 (not From Account, To Account)
        formatted_df = pd.DataFrame()

        # Required columns for GNN (in the exact format GNN expects)
        # Use .get() on columns dict, not on DataFrame
        cols = cluster_txns.columns.tolist()

        formatted_df["Timestamp"] = cluster_txns["Timestamp"] if "Timestamp" in cols else ""
        formatted_df["From Bank"] = cluster_txns["from_bank_id"] if "from_bank_id" in cols else (cluster_txns["From Bank"] if "From Bank" in cols else "")
        formatted_df["Account"] = cluster_txns["from_account_number"] if "from_account_number" in cols else (cluster_txns["From Account"] if "From Account" in cols else "")
        formatted_df["To Bank"] = cluster_txns["to_bank_id"] if "to_bank_id" in cols else (cluster_txns["To Bank"] if "To Bank" in cols else "")
        formatted_df["Account.1"] = cluster_txns["to_account_number"] if "to_account_number" in cols else (cluster_txns["To Account"] if "To Account" in cols else "")
        formatted_df["Amount Received"] = cluster_txns["Amount Received"] if "Amount Received" in cols else (cluster_txns["amount"] if "amount" in cols else 0)
        formatted_df["Receiving Currency"] = cluster_txns["Receiving Currency"] if "Receiving Currency" in cols else "USD"
        formatted_df["Amount Paid"] = cluster_txns["Amount Paid"] if "Amount Paid" in cols else (cluster_txns["amount"] if "amount" in cols else 0)
        formatted_df["Payment Currency"] = cluster_txns["Payment Currency"] if "Payment Currency" in cols else "USD"
        formatted_df["Payment Format"] = cluster_txns["Payment Format"] if "Payment Format" in cols else "Reinvestment"
        formatted_df["Is Laundering"] = cluster_txns["Is Laundering"] if "Is Laundering" in cols else 0

        return formatted_df

    def _compute_data_fingerprint(
        self,
        cluster_nodes: List[str],
        transactions_df: pd.DataFrame,
        accounts_df: pd.DataFrame
    ) -> str:
        """Compute a fingerprint (hash) for the cluster data.

        This is used for caching to avoid redundant inference calls.

        Args:
            cluster_nodes: List of node IDs in the cluster
            transactions_df: Full transactions DataFrame
            accounts_df: Full accounts DataFrame

        Returns:
            str: MD5 hash combining cluster nodes and relevant data content
        """
        # Sort nodes for consistent hashing
        sorted_nodes = sorted(cluster_nodes)
        nodes_str = ",".join(map(str, sorted_nodes))

        # Extract cluster-relevant transactions (edges within cluster)
        cluster_nodes_set = set(cluster_nodes)
        cluster_mask = (
            transactions_df["from_node_id"].isin(cluster_nodes_set) &
            transactions_df["to_node_id"].isin(cluster_nodes_set)
        )
        cluster_transactions = transactions_df[cluster_mask]

        # Create a hash of the transaction data (use all columns)
        if len(cluster_transactions) > 0:
            # Sort by edge_id for consistent ordering
            cluster_transactions_sorted = cluster_transactions.sort_values("edge_id")
            trans_hash_data = cluster_transactions_sorted.to_csv(index=False, header=False)
        else:
            trans_hash_data = ""

        # Combine nodes and transaction data
        combined_str = f"{nodes_str}|{trans_hash_data}"

        # Compute MD5 hash
        fingerprint = hashlib.md5(combined_str.encode()).hexdigest()
        return fingerprint

    def _run_gnn_inference(self, data_name: str, raw_data_file: str) -> Dict[str, float]:
        """Run GNN inference on prepared data using subprocess.

        Args:
            data_name: Name of the data file (without .csv extension)
            raw_data_file: Path to the raw data CSV file

        Returns:
            Dict with metrics (f1, auc, ap)
        """
        original_cwd = os.getcwd()
        try:
            os.chdir(self.gnn_project_path)

            # Step 1: Run format_kaggle_files.py to format the data
            format_script = self.gnn_project_path / "format_kaggle_files.py"
            format_cmd = [sys.executable, str(format_script), raw_data_file]

            print(f"[GNN Monitor] Formatting data: {' '.join(format_cmd)}")
            t_format_start = time.time()
            result = subprocess.run(
                format_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            t_format_end = time.time()
            print(f"[GNN Monitor] ⏱️  Data formatting took: {t_format_end - t_format_start:.2f}s")

            # The formatted file will be named "formatted_{data_name}.csv"
            formatted_data_name = f"formatted_{data_name}"

            # Step 2: Run main.py with inference arguments
            inference_cmd = [
                sys.executable, "main.py",
                "--data", formatted_data_name,
                "--model", self.model_name,
                "--emlps",
                "--reverse_mp",
                "--ego",
                "--ports",
                "--inference",
                "--inference_mode", "full"
            ]

            print(f"[GNN Monitor] Running inference: {' '.join(inference_cmd)}")
            t_inference_start = time.time()
            result = subprocess.run(
                inference_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            t_inference_end = time.time()
            print(f"[GNN Monitor] ⏱️  GNN inference took: {t_inference_end - t_inference_start:.2f}s")

            # Step 3: Read the JSON metrics output
            t_read_start = time.time()
            out_dir = self.gnn_project_path / "out"
            metrics_file = out_dir / f"{formatted_data_name}_{self.model_name}_full_metrics.json"

            if not metrics_file.exists():
                raise FileNotFoundError(
                    f"GNN metrics output not found: {metrics_file}"
                )

            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            t_read_end = time.time()
            print(f"[GNN Monitor] ⏱️  Reading results took: {t_read_end - t_read_start:.3f}s")

            # Print total time breakdown
            total_time = t_read_end - t_format_start
            print(f"[GNN Monitor] ⏱️  Total subprocess time: {total_time:.2f}s")
            print(f"[GNN Monitor]   - Formatting: {t_format_end - t_format_start:.2f}s ({(t_format_end - t_format_start)/total_time*100:.1f}%)")
            print(f"[GNN Monitor]   - Inference: {t_inference_end - t_inference_start:.2f}s ({(t_inference_end - t_inference_start)/total_time*100:.1f}%)")
            print(f"[GNN Monitor]   - Reading: {t_read_end - t_read_start:.3f}s ({(t_read_end - t_read_start)/total_time*100:.1f}%)")

            return metrics

        finally:
            # Restore original working directory
            os.chdir(original_cwd)

    def predict_proba(
        self,
        transactions_df: pd.DataFrame,
        accounts_df: pd.DataFrame,
        cluster_nodes: List[str],
        debug_label: str = None,
        trajectory_id: int = None,
        step_idx: int = None
    ) -> float:
        """Predict probability of money laundering using GNN model.

        Args:
            transactions_df: Full transactions DataFrame
            accounts_df: Full accounts DataFrame
            cluster_nodes: List of node IDs in the cluster to evaluate
            debug_label: Optional label for debug output (e.g., "before", "after")
            trajectory_id: Optional trajectory ID for grouping debug output
            step_idx: Optional step index within trajectory

        Returns:
            float: Probability score between 0 and 1 (aggregate metric from GNN)
        """
        t_total_start = time.time()

        # Compute data fingerprint for caching
        t_fingerprint_start = time.time()
        fingerprint = self._compute_data_fingerprint(cluster_nodes, transactions_df, accounts_df)
        t_fingerprint_end = time.time()

        # Check cache if enabled
        if self.enable_cache and fingerprint in self._result_cache:
            cached_result = self._result_cache[fingerprint]
            t_total_end = time.time()
            print(f"[GNN Monitor] 🚀 Using cached result for fingerprint {fingerprint[:8]}...")
            print(f"[GNN Monitor] ⏱️  Total time (cached): {t_total_end - t_total_start:.3f}s")

            # Still save debug output if requested
            if self.save_debug_output and self.debug_output_dir:
                self._save_debug_output(
                    metrics=cached_result["metrics"],
                    score=cached_result["score"],
                    cluster_nodes=cluster_nodes,
                    debug_label=debug_label,
                    trajectory_id=trajectory_id,
                    step_idx=step_idx
                )

            return cached_result["score"]

        # Only increment counter for "before" or no label to keep before/after/summary aligned
        if debug_label != "after":
            self.call_count += 1

        # Prepare transaction data
        t_prepare_start = time.time()
        formatted_df = self._prepare_transaction_data(
            transactions_df, accounts_df, cluster_nodes
        )
        t_prepare_end = time.time()
        print(f"[GNN Monitor] ⏱️  Data preparation took: {t_prepare_end - t_prepare_start:.3f}s")

        if len(formatted_df) == 0:
            return 0.0

        # Generate unique data name for this inference
        data_name = f"gnn_monitor_inference_{self.call_count}"

        # Save prepared data to GNN data directory
        t_save_start = time.time()
        data_dir = self.data_config["paths"]["aml_data"]
        os.makedirs(data_dir, exist_ok=True)
        data_file = os.path.join(data_dir, f"{data_name}.csv")
        formatted_df.to_csv(data_file, index=False)
        t_save_end = time.time()
        print(f"[GNN Monitor] ⏱️  Saving CSV took: {t_save_end - t_save_start:.3f}s")

        try:
            # Run GNN inference (format + inference + read metrics)
            metrics = self._run_gnn_inference(data_name, data_file)

            # Select score based on configured metric
            t_score_start = time.time()
            if self.score_metric == "weighted":
                # Weighted combination of metrics
                score = 0.0
                total_weight = 0.0
                for metric_name, weight in self.metric_weights.items():
                    if metric_name in metrics:
                        score += metrics[metric_name] * weight
                        total_weight += weight

                if total_weight > 0:
                    score = score / total_weight
                else:
                    # Fallback to f1 if weighted fails
                    score = metrics.get("f1", 0.5)

            elif self.score_metric in metrics:
                # Use specific metric
                score = metrics[self.score_metric]
            else:
                # Default to f1
                print(f"Warning: {self.score_metric} not available, using f1")
                score = metrics.get("f1", 0.5)
            t_score_end = time.time()

            # Cache the result if caching is enabled
            if self.enable_cache:
                self._result_cache[fingerprint] = {
                    "metrics": metrics,
                    "score": score
                }
                print(f"[GNN Monitor] Cached result for fingerprint {fingerprint[:8]}")

            # Log metrics for debugging
            print(f"[GNN Monitor] Metrics: {metrics}")
            print(f"[GNN Monitor] Selected metric '{self.score_metric}': {score:.4f}")

            # Save debug output if enabled
            if self.save_debug_output and self.debug_output_dir:
                self._save_debug_output(
                    metrics=metrics,
                    score=score,
                    cluster_nodes=cluster_nodes,
                    debug_label=debug_label,
                    trajectory_id=trajectory_id,
                    step_idx=step_idx
                )

            t_total_end = time.time()

            # Print comprehensive timing summary
            print(f"\n[GNN Monitor] ⏱️  === TIMING SUMMARY ===")
            print(f"[GNN Monitor] ⏱️  Total time: {t_total_end - t_total_start:.2f}s")
            print(f"[GNN Monitor]   - Fingerprint: {t_fingerprint_end - t_fingerprint_start:.3f}s")
            print(f"[GNN Monitor]   - Data prep: {t_prepare_end - t_prepare_start:.3f}s")
            print(f"[GNN Monitor]   - Save CSV: {t_save_end - t_save_start:.3f}s")
            print(f"[GNN Monitor]   - Score computation: {t_score_end - t_score_start:.3f}s")
            print(f"[GNN Monitor] ⏱️  ======================\n")

            return float(np.clip(score, 0.0, 1.0))

        except Exception as e:
            print(f"Warning: GNN inference failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.5

        finally:
            # Clean up temporary data file (unless keep_formatted_files is True)
            if not self.keep_formatted_files and os.path.exists(data_file):
                try:
                    os.remove(data_file)
                except Exception:
                    pass

    def _save_debug_output(
        self,
        metrics: Dict[str, float],
        score: float,
        cluster_nodes: List[str],
        debug_label: str = None,
        trajectory_id: int = None,
        step_idx: int = None
    ):
        """Save debug output for GNN predictions.

        Args:
            metrics: Dict of aggregate metrics (f1, auc, ap)
            score: Final score used for reward
            cluster_nodes: List of cluster node IDs
            debug_label: Optional label (e.g., "before", "after")
            trajectory_id: Optional trajectory ID
            step_idx: Optional step index
        """
        debug_dir = os.path.join(self.debug_output_dir, "gnn_debug")
        os.makedirs(debug_dir, exist_ok=True)

        # Save summary file with aggregate metrics
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
                with open(summary_file, mode) as f:
                    if debug_label == "before":
                        f.write(f"=== Step {step_idx + 1} BEFORE ===\n")
                    else:
                        f.write(f"\n=== Step {step_idx + 1} AFTER ===\n")

                    f.write(f"Cluster nodes: {len(cluster_nodes)}\n\n")

                    f.write(f"GNN Aggregate Metrics:\n")
                    for metric_name, metric_value in metrics.items():
                        f.write(f"  {metric_name}: {metric_value:.6f}\n")

                    f.write(f"\nSelected metric '{self.score_metric}': {score:.6f}\n")

        else:
            # Fallback: old behavior for backward compatibility
            summary_file = os.path.join(debug_dir, f"summary_{self.call_count}.txt")

            with open(summary_file, 'w') as f:
                f.write(f"=== GNN Prediction Summary ===\n")
                f.write(f"Cluster nodes: {len(cluster_nodes)}\n\n")

                f.write(f"GNN Aggregate Metrics:\n")
                for metric_name, metric_value in metrics.items():
                    f.write(f"  {metric_name}: {metric_value:.6f}\n")

                f.write(f"\nSelected metric '{self.score_metric}': {score:.6f}\n")
