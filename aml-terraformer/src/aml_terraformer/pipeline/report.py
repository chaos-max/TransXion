"""Generate summary report."""

import json
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict


def generate_summary_report(
    log_path: str,
    output_path: str,
    cli_args: Dict[str, Any]
):
    """Generate summary report from log file.

    Args:
        log_path: Path to perturb_log.jsonl
        output_path: Path to output summary_report.json
        cli_args: CLI arguments dictionary
    """
    # Read log entries
    entries = []
    with open(log_path, 'r') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    # Aggregate statistics
    clusters_total = 0
    clusters_touched = 0
    actions_count = defaultdict(int)
    created_accounts_count = 0
    created_edges_count = 0
    removed_edges_count = 0
    self_loops_dropped_count = 0
    fallback_count = 0
    rejection_count = defaultdict(int)

    # Track per cluster
    cluster_stats = defaultdict(lambda: {
        "steps_successful": 0,
        "steps_attempted": 0,
        "end_reason": None
    })

    for entry in entries:
        entry_type = entry.get("type")

        if entry_type == "agent_decision":
            cluster_id = entry["cluster_id"]
            if not entry["is_valid"]:
                reason = entry.get("invalid_reason", "unknown")
                rejection_count[reason] += 1

        elif entry_type == "step_result":
            cluster_id = entry["cluster_id"]
            tool = entry["tool"]
            ok = entry["ok"]

            if ok:
                cluster_stats[cluster_id]["steps_successful"] += 1
                actions_count[tool] += 1

                # Count created resources
                created_accounts_count += len(entry.get("created_accounts", []))
                created_edges_count += len(entry.get("new_edge_ids", []))

                # Inject removes original edges
                if tool == "inject_intermediary":
                    removed_edges_count += len(entry.get("affected_edge_ids", []))

            cluster_stats[cluster_id]["steps_attempted"] += 1

            if entry.get("fallback_used"):
                fallback_count += 1

        elif entry_type == "cluster_end":
            cluster_id = entry["cluster_id"]
            clusters_total += 1

            if cluster_stats[cluster_id]["steps_successful"] > 0:
                clusters_touched += 1

            cluster_stats[cluster_id]["end_reason"] = entry["end_reason"]

    # Build report
    report = {
        "clusters_total": clusters_total,
        "clusters_touched": clusters_touched,
        "actions_count_by_type": dict(actions_count),
        "created_accounts_count": created_accounts_count,
        "created_edges_count": created_edges_count,
        "removed_edges_count": removed_edges_count,
        "self_loops_dropped_count": self_loops_dropped_count,
        "fallback_count": fallback_count,
        "rejection_count_by_reason": dict(rejection_count),
        "cli_args": cli_args,
    }

    # Write report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
