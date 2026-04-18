"""Perturbation logging to JSONL."""

import json
from pathlib import Path
from typing import Dict, Any


class PerturbationLogger:
    """Logger for perturbation operations."""

    def __init__(self, log_path: str):
        """Initialize logger.

        Args:
            log_path: Path to output JSONL file
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Clear existing file
        with open(self.log_path, 'w') as f:
            pass

    def log_agent_decision(
        self,
        cluster_id: int,
        step_idx: int,
        allowed_tools: list,
        decision_raw: str,
        decision_parsed: Dict[str, Any],
        is_valid: bool,
        invalid_reason: str = None,
        tool: str = None,
        args: Dict = None,
        rationale: str = None
    ):
        """Log agent decision.

        Args:
            cluster_id: Cluster ID
            step_idx: Step index
            allowed_tools: List of allowed tools
            decision_raw: Raw decision from LLM
            decision_parsed: Parsed decision
            is_valid: Whether decision is valid
            invalid_reason: Reason if invalid
            tool: Tool name if valid
            args: Tool arguments if valid
            rationale: Rationale if present
        """
        entry = {
            "type": "agent_decision",
            "cluster_id": cluster_id,
            "step_idx": step_idx,
            "allowed_tools": allowed_tools,
            "decision_raw": decision_raw,
            "decision_parsed": decision_parsed,
            "is_valid": is_valid,
        }

        if not is_valid:
            entry["invalid_reason"] = invalid_reason
        else:
            entry["tool"] = tool
            entry["args"] = args
            entry["rationale"] = rationale

        self._write_entry(entry)

    def log_step_result(
        self,
        cluster_id: int,
        step_idx: int,
        attempt_idx: int,
        tool: str,
        args: Dict,
        ok: bool,
        violations: list = None,
        affected_edge_ids: list = None,
        new_edge_ids: list = None,
        created_accounts: list = None,
        fallback_used: bool = False,
        error_message: str = None
    ):
        """Log step result.

        Args:
            cluster_id: Cluster ID
            step_idx: Step index
            attempt_idx: Attempt index
            tool: Tool name
            args: Tool arguments
            ok: Whether operation succeeded
            violations: List of violations if failed
            affected_edge_ids: List of affected edge IDs
            new_edge_ids: List of new edge IDs
            created_accounts: List of created accounts
            fallback_used: Whether fallback was used
            error_message: Error message if failed
        """
        entry = {
            "type": "step_result",
            "cluster_id": cluster_id,
            "step_idx": step_idx,
            "attempt_idx": attempt_idx,
            "tool": tool,
            "args": args,
            "ok": ok,
            "fallback_used": fallback_used,
        }

        if ok:
            entry["affected_edge_ids"] = affected_edge_ids or []
            entry["new_edge_ids"] = new_edge_ids or []
            entry["created_accounts"] = created_accounts or []
        else:
            entry["violations"] = violations or []
            entry["error_message"] = error_message or ""

        self._write_entry(entry)

    def log_cluster_end(
        self,
        cluster_id: int,
        steps_attempted: int,
        steps_successful: int,
        end_reason: str
    ):
        """Log cluster end.

        Args:
            cluster_id: Cluster ID
            steps_attempted: Number of steps attempted
            steps_successful: Number of successful steps
            end_reason: Reason for ending (max_steps, stop, hard_failure)
        """
        entry = {
            "type": "cluster_end",
            "cluster_id": cluster_id,
            "steps_attempted": steps_attempted,
            "steps_successful": steps_successful,
            "end_reason": end_reason,
        }

        self._write_entry(entry)

    def _write_entry(self, entry: Dict[str, Any]):
        """Write entry to JSONL file.

        Args:
            entry: Entry dictionary
        """
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
