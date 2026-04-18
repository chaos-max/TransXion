"""Main perturbation runner with agent loop."""

import pandas as pd
from typing import Dict, Any, List
from ..core.clusters import LaunderingCluster
from ..core.candidates import generate_candidates
from ..core.budget import BudgetTracker
from ..core.validator import (
    create_snapshot, rollback_to_snapshot, validate_state,
    validate_inject_chain, validate_split_result, validate_merge_bank_constraint
)
from ..core.pre_validator import pre_validate_decision
from ..agent import LLMAgent, get_deterministic_fallback
from ..tools import inject_intermediary, merge_accounts, split_account, adjust_transaction
from .logging import PerturbationLogger


class PerturbationRunner:
    """Main runner for perturbation pipeline."""

    def __init__(
        self,
        agent: LLMAgent,
        logger: PerturbationLogger,
        config: Dict[str, Any]
    ):
        """Initialize runner.

        Args:
            agent: LLM agent
            logger: Logger instance
            config: Configuration dictionary with:
                - seed: Random seed
                - topk_candidates: Top-K candidates to include
                - max_steps_per_cluster: Max steps per cluster
                - max_attempts_per_step: Max attempts per step
                - fail_limit: Max consecutive failures before hard stop
                - max_merges_per_cluster: Max merge operations
                - max_splits_per_cluster: Max split operations
                - max_new_nodes_ratio: Max new nodes ratio
                - max_new_edges_ratio: Max new edges ratio
                - timestamp_output_format: Timestamp format
        """
        self.agent = agent
        self.logger = logger
        self.config = config
        self.allowed_params = self._build_allowed_params()

    def run_cluster(
        self,
        cluster: LaunderingCluster,
        transactions_df: pd.DataFrame,
        accounts_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, set]:
        """Run perturbation for a single cluster.

        Args:
            cluster: LaunderingCluster object
            transactions_df: Full transactions DataFrame
            accounts_df: Full accounts DataFrame

        Returns:
            Tuple of (updated_transactions_df, updated_accounts_df, affected_nodes)
        """
        cluster_id = cluster.cluster_id
        cluster_nodes = cluster.nodes_in_cluster
        cluster_internal_edges = cluster.laundering_edges_internal

        # Initialize budget
        initial_nodes = len(cluster_nodes)
        initial_edges = len(cluster_internal_edges)

        budget = BudgetTracker(
            max_steps=self.config["max_steps_per_cluster"],
            max_merges=self.config["max_merges_per_cluster"],
            max_splits=self.config["max_splits_per_cluster"],
            max_new_nodes=int(initial_nodes * self.config["max_new_nodes_ratio"]),
            max_new_edges=int(initial_edges * self.config["max_new_edges_ratio"])
        )

        # Track state
        step_idx = 0
        fail_streak = 0
        steps_successful = 0
        inject_count = 0
        last_failure = None
        cluster_affected_nodes = set()  # Collect all affected nodes in this cluster

        # Main loop
        while step_idx < self.config["max_steps_per_cluster"]:
            # Generate candidates
            candidates = generate_candidates(
                transactions_df,
                accounts_df,
                cluster_nodes,
                cluster_internal_edges,
                topk=self.config["topk_candidates"],
                seed=self.config["seed"] + cluster_id * 1000 + step_idx
            )

            # Determine allowed tools
            allowed_tools = self._determine_allowed_tools(budget, candidates)

            # Build state_json
            state_json = self._build_state_json(
                cluster,
                cluster_id,
                step_idx,
                transactions_df,
                candidates,
                budget,
                allowed_tools,
                inject_count,
                last_failure
            )

            # Call agent (MUST call even if allowed_tools is empty)
            decision = self.agent.decide_tool(state_json)

            # Log decision
            self.logger.log_agent_decision(
                cluster_id=cluster_id,
                step_idx=step_idx,
                allowed_tools=allowed_tools,
                decision_raw=decision.get("decision_raw", ""),
                decision_parsed=decision.get("decision_parsed"),
                is_valid=decision["is_valid"],
                invalid_reason=decision.get("invalid_reason"),
                tool=decision.get("tool"),
                args=decision.get("args"),
                rationale=decision.get("rationale")
            )

            # Check if stop
            if decision.get("tool") == "stop":
                self.logger.log_cluster_end(
                    cluster_id=cluster_id,
                    steps_attempted=step_idx,
                    steps_successful=steps_successful,
                    end_reason="stop"
                )
                break

            # Try to execute (with retry logic)
            attempt_idx = 0
            success = False

            while attempt_idx < self.config["max_attempts_per_step"]:
                # Get decision to try (first attempt or fallback)
                if attempt_idx == 0:
                    if not decision["is_valid"]:
                        # Invalid decision, use fallback on second attempt
                        attempt_idx += 1
                        continue

                    # Pre-validate LLM decision (Solution 2B)
                    is_valid, error_msg = pre_validate_decision(
                        decision, transactions_df, accounts_df, cluster_nodes
                    )
                    if not is_valid:
                        # Pre-validation failed, skip to fallback
                        # Store error for potential debugging
                        last_failure = error_msg
                        attempt_idx += 1
                        continue

                    tool_decision = decision
                else:
                    # Fallback - regenerate candidates with fresh state (Solution 1A)
                    # This ensures we don't reference any accounts created in the failed first attempt
                    fresh_candidates = generate_candidates(
                        transactions_df,
                        accounts_df,
                        cluster_nodes,
                        cluster_internal_edges,
                        topk=self.config["topk_candidates"],
                        seed=self.config["seed"] + cluster_id * 1000 + step_idx + attempt_idx
                    )
                    tool_decision = get_deterministic_fallback(allowed_tools, fresh_candidates, self.allowed_params)
                    tool_decision["is_valid"] = True

                tool = tool_decision["tool"]
                args = tool_decision["args"]
                fallback_used = tool_decision.get("fallback_used", False)

                if tool == "stop":
                    # Fallback is stop
                    self.logger.log_step_result(
                        cluster_id=cluster_id,
                        step_idx=step_idx,
                        attempt_idx=attempt_idx,
                        tool=tool,
                        args=args,
                        ok=True,
                        fallback_used=fallback_used
                    )
                    self.logger.log_cluster_end(
                        cluster_id=cluster_id,
                        steps_attempted=step_idx,
                        steps_successful=steps_successful,
                        end_reason="stop"
                    )
                    return transactions_df, accounts_df, cluster_affected_nodes

                # Execute tool
                ok, result = self._execute_tool(
                    tool, args, transactions_df, accounts_df, cluster_nodes
                )

                # Update state (whether success or failure - failure returns rolled-back state)
                if "transactions_df" in result:
                    transactions_df = result["transactions_df"]
                if "accounts_df" in result:
                    accounts_df = result["accounts_df"]

                # Log result
                if ok:
                    self.logger.log_step_result(
                        cluster_id=cluster_id,
                        step_idx=step_idx,
                        attempt_idx=attempt_idx,
                        tool=tool,
                        args=args,
                        ok=True,
                        affected_edge_ids=result.get("affected_edge_ids", []),
                        new_edge_ids=result.get("new_edge_ids", []),
                        created_accounts=result.get("created_accounts", []),
                        fallback_used=fallback_used
                    )

                    # Collect affected nodes from this operation
                    cluster_affected_nodes.update(result.get("affected_nodes", []))

                    # Update budget
                    if tool == "inject_intermediary":
                        budget.use_edges(len(result.get("new_edge_ids", [])))
                        budget.use_nodes(len(result.get("created_accounts", [])))
                        inject_count += 1
                    elif tool == "merge_accounts":
                        budget.use_merge()
                        budget.use_nodes(len(result.get("created_accounts", [])))
                    elif tool == "split_account":
                        budget.use_split()
                        budget.use_nodes(len(result.get("created_accounts", [])))
                    elif tool == "adjust_transaction":
                        # adjust does not create new nodes or edges, only modifies existing transactions
                        pass

                    budget.use_step()
                    steps_successful += 1
                    fail_streak = 0
                    last_failure = None
                    success = True
                    break
                else:
                    # Failed
                    self.logger.log_step_result(
                        cluster_id=cluster_id,
                        step_idx=step_idx,
                        attempt_idx=attempt_idx,
                        tool=tool,
                        args=args,
                        ok=False,
                        violations=result.get("violations", []),
                        fallback_used=fallback_used,
                        error_message=result.get("error_message", "")
                    )

                    last_failure = result.get("error_message", "Unknown error")
                    attempt_idx += 1

            # Check if failed all attempts
            if not success:
                fail_streak += 1

                if fail_streak >= self.config["fail_limit"]:
                    # Hard failure: call agent one more time with allowed_tools=[]
                    state_json_final = self._build_state_json(
                        cluster,
                        cluster_id,
                        step_idx,
                        transactions_df,
                        candidates,
                        budget,
                        allowed_tools=[],
                        inject_count=inject_count,
                        last_failure=last_failure
                    )

                    final_decision = self.agent.decide_tool(state_json_final)

                    self.logger.log_agent_decision(
                        cluster_id=cluster_id,
                        step_idx=step_idx,
                        allowed_tools=[],
                        decision_raw=final_decision.get("decision_raw", ""),
                        decision_parsed=final_decision.get("decision_parsed"),
                        is_valid=final_decision["is_valid"],
                        invalid_reason=final_decision.get("invalid_reason"),
                        tool=final_decision.get("tool"),
                        args=final_decision.get("args"),
                        rationale=final_decision.get("rationale")
                    )

                    self.logger.log_cluster_end(
                        cluster_id=cluster_id,
                        steps_attempted=step_idx + 1,
                        steps_successful=steps_successful,
                        end_reason="hard_failure"
                    )
                    break

            step_idx += 1

        # If we exited loop normally (max steps reached)
        if step_idx >= self.config["max_steps_per_cluster"]:
            self.logger.log_cluster_end(
                cluster_id=cluster_id,
                steps_attempted=step_idx,
                steps_successful=steps_successful,
                end_reason="max_steps"
            )

        return transactions_df, accounts_df, cluster_affected_nodes

    def _determine_allowed_tools(self, budget: BudgetTracker, candidates: Dict) -> List[str]:
        """Determine which tools are allowed based on budget and candidates.

        Args:
            budget: BudgetTracker
            candidates: Candidates dictionary

        Returns:
            List of allowed tool names
        """
        allowed = []

        # Inject
        inject_cands = candidates.get("inject_candidates", [])
        if len(inject_cands) > 0 and budget.can_inject(1):
            allowed.append("inject_intermediary")

        # Merge
        merge_cands = candidates.get("merge_candidates", [])
        if len(merge_cands) > 0 and budget.can_merge():
            allowed.append("merge_accounts")

        # Split
        split_cands = candidates.get("split_candidates", [])
        if len(split_cands) > 0 and budget.can_split():
            allowed.append("split_account")

        # Adjust
        adjust_cands = candidates.get("adjust_candidates", [])
        if len(adjust_cands) > 0:
            allowed.append("adjust_transaction")

        return allowed

    def _build_state_json(
        self,
        cluster: LaunderingCluster,
        cluster_id: int,
        step_idx: int,
        transactions_df: pd.DataFrame,
        candidates: Dict,
        budget: BudgetTracker,
        allowed_tools: List[str],
        inject_count: int,
        last_failure: str = None
    ) -> Dict[str, Any]:
        """Build state JSON for agent.

        Args:
            cluster: LaunderingCluster
            cluster_id: Cluster ID
            step_idx: Step index
            transactions_df: Transactions DataFrame
            candidates: Candidates dictionary
            budget: BudgetTracker
            allowed_tools: List of allowed tools
            inject_count: Number of inject operations performed
            last_failure: Last failure message

        Returns:
            State JSON dictionary
        """
        # Compute cluster summary
        cluster_nodes = cluster.nodes_in_cluster
        cluster_internal_edges = cluster.laundering_edges_internal

        # Compute degrees - VECTORIZED
        from collections import defaultdict
        node_in_deg = transactions_df["to_node_id"].value_counts().to_dict()
        node_out_deg = transactions_df["from_node_id"].value_counts().to_dict()

        # Convert to defaultdict for safe access
        node_in_deg = defaultdict(int, node_in_deg)
        node_out_deg = defaultdict(int, node_out_deg)

        max_in = max([node_in_deg[n] for n in cluster_nodes], default=0)
        max_out = max([node_out_deg[n] for n in cluster_nodes], default=0)
        avg_deg = sum([node_in_deg[n] + node_out_deg[n] for n in cluster_nodes]) / len(cluster_nodes) if len(cluster_nodes) > 0 else 0

        # Time span
        internal_df = transactions_df[transactions_df["edge_id"].isin(cluster_internal_edges)]
        if len(internal_df) > 0:
            time_span = internal_df["ts_int"].max() - internal_df["ts_int"].min()
        else:
            time_span = 0

        # Top currencies
        from collections import Counter
        currency_counter = Counter(internal_df["Payment Currency"].values)
        currency_top3 = [c for c, _ in currency_counter.most_common(3)]

        cluster_summary = {
            "n_nodes": len(cluster_nodes),
            "laundering_edge_count_internal": len(cluster_internal_edges),
            "max_in_deg": max_in,
            "max_out_deg": max_out,
            "avg_deg": avg_deg,
            "time_span_seconds": time_span,
            "currency_top3": currency_top3,
        }

        state_json = {
            "cluster_id": cluster_id,
            "step_idx": step_idx,
            "cluster_summary": cluster_summary,
            "candidates": candidates,
            "budget_left": budget.get_remaining(),
            "allowed_tools": allowed_tools,
            "allowed_params": self.allowed_params,
            "history": {
                "inject_count": inject_count,
                "merge_count": budget.merges_used,
                "split_count": budget.splits_used,
            },
        }

        if last_failure:
            state_json["last_failure"] = last_failure

        return state_json

    def _execute_tool(
        self,
        tool: str,
        args: Dict,
        transactions_df: pd.DataFrame,
        accounts_df: pd.DataFrame,
        cluster_nodes: set = None
    ) -> tuple[bool, Dict[str, Any]]:
        """Execute a tool with precheck, apply, and postcheck.

        Args:
            tool: Tool name
            args: Tool arguments
            transactions_df: Transactions DataFrame
            accounts_df: Accounts DataFrame
            cluster_nodes: Set of node IDs in current cluster (optional)

        Returns:
            Tuple of (success, result_dict)
        """
        # Create snapshot for rollback
        snapshot = create_snapshot(transactions_df, accounts_df)

        try:
            # Precheck
            if tool == "merge_accounts":
                for pair in args["pairs"]:
                    ok, violations = validate_merge_bank_constraint(pair[0], pair[1])
                    if not ok:
                        return False, {"error_message": "; ".join(violations), "violations": violations}

            # Execute tool
            if tool == "inject_intermediary":
                result = inject_intermediary(
                    transactions_df,
                    accounts_df,
                    edge_ids=args["edge_ids"],
                    depth=args["depth"],
                    time_delta_seconds=args["time_delta_seconds"],
                    timestamp_format=self.config["timestamp_output_format"],
                    seed=self.config["seed"],
                    cluster_nodes=cluster_nodes
                )
            elif tool == "merge_accounts":
                result = merge_accounts(
                    transactions_df,
                    accounts_df,
                    pairs=args["pairs"],
                    drop_self_loops=args["drop_self_loops"],
                    seed=self.config["seed"]
                )
            elif tool == "split_account":
                result = split_account(
                    transactions_df,
                    accounts_df,
                    node_ids=args["node_ids"],
                    split_ratio=args["split_ratio"],
                    move_direction=args["move_direction"],
                    edge_sampling=args["edge_sampling"],
                    seed=self.config["seed"]
                )
            elif tool == "adjust_transaction":
                result = adjust_transaction(
                    transactions_df,
                    accounts_df,
                    edge_ids=args["edge_ids"],
                    time_offset_seconds=args["time_offset_seconds"],
                    amount_multiplier=args["amount_multiplier"],
                    timestamp_format=self.config["timestamp_output_format"]
                )
            else:
                return False, {"error_message": f"Unknown tool: {tool}", "violations": []}

            # Update DataFrames
            transactions_df = result["transactions_df"]
            accounts_df = result["accounts_df"]

            # Postcheck: general validation
            ok, violations = validate_state(transactions_df, accounts_df)
            if not ok:
                # Rollback
                transactions_df, accounts_df = rollback_to_snapshot(snapshot)
                return False, {
                    "error_message": "Validation failed",
                    "violations": violations,
                    "transactions_df": transactions_df,
                    "accounts_df": accounts_df
                }

            # Tool-specific postcheck
            # NOTE: Inject chain validation disabled to allow more flexible inject operations
            # if tool == "inject_intermediary":
            #     ok, violations = validate_inject_chain(transactions_df, result.get("new_edge_ids", []))
            #     if not ok:
            #         transactions_df, accounts_df = rollback_to_snapshot(snapshot)
            #         return False, {"error_message": "Inject chain validation failed", "violations": violations}

            if tool == "split_account":
                for node_id in args["node_ids"]:
                    created = result.get("created_accounts", [])
                    if len(created) > 0:
                        # Assume created accounts correspond to node_ids in order
                        idx = args["node_ids"].index(node_id)
                        if idx < len(created):
                            new_node = created[idx]
                            ok, violations = validate_split_result(
                                transactions_df,
                                node_id,
                                new_node,
                                args["move_direction"],
                                result.get("moved_edge_ids", [])
                            )
                            if not ok:
                                transactions_df, accounts_df = rollback_to_snapshot(snapshot)
                                return False, {
                                    "error_message": "Split validation failed",
                                    "violations": violations,
                                    "transactions_df": transactions_df,
                                    "accounts_df": accounts_df
                                }

            # Success
            result["transactions_df"] = transactions_df
            result["accounts_df"] = accounts_df
            return True, result

        except Exception as e:
            # Rollback on exception
            transactions_df, accounts_df = rollback_to_snapshot(snapshot)
            return False, {
                "error_message": str(e),
                "violations": [str(e)],
                "transactions_df": transactions_df,
                "accounts_df": accounts_df
            }

    def _build_allowed_params(self) -> Dict[str, Any]:
        """Build allowed parameters dictionary.

        Returns:
            Allowed parameters dictionary
        """
        return {
            "inject": {
                "depth": [1, 2],
                "time_delta_seconds": [1, 5, 60],
                "max_edge_ids": 3,
            },
            "merge": {
                "drop_self_loops": [False, True],
                "max_pairs": 2,
            },
            "split": {
                "split_ratio": [0.2, 0.3, 0.4],
                "move_direction": ["out", "in", "both"],
                "edge_sampling": ["time_stratified", "random_within_currency", "random"],
                "max_node_ids": 2,
            },
            "adjust": {
                "time_offset_seconds": [-3600, -60, 60, 3600],
                "amount_multiplier": [0.9, 0.95, 1.05, 1.1],
                "max_edge_ids": 3,
            },
        }
