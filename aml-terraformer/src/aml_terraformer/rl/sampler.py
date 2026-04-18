"""Trajectory sampler for GRPO training."""

import pandas as pd
import time
from typing import Dict, Any, List, Tuple
from multiprocessing import Pool, get_context
import os
from ..core.clusters import LaunderingCluster
from ..core.candidates import generate_candidates
from ..core.budget import BudgetTracker
from ..agent import LLMAgent
from ..tools import inject_intermediary, merge_accounts, split_account, adjust_transaction
from ..core.validator import (
    create_snapshot, rollback_to_snapshot, validate_state,
    validate_inject_chain, validate_split_result, validate_merge_bank_constraint
)
from .reward import GRPOReward


# Global worker function for parallel sampling (must be at module level for pickle)
def _sample_trajectory_worker(args):
    """Worker function for parallel trajectory sampling.

    This function must be at module level to be picklable by multiprocessing.
    """
    (cluster, transactions_df, accounts_df, sample_id,
     agent, reward_fn, config) = args

    # Import here to avoid circular imports in worker process
    from .sampler import GRPOSampler

    # Create a temporary sampler instance for this worker
    sampler = GRPOSampler(agent, reward_fn, config)

    # Sample single trajectory
    trajectory = sampler._sample_single_trajectory(
        cluster, transactions_df, accounts_df, sample_id
    )

    return trajectory


class Trajectory:
    """Single trajectory (episode) in GRPO.

    A trajectory consists of a sequence of (state, action, reward, info) tuples.
    """

    def __init__(self, cluster_id: int):
        """Initialize trajectory.

        Args:
            cluster_id: ID of the cluster being perturbed
        """
        self.cluster_id = cluster_id
        self.steps = []  # List of step dicts
        self.total_return = 0.0
        self.final_detection_score = None

    def add_step(self, step_data: Dict[str, Any]):
        """Add a step to the trajectory.

        Args:
            step_data: Dict containing:
                - step_idx: Step index
                - state: State dict
                - action: Action dict (tool + args)
                - reward: Reward value
                - info: Additional info dict
                - prompt: Prompt sent to LLM
                - decision: LLM decision output
        """
        self.steps.append(step_data)
        self.total_return += step_data["reward"]

    def set_final_score(self, score: float):
        """Set final detection score."""
        self.final_detection_score = score

    def get_return(self) -> float:
        """Get total return (sum of rewards)."""
        return self.total_return

    def get_length(self) -> int:
        """Get trajectory length."""
        return len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cluster_id": self.cluster_id,
            "total_return": self.total_return,
            "length": len(self.steps),
            "final_detection_score": self.final_detection_score,
            "steps": self.steps,
        }


class GRPOSampler:
    """Trajectory sampler for GRPO.

    Samples K trajectories from each cluster using the current policy (LLM agent).
    Each trajectory is a complete episode from initial state to terminal state.

    Args:
        agent: LLM agent for decision making
        reward_fn: Reward function
        config: Configuration dict
    """

    def __init__(
        self,
        agent: LLMAgent,
        reward_fn: GRPOReward,
        config: Dict[str, Any]
    ):
        """Initialize sampler."""
        self.agent = agent
        self.reward_fn = reward_fn
        self.config = config
        self.allowed_params = self._build_allowed_params()

    def sample_trajectories(
        self,
        cluster: LaunderingCluster,
        transactions_df: pd.DataFrame,
        accounts_df: pd.DataFrame,
        num_samples: int = 4
    ) -> List[Trajectory]:
        """Sample multiple trajectories for a cluster.

        Args:
            cluster: LaunderingCluster object
            transactions_df: Transactions DataFrame
            accounts_df: Accounts DataFrame
            num_samples: Number of trajectories to sample (K in GRPO)

        Returns:
            List of Trajectory objects
        """
        # Check if parallel sampling is enabled
        use_parallel = self.config.get("use_parallel_sampling", True)
        num_workers = self.config.get("num_sampling_workers", 4)

        if use_parallel and num_samples > 1:
            # Parallel sampling using multiprocessing
            trajectories = self._sample_trajectories_parallel(
                cluster, transactions_df, accounts_df, num_samples, num_workers
            )
        else:
            # Sequential sampling (original behavior)
            trajectories = []
            for k in range(num_samples):
                trajectory = self._sample_single_trajectory(
                    cluster,
                    transactions_df.copy(deep=True),
                    accounts_df.copy(deep=True),
                    sample_id=k
                )
                trajectories.append(trajectory)

        return trajectories

    def _sample_trajectories_parallel(
        self,
        cluster: LaunderingCluster,
        transactions_df: pd.DataFrame,
        accounts_df: pd.DataFrame,
        num_samples: int,
        num_workers: int
    ) -> List[Trajectory]:
        """Sample multiple trajectories in parallel using multiprocessing.

        Args:
            cluster: LaunderingCluster object
            transactions_df: Transactions DataFrame
            accounts_df: Accounts DataFrame
            num_samples: Number of trajectories to sample
            num_workers: Number of parallel workers

        Returns:
            List of Trajectory objects
        """
        # Limit workers to num_samples
        num_workers = min(num_workers, num_samples)

        # Prepare arguments for each worker
        args_list = [
            (cluster,
             transactions_df.copy(deep=True),
             accounts_df.copy(deep=True),
             k,
             self.agent,
             self.reward_fn,
             self.config)
            for k in range(num_samples)
        ]

        # Use 'spawn' start method for CUDA compatibility
        # This creates fresh Python processes instead of forking
        ctx = get_context('spawn')
        with ctx.Pool(num_workers) as pool:
            trajectories = pool.map(_sample_trajectory_worker, args_list)

        return trajectories

    def _sample_single_trajectory(
        self,
        cluster: LaunderingCluster,
        transactions_df: pd.DataFrame,
        accounts_df: pd.DataFrame,
        sample_id: int
    ) -> Trajectory:
        """Sample a single trajectory.

        Args:
            cluster: LaunderingCluster object
            transactions_df: Transactions DataFrame (will be modified)
            accounts_df: Accounts DataFrame (will be modified)
            sample_id: Sample ID for seeding

        Returns:
            Trajectory object
        """
        
        cluster_id = cluster.cluster_id
        cluster_nodes = cluster.nodes_in_cluster
        cluster_internal_edges = cluster.laundering_edges_internal

        trajectory = Trajectory(cluster_id)

        # Track timing for this trajectory
        trajectory_start_time = time.time()
        total_candidates_time = 0.0
        total_agent_time = 0.0
        total_tool_time = 0.0
        total_reward_time = 0.0

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
        inject_count = 0

        # Main rollout loop
        while step_idx < self.config["max_steps_per_cluster"]:
            # Save state before action
            state_before = {
                "transactions_df": transactions_df.copy(deep=True),
                "accounts_df": accounts_df.copy(deep=True),
                "cluster_nodes": cluster_nodes.copy(),
            }

            # Generate candidates
            t_candidates_start = time.time()
            candidates = generate_candidates(
                transactions_df,
                accounts_df,
                cluster_nodes,
                cluster_internal_edges,
                topk=self.config["topk_candidates"],
                seed=self.config["seed"] + cluster_id * 1000 + step_idx + sample_id * 100
            )
            t_candidates_end = time.time()
            total_candidates_time += (t_candidates_end - t_candidates_start)

            # Determine allowed tools
            allowed_tools = self._determine_allowed_tools(budget, candidates)

            # Build state JSON for agent
            state_json = self._build_state_json(
                cluster,
                cluster_id,
                step_idx,
                transactions_df,
                candidates,
                budget,
                allowed_tools,
                inject_count
            )

            # Get decision from agent
            t_agent_start = time.time()
            decision = self.agent.decide_tool(state_json)
            t_agent_end = time.time()
            total_agent_time += (t_agent_end - t_agent_start)

            # Build prompt (for logging)
            from ..agent.prompt import build_prompt
            prompt = build_prompt(state_json)

            # Execute action
            if not decision["is_valid"]:
                # Invalid decision (JSON parse error, etc.)
                action_result = {
                    "ok": False,  # Mark as failed
                    "tool": "invalid",
                    "budget_used": 0,
                    "error": decision.get("error", "Invalid decision")
                }
                state_after = state_before
                is_terminal = True
            elif decision.get("tool") == "stop":
                # Valid stop decision
                action_result = {
                    "ok": True,
                    "tool": "stop",
                    "budget_used": 0,
                }
                state_after = state_before
                is_terminal = True
            else:
                # Execute tool
                tool = decision["tool"]
                args = decision["args"]

                t_tool_start = time.time()
                ok, result = self._execute_tool(
                    tool, args, transactions_df, accounts_df
                )
                t_tool_end = time.time()
                total_tool_time += (t_tool_end - t_tool_start)

                if ok:
                    # Update state
                    transactions_df = result["transactions_df"]
                    accounts_df = result["accounts_df"]

                    # Update budget
                    budget_used = 0
                    if tool == "inject_intermediary":
                        budget.use_edges(len(result.get("new_edge_ids", [])))
                        budget.use_nodes(len(result.get("created_accounts", [])))
                        inject_count += 1
                        budget_used = len(result.get("new_edge_ids", [])) + len(result.get("created_accounts", []))
                    elif tool == "merge_accounts":
                        budget.use_merge()
                        budget.use_nodes(len(result.get("created_accounts", [])))
                        budget_used = 1 + len(result.get("created_accounts", []))
                    elif tool == "split_account":
                        budget.use_split()
                        budget.use_nodes(len(result.get("created_accounts", [])))
                        budget_used = 1 + len(result.get("created_accounts", []))
                    elif tool == "adjust_transaction":
                        # Adjust doesn't create new nodes/edges, minimal budget cost
                        budget_used = len(result.get("affected_edge_ids", []))

                    budget.use_step()

                    action_result = {
                        "ok": True,
                        "tool": tool,
                        "args": args,
                        "budget_used": budget_used,
                    }
                else:
                    # Failed execution
                    action_result = {
                        "ok": False,
                        "tool": tool,
                        "args": args,
                        "budget_used": 0,
                        "error": result.get("error_message", "Unknown error"),
                    }

                state_after = {
                    "transactions_df": transactions_df.copy(deep=True),
                    "accounts_df": accounts_df.copy(deep=True),
                    "cluster_nodes": cluster_nodes.copy(),
                }

                is_terminal = False

            # Compute reward
            t_reward_start = time.time()
            reward, reward_info = self.reward_fn.compute_step_reward(
                state_before,
                state_after,
                action_result,
                is_terminal=is_terminal,
                trajectory_id=sample_id,
                step_idx=step_idx
            )
            t_reward_end = time.time()
            total_reward_time += (t_reward_end - t_reward_start)

            # Add step to trajectory
            step_data = {
                "step_idx": step_idx,
                "state": state_json,  # State representation for logging
                "action": {
                    "tool": decision.get("tool"),
                    "args": decision.get("args"),
                },
                "action_result": action_result,  # Add execution result for debugging
                "reward": reward,
                "info": reward_info,
                "prompt": prompt,
                "decision": decision,
                "is_terminal": is_terminal,
            }

            trajectory.add_step(step_data)

            # Check terminal
            if is_terminal:
                trajectory.set_final_score(reward_info.get("score_after", 1.0))
                break

            step_idx += 1

        # Ensure final_detection_score is always set
        if trajectory.final_detection_score is None:
            # Use the last step's score_after as final score
            if len(trajectory.steps) > 0:
                last_step_info = trajectory.steps[-1]["info"]
                score_after = last_step_info.get("score_after")
                # If score_after is None (failed action), use default 1.0
                trajectory.set_final_score(score_after if score_after is not None else 1.0)
            else:
                # No steps executed, set default score
                trajectory.set_final_score(1.0)

        # Print timing summary for this trajectory
        trajectory_total_time = time.time() - trajectory_start_time
        print(f"\n[Sampler] ⏱️  Trajectory {sample_id} timing breakdown:")
        print(f"  Total time: {trajectory_total_time:.2f}s")
        print(f"    - Candidates: {total_candidates_time:.2f}s ({total_candidates_time/trajectory_total_time*100:.1f}%)")
        print(f"    - Agent decisions: {total_agent_time:.2f}s ({total_agent_time/trajectory_total_time*100:.1f}%)")
        print(f"    - Tool execution: {total_tool_time:.2f}s ({total_tool_time/trajectory_total_time*100:.1f}%)")
        print(f"    - Reward computation: {total_reward_time:.2f}s ({total_reward_time/trajectory_total_time*100:.1f}%)")

        return trajectory

    def _determine_allowed_tools(self, budget: BudgetTracker, candidates: Dict) -> List[str]:
        """Determine which tools are allowed based on budget and candidates."""
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

        # Shuffle to avoid position bias
        import random
        random.shuffle(allowed)

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
        inject_count: int
    ) -> Dict[str, Any]:
        """Build state JSON for agent (same as in runner.py)."""
        cluster_nodes = cluster.nodes_in_cluster
        cluster_internal_edges = cluster.laundering_edges_internal

        # Compute cluster summary - VECTORIZED
        from collections import defaultdict
        node_in_deg = transactions_df["to_node_id"].value_counts().to_dict()
        node_out_deg = transactions_df["from_node_id"].value_counts().to_dict()

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

        return state_json

    def _execute_tool(
        self,
        tool: str,
        args: Dict,
        transactions_df: pd.DataFrame,
        accounts_df: pd.DataFrame
    ) -> Tuple[bool, Dict[str, Any]]:
        """Execute a tool with validation (same as in runner.py)."""
        snapshot = create_snapshot(transactions_df, accounts_df)

        try:
            # Precheck
            if tool == "merge_accounts":
                for pair in args["pairs"]:
                    # Handle both tuple and dict formats
                    if isinstance(pair, tuple):
                        a, b = pair
                    else:
                        a, b = pair["a"], pair["b"]
                    ok, violations = validate_merge_bank_constraint(a, b)
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
                    seed=self.config["seed"]
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
                return False, {"error_message": f"Unknown tool: {tool}"}

            # Update DataFrames
            transactions_df = result["transactions_df"]
            accounts_df = result["accounts_df"]

            # Postcheck
            ok, violations = validate_state(transactions_df, accounts_df)
            if not ok:
                transactions_df, accounts_df = rollback_to_snapshot(snapshot)
                return False, {"error_message": "Validation failed", "violations": violations}

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
                                return False, {"error_message": "Split validation failed", "violations": violations}

            # Success
            result["transactions_df"] = transactions_df
            result["accounts_df"] = accounts_df
            return True, result

        except Exception as e:
            transactions_df, accounts_df = rollback_to_snapshot(snapshot)
            return False, {"error_message": str(e)}

    def _build_allowed_params(self) -> Dict[str, Any]:
        """Build allowed parameters dictionary."""
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
                "max_edge_ids": 3,
            },
        }
