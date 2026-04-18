"""Sanitize and validate LLM decisions."""

from typing import Dict, Any, List


def sanitize_decision(decision: Dict[str, Any], state_json: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize and validate LLM decision.

    Args:
        decision: Parsed decision from LLM
        state_json: Current state with candidates and allowed_tools

    Returns:
        Dictionary with:
        - is_valid: Whether decision is valid
        - invalid_reason: Reason if invalid
        - tool: Tool name (if valid)
        - args: Tool arguments (if valid)
        - rationale: Rationale (if present)
    """
    # Check required fields
    if "tool" not in decision:
        return {
            "is_valid": False,
            "invalid_reason": "Missing 'tool' field in decision",
        }

    tool = decision["tool"]
    args = decision.get("args", {})
    rationale = decision.get("rationale", "")

    # Get allowed tools and candidates
    allowed_tools = state_json.get("allowed_tools", [])
    candidates = state_json.get("candidates", {})
    allowed_params = state_json.get("allowed_params", {})

    # Check if tool is allowed
    if tool != "stop" and tool not in allowed_tools:
        return {
            "is_valid": False,
            "invalid_reason": f"Tool '{tool}' not in allowed_tools: {allowed_tools}",
        }

    # Validate stop
    if tool == "stop":
        return {
            "is_valid": True,
            "tool": "stop",
            "args": {},
            "rationale": rationale,
        }

    # Validate inject_intermediary
    if tool == "inject_intermediary":
        return validate_inject(args, candidates, allowed_params, rationale)

    # Validate merge_accounts
    if tool == "merge_accounts":
        return validate_merge(args, candidates, allowed_params, rationale)

    # Validate split_account
    if tool == "split_account":
        return validate_split(args, candidates, allowed_params, rationale)

    # Validate adjust_transaction
    if tool == "adjust_transaction":
        return validate_adjust(args, candidates, allowed_params, rationale)

    return {
        "is_valid": False,
        "invalid_reason": f"Unknown tool: {tool}",
    }


def validate_inject(args: Dict, candidates: Dict, allowed_params: Dict, rationale: str) -> Dict[str, Any]:
    """Validate inject_intermediary arguments."""
    inject_cands = candidates.get("inject_candidates", [])
    inject_params = allowed_params.get("inject", {})

    # Check required args
    if "edge_ids" not in args:
        return {"is_valid": False, "invalid_reason": "Missing 'edge_ids' in args"}
    if "depth" not in args:
        return {"is_valid": False, "invalid_reason": "Missing 'depth' in args"}
    if "time_delta_seconds" not in args:
        return {"is_valid": False, "invalid_reason": "Missing 'time_delta_seconds' in args"}

    edge_ids = args["edge_ids"]
    depth = args["depth"]
    time_delta = args["time_delta_seconds"]

    # Validate types
    if not isinstance(edge_ids, list):
        return {"is_valid": False, "invalid_reason": "'edge_ids' must be a list"}

    # Check length
    max_edge_ids = inject_params.get("max_edge_ids", 3)
    if len(edge_ids) == 0 or len(edge_ids) > max_edge_ids:
        return {"is_valid": False, "invalid_reason": f"edge_ids length must be 1..{max_edge_ids}"}

    # Check each edge_id is in candidates
    valid_edge_ids = set(c["edge_id"] for c in inject_cands)
    for eid in edge_ids:
        if eid not in valid_edge_ids:
            return {"is_valid": False, "invalid_reason": f"edge_id '{eid}' not in inject_candidates"}

    # Check depth
    allowed_depths = inject_params.get("depth", [1, 2])
    if depth not in allowed_depths:
        return {"is_valid": False, "invalid_reason": f"depth must be in {allowed_depths}"}

    # Check time_delta
    allowed_deltas = inject_params.get("time_delta_seconds", [1, 5, 60])
    if time_delta not in allowed_deltas:
        return {"is_valid": False, "invalid_reason": f"time_delta_seconds must be in {allowed_deltas}"}

    return {
        "is_valid": True,
        "tool": "inject_intermediary",
        "args": args,
        "rationale": rationale,
    }


def validate_merge(args: Dict, candidates: Dict, allowed_params: Dict, rationale: str) -> Dict[str, Any]:
    """Validate merge_accounts arguments."""
    merge_cands = candidates.get("merge_candidates", [])
    merge_params = allowed_params.get("merge", {})

    # Check required args
    if "pairs" not in args:
        return {"is_valid": False, "invalid_reason": "Missing 'pairs' in args"}
    if "drop_self_loops" not in args:
        return {"is_valid": False, "invalid_reason": "Missing 'drop_self_loops' in args"}

    pairs = args["pairs"]
    drop_self_loops = args["drop_self_loops"]

    # Validate types
    if not isinstance(pairs, list):
        return {"is_valid": False, "invalid_reason": "'pairs' must be a list"}

    # Check length
    max_pairs = merge_params.get("max_pairs", 2)
    if len(pairs) == 0 or len(pairs) > max_pairs:
        return {"is_valid": False, "invalid_reason": f"pairs length must be 1..{max_pairs}"}

    # Check each pair is in candidates
    valid_pairs = set()
    for c in merge_cands:
        # Create unordered pair
        pair = tuple(sorted([c["a"], c["b"]]))
        valid_pairs.add(pair)

    for p in pairs:
        if not isinstance(p, dict) or "a" not in p or "b" not in p:
            return {"is_valid": False, "invalid_reason": "Each pair must have 'a' and 'b' fields"}

        pair = tuple(sorted([p["a"], p["b"]]))
        if pair not in valid_pairs:
            return {"is_valid": False, "invalid_reason": f"pair {p} not in merge_candidates"}

    # Convert pairs to tuples for tool
    pairs_tuples = [(p["a"], p["b"]) for p in pairs]

    # Check drop_self_loops
    allowed_drop = merge_params.get("drop_self_loops", [False, True])
    if drop_self_loops not in allowed_drop:
        return {"is_valid": False, "invalid_reason": f"drop_self_loops must be in {allowed_drop}"}

    return {
        "is_valid": True,
        "tool": "merge_accounts",
        "args": {"pairs": pairs_tuples, "drop_self_loops": drop_self_loops},
        "rationale": rationale,
    }


def validate_split(args: Dict, candidates: Dict, allowed_params: Dict, rationale: str) -> Dict[str, Any]:
    """Validate split_account arguments."""
    split_cands = candidates.get("split_candidates", [])
    split_params = allowed_params.get("split", {})

    # Check required args
    required = ["node_ids", "split_ratio", "move_direction", "edge_sampling"]
    for field in required:
        if field not in args:
            return {"is_valid": False, "invalid_reason": f"Missing '{field}' in args"}

    node_ids = args["node_ids"]
    split_ratio = args["split_ratio"]
    move_direction = args["move_direction"]
    edge_sampling = args["edge_sampling"]

    # Validate types
    if not isinstance(node_ids, list):
        return {"is_valid": False, "invalid_reason": "'node_ids' must be a list"}

    # Check length
    max_node_ids = split_params.get("max_node_ids", 2)
    if len(node_ids) == 0 or len(node_ids) > max_node_ids:
        return {"is_valid": False, "invalid_reason": f"node_ids length must be 1..{max_node_ids}"}

    # Check each node_id is in candidates
    valid_node_ids = set(c["node_id"] for c in split_cands)
    for nid in node_ids:
        if nid not in valid_node_ids:
            return {"is_valid": False, "invalid_reason": f"node_id '{nid}' not in split_candidates"}

    # Check split_ratio
    allowed_ratios = split_params.get("split_ratio", [0.2, 0.3, 0.4])
    if split_ratio not in allowed_ratios:
        return {"is_valid": False, "invalid_reason": f"split_ratio must be in {allowed_ratios}"}

    # Check move_direction
    allowed_directions = split_params.get("move_direction", ["out", "in", "both"])
    if move_direction not in allowed_directions:
        return {"is_valid": False, "invalid_reason": f"move_direction must be in {allowed_directions}"}

    # Check edge_sampling
    allowed_sampling = split_params.get("edge_sampling", ["time_stratified", "random_within_currency", "random"])
    if edge_sampling not in allowed_sampling:
        return {"is_valid": False, "invalid_reason": f"edge_sampling must be in {allowed_sampling}"}

    return {
        "is_valid": True,
        "tool": "split_account",
        "args": args,
        "rationale": rationale,
    }


def validate_adjust(args: Dict, candidates: Dict, allowed_params: Dict, rationale: str) -> Dict[str, Any]:
    """Validate adjust_transaction arguments."""
    adjust_cands = candidates.get("adjust_candidates", [])
    adjust_params = allowed_params.get("adjust", {})

    # Check required args
    required = ["edge_ids", "time_offset_seconds", "amount_multiplier"]
    for field in required:
        if field not in args:
            return {"is_valid": False, "invalid_reason": f"Missing '{field}' in args"}

    edge_ids = args["edge_ids"]
    time_offset_seconds = args["time_offset_seconds"]
    amount_multiplier = args["amount_multiplier"]

    # Validate types
    if not isinstance(edge_ids, list):
        return {"is_valid": False, "invalid_reason": "'edge_ids' must be a list"}

    if not isinstance(time_offset_seconds, (int, float)):
        return {"is_valid": False, "invalid_reason": "'time_offset_seconds' must be a number"}

    if not isinstance(amount_multiplier, (int, float)):
        return {"is_valid": False, "invalid_reason": "'amount_multiplier' must be a number"}

    # Check length
    max_edge_ids = adjust_params.get("max_edge_ids", 3)
    if len(edge_ids) == 0 or len(edge_ids) > max_edge_ids:
        return {"is_valid": False, "invalid_reason": f"edge_ids length must be 1..{max_edge_ids}"}

    # Check each edge_id is in candidates
    valid_edge_ids = set(c["edge_id"] for c in adjust_cands)
    for eid in edge_ids:
        if eid not in valid_edge_ids:
            return {"is_valid": False, "invalid_reason": f"edge_id '{eid}' not in adjust_candidates"}

    # Validate amount_multiplier range (should be positive)
    if amount_multiplier <= 0:
        return {"is_valid": False, "invalid_reason": "amount_multiplier must be positive"}

    return {
        "is_valid": True,
        "tool": "adjust_transaction",
        "args": args,
        "rationale": rationale,
    }
