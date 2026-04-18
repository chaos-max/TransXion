"""Deterministic fallback logic."""

from typing import Dict, Any, List


def get_deterministic_fallback(
    allowed_tools: List[str],
    candidates: Dict[str, List],
    allowed_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Get deterministic fallback decision.

    Priority:
    1. inject (if allowed and candidates available)
    2. split (if allowed and candidates available)
    3. merge (if allowed and candidates available)
    4. stop

    Args:
        allowed_tools: List of allowed tool names
        candidates: Dictionary of candidates
        allowed_params: Dictionary of allowed parameters

    Returns:
        Dictionary with:
        - tool: Tool name
        - args: Tool arguments
        - rationale: Rationale
        - fallback_used: True
    """
    inject_cands = candidates.get("inject_candidates", [])
    merge_cands = candidates.get("merge_candidates", [])
    split_cands = candidates.get("split_candidates", [])

    # Try inject first
    if "inject_intermediary" in allowed_tools and len(inject_cands) > 0:
        first_edge = inject_cands[0]["edge_id"]
        return {
            "tool": "inject_intermediary",
            "args": {
                "edge_ids": [first_edge],
                "depth": 1,
                "time_delta_seconds": 1,
            },
            "rationale": "Deterministic fallback: inject first candidate",
            "fallback_used": True,
        }

    # Try split
    if "split_account" in allowed_tools and len(split_cands) > 0:
        first_node = split_cands[0]["node_id"]
        return {
            "tool": "split_account",
            "args": {
                "node_ids": [first_node],
                "split_ratio": 0.2,
                "move_direction": "out",
                "edge_sampling": "random",
            },
            "rationale": "Deterministic fallback: split first candidate",
            "fallback_used": True,
        }

    # Try merge
    if "merge_accounts" in allowed_tools and len(merge_cands) > 0:
        first_pair = merge_cands[0]
        return {
            "tool": "merge_accounts",
            "args": {
                "pairs": [(first_pair["a"], first_pair["b"])],
                "drop_self_loops": False,
            },
            "rationale": "Deterministic fallback: merge first candidate pair",
            "fallback_used": True,
        }

    # Stop
    return {
        "tool": "stop",
        "args": {},
        "rationale": "Deterministic fallback: no tools available",
        "fallback_used": True,
    }
