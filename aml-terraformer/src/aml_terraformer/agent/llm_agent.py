"""LLM Agent for decision making."""

import json
import re
from typing import Dict, Any, Optional
from .client_interface import LLMClient
from .prompt import build_prompt
from .sanitize import sanitize_decision


def extract_json_from_text(text: str) -> Optional[str]:
    """Extract JSON object from text that may contain additional content.

    Args:
        text: Raw text that may contain JSON plus extra text

    Returns:
        Extracted JSON string, or None if not found
    """
    # Try to find JSON object using regex
    # Look for pattern: {...}
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)

    if not matches:
        return None

    # Try each match to see if it's valid JSON
    for match in matches:
        try:
            # Test if it's valid JSON
            json.loads(match)
            return match
        except json.JSONDecodeError:
            continue

    return None


class LLMAgent:
    """LLM-based agent for perturbation decisions."""

    def __init__(self, client: LLMClient):
        """Initialize LLM agent.

        Args:
            client: LLM client implementation
        """
        self.client = client

    def decide_tool(self, state_json: Dict[str, Any]) -> Dict[str, Any]:
        """Decide which tool to use based on state.

        Args:
            state_json: Current state including candidates, budget, etc.

        Returns:
            Dictionary with:
            - decision_raw: Raw LLM output
            - decision_parsed: Parsed decision (if valid)
            - is_valid: Whether decision is valid
            - invalid_reason: Reason if invalid
            - tool: Tool name (if valid)
            - args: Tool arguments (if valid)
            - rationale: Rationale (if present)
        """
        # Build prompt
        prompt = build_prompt(state_json)

        # Get LLM response
        try:
            raw_output = self.client.complete(prompt)
        except Exception as e:
            return {
                "decision_raw": str(e),
                "decision_parsed": None,
                "is_valid": False,
                "invalid_reason": f"LLM API error: {e}",
                "tool": None,
                "args": None,
                "rationale": None,
            }

        # Parse and sanitize
        try:
            # First, try to parse raw output directly
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as e:
            # If direct parsing fails, try to extract JSON from text
            json_str = extract_json_from_text(raw_output)
            if json_str:
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError as e2:
                    return {
                        "decision_raw": raw_output,
                        "decision_parsed": None,
                        "is_valid": False,
                        "invalid_reason": f"Invalid JSON after extraction: {e2}",
                        "tool": None,
                        "args": None,
                        "rationale": None,
                    }
            else:
                return {
                    "decision_raw": raw_output,
                    "decision_parsed": None,
                    "is_valid": False,
                    "invalid_reason": f"Invalid JSON: {e}",
                    "tool": None,
                    "args": None,
                    "rationale": None,
                }

        # Handle case where LLM returns an array instead of object
        if isinstance(parsed, list):
            if len(parsed) > 0 and isinstance(parsed[0], dict):
                # Take the first element if it's a valid decision
                parsed = parsed[0]
            else:
                return {
                    "decision_raw": raw_output,
                    "decision_parsed": None,
                    "is_valid": False,
                    "invalid_reason": "LLM returned array instead of object",
                    "tool": None,
                    "args": None,
                    "rationale": None,
                }

        # Sanitize decision
        sanitized = sanitize_decision(parsed, state_json)

        return {
            "decision_raw": raw_output,
            "decision_parsed": parsed,
            "is_valid": sanitized["is_valid"],
            "invalid_reason": sanitized.get("invalid_reason"),
            "tool": sanitized.get("tool"),
            "args": sanitized.get("args"),
            "rationale": sanitized.get("rationale"),
        }
