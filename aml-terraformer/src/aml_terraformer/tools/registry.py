"""Tool registry for managing available tools."""

from typing import Dict, Any, Callable


class ToolRegistry:
    """Registry for perturbation tools."""

    def __init__(self):
        self.tools = {}

    def register(self, name: str, func: Callable):
        """Register a tool.

        Args:
            name: Tool name
            func: Tool function
        """
        self.tools[name] = func

    def get(self, name: str) -> Callable:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool function

        Raises:
            KeyError: If tool not found
        """
        if name not in self.tools:
            raise KeyError(f"Tool not found: {name}")
        return self.tools[name]

    def list_tools(self) -> list:
        """List available tool names.

        Returns:
            List of tool names
        """
        return list(self.tools.keys())


def create_default_registry() -> ToolRegistry:
    """Create registry with default tools.

    Returns:
        ToolRegistry with inject_intermediary, merge_accounts, split_account
    """
    from .inject import inject_intermediary
    from .merge import merge_accounts
    from .split import split_account

    registry = ToolRegistry()
    registry.register("inject_intermediary", inject_intermediary)
    registry.register("merge_accounts", merge_accounts)
    registry.register("split_account", split_account)

    return registry
