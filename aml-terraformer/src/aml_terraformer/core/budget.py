"""Budget tracking for perturbation operations."""


class BudgetTracker:
    """Track and manage operation budgets for a cluster."""

    def __init__(
        self,
        max_steps: int,
        max_merges: int,
        max_splits: int,
        max_new_nodes: int,
        max_new_edges: int
    ):
        """Initialize budget tracker.

        Args:
            max_steps: Maximum number of steps
            max_merges: Maximum number of merge operations
            max_splits: Maximum number of split operations
            max_new_nodes: Maximum number of new nodes to create
            max_new_edges: Maximum number of new edges to create
        """
        self.max_steps = max_steps
        self.max_merges = max_merges
        self.max_splits = max_splits
        self.max_new_nodes = max_new_nodes
        self.max_new_edges = max_new_edges

        # Current usage
        self.steps_used = 0
        self.merges_used = 0
        self.splits_used = 0
        self.new_nodes_used = 0
        self.new_edges_used = 0

    def get_remaining(self) -> dict:
        """Get remaining budget.

        Returns:
            Dictionary with remaining amounts
        """
        return {
            "steps_left": self.max_steps - self.steps_used,
            "merges_left": self.max_merges - self.merges_used,
            "splits_left": self.max_splits - self.splits_used,
            "new_nodes_left": self.max_new_nodes - self.new_nodes_used,
            "new_edges_left": self.max_new_edges - self.new_edges_used,
        }

    def can_inject(self, num_edges: int) -> bool:
        """Check if inject operation is allowed.

        Args:
            num_edges: Number of new edges to create

        Returns:
            True if operation is within budget
        """
        return self.new_edges_used + num_edges <= self.max_new_edges

    def can_merge(self) -> bool:
        """Check if merge operation is allowed."""
        return self.merges_used < self.max_merges

    def can_split(self) -> bool:
        """Check if split operation is allowed."""
        return self.splits_used < self.max_splits

    def can_create_nodes(self, num_nodes: int) -> bool:
        """Check if creating nodes is allowed.

        Args:
            num_nodes: Number of new nodes to create

        Returns:
            True if operation is within budget
        """
        return self.new_nodes_used + num_nodes <= self.max_new_nodes

    def use_step(self):
        """Increment step counter."""
        self.steps_used += 1

    def use_merge(self):
        """Increment merge counter."""
        self.merges_used += 1

    def use_split(self):
        """Increment split counter."""
        self.splits_used += 1

    def use_nodes(self, count: int):
        """Increment new nodes counter."""
        self.new_nodes_used += count

    def use_edges(self, count: int):
        """Increment new edges counter."""
        self.new_edges_used += count

    def get_history(self) -> dict:
        """Get operation history.

        Returns:
            Dictionary with operation counts
        """
        return {
            "inject_count": 0,  # Will be tracked separately in pipeline
            "merge_count": self.merges_used,
            "split_count": self.splits_used,
        }
