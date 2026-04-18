"""Node and edge identifier generation and parsing."""


def make_node_id(bank_id: str, account_number: str) -> str:
    """Create node ID from bank ID and account number.

    Format: {bank_id}|{account_number}
    """
    return f"{bank_id}|{account_number}"


def parse_node_id(node_id: str) -> tuple[str, str]:
    """Parse node ID into bank ID and account number.

    Args:
        node_id: Node ID in format {bank_id}|{account_number}

    Returns:
        Tuple of (bank_id, account_number)
    """
    parts = node_id.split("|", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid node_id format: {node_id}")
    return parts[0], parts[1]


def make_edge_id(row_index: int, insert_index: int = None) -> str:
    """Create edge ID.

    Args:
        row_index: Original row index in transactions
        insert_index: If this is an inserted edge, the insertion index (0-based)

    Returns:
        Edge ID in format 'row{row_index}' or 'row{row_index}_ins{insert_index}'
    """
    if insert_index is None:
        return f"row{row_index}"
    return f"row{row_index}_ins{insert_index}"
