"""Timestamp parsing and formatting utilities."""

from datetime import datetime
from typing import Union


def parse_timestamp_to_int_seconds(ts: Union[str, int, float]) -> int:
    """Parse timestamp to integer seconds.

    Supports:
    - Unix seconds (int)
    - Unix milliseconds (int > 1e12, converted to seconds)
    - Datetime strings (ISO format or common formats)

    Returns:
        Integer seconds since Unix epoch
    """
    if isinstance(ts, int):
        # If > 1e12, treat as milliseconds
        if ts > 1e12:
            return int(ts / 1000)
        return ts

    if isinstance(ts, float):
        # If > 1e12, treat as milliseconds
        if ts > 1e12:
            return int(ts / 1000)
        return int(ts)

    # String: try to parse as datetime
    if isinstance(ts, str):
        # Try common formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",  # Support format without seconds
            "%m/%d/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(ts, fmt)
                return int(dt.timestamp())
            except ValueError:
                continue

        # Try as unix timestamp string
        try:
            val = float(ts)
            if val > 1e12:
                return int(val / 1000)
            return int(val)
        except ValueError:
            pass

        raise ValueError(f"Unable to parse timestamp: {ts}")

    raise TypeError(f"Unsupported timestamp type: {type(ts)}")


def format_timestamp(ts_int: int, format_type: str = "iso", original_value: Union[str, int, float] = None) -> str:
    """Format integer seconds timestamp to string.

    Args:
        ts_int: Integer seconds since Unix epoch
        format_type: One of 'iso', 'unix', 'original'
        original_value: If format_type='original', try to match original format

    Returns:
        Formatted timestamp string
    """
    if format_type == "unix":
        return str(ts_int)

    elif format_type == "iso":
        dt = datetime.fromtimestamp(ts_int)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    elif format_type == "original":
        if original_value is None:
            # Fallback to ISO
            dt = datetime.fromtimestamp(ts_int)
            return dt.strftime("%Y-%m-%d %H:%M:%S")

        # Try to detect original format
        if isinstance(original_value, int):
            if original_value > 1e12:
                return str(ts_int * 1000)
            return str(ts_int)

        if isinstance(original_value, str):
            # Try to detect format from original string
            if 'T' in original_value:
                dt = datetime.fromtimestamp(ts_int)
                if original_value.endswith('Z'):
                    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                return dt.strftime("%Y-%m-%dT%H:%M:%S")
            elif '/' in original_value:
                dt = datetime.fromtimestamp(ts_int)
                # Check if original has seconds (by counting colons or checking length)
                # Format: "2022/09/01 00:20" (16 chars, no seconds) vs "2022/09/01 00:20:30" (19 chars, with seconds)
                if original_value.count(':') >= 2 or len(original_value) >= 19:
                    # Has seconds
                    return dt.strftime("%Y/%m/%d %H:%M:%S")
                else:
                    # No seconds
                    return dt.strftime("%Y/%m/%d %H:%M")
            else:
                dt = datetime.fromtimestamp(ts_int)
                return dt.strftime("%Y-%m-%d %H:%M:%S")

        # Fallback
        dt = datetime.fromtimestamp(ts_int)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    else:
        raise ValueError(f"Unsupported format_type: {format_type}")
