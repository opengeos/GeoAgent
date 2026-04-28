"""Optional NASA Earthdata tools — placeholder."""

from __future__ import annotations

from typing import Any


def earthdata_tools() -> list[Any]:
    """Return Earthdata tools when ``earthaccess`` is installed."""
    try:
        import earthaccess  # noqa: F401
    except ImportError:
        return []
    return []
