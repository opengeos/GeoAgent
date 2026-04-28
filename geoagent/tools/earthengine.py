"""Optional Earth Engine / geemap tools — placeholder."""

from __future__ import annotations

from typing import Any


def earthengine_tools() -> list[Any]:
    """Return EE tools when ``earthengine-api`` is available."""
    try:
        import ee  # noqa: F401
    except ImportError:
        return []
    return []
