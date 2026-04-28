"""Optional geoai tools — placeholder."""

from __future__ import annotations

from typing import Any


def geoai_tools() -> list[Any]:
    """Return GeoAI tools when ``geoai`` is installed."""
    try:
        import geoai  # noqa: F401
    except ImportError:
        return []
    return []
