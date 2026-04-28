"""Optional STAC tools — stub until Phase 2 restores search/add workflows."""

from __future__ import annotations

from typing import Any


def stac_tools() -> list[Any]:
    """Return STAC-related tools when optional dependencies exist."""
    try:
        import pystac_client  # noqa: F401
    except ImportError:
        return []
    return []
