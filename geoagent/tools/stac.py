"""Optional STAC tools — stub until Phase 2 restores search/add workflows.

STAC search/add helpers are deferred and will return in a later milestone.
:func:`stac_tools` always returns ``[]`` regardless of whether
``pystac-client`` is installed; the import probe is kept so the placeholder
can be evolved into a real factory without changing the call site.
"""

from __future__ import annotations

from typing import Any


def stac_tools() -> list[Any]:
    """Return ``[]`` (placeholder until STAC tools are reintroduced).

    The empty list is intentional even when ``pystac_client`` imports
    successfully; treat this factory as a stub. See module docstring.
    """
    try:
        import pystac_client  # noqa: F401
    except ImportError:
        return []
    return []
