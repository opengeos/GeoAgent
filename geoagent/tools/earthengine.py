"""Optional Earth Engine / geemap tools — placeholder.

Earth Engine integrations are deferred and will return in a later milestone.
:func:`earthengine_tools` always returns ``[]`` regardless of whether
``earthengine-api`` is installed; the import probe is kept so the placeholder
can be evolved into a real factory without changing the call site.
"""

from __future__ import annotations

from typing import Any


def earthengine_tools() -> list[Any]:
    """Return ``[]`` (placeholder until Earth Engine tools are reintroduced).

    The empty list is intentional even when ``ee`` imports successfully;
    treat this factory as a stub. See module docstring.
    """
    try:
        import ee  # noqa: F401
    except ImportError:
        return []
    return []
