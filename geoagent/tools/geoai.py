"""Optional geoai tools — placeholder.

The geoai integrations are deferred and will return in a later milestone.
:func:`geoai_tools` always returns ``[]`` regardless of whether ``geoai``
is installed; the import probe is kept so the placeholder can later be
fleshed out without changing the call site.
"""

from __future__ import annotations

from typing import Any


def geoai_tools() -> list[Any]:
    """Return ``[]`` (placeholder until geoai tools are reintroduced).

    The empty list is intentional even when ``geoai`` imports successfully;
    treat this factory as a stub. See module docstring.
    """
    try:
        import geoai  # noqa: F401
    except ImportError:
        return []
    return []
