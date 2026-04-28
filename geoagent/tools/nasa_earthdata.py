"""Optional NASA Earthdata tools — placeholder.

Earthdata integrations are deferred and will return in a later milestone.
:func:`earthdata_tools` always returns ``[]`` regardless of whether
``earthaccess`` is installed; the import probe is kept so the placeholder
can be evolved into a real factory without changing the call site.
"""

from __future__ import annotations

from typing import Any


def earthdata_tools() -> list[Any]:
    """Return ``[]`` (placeholder until Earthdata tools are reintroduced).

    The empty list is intentional even when ``earthaccess`` imports
    successfully; treat this factory as a stub. See module docstring.
    """
    try:
        import earthaccess  # noqa: F401
    except ImportError:
        return []
    return []
