"""NASA Earthdata subagent — search and download CMR granules.

Returns ``None`` when ``earthaccess`` is not installed.
"""

from __future__ import annotations

from typing import Any, Optional


def earthdata_subagent() -> Optional[dict[str, Any]]:
    """Build the NASA Earthdata :class:`deepagents.SubAgent` spec.

    Returns:
        A subagent dict, or ``None`` when ``earthaccess`` is unavailable.
    """
    from geoagent.tools.nasa_earthdata import earthdata_tools

    tools = earthdata_tools()
    if not tools:
        return None

    from geoagent.core.prompts import EARTHDATA_PROMPT

    return {
        "name": "earthdata",
        "description": (
            "Search NASA's CMR for granules and (with explicit user "
            "approval) download them via earthaccess."
        ),
        "system_prompt": EARTHDATA_PROMPT,
        "tools": tools,
    }


__all__ = ["earthdata_subagent"]
