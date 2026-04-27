"""GeoAI subagent — segmentation, object detection, classification.

Returns ``None`` when the optional ``geoai`` package is not installed,
so :func:`geoagent.agents.coordinator.default_subagents` can transparently
skip it.
"""

from __future__ import annotations

from typing import Any, Optional


def geoai_subagent() -> Optional[dict[str, Any]]:
    """Build the GeoAI :class:`deepagents.SubAgent` spec.

    Returns:
        A subagent dict, or ``None`` when ``geoai`` is unavailable.
    """
    from geoagent.tools.geoai import geoai_tools

    tools = geoai_tools()
    if not tools:
        return None

    from geoagent.core.prompts import GEOAI_PROMPT

    return {
        "name": "geoai",
        "description": (
            "Run segmentation, object detection, or whole-image "
            "classification on raster imagery via the geoai package."
        ),
        "system_prompt": GEOAI_PROMPT,
        "tools": tools,
    }


__all__ = ["geoai_subagent"]
