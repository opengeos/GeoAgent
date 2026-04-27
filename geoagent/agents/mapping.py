"""Mapping subagent — bound to a live ``leafmap.Map`` or ``anymap.Map``.

The mapping subagent absorbs the responsibilities of the legacy
``geoagent/core/viz_agent.py`` ``VizAgent`` class for the case where an
interactive map widget is part of the runtime context. For one-off map
rendering without a live widget, the data-side viz tools (under
:mod:`geoagent.tools.data.viz`) handle that path via the analysis
subagent.
"""

from __future__ import annotations

from typing import Any, Optional


def mapping_subagent(map_obj: Any) -> Optional[dict[str, Any]]:
    """Build the Mapping :class:`deepagents.SubAgent` spec.

    Args:
        map_obj: A live ``leafmap.Map`` (or ``leafmap.maplibregl.Map``)
            or ``anymap.Map`` instance. The subagent's tools are bound
            to it via closure. Passing ``None`` returns ``None`` so the
            caller can skip the spec.

    Returns:
        A subagent dict, or ``None`` when ``map_obj`` is ``None``.
    """
    if map_obj is None:
        return None

    from geoagent.core.prompts import MAPPING_PROMPT
    from geoagent.tools.anymap import anymap_tools
    from geoagent.tools.leafmap import leafmap_tools
    from geoagent.tools.stac import stac_tools

    is_anymap = "anymap" in type(map_obj).__module__
    base_tools = anymap_tools(map_obj) if is_anymap else leafmap_tools(map_obj)
    # stac_tools() returns [] when pystac_client is not installed, so the
    # mapping subagent stays usable even on a minimal install. When STAC
    # is available, the subagent can resolve a natural-language query
    # ("Sentinel-2 RGB over Knoxville, July 2024") into a concrete URL
    # via search_stac, then add it via add_cog_layer / add_stac_layer.
    tools = base_tools + stac_tools()

    return {
        "name": "mapping",
        "description": (
            "Manipulate the active interactive map widget: list, add, "
            "and remove layers; change the basemap; set centre / zoom; "
            "save the map to HTML. Use this whenever the user wants to "
            "see results on their existing map."
        ),
        "system_prompt": MAPPING_PROMPT,
        "tools": tools,
    }


__all__ = ["mapping_subagent"]
