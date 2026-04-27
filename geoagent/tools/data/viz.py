"""Data-side visualisation tools (map snapshot helpers).

Wraps the v0.x visualisation helpers (:mod:`geoagent.core.tools.viz`) with
GeoAgent metadata. These differ from the live-map control tools in
:mod:`geoagent.tools.leafmap`: viz tools build a new map for one-shot
rendering, whereas the leafmap tools mutate an existing map widget.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from geoagent.tools.stac import _stamp


def viz_tools() -> list[BaseTool]:
    """Build the viz tool set.

    Returns:
        ``show_on_map``, ``add_cog_layer``, ``add_vector_layer``,
        ``split_map``, ``create_choropleth_map``, ``add_pmtiles_layer``,
        ``create_3d_terrain_map``, ``save_map``.
    """
    try:
        from geoagent.core.tools.viz import (
            show_on_map,
            add_cog_layer,
            add_vector_layer,
            split_map,
            create_choropleth_map,
            add_pmtiles_layer,
            create_3d_terrain_map,
            save_map,
        )
    except ImportError:
        return []

    requirements = ["leafmap"]
    safe_tools = (
        show_on_map,
        add_cog_layer,
        add_vector_layer,
        split_map,
        create_choropleth_map,
        add_pmtiles_layer,
        create_3d_terrain_map,
    )
    for tool in safe_tools:
        _stamp(
            tool,
            category="map",
            requires_confirmation=False,
            requires_packages=requirements,
        )
    _stamp(
        save_map,
        category="io",
        requires_confirmation=True,
        requires_packages=requirements,
    )
    return list(safe_tools) + [save_map]


__all__ = ["viz_tools"]
