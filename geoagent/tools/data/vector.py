"""Vector analysis tools.

Wraps the v0.x vector tools (:mod:`geoagent.core.tools.vector`) with
GeoAgent metadata.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from geoagent.tools.stac import _stamp


def vector_tools() -> list[BaseTool]:
    """Build the vector tool set.

    Returns:
        ``read_vector``, ``spatial_filter``, ``buffer_analysis``,
        ``spatial_join``, ``analyze_geometries``.
    """
    try:
        from geoagent.core.tools.vector import (
            read_vector,
            spatial_filter,
            buffer_analysis,
            spatial_join,
            analyze_geometries,
        )
    except ImportError:
        return []

    requirements = ["geopandas", "shapely"]
    tools = (
        read_vector,
        spatial_filter,
        buffer_analysis,
        spatial_join,
        analyze_geometries,
    )
    for tool in tools:
        _stamp(
            tool,
            category="data",
            requires_confirmation=False,
            requires_packages=requirements,
        )
    return list(tools)


__all__ = ["vector_tools"]
