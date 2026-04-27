"""Raster analysis tools.

Wraps the v0.x raster tools (:mod:`geoagent.core.tools.raster`) with GeoAgent
metadata so they show up in the new registry under ``category="data"``.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from geoagent.tools.stac import _stamp


def raster_tools() -> list[BaseTool]:
    """Build the raster tool set.

    Returns:
        ``load_raster``, ``compute_index``, ``raster_to_array``,
        ``zonal_stats`` — each a LangChain ``BaseTool``.
    """
    try:
        from geoagent.core.tools.raster import (
            load_raster,
            compute_index,
            raster_to_array,
            zonal_stats,
        )
    except ImportError:
        return []

    requirements = ["rasterio", "rioxarray", "xarray"]
    for tool in (load_raster, compute_index, raster_to_array, zonal_stats):
        _stamp(
            tool,
            category="data",
            requires_confirmation=False,
            requires_packages=requirements,
        )

    return [load_raster, compute_index, raster_to_array, zonal_stats]


__all__ = ["raster_tools"]
