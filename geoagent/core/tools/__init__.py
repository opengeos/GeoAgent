"""GeoAgent tools module.

This module contains all the LangChain tool functions for geospatial analysis.
"""

from .stac import search_stac, get_stac_collections
from .duckdb_tool import query_spatial_data, query_overture, analyze_spatial_data
from .raster import load_raster, compute_index, raster_to_array, zonal_stats
from .vector import (
    read_vector,
    spatial_filter,
    buffer_analysis,
    spatial_join,
    analyze_geometries,
)
from .viz import (
    show_on_map,
    add_cog_layer,
    add_vector_layer,
    split_map,
    create_choropleth_map,
    add_pmtiles_layer,
    create_3d_terrain_map,
    save_map,
)

__all__ = [
    # STAC tools
    "search_stac",
    "get_stac_collections",
    # Spatial SQL tools
    "query_spatial_data",
    "query_overture",
    "analyze_spatial_data",
    # Raster tools
    "load_raster",
    "compute_index",
    "raster_to_array",
    "zonal_stats",
    # Vector tools
    "read_vector",
    "spatial_filter",
    "buffer_analysis",
    "spatial_join",
    "analyze_geometries",
    # Visualization tools (MapLibre backend)
    "show_on_map",
    "add_cog_layer",
    "add_vector_layer",
    "split_map",
    "create_choropleth_map",
    "add_pmtiles_layer",
    "create_3d_terrain_map",
    "save_map",
]
