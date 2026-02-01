"""GeoAgent - An AI agent for geospatial data analysis and visualization."""

__author__ = """Qiusheng Wu"""
__email__ = "giswqs@gmail.com"
__version__ = "0.1.0"

from geoagent.core.llm import get_llm, get_default_llm

try:
    from geoagent.core.agent import GeoAgent
except ImportError:
    GeoAgent = None

# Import tool functions for direct use
try:
    from geoagent.core.tools.stac import search_stac, get_stac_collections
    from geoagent.core.tools.duckdb_tool import (
        query_spatial_data,
        query_overture,
        analyze_spatial_data,
    )
    from geoagent.core.tools.raster import (
        load_raster,
        compute_index,
        raster_to_array,
        zonal_stats,
    )
    from geoagent.core.tools.vector import (
        read_vector,
        spatial_filter,
        buffer_analysis,
        spatial_join,
        analyze_geometries,
    )
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
    # Tools may not be available if dependencies are missing
    pass

__all__ = [
    "GeoAgent",
    "get_llm",
    "get_default_llm",
    # Tool functions
    "search_stac",
    "get_stac_collections",
    "query_spatial_data",
    "query_overture",
    "analyze_spatial_data",
    "load_raster",
    "compute_index",
    "raster_to_array",
    "zonal_stats",
    "read_vector",
    "spatial_filter",
    "buffer_analysis",
    "spatial_join",
    "analyze_geometries",
    "show_on_map",
    "add_cog_layer",
    "add_vector_layer",
    "split_map",
    "create_choropleth_map",
    "add_pmtiles_layer",
    "create_3d_terrain_map",
    "save_map",
]
