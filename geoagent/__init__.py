"""GeoAgent - centralized AI agent framework for Open Geospatial tools."""

__author__ = """Qiusheng Wu"""
__email__ = "giswqs@gmail.com"
__version__ = "1.0.0"

# LLM provider factory (always available)
from geoagent.core.llm import get_llm, get_default_llm, resolve_model

# New v1 surface
from geoagent.core.context import GeoAgentContext
from geoagent.core.decorators import (
    geo_tool,
    get_geo_meta,
    needs_confirmation,
)
from geoagent.core.result import GeoAgentResponse
from geoagent.core.safety import (
    ConfirmCallback,
    ConfirmRequest,
    auto_approve_all,
    auto_approve_safe_only,
    build_interrupt_on,
)

# Factory functions are lazily importable so deepagents missing doesn't
# break `import geoagent`.
try:
    from geoagent.core.factory import (
        create_geo_agent,
        for_anymap,
        for_leafmap,
        for_qgis,
    )
except ImportError:  # pragma: no cover - exercised when deepagents missing
    create_geo_agent = None  # type: ignore[assignment]
    for_anymap = None  # type: ignore[assignment]
    for_leafmap = None  # type: ignore[assignment]
    for_qgis = None  # type: ignore[assignment]

# Backward-compatible legacy GeoAgent class (Phase 2 will rebuild this on
# top of the deepagents factory).
try:
    from geoagent.core.agent import GeoAgent
except ImportError:
    GeoAgent = None  # type: ignore[assignment]

# Legacy v0.x tool re-exports (preserve names from the previous public API).
try:
    from geoagent.core.tools.stac import search_stac, get_stac_collections
    from geoagent.core.tools.duckdb_tool import (
        analyze_spatial_data,
        query_overture,
        query_spatial_data,
    )
    from geoagent.core.tools.raster import (
        compute_index,
        load_raster,
        raster_to_array,
        zonal_stats,
    )
    from geoagent.core.tools.vector import (
        analyze_geometries,
        buffer_analysis,
        read_vector,
        spatial_filter,
        spatial_join,
    )
    from geoagent.core.tools.viz import (
        add_cog_layer,
        add_pmtiles_layer,
        add_vector_layer,
        create_3d_terrain_map,
        create_choropleth_map,
        save_map,
        show_on_map,
        split_map,
    )
except ImportError:
    pass

__all__ = [
    # Versioning
    "__version__",
    # New v1 surface
    "create_geo_agent",
    "for_leafmap",
    "for_anymap",
    "for_qgis",
    "GeoAgentContext",
    "GeoAgentResponse",
    "geo_tool",
    "get_geo_meta",
    "needs_confirmation",
    "ConfirmCallback",
    "ConfirmRequest",
    "auto_approve_all",
    "auto_approve_safe_only",
    "build_interrupt_on",
    # LLM providers
    "get_llm",
    "get_default_llm",
    "resolve_model",
    # Backward-compat legacy class
    "GeoAgent",
    # Legacy tool functions
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
