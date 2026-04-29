"""GeoAgent — centralized geospatial agent layer (Strands Agents, 1.0)."""

__author__ = """Qiusheng Wu"""
__email__ = "giswqs@gmail.com"
__version__ = "1.1.1"

from geoagent.core.config import GeoAgentConfig
from geoagent.core.context import GeoAgentContext
from geoagent.core.decorators import (
    geo_tool,
    get_geo_meta,
    needs_confirmation,
    stamp_geo_meta,
)
from geoagent.core.model import get_default_model, get_llm, resolve_model
from geoagent.core.result import GeoAgentResponse
from geoagent.core.safety import (
    ConfirmCallback,
    ConfirmRequest,
    auto_approve_all,
    auto_approve_safe_only,
    build_interrupt_on,
)
from geoagent.core.factory import (
    create_agent,
    for_anymap,
    for_gee_data_catalogs,
    for_leafmap,
    for_nasa_earthdata,
    for_nasa_opera,
    for_qgis,
    for_whitebox,
)
from geoagent.core.agent import GeoAgent
from geoagent.core.registry import GeoToolMeta, GeoToolRegistry

__all__ = [
    "__version__",
    "GeoAgent",
    "GeoAgentConfig",
    "GeoAgentContext",
    "GeoAgentResponse",
    "GeoToolMeta",
    "GeoToolRegistry",
    "create_agent",
    "for_anymap",
    "for_gee_data_catalogs",
    "for_leafmap",
    "for_nasa_earthdata",
    "for_nasa_opera",
    "for_qgis",
    "for_whitebox",
    "geo_tool",
    "get_geo_meta",
    "stamp_geo_meta",
    "needs_confirmation",
    "ConfirmCallback",
    "ConfirmRequest",
    "auto_approve_all",
    "auto_approve_safe_only",
    "build_interrupt_on",
    "resolve_model",
    "get_llm",
    "get_default_model",
]
