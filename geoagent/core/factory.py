"""Build :class:`strands.agent.agent.Agent` and :class:`geoagent.GeoAgent` instances."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from geoagent.core.config import GeoAgentConfig
from geoagent.core.context import GeoAgentContext
from geoagent.core.registry import (
    GeoToolRegistry,
    collect_tools_for_context,
    packages_available,
)
from geoagent.core.safety import ConfirmCallback
from geoagent.core.agent import GeoAgent
from geoagent.tools.anymap import anymap_tools
from geoagent.tools.gee_data_catalogs import gee_data_catalogs_tools
from geoagent.tools.leafmap import leafmap_tools
from geoagent.tools.nasa_earthdata import earthdata_tools
from geoagent.tools.nasa_opera import nasa_opera_tools
from geoagent.tools.qgis import qgis_tools

NASA_EARTHDATA_SYSTEM_PROMPT = """\
You are an AI assistant embedded in QGIS for the NASA Earthdata plugin.
Use the NASA Earthdata tools to search the dataset catalog, search CMR
granules, display footprints, and load raster assets into the current QGIS
project.

Workflow guidance:
- Search the catalog first when the user names a topic rather than an exact
  NASA Earthdata short name.
- If the user gives no location, use the current QGIS map extent.
- Search granules before displaying footprints or loading rasters.
- For raster display, choose a specific COG or GeoTIFF data link from search
  results. Loading data can download protected assets and requires user
  confirmation.
- Keep responses concise and include dataset short names, result counts,
  date ranges, and relevant first-result identifiers when available.
"""

NASA_OPERA_SYSTEM_PROMPT = """\
You are an AI assistant embedded in QGIS for NASA OPERA satellite data.
Use the OPERA tools to search, display, and summarize NASA OPERA products.

Dataset guidance:
- Water, flood, inundation: prefer OPERA_L3_DSWX-HLS_V1 or OPERA_L3_DSWX-S1_V1.
- Deforestation, fire damage, vegetation loss, land disturbance: prefer
  OPERA_L3_DIST-ALERT-HLS_V1 or OPERA_L3_DIST-ANN-HLS_V1.
- SAR/radar/backscatter: prefer OPERA_L2_RTC-S1_V1.
- Interferometry/phase workflows: prefer OPERA_L2_CSLC-S1_V1.

Workflow guidance:
- Search before displaying footprints or rasters.
- If the user gives no location, use the current QGIS map extent.
- For raster display, choose a specific data link from search results.
- Keep responses concise and include result counts, date range, and relevant
  first-result identifiers when available.
"""

GEE_DATA_CATALOGS_SYSTEM_PROMPT = """\
You are an AI assistant embedded in QGIS for Google Earth Engine data catalogs.
Use the GEE Data Catalogs tools to search official and community datasets,
inspect metadata, initialize Earth Engine, configure plugin panels, and load
Earth Engine layers into the current QGIS project.

Workflow guidance:
- Search the catalog before loading when the user names a topic rather than an
  exact Earth Engine asset id.
- Prefer the current QGIS map extent or a user-provided bbox for spatially
  constrained ImageCollection requests.
- When the user asks to clip a raster/Image/ImageCollection to an administrative
  boundary or other vector region, use load_gee_dataset with
  clip_collection_asset_id and clip_filter_property/value. The tool applies
  ee.Image.clipToCollection to the raster output. For Tennessee, use
  TIGER/2018/States with NAME=Tennessee.
- When the user asks for normalized difference indexes such as NDVI, NDWI,
  MNDWI, NDMI, or NBR, use calculate_gee_normalized_difference. Do not display
  a single source band as a proxy. Common Sentinel-2/HLS S30 pairs: NDVI B8/B4,
  NDWI B3/B8, MNDWI B3/B11, NBR B8/B12.
- Ask for or infer visualization bands only when needed; common RGB defaults
  are acceptable for Landsat and Sentinel imagery.
- Keep responses concise and include asset ids, layer names, and filters used.
"""


def _filter_by_imports(tools: list[Any]) -> list[Any]:
    """Drop tools whose declared optional packages are unavailable."""
    out: list[Any] = []
    for t in tools:
        meta = getattr(t, "_geoagent_meta", None)
        if (
            meta
            and meta.requires_packages
            and not packages_available(meta.requires_packages)
        ):
            continue
        out.append(t)
    return out


def register_all_tools(registry: GeoToolRegistry, tools: Iterable[Any]) -> None:
    """Populate registry from decorated tools."""
    for t in tools:
        meta = getattr(t, "_geoagent_meta", None)
        if meta is not None:
            registry.register_tool(t, meta)


def assemble_tools(
    *,
    context: GeoAgentContext,
    extra_tools: Optional[list[Any]] = None,
    include_leafmap: bool = False,
    include_anymap: bool = False,
    include_qgis: bool = False,
    include_nasa_earthdata: bool = False,
    include_nasa_opera: bool = False,
    include_gee_data_catalogs: bool = False,
    nasa_earthdata_plugin: Any | None = None,
    gee_data_catalogs_plugin: Any | None = None,
    fast: bool = False,
) -> tuple[list[Any], GeoToolRegistry]:
    """Collect tools for a context and build a metadata registry."""
    registry = GeoToolRegistry()
    collected: list[Any] = []
    if include_leafmap and context.map_obj is not None:
        lt = _filter_by_imports(leafmap_tools(context.map_obj))
        register_all_tools(registry, lt)
        collected.extend(lt)
    if include_anymap and context.map_obj is not None:
        at = _filter_by_imports(anymap_tools(context.map_obj))
        register_all_tools(registry, at)
        collected.extend(at)
    if include_qgis:
        qt = _filter_by_imports(qgis_tools(context.qgis_iface, context.qgis_project))
        register_all_tools(registry, qt)
        collected.extend(qt)
    if include_nasa_earthdata:
        earthdata_tool_list = _filter_by_imports(
            earthdata_tools(
                context.qgis_iface,
                context.qgis_project,
                plugin=nasa_earthdata_plugin,
            )
        )
        register_all_tools(registry, earthdata_tool_list)
        collected.extend(earthdata_tool_list)
    if include_nasa_opera:
        opera_tools = _filter_by_imports(
            nasa_opera_tools(context.qgis_iface, context.qgis_project)
        )
        register_all_tools(registry, opera_tools)
        collected.extend(opera_tools)
    if include_gee_data_catalogs:
        gee_tools = _filter_by_imports(
            gee_data_catalogs_tools(
                context.qgis_iface,
                plugin=gee_data_catalogs_plugin,
            )
        )
        register_all_tools(registry, gee_tools)
        collected.extend(gee_tools)
    if extra_tools:
        register_all_tools(registry, extra_tools)
        collected.extend(extra_tools)
    tools = collect_tools_for_context(collected, fast=fast, registry=registry)
    return tools, registry


def create_agent(
    *,
    context: GeoAgentContext | None = None,
    tools: list[Any] | None = None,
    config: GeoAgentConfig | None = None,
    model: Any | None = None,
    provider: str | None = None,
    model_id: str | None = None,
    fast: bool = False,
    confirm: ConfirmCallback | None = None,
) -> GeoAgent:
    """Create a :class:`GeoAgent` with explicit tools and optional model."""
    ctx = context or GeoAgentContext()
    cfg = config or GeoAgentConfig()
    if provider is not None:
        cfg = cfg.model_copy(update={"provider": provider})
    if model_id is not None:
        cfg = cfg.model_copy(update={"model": model_id})
    registry = GeoToolRegistry()
    tool_list = _filter_by_imports(list(tools or []))
    register_all_tools(registry, tool_list)
    tool_list = collect_tools_for_context(tool_list, fast=fast, registry=registry)
    return GeoAgent(
        context=ctx,
        config=cfg,
        tools=tool_list,
        registry=registry,
        model=model,
        provider=provider,
        model_id=model_id,
        fast=fast,
        confirm=confirm,
    )


def for_leafmap(
    m: Any,
    *,
    config: GeoAgentConfig | None = None,
    model: Any | None = None,
    provider: str | None = None,
    model_id: str | None = None,
    fast: bool = False,
    confirm: ConfirmCallback | None = None,
    extra_tools: Optional[list[Any]] = None,
) -> GeoAgent:
    """Bind an agent to a leafmap-compatible map instance."""
    ctx = GeoAgentContext(map_obj=m)
    tools, registry = assemble_tools(
        context=ctx,
        include_leafmap=True,
        extra_tools=extra_tools,
        fast=fast,
    )
    cfg = config or GeoAgentConfig()
    if provider is not None:
        cfg = cfg.model_copy(update={"provider": provider})
    if model_id is not None:
        cfg = cfg.model_copy(update={"model": model_id})
    return GeoAgent(
        context=ctx,
        config=cfg,
        tools=tools,
        registry=registry,
        model=model,
        provider=provider,
        model_id=model_id,
        fast=fast,
        confirm=confirm,
    )


def for_anymap(
    m: Any,
    *,
    config: GeoAgentConfig | None = None,
    model: Any | None = None,
    provider: str | None = None,
    model_id: str | None = None,
    fast: bool = False,
    confirm: ConfirmCallback | None = None,
    extra_tools: Optional[list[Any]] = None,
) -> GeoAgent:
    """Bind an agent to an anymap map instance."""
    ctx = GeoAgentContext(map_obj=m)
    tools, registry = assemble_tools(
        context=ctx,
        include_anymap=True,
        extra_tools=extra_tools,
        fast=fast,
    )
    cfg = config or GeoAgentConfig()
    if provider is not None:
        cfg = cfg.model_copy(update={"provider": provider})
    if model_id is not None:
        cfg = cfg.model_copy(update={"model": model_id})
    return GeoAgent(
        context=ctx,
        config=cfg,
        tools=tools,
        registry=registry,
        model=model,
        provider=provider,
        model_id=model_id,
        fast=fast,
        confirm=confirm,
    )


def for_qgis(
    iface: Any,
    project: Any = None,
    *,
    config: GeoAgentConfig | None = None,
    model: Any | None = None,
    provider: str | None = None,
    model_id: str | None = None,
    fast: bool = False,
    confirm: ConfirmCallback | None = None,
    extra_tools: Optional[list[Any]] = None,
) -> GeoAgent:
    """Bind an agent to QGIS ``iface`` (and optional ``project``)."""
    ctx = GeoAgentContext(qgis_iface=iface, qgis_project=project)
    tools, registry = assemble_tools(
        context=ctx,
        include_qgis=True,
        extra_tools=extra_tools,
        fast=fast,
    )
    cfg = config or GeoAgentConfig()
    if provider is not None:
        cfg = cfg.model_copy(update={"provider": provider})
    if model_id is not None:
        cfg = cfg.model_copy(update={"model": model_id})
    return GeoAgent(
        context=ctx,
        config=cfg,
        tools=tools,
        registry=registry,
        model=model,
        provider=provider,
        model_id=model_id,
        fast=fast,
        confirm=confirm,
        qgis_safe_mode=True,
    )


def for_nasa_opera(
    iface: Any,
    project: Any = None,
    *,
    config: GeoAgentConfig | None = None,
    model: Any | None = None,
    provider: str | None = None,
    model_id: str | None = None,
    fast: bool = False,
    confirm: ConfirmCallback | None = None,
    extra_tools: Optional[list[Any]] = None,
    include_qgis: bool = True,
) -> GeoAgent:
    """Bind an agent to the NASA OPERA QGIS plugin runtime.

    The factory exposes native GeoAgent OPERA tools and, by default, the
    general QGIS map/project tools used for navigation and layer management.
    """
    ctx = GeoAgentContext(
        qgis_iface=iface,
        qgis_project=project,
        metadata={
            "integration": "nasa_opera",
            "system_prompt": NASA_OPERA_SYSTEM_PROMPT,
        },
    )
    tools, registry = assemble_tools(
        context=ctx,
        include_qgis=include_qgis,
        include_nasa_opera=True,
        extra_tools=extra_tools,
        fast=fast,
    )
    cfg = config or GeoAgentConfig()
    if provider is not None:
        cfg = cfg.model_copy(update={"provider": provider})
    if model_id is not None:
        cfg = cfg.model_copy(update={"model": model_id})
    return GeoAgent(
        context=ctx,
        config=cfg,
        tools=tools,
        registry=registry,
        model=model,
        provider=provider,
        model_id=model_id,
        fast=fast,
        confirm=confirm,
        qgis_safe_mode=True,
    )


def for_nasa_earthdata(
    iface: Any,
    project: Any = None,
    *,
    plugin: Any | None = None,
    config: GeoAgentConfig | None = None,
    model: Any | None = None,
    provider: str | None = None,
    model_id: str | None = None,
    fast: bool = False,
    confirm: ConfirmCallback | None = None,
    extra_tools: Optional[list[Any]] = None,
    include_qgis: bool = True,
) -> GeoAgent:
    """Bind an agent to the NASA Earthdata QGIS plugin runtime.

    The factory exposes native NASA Earthdata tools and, by default, the
    general QGIS map/project tools used for navigation and layer management.
    """
    ctx = GeoAgentContext(
        qgis_iface=iface,
        qgis_project=project,
        metadata={
            "integration": "nasa_earthdata",
            "system_prompt": NASA_EARTHDATA_SYSTEM_PROMPT,
        },
    )
    tools, registry = assemble_tools(
        context=ctx,
        include_qgis=include_qgis,
        include_nasa_earthdata=True,
        nasa_earthdata_plugin=plugin,
        extra_tools=extra_tools,
        fast=fast,
    )
    cfg = config or GeoAgentConfig()
    if provider is not None:
        cfg = cfg.model_copy(update={"provider": provider})
    if model_id is not None:
        cfg = cfg.model_copy(update={"model": model_id})
    return GeoAgent(
        context=ctx,
        config=cfg,
        tools=tools,
        registry=registry,
        model=model,
        provider=provider,
        model_id=model_id,
        fast=fast,
        confirm=confirm,
        qgis_safe_mode=True,
    )


def for_gee_data_catalogs(
    iface: Any,
    project: Any = None,
    *,
    plugin: Any | None = None,
    config: GeoAgentConfig | None = None,
    model: Any | None = None,
    provider: str | None = None,
    model_id: str | None = None,
    fast: bool = False,
    confirm: ConfirmCallback | None = None,
    extra_tools: Optional[list[Any]] = None,
    include_qgis: bool = True,
) -> GeoAgent:
    """Bind an agent to the QGIS GEE Data Catalogs plugin runtime.

    The factory exposes native GEE Data Catalogs tools and, by default, the
    general QGIS map/project tools used for navigation and layer management.
    """
    ctx = GeoAgentContext(
        qgis_iface=iface,
        qgis_project=project,
        metadata={
            "integration": "gee_data_catalogs",
            "system_prompt": GEE_DATA_CATALOGS_SYSTEM_PROMPT,
        },
    )
    tools, registry = assemble_tools(
        context=ctx,
        include_qgis=include_qgis,
        include_gee_data_catalogs=True,
        gee_data_catalogs_plugin=plugin,
        extra_tools=extra_tools,
        fast=fast,
    )
    cfg = config or GeoAgentConfig()
    if provider is not None:
        cfg = cfg.model_copy(update={"provider": provider})
    if model_id is not None:
        cfg = cfg.model_copy(update={"model": model_id})
    return GeoAgent(
        context=ctx,
        config=cfg,
        tools=tools,
        registry=registry,
        model=model,
        provider=provider,
        model_id=model_id,
        fast=fast,
        confirm=confirm,
        qgis_safe_mode=True,
    )


__all__ = [
    "assemble_tools",
    "create_agent",
    "for_anymap",
    "for_gee_data_catalogs",
    "for_leafmap",
    "for_nasa_earthdata",
    "for_nasa_opera",
    "for_qgis",
    "register_all_tools",
]
