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
from geoagent.tools.whitebox import whitebox_tools

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
- Only pass bbox to load_gee_dataset when the current user request explicitly
  asks for a region/place/current map extent or provides bbox coordinates. Do
  not reuse a previous conversation location for a new dataset request unless
  the current request says "same area", "there", or otherwise clearly refers
  to that prior location. If the user asks for a global layer, or gives no
  location, omit bbox. For ImageCollections, regional display should use
  filterBounds. When a specific FeatureCollection exists for the requested
  region, prefer load_gee_dataset bounds_collection_asset_id and
  bounds_filter_property/value over bbox coordinates; for example, use
  TIGER/2018/States with NAME=Tennessee for Tennessee. Use bbox only when no
  appropriate FeatureCollection is known, the user provides exact bbox
  coordinates, or the bbox is already known and should be used only to zoom
  QGIS to the requested region after loading. If you have both a
  FeatureCollection filter and an already-known bbox for the same region, pass
  both; the tool will use the FeatureCollection for filterBounds and the bbox
  for QGIS zoom. Do not compute a bbox from the FeatureCollection geometry.
- Earth Engine clip operations are computationally intensive. Do not use
  ee.Image.clip or ee.Image.clipToCollection just because the user asks to show
  data for a region; use filterBounds through bounds_collection_asset_id or
  bbox instead. Only when the user specifically asks to clip, crop, or mask the
  raster/Image/ImageCollection to an administrative boundary or other vector
  region, use load_gee_dataset with clip_collection_asset_id and
  clip_filter_property/value. The tool applies ee.Image.clipToCollection to the
  raster output.
- When the user asks for normalized difference indexes such as NDVI, NDWI,
  MNDWI, NDMI, or NBR, use calculate_gee_normalized_difference. Do not display
  a single source band as a proxy. Common Sentinel-2/HLS S30 pairs: NDVI B8/B4,
  NDWI B3/B8, MNDWI B3/B11, NBR B8/B12.
- When the user asks for an Earth Engine operation that has no dedicated
  GeoAgent tool, write a short Earth Engine Python snippet and run it with
  run_gee_python_snippet. Prefer official Earth Engine API functions such as
  ee.Terrain.hillshade for DEM hillshade. Do not say a task is impossible only
  because no named GeoAgent tool exists.
- When the user asks for scalar raster statistics such as mean elevation,
  min/max, count, or summary values from a loaded Earth Engine layer, use
  calculate_gee_layer_statistics. Do not use run_gee_python_snippet with
  reduceRegion/getInfo for those requests. For large regions, use the default
  coarse best-effort scale and report that the result is approximate.
- When the user refers to an existing Earth Engine layer, call
  list_loaded_gee_layers and reuse it with get_ee_layer(name) inside
  run_gee_python_snippet instead of reloading data when possible.
- For ImageCollections, call the selected aggregation a composite method.
  ``mosaic`` is valid, but it is an ImageCollection method, not an ee.Reducer.
- For OPERA DSWx water maps, use OPERA/DSWX/L3_V1/HLS by default. Use
  OPERA/DSWX/L3_V1/S1 only when the user explicitly asks for Sentinel-1 or S1.
  Use WTR_Water_classification or BWTR_Binary_water band names. The tool masks
  invalid classes, defaults HLS composites to mode and S1 composites to max,
  and remaps class values for QGIS rendering.
- Do not report that Earth Engine returned an empty collection or no bands
  unless load_gee_dataset returns success=false with diagnostics proving that
  specific condition. If layer insertion or tile generation fails, report that
  actual error instead.
- Ask for or infer visualization bands only when needed; common RGB defaults
  are acceptable for Landsat and Sentinel imagery.
- Keep responses concise and include asset ids, layer names, and filters used.
- If load_gee_dataset returns a bbox field, include the bbox coordinates in the
  response in west,south,east,north order.
"""

QGIS_SYSTEM_PROMPT = """\
You are an AI assistant embedded in QGIS with access to PyQGIS-backed tools.

Workflow guidance:
- Use QGIS layer names as input values only when the layer is backed by a
  local file. Otherwise ask the user to export the layer or provide a file.
- Generated outputs are added back to QGIS when possible.
- When the user asks for a QGIS API operation that has no dedicated tool, such
  as raster renderer/band styling, labeling, layer tree tweaks, or other
  project/canvas changes, write a short PyQGIS script and run it with
  run_pyqgis_script. Do not merely provide a script for the user to paste when
  run_pyqgis_script can safely perform the requested QGIS change.
- Keep responses concise and include the tool name, output path, and loaded
  layer names when available.
"""

WHITEBOX_SYSTEM_PROMPT = """\
You are an AI assistant embedded in QGIS with access to WhiteboxTools.
WhiteboxTools exposes hundreds of geospatial analysis commands through a
routed tool interface.

Workflow guidance:
- Do not guess exact Whitebox command parameters. Search first when the user
  describes an analysis task, then inspect the selected tool's parameter
  metadata before running it.
- Use search_whitebox_tools to find candidate commands, get_whitebox_tool_info
  to inspect required parameters, then run_whitebox_tool to execute.
- For active DEM flow accumulation requests, prefer
  run_whitebox_flow_accumulation.
- For active DEM sink/depression filling requests, prefer
  run_whitebox_fill_sinks with method="fill_depressions"; this is the default
  and is preferred before flow direction or flow accumulation.
- Use run_whitebox_fill_sinks with method="breach_depressions" only when the
  user explicitly asks to breach, carve, or channel depressions.
- Run each long-running Whitebox command at most once per user request unless
  the tool returns an error. Do not repeat an identical tool call after it has
  already produced an output.
- For active DEM color shaded relief requests, prefer
  run_whitebox_color_shaded_relief.
- For active vector layer buffer requests, prefer buffer_active_layer.
- Use QGIS layer names as input values only when the layer is backed by a
  local file. Otherwise ask the user to export the layer or provide a file.
- Generated outputs are added back to QGIS when possible.
- When the user asks for a QGIS API operation that has no dedicated tool, such
  as raster renderer/band styling, labeling, layer tree tweaks, or other
  project/canvas changes, write a short PyQGIS script and run it with
  run_pyqgis_script. Do not merely provide a script for the user to paste when
  run_pyqgis_script can safely perform the requested QGIS change.
- Keep responses concise and include the Whitebox tool name, output path, and
  loaded layer names when available.
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
    include_whitebox: bool = False,
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
    if include_whitebox:
        whitebox_tool_list = _filter_by_imports(
            whitebox_tools(context.qgis_iface, context.qgis_project)
        )
        register_all_tools(registry, whitebox_tool_list)
        collected.extend(whitebox_tool_list)
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
    ctx = GeoAgentContext(
        qgis_iface=iface,
        qgis_project=project,
        metadata={"system_prompt": QGIS_SYSTEM_PROMPT},
    )
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


def for_whitebox(
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
    """Bind an agent to QGIS with WhiteboxTools analysis support.

    The factory exposes a routed WhiteboxTools broker surface and, by default,
    the general QGIS map/project tools used for inspection and navigation.
    """
    ctx = GeoAgentContext(
        qgis_iface=iface,
        qgis_project=project,
        metadata={
            "integration": "whitebox",
            "system_prompt": WHITEBOX_SYSTEM_PROMPT,
        },
    )
    tools, registry = assemble_tools(
        context=ctx,
        include_qgis=include_qgis,
        include_whitebox=True,
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
    "for_whitebox",
    "register_all_tools",
]
