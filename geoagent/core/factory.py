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
from geoagent.tools.images import image_generation_tools
from geoagent.tools.leafmap import leafmap_tools
from geoagent.tools.nasa_earthdata import earthdata_tools
from geoagent.tools.nasa_opera import nasa_opera_tools
from geoagent.tools.qgis import qgis_tools
from geoagent.tools.stac import stac_tools
from geoagent.tools.timelapse import timelapse_tools
from geoagent.tools.vantor import vantor_tools
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

VANTOR_SYSTEM_PROMPT = """\
You are an AI assistant embedded in QGIS for the Vantor Open Data plugin.
Use the Vantor tools to list event collections, inspect event metadata, search
STAC items, display footprints, and load COG imagery into the current QGIS
project.

Workflow guidance:
- List Vantor events first when the user names a disaster, place, or event
  imprecisely.
- Search a specific event collection before displaying footprints or loading
  imagery.
- If the user asks for the current map area, call
  get_current_vantor_search_extent and pass the returned bbox to
  search_vantor_items.
- Use the phase filter for pre-event or post-event requests.
- For raster display, use a specific item id from search results or a concrete
  COG URL. Loading rasters and displaying footprints change the QGIS project
  and require user confirmation.
- Keep responses concise and include event names, item counts, item ids, phase,
  acquisition date, sensor, and loaded layer names when available.
"""

TIMELAPSE_SYSTEM_PROMPT = """\
You are an AI assistant embedded in QGIS for the Timelapse plugin.
Use the Timelapse tools to inspect available imagery types, get the current
map extent, initialize Earth Engine, create timelapse GIFs, and open plugin
panels when the user wants the visual UI.

Workflow guidance:
- If the user gives no bbox, use the current QGIS map extent.
- Confirm the imagery type and time window before launching long timelapse
  generation unless the request already provides them clearly.
- Prefer Landsat for long historical change, Sentinel-2 for recent optical
  detail, Sentinel-1 for radar/cloud-tolerant change, NAIP for US aerial
  detail, MODIS NDVI for vegetation phenology, and GOES for weather animation.
- Timelapse generation can take a while and requires user confirmation.
- Keep responses concise and include imagery type, bbox, time window, and
  output path when available.
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
- When the user asks to create, draw, render, or generate an image or picture,
  or provides a standalone visual description after discussing image
  generation, call generate_image if it is available. Do not reply with only a
  prompt for another image generator unless the tool reports that image
  generation is not configured.
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

STAC_SYSTEM_PROMPT = """\
You are an AI assistant embedded in QGIS with access to STAC catalog tools.

Workflow guidance:
- If the user does not name a STAC catalog, use the Planetary Computer STAC
  catalog: https://planetarycomputer.microsoft.com/api/stac/v1.
- Use list_stac_collections when the user has a catalog URL but no collection.
- Use search_stac_items for catalog search, including the current QGIS map
  extent when the user asks for the current area. Use
  get_current_stac_search_extent for the current QGIS map extent; do not use
  run_pyqgis_script to calculate STAC search bounds.
- When the user asks for cloud-free, clear-sky, or low-cloud imagery, pass
  max_cloud_cover=10 to search_stac_items and select the returned item with the
  best spatial fit and lowest cloud_cover rather than the newest item. Prefer
  items where contains_query_center is true and bbox_overlap_ratio is high.
  Mention cloud_cover in the response when available.
- For DEM/elevation requests, do not pass max_cloud_cover. Prefer Planetary
  Computer collection cop-dem-glo-30 and load the data/elevation asset rather
  than a rendered preview.
- Inspect item assets before loading. Prefer cloud-optimized raster assets
  such as COG, GeoTIFF, visual, analytic, or band-specific raster assets.
- If search_stac_items returns a suitable preferred_assets entry, use that href
  directly with add_stac_asset_to_qgis instead of making a separate
  get_stac_item_assets call.
- Use add_stac_asset_to_qgis only for a concrete asset href. If QGIS cannot
  load the asset directly, report the returned asset URL and reason instead of
  claiming the layer was added. If the tool reports queued=True, tell the user
  the QGIS background task has started and the layer will appear when QGIS
  validates the raster.
- Keep responses concise and include catalog URL, collection, item id, asset
  key, and layer name when available.
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


def _permission_allows_tool(permission_profile: str | None, tool: Any) -> bool:
    """Return whether a QGIS/plugin tool should be exposed for a profile."""
    profile = permission_profile or "Trusted auto-approve"
    if profile == "Execute PyQGIS":
        profile = "Execute Scripts"
    name = (
        getattr(tool, "tool_name", "")
        or getattr(tool, "__name__", "")
        or getattr(tool, "name", "")
    )
    meta = getattr(tool, "_geoagent_meta", None)
    category = str(getattr(meta, "category", "") or "")
    requires_confirmation = bool(getattr(meta, "requires_confirmation", False))
    destructive = bool(getattr(meta, "destructive", False))
    long_running = bool(getattr(meta, "long_running", False))

    if profile == "Trusted auto-approve":
        return True
    if profile == "Execute Scripts":
        return True
    if name == "run_pyqgis_script":
        return False
    if profile == "Run processing":
        return True
    if category in {
        "whitebox",
        "nasa_earthdata",
        "nasa_opera",
        "gee_data_catalogs",
        "timelapse",
        "vantor",
    }:
        return profile in {"Run processing", "Execute Scripts", "Trusted auto-approve"}
    if profile == "Edit layers":
        return not destructive and name != "run_processing_algorithm"
    return not (requires_confirmation or destructive or long_running)


def _filter_by_permission(
    tools: list[Any], permission_profile: str | None
) -> list[Any]:
    """Filter QGIS-related tool surfaces according to a permission profile."""
    if not permission_profile:
        return tools
    return [tool for tool in tools if _permission_allows_tool(permission_profile, tool)]


def _drop_tools_by_name(tools: list[Any], excluded: set[str]) -> list[Any]:
    """Return tools except those whose Strands name is excluded."""
    if not excluded:
        return tools
    out = []
    for tool in tools:
        name = (
            getattr(tool, "tool_name", "")
            or getattr(tool, "__name__", "")
            or getattr(tool, "name", "")
        )
        if str(name) not in excluded:
            out.append(tool)
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
    include_timelapse: bool = False,
    include_vantor: bool = False,
    include_whitebox: bool = False,
    include_stac: bool = False,
    include_image_generation: bool = False,
    nasa_earthdata_plugin: Any | None = None,
    gee_data_catalogs_plugin: Any | None = None,
    timelapse_plugin: Any | None = None,
    vantor_plugin: Any | None = None,
    fast: bool = False,
    permission_profile: str | None = None,
    exclude_tool_names: set[str] | None = None,
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
    if include_timelapse:
        timelapse_tool_list = _filter_by_imports(
            timelapse_tools(
                context.qgis_iface,
                context.qgis_project,
                plugin=timelapse_plugin,
            )
        )
        register_all_tools(registry, timelapse_tool_list)
        collected.extend(timelapse_tool_list)
    if include_vantor:
        vantor_tool_list = _filter_by_imports(
            vantor_tools(
                context.qgis_iface,
                context.qgis_project,
                plugin=vantor_plugin,
            )
        )
        register_all_tools(registry, vantor_tool_list)
        collected.extend(vantor_tool_list)
    if include_whitebox:
        whitebox_tool_list = _filter_by_imports(
            whitebox_tools(context.qgis_iface, context.qgis_project)
        )
        register_all_tools(registry, whitebox_tool_list)
        collected.extend(whitebox_tool_list)
    if include_stac:
        stac_tool_list = _filter_by_imports(
            stac_tools(context.qgis_iface, context.qgis_project)
        )
        register_all_tools(registry, stac_tool_list)
        collected.extend(stac_tool_list)
    if include_image_generation:
        image_tools = _filter_by_imports(image_generation_tools())
        register_all_tools(registry, image_tools)
        collected.extend(image_tools)
    if extra_tools:
        register_all_tools(registry, extra_tools)
        collected.extend(extra_tools)
    collected = _filter_by_permission(collected, permission_profile)
    collected = _drop_tools_by_name(collected, exclude_tool_names or set())
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
        include_image_generation=True,
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
        include_image_generation=True,
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
    permission_profile: str | None = None,
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
        include_image_generation=True,
        extra_tools=extra_tools,
        fast=fast,
        permission_profile=permission_profile,
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
    permission_profile: str | None = None,
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
        include_image_generation=True,
        extra_tools=extra_tools,
        fast=fast,
        permission_profile=permission_profile,
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
    permission_profile: str | None = None,
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
        include_image_generation=True,
        nasa_earthdata_plugin=plugin,
        extra_tools=extra_tools,
        fast=fast,
        permission_profile=permission_profile,
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
    permission_profile: str | None = None,
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
        include_image_generation=True,
        gee_data_catalogs_plugin=plugin,
        extra_tools=extra_tools,
        fast=fast,
        permission_profile=permission_profile,
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


def for_vantor(
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
    permission_profile: str | None = None,
) -> GeoAgent:
    """Bind an agent to the QGIS Vantor plugin runtime.

    The factory exposes native Vantor Open Data STAC tools and, by default,
    the general QGIS map/project tools used for navigation and layer
    management.
    """
    ctx = GeoAgentContext(
        qgis_iface=iface,
        qgis_project=project,
        metadata={
            "integration": "vantor",
            "system_prompt": VANTOR_SYSTEM_PROMPT,
        },
    )
    tools, registry = assemble_tools(
        context=ctx,
        include_qgis=include_qgis,
        include_vantor=True,
        include_image_generation=True,
        vantor_plugin=plugin,
        extra_tools=extra_tools,
        fast=fast,
        permission_profile=permission_profile,
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


def for_timelapse(
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
    permission_profile: str | None = None,
) -> GeoAgent:
    """Bind an agent to the QGIS Timelapse plugin runtime.

    The factory exposes native Timelapse tools and, by default, the general
    QGIS map/project tools used for inspection and navigation.
    """
    ctx = GeoAgentContext(
        qgis_iface=iface,
        qgis_project=project,
        metadata={
            "integration": "timelapse",
            "system_prompt": TIMELAPSE_SYSTEM_PROMPT,
        },
    )
    tools, registry = assemble_tools(
        context=ctx,
        include_qgis=include_qgis,
        include_timelapse=True,
        include_image_generation=True,
        timelapse_plugin=plugin,
        extra_tools=extra_tools,
        fast=fast,
        permission_profile=permission_profile,
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
    permission_profile: str | None = None,
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
        include_image_generation=True,
        extra_tools=extra_tools,
        fast=fast,
        permission_profile=permission_profile,
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


def for_stac(
    iface: Any = None,
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
    permission_profile: str | None = None,
) -> GeoAgent:
    """Bind an agent to STAC catalog workflows and optional QGIS loading."""
    ctx = GeoAgentContext(
        qgis_iface=iface,
        qgis_project=project,
        metadata={
            "integration": "stac",
            "system_prompt": STAC_SYSTEM_PROMPT,
        },
    )
    tools, registry = assemble_tools(
        context=ctx,
        include_qgis=include_qgis,
        include_stac=True,
        include_image_generation=True,
        extra_tools=extra_tools,
        fast=fast,
        permission_profile=permission_profile,
        exclude_tool_names={"run_pyqgis_script"},
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
        qgis_safe_mode=iface is not None,
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
    "for_stac",
    "for_timelapse",
    "for_vantor",
    "for_whitebox",
    "register_all_tools",
]
