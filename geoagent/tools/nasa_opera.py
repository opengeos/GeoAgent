"""Tool adapters for the NASA OPERA QGIS plugin.

The tools in this module are native GeoAgent tools. They intentionally do not
wrap ``nasa_opera.ai.tools`` because that plugin-local agent surface is being
replaced by GeoAgent. The module also avoids importing NASA OPERA plugin UI
modules because they import QGIS/PyQt objects at module scope and are unsafe
from background task threads.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from datetime import datetime
from typing import Any, Callable, Optional
from urllib.parse import urlparse

from geoagent.core.decorators import geo_tool
from geoagent.tools._qt_marshal import run_on_qt_gui_thread

_QGIS_TASKS: list[Any] = []

OPERA_DATASETS: dict[str, dict[str, str]] = {
    "OPERA_L3_DSWX-HLS_V1": {
        "title": "Dynamic Surface Water Extent from Harmonized Landsat Sentinel-2 (Version 1)",
        "short_title": "DSWX-HLS",
        "description": "Surface water extent derived from HLS data",
    },
    "OPERA_L3_DSWX-S1_V1": {
        "title": "Dynamic Surface Water Extent from Sentinel-1 (Version 1)",
        "short_title": "DSWX-S1",
        "description": "Surface water extent derived from Sentinel-1 SAR data",
    },
    "OPERA_L3_DIST-ALERT-HLS_V1": {
        "title": "Land Surface Disturbance Alert from HLS (Version 1)",
        "short_title": "DIST-ALERT",
        "description": "Near real-time disturbance alerts",
    },
    "OPERA_L3_DIST-ANN-HLS_V1": {
        "title": "Land Surface Disturbance Annual from HLS (Version 1)",
        "short_title": "DIST-ANN",
        "description": "Annual land surface disturbance product",
    },
    "OPERA_L2_RTC-S1_V1": {
        "title": "Radiometric Terrain Corrected SAR Backscatter from Sentinel-1 (Version 1)",
        "short_title": "RTC-S1",
        "description": "Analysis-ready SAR backscatter data",
    },
    "OPERA_L2_RTC-S1-STATIC_V1": {
        "title": "RTC-S1 Static Layers (Version 1)",
        "short_title": "RTC-S1-STATIC",
        "description": "Static layers for RTC-S1 product",
    },
    "OPERA_L2_CSLC-S1_V1": {
        "title": "Coregistered Single-Look Complex from Sentinel-1 (Version 1)",
        "short_title": "CSLC-S1",
        "description": "SLC data coregistered to a common reference",
    },
    "OPERA_L2_CSLC-S1-STATIC_V1": {
        "title": "CSLC-S1 Static Layers (Version 1)",
        "short_title": "CSLC-S1-STATIC",
        "description": "Static layers for CSLC-S1 product",
    },
}


def _on_gui(fn: Any) -> Any:
    """Run ``fn`` on the Qt GUI thread when QGIS is available."""
    return run_on_qt_gui_thread(fn)


def _datasets() -> dict[str, dict[str, str]]:
    """Return OPERA dataset metadata."""
    return dict(OPERA_DATASETS)


def _earthdata_login() -> None:
    """Authenticate with NASA Earthdata using earthaccess env/netrc defaults."""
    import earthaccess

    for strategy in ("environment", "netrc"):
        try:
            if earthaccess.login(strategy=strategy):
                return
        except Exception:
            continue
    raise RuntimeError(
        "NASA Earthdata authentication failed. Configure Earthdata credentials "
        "in the NASA OPERA plugin settings, environment, or ~/.netrc."
    )


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(value: str, *, fallback: str = "opera") -> str:
    """Return a filesystem-safe basename derived from a user-provided string.

    Strips path separators, query strings, and other non-portable characters so
    the result can be joined with a temp/cache directory without escaping it.
    """
    candidate = os.path.basename(value or "").strip()
    candidate = _SAFE_NAME_RE.sub("_", candidate).strip("._")
    return candidate or fallback


def _filename_from_url(url: str, *, fallback: str = "opera") -> str:
    """Return a stable basename for ``url`` ignoring query strings."""
    parsed = urlparse(url)
    raw = os.path.basename(parsed.path) if parsed.scheme else os.path.basename(url)
    return _safe_filename(raw, fallback=fallback)


def _setup_gdal_for_earthdata() -> tuple[bool, Optional[str]]:
    """Configure GDAL for Earthdata S3 access.

    TLS verification stays enabled by default. To work around hosts with broken
    certificate chains, set ``GEOAGENT_OPERA_INSECURE_SSL=1`` in the
    environment. The unsafe option is opt-in because it disables verification
    for every subsequent GDAL HTTP read in the QGIS process.
    """
    try:
        import earthaccess
        from osgeo import gdal

        _earthdata_login()
        creds = earthaccess.get_s3_credentials(daac="PODAAC")
        gdal.SetConfigOption("AWS_ACCESS_KEY_ID", creds["accessKeyId"])
        gdal.SetConfigOption("AWS_SECRET_ACCESS_KEY", creds["secretAccessKey"])
        gdal.SetConfigOption("AWS_SESSION_TOKEN", creds["sessionToken"])
        gdal.SetConfigOption("AWS_REGION", "us-west-2")
        gdal.SetConfigOption("AWS_S3_ENDPOINT", "s3.us-west-2.amazonaws.com")
        gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
        gdal.SetConfigOption(
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF,.tiff,.TIFF"
        )
        if os.environ.get("GEOAGENT_OPERA_INSECURE_SSL", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            gdal.SetConfigOption("GDAL_HTTP_UNSAFESSL", "YES")
        cookies = os.path.expanduser("~/cookies.txt")
        gdal.SetConfigOption("GDAL_HTTP_COOKIEFILE", cookies)
        gdal.SetConfigOption("GDAL_HTTP_COOKIEJAR", cookies)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _vsicurl_path(url: str) -> str:
    """Return a GDAL virtual filesystem path for an OPERA asset URL."""
    if url.startswith("s3://"):
        return f"/vsis3/{url[5:]}"
    if url.startswith(("https://", "http://")):
        return f"/vsicurl/{url}"
    return url


def _parse_bbox(bbox: Optional[str]) -> tuple[float, float, float, float] | None:
    """Parse a west,south,east,north bbox string."""
    if not bbox:
        return None
    parts = [float(part.strip()) for part in bbox.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have exactly four values: west,south,east,north")
    west, south, east, north = parts
    if west >= east or south >= north:
        raise ValueError("bbox coordinates must satisfy west < east and south < north")
    return west, south, east, north


def _current_bbox_wgs84(iface: Any) -> tuple[float, float, float, float]:
    """Return the current QGIS canvas extent as WGS84 coordinates."""

    def _run() -> tuple[float, float, float, float]:
        canvas = iface.mapCanvas()
        extent = canvas.extent()
        crs = canvas.mapSettings().destinationCrs()
        if crs.authid() != "EPSG:4326":
            from qgis.core import (  # type: ignore[import-not-found]
                QgsCoordinateReferenceSystem,
                QgsCoordinateTransform,
                QgsProject,
            )

            transform = QgsCoordinateTransform(
                crs,
                QgsCoordinateReferenceSystem("EPSG:4326"),
                QgsProject.instance(),
            )
            extent = transform.transformBoundingBox(extent)
        return (
            float(extent.xMinimum()),
            float(extent.yMinimum()),
            float(extent.xMaximum()),
            float(extent.yMaximum()),
        )

    return _on_gui(_run)


def _add_qgis_raster_layer(
    iface: Any,
    project_getter: Callable[[], Any],
    path_or_uri: str,
    layer_name: str,
) -> dict[str, Any]:
    """Add a raster layer to QGIS on the GUI thread."""

    def _run() -> dict[str, Any]:
        from qgis.core import QgsRasterLayer  # type: ignore[import-not-found]

        layer = QgsRasterLayer(path_or_uri, layer_name)
        if not layer.isValid():
            return {"error": f"Failed to load raster from {path_or_uri}."}
        project_getter().addMapLayer(layer)
        iface.mapCanvas().refresh()
        return {"success": True, "layer_name": layer_name}

    return _on_gui(_run)


def _granule_links(granule: Any) -> list[str]:
    """Return data links from an earthaccess granule."""
    if hasattr(granule, "data_links"):
        try:
            return list(granule.data_links())
        except Exception:
            return []
    links = (
        granule.get("umm", {}).get("RelatedUrls", []) if hasattr(granule, "get") else []
    )
    out: list[str] = []
    for link in links:
        url = link.get("URL") if isinstance(link, dict) else None
        if url:
            out.append(str(url))
    return out


def _granule_geometry(granule: Any) -> dict[str, Any] | None:
    """Extract a GeoJSON geometry from an earthaccess granule."""
    umm = granule.get("umm", {}) if hasattr(granule, "get") else {}
    spatial = umm.get("SpatialExtent", {})
    horizontal = spatial.get("HorizontalSpatialDomain", {})
    geometry = horizontal.get("Geometry", {})

    rects = geometry.get("BoundingRectangles", [])
    if rects:
        rect = rects[0]
        west = rect.get("WestBoundingCoordinate", 0)
        south = rect.get("SouthBoundingCoordinate", 0)
        east = rect.get("EastBoundingCoordinate", 0)
        north = rect.get("NorthBoundingCoordinate", 0)
        return {
            "type": "Polygon",
            "coordinates": [
                [
                    [west, south],
                    [east, south],
                    [east, north],
                    [west, north],
                    [west, south],
                ]
            ],
        }

    polygons = geometry.get("GPolygons", [])
    if polygons:
        points = polygons[0].get("Boundary", {}).get("Points", [])
        coords = [[p.get("Longitude", 0), p.get("Latitude", 0)] for p in points]
        if coords:
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            return {"type": "Polygon", "coordinates": [coords]}
    return None


def _granule_summary(granule: Any, *, max_links: int = 5) -> dict[str, Any]:
    """Return a compact, JSON-serializable granule summary."""
    meta = granule.get("meta", {}) if hasattr(granule, "get") else {}
    umm = granule.get("umm", {}) if hasattr(granule, "get") else {}
    temporal = umm.get("TemporalExtent", {}).get("RangeDateTime", {})
    links = _granule_links(granule)
    return {
        "native_id": meta.get("native-id", ""),
        "producer_granule_id": meta.get("producer-granule-id", ""),
        "concept_id": meta.get("concept-id", ""),
        "begin_date": temporal.get("BeginningDateTime", ""),
        "end_date": temporal.get("EndingDateTime", ""),
        "num_links": len(links),
        "data_links": links[:max_links],
    }


def _qgis_log(iface: Any, message: Any, level: Any = None) -> None:
    """Log to QGIS message log and message bar on the GUI thread."""
    try:
        text = str(message)

        def _run() -> None:
            from qgis.core import Qgis, QgsMessageLog  # type: ignore[import-not-found]

            msg_level = level if level is not None else Qgis.MessageLevel.Info
            QgsMessageLog.logMessage(text, "GeoAgent NASA OPERA", msg_level)
            try:
                iface.messageBar().pushMessage(
                    "GeoAgent NASA OPERA",
                    text,
                    level=msg_level,
                    duration=5,
                )
            except Exception:
                pass

        _on_gui(_run)
    except Exception:
        print(message)


def nasa_opera_tools(iface: Any, project: Optional[Any] = None) -> list[Any]:
    """Return GeoAgent tools for NASA OPERA workflows in QGIS.

    Args:
        iface: The QGIS ``QgisInterface`` from the NASA OPERA plugin runtime.
        project: Optional ``QgsProject`` instance. The tools fall back to
            ``QgsProject.instance()`` when omitted.
    """
    if iface is None:
        return []

    state: dict[str, Any] = {}

    def _project() -> Any:
        """Return the bound or singleton QGIS project."""
        if project is not None:
            return project
        from qgis.core import QgsProject  # type: ignore[import-not-found]

        return QgsProject.instance()

    @geo_tool(
        category="nasa_opera",
        name="get_available_datasets",
        available_in=("full", "fast"),
    )
    def get_available_datasets() -> list[dict[str, str]]:
        """List NASA OPERA product short names, titles, and descriptions."""
        return [
            {
                "short_name": short_name,
                "title": info["title"],
                "short_title": info["short_title"],
                "description": info["description"],
            }
            for short_name, info in _datasets().items()
        ]

    @geo_tool(
        category="nasa_opera",
        name="get_dataset_info",
        available_in=("full", "fast"),
    )
    def get_dataset_info(dataset_name: str) -> list[dict[str, str]]:
        """Find OPERA datasets by short name, title, short title, or keyword."""
        query = dataset_name.strip().upper()
        matches: list[dict[str, str]] = []
        for short_name, info in _datasets().items():
            haystack = " ".join(
                [
                    short_name,
                    info["title"],
                    info["short_title"],
                    info["description"],
                ]
            ).upper()
            if query in haystack:
                matches.append(
                    {
                        "short_name": short_name,
                        "title": info["title"],
                        "short_title": info["short_title"],
                        "description": info["description"],
                    }
                )
        return matches

    @geo_tool(
        category="nasa_opera",
        name="search_opera_data",
        requires_packages=("earthaccess",),
    )
    def search_opera_data(
        dataset: str,
        bbox: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: int = 20,
    ) -> dict[str, Any]:
        """Search NASA OPERA granules by product, bbox, and date range.

        Args:
            dataset: OPERA dataset short name, such as
                ``OPERA_L3_DSWX-HLS_V1``.
            bbox: Optional ``west,south,east,north`` WGS84 bounding box. When
                omitted, the current QGIS map extent is used.
            start_date: Optional start date in ``YYYY-MM-DD`` format.
            end_date: Optional end date in ``YYYY-MM-DD`` format.
            max_results: Maximum number of granules to return.
        """
        import earthaccess

        parsed_bbox = _parse_bbox(bbox)
        if parsed_bbox is None:
            parsed_bbox = _current_bbox_wgs84(iface)

        _earthdata_login()
        count = max(1, int(max_results))
        search_params: dict[str, Any] = {
            "short_name": dataset,
            "count": count,
            "bounding_box": parsed_bbox,
        }
        if start_date and end_date:
            search_params["temporal"] = (start_date, end_date)
        elif start_date:
            search_params["temporal"] = (
                start_date,
                datetime.today().strftime("%Y-%m-%d"),
            )

        results = list(earthaccess.search_data(**search_params))
        state["last_search_results"] = results
        state["last_search_dataset"] = dataset
        state["last_search_bbox"] = parsed_bbox

        return {
            "count": len(results),
            "dataset": dataset,
            "bbox": parsed_bbox,
            "granules": [_granule_summary(granule) for granule in results],
        }

    @geo_tool(
        category="nasa_opera",
        name="display_footprints",
    )
    def display_footprints(layer_name: str = "OPERA Footprints") -> dict[str, Any]:
        """Display footprints for the most recent OPERA search results."""

        def _run() -> dict[str, Any]:
            results = state.get("last_search_results")
            if not results:
                return {"error": "No OPERA search results are available."}

            features: list[dict[str, Any]] = []
            for granule in results:
                geometry = _granule_geometry(granule)
                if geometry is None:
                    continue
                summary = _granule_summary(granule, max_links=0)
                features.append(
                    {
                        "type": "Feature",
                        "geometry": geometry,
                        "properties": summary,
                    }
                )

            if not features:
                return {"error": "No valid footprint geometries found."}

            geojson = {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
                "features": features,
            }
            path = os.path.join(
                tempfile.gettempdir(), "geoagent_opera_footprints.geojson"
            )
            with open(path, "w", encoding="utf-8") as f:
                json.dump(geojson, f)

            from qgis.PyQt.QtGui import QColor  # type: ignore[import-not-found]
            from qgis.core import (  # type: ignore[import-not-found]
                QgsFillSymbol,
                QgsVectorLayer,
            )

            proj = _project()
            for existing in proj.mapLayersByName(layer_name):
                proj.removeMapLayer(existing.id())

            layer = QgsVectorLayer(path, layer_name, "ogr")
            if not layer.isValid():
                return {"error": "Failed to create OPERA footprint layer."}

            symbol = QgsFillSymbol.createSimple({})
            fill = symbol.symbolLayer(0)
            fill.setColor(QColor(25, 118, 210, 50))
            fill.setStrokeColor(QColor(25, 118, 210, 200))
            fill.setStrokeWidth(0.5)
            layer.renderer().setSymbol(symbol)

            proj.addMapLayer(layer)
            iface.mapCanvas().refresh()
            return {
                "success": True,
                "layer_name": layer_name,
                "feature_count": len(features),
                "path": path,
            }

        return _on_gui(_run)

    @geo_tool(
        category="nasa_opera",
        name="display_raster",
        requires_confirmation=True,
        long_running=True,
        requires_packages=("earthaccess",),
    )
    def display_raster(
        url: str,
        layer_name: Optional[str] = None,
        prefer_streaming: bool = True,
        cache_dir: Optional[str] = None,
    ) -> dict[str, Any]:
        """Load an OPERA raster asset URL into QGIS.

        The tool tries GDAL virtual-file streaming first, then falls back to
        downloading the file through ``earthaccess``.
        """
        url_basename = _filename_from_url(url, fallback="opera_raster")
        name = layer_name or url_basename or "OPERA Raster"
        if prefer_streaming:
            success, error = _setup_gdal_for_earthdata()
            if success:
                result = _add_qgis_raster_layer(
                    iface,
                    _project,
                    _vsicurl_path(url),
                    name,
                )
                if result.get("success"):
                    result["mode"] = "streaming"
                    return result
            state["last_streaming_error"] = error

        import earthaccess

        _earthdata_login()
        target_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), "nasa_opera_cache"
        )
        os.makedirs(target_dir, exist_ok=True)
        local_path = os.path.join(target_dir, url_basename)
        if not os.path.exists(local_path):
            downloaded = earthaccess.download([url], local_path=target_dir, threads=1)
            if downloaded:
                local_path = str(downloaded[0])

        result = _add_qgis_raster_layer(iface, _project, local_path, name)
        if result.get("success"):
            return {
                "success": True,
                "layer_name": name,
                "mode": "download",
                "path": local_path,
            }
        return result

    @geo_tool(
        category="nasa_opera",
        name="create_mosaic",
        requires_confirmation=True,
        long_running=True,
        requires_packages=("osgeo", "earthaccess"),
    )
    def create_mosaic(
        urls: list[str],
        layer_name: str = "OPERA Mosaic",
    ) -> dict[str, Any]:
        """Create and display a GDAL VRT mosaic from OPERA raster URLs."""
        if not urls:
            return {"error": "No URLs provided."}

        try:
            from osgeo import gdal  # type: ignore[import-not-found]
        except ImportError as exc:
            return {"error": f"GDAL Python bindings are not available: {exc}"}

        success, error = _setup_gdal_for_earthdata()
        if not success:
            return {"error": f"Failed to configure GDAL: {error}"}

        accessible: list[str] = []
        for path in [_vsicurl_path(url) for url in urls]:
            ds = gdal.Open(path)
            if ds:
                accessible.append(path)
                ds = None

        if not accessible:
            return {"error": "None of the provided OPERA URLs are accessible."}

        safe_stem = _safe_filename(layer_name, fallback="opera_mosaic")
        vrt_path = os.path.join(
            tempfile.gettempdir(), f"{safe_stem}_{uuid.uuid4().hex}.vrt"
        )
        vrt_ds = gdal.BuildVRT(vrt_path, accessible)
        if vrt_ds is None:
            return {"error": "Failed to build OPERA VRT mosaic."}
        vrt_ds.FlushCache()
        vrt_ds = None

        result = _add_qgis_raster_layer(iface, _project, vrt_path, layer_name)
        if result.get("success"):
            result["path"] = vrt_path
            result["source_count"] = len(accessible)
        return result

    return [
        get_available_datasets,
        get_dataset_info,
        search_opera_data,
        display_footprints,
        display_raster,
        create_mosaic,
    ]


def submit_nasa_opera_search_task(
    iface: Any,
    *,
    dataset: str,
    bbox: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_results: int = 20,
    project: Optional[Any] = None,
    display_footprints: bool = True,
    layer_name: str = "OPERA Footprints",
    on_finished: Optional[Callable[[dict[str, Any] | None], None]] = None,
) -> Any:
    """Submit an OPERA search as a QGIS-managed background task.

    This helper is intended for the QGIS Python console and plugin UI code. It
    reports progress through QGIS's Log Messages panel and message bar, and it
    retains the returned task in a module-level list so the task remains alive.
    """
    if iface is None:
        raise ValueError(
            "submit_nasa_opera_search_task requires a QGIS iface; got None."
        )

    from qgis.core import Qgis, QgsApplication, QgsTask  # type: ignore[import-not-found]

    tools = {tool.tool_name: tool for tool in nasa_opera_tools(iface, project)}
    required = ("search_opera_data", "display_footprints")
    missing = [name for name in required if name not in tools]
    if missing:
        raise RuntimeError(
            "NASA OPERA tool registration missing required tools: " + ", ".join(missing)
        )

    def _run(task: Any) -> dict[str, Any]:
        task.setProgress(1)
        result = tools["search_opera_data"](
            dataset=dataset,
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            max_results=max_results,
        )
        task.setProgress(90)
        return result

    def _finished(
        exception: BaseException | None, result: dict[str, Any] | None = None
    ) -> None:
        try:
            if exception is not None:
                _qgis_log(
                    iface,
                    f"NASA OPERA search failed: {exception}",
                    Qgis.MessageLevel.Critical,
                )
                if on_finished is not None:
                    on_finished(None)
                return
            if result is None:
                _qgis_log(
                    iface,
                    "NASA OPERA search was cancelled.",
                    Qgis.MessageLevel.Warning,
                )
                if on_finished is not None:
                    on_finished(None)
                return

            _qgis_log(
                iface,
                f"NASA OPERA search finished: {result['count']} granules found.",
            )
            if display_footprints:
                footprint_result = tools["display_footprints"](layer_name=layer_name)
                _qgis_log(iface, footprint_result)
            if on_finished is not None:
                on_finished(result)
        finally:
            try:
                _QGIS_TASKS.remove(task)
            except ValueError:
                pass

    task = QgsTask.fromFunction(
        "GeoAgent NASA OPERA search",
        _run,
        on_finished=_finished,
    )
    _QGIS_TASKS.append(task)
    QgsApplication.taskManager().addTask(task)
    _qgis_log(iface, "Submitted GeoAgent NASA OPERA search task.")
    return task


def submit_nasa_opera_chat_task(
    agent: Any,
    query: str,
    *,
    on_finished: Optional[Callable[[Any | None], None]] = None,
) -> Any:
    """Fail closed for NASA OPERA chat inside QGIS.

    LLM-driven chat can choose QGIS tools dynamically, which has repeatedly
    triggered Qt thread-affinity crashes in the QGIS process. Use
    :func:`submit_nasa_opera_search_task` or direct tools instead.
    """
    from qgis.core import Qgis  # type: ignore[import-not-found]

    iface = agent.context.qgis_iface
    _qgis_log(
        iface,
        "NASA OPERA chat is disabled in QGIS. Use submit_nasa_opera_search_task(...) "
        "or direct nasa_opera_tools(...) calls instead.",
        Qgis.MessageLevel.Critical,
    )
    if on_finished is not None:
        on_finished(None)
    return None


__all__ = [
    "OPERA_DATASETS",
    "nasa_opera_tools",
    "submit_nasa_opera_chat_task",
    "submit_nasa_opera_search_task",
]
