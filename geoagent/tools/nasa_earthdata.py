"""Tool adapters for the NASA Earthdata QGIS plugin.

The module is import-safe outside QGIS and outside the plugin. Tool bodies
resolve QGIS, earthaccess, and plugin UI objects lazily so ordinary GeoAgent
imports keep working in non-QGIS environments.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
import uuid
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urlparse
from urllib.request import urlopen

from geoagent.core.decorators import geo_tool
from geoagent.tools._qt_marshal import run_on_qt_gui_thread

NASA_DATA_URL = (
    "https://github.com/opengeos/NASA-Earth-Data/raw/main/nasa_earth_data.tsv"
)


def _on_gui(fn: Any) -> Any:
    """Run ``fn`` on the Qt GUI thread when QGIS is available."""
    return run_on_qt_gui_thread(fn)


def _catalog_url_from_settings() -> str:
    """Return the plugin-configured catalog URL or the built-in default."""
    try:
        from qgis.PyQt.QtCore import QSettings  # type: ignore[import-not-found]

        value = QSettings().value("NASAEarthdata/catalog_url", "", type=str)
        value = value.strip() if value else ""
        if value:
            return value
    except Exception:
        pass
    return NASA_DATA_URL


def _load_catalog_rows(catalog_url: str | None = None) -> list[dict[str, str]]:
    """Load NASA Earthdata catalog rows from the TSV catalog."""
    url = catalog_url or _catalog_url_from_settings()
    if not str(url).lower().startswith("https://"):
        raise ValueError("NASA Earthdata catalog URL must use HTTPS.")
    with urlopen(url, timeout=30) as response:  # nosec B310 - HTTPS enforced above
        text = response.read().decode("utf-8")
    return list(csv.DictReader(text.splitlines(), delimiter="\t"))


def _compact_catalog_row(row: dict[str, str]) -> dict[str, str]:
    """Return a compact, JSON-friendly catalog row."""
    keys = [
        "ShortName",
        "EntryTitle",
        "Platform",
        "Instrument",
        "ProcessingLevel",
        "Provider",
        "StartTime",
        "EndTime",
        "OnlineAccessURLs",
    ]
    out = {key: row.get(key, "") for key in keys if row.get(key)}
    summary = row.get("Summary") or row.get("Abstract") or row.get("Description")
    if summary:
        out["Summary"] = summary[:600]
    return out


def _earthdata_login(username: str | None = None, password: str | None = None) -> None:
    """Authenticate with NASA Earthdata using explicit, env, or netrc credentials."""
    import earthaccess

    username = (username or os.environ.get("EARTHDATA_USERNAME", "")).strip()
    password = (password or os.environ.get("EARTHDATA_PASSWORD", "")).strip()
    if username and password:
        os.environ["EARTHDATA_USERNAME"] = username
        os.environ["EARTHDATA_PASSWORD"] = password
        auth = earthaccess.login(strategy="environment")
        if getattr(auth, "authenticated", bool(auth)):
            return

    for strategy in ("environment", "netrc"):
        try:
            auth = earthaccess.login(strategy=strategy)
            if getattr(auth, "authenticated", bool(auth)):
                return
        except Exception:
            continue
    raise RuntimeError(
        "NASA Earthdata authentication failed. Configure credentials in the "
        "NASA Earthdata plugin settings, environment, or ~/.netrc."
    )


def _parse_bbox(bbox: str | list[float] | tuple[float, ...] | None) -> (
    tuple[
        float,
        float,
        float,
        float,
    ]
    | None
):
    """Parse west,south,east,north bounds."""
    if bbox is None or bbox == "":
        return None
    values = list(bbox) if isinstance(bbox, (list, tuple)) else str(bbox).split(",")
    if len(values) != 4:
        raise ValueError("bbox must contain west,south,east,north")
    west, south, east, north = [float(value) for value in values]
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
        "collection_concept_id": meta.get("collection-concept-id", ""),
        "begin_date": temporal.get("BeginningDateTime", ""),
        "end_date": temporal.get("EndingDateTime", ""),
        "size_mb": meta.get("size-mb", ""),
        "num_links": len(links),
        "data_links": links[:max_links],
    }


def _safe_basename(value: str, *, fallback: str = "earthdata") -> str:
    """Return a filesystem-safe basename."""
    parsed = urlparse(value)
    raw = os.path.basename(parsed.path) if parsed.scheme else os.path.basename(value)
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)
    return cleaned.strip("._") or fallback


def _add_qgis_raster_layer(
    iface: Any,
    project_getter: Any,
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


def earthdata_tools(
    iface: Any | None = None,
    project: Optional[Any] = None,
    *,
    plugin: Any | None = None,
) -> list[Any]:
    """Return GeoAgent tools for NASA Earthdata workflows in QGIS."""
    if iface is None:
        return []

    state: dict[str, Any] = {}

    def _project() -> Any:
        if project is not None:
            return project
        from qgis.core import QgsProject  # type: ignore[import-not-found]

        return QgsProject.instance()

    def _ensure_plugin_dock(kind: str) -> dict[str, Any]:
        """Show a plugin dock if a plugin instance was supplied."""
        if plugin is None:
            return {"success": False, "error": "No NASA Earthdata plugin instance."}

        def _run() -> dict[str, Any]:
            if kind == "settings":
                attr = "_settings_dock"
                creator = "toggle_settings_dock"
            else:
                attr = "_earthdata_dock"
                creator = "toggle_earthdata_dock"
            dock = getattr(plugin, attr, None)
            if dock is None and hasattr(plugin, creator):
                getattr(plugin, creator)()
                dock = getattr(plugin, attr, None)
            if dock is None:
                return {"success": False, "error": f"Could not create {kind} dock."}
            dock.show()
            dock.raise_()
            return {"success": True, "dock": kind}

        return _on_gui(_run)

    @geo_tool(
        category="nasa_earthdata",
        name="search_earthdata_catalog",
        available_in=("full", "fast"),
    )
    def search_earthdata_catalog(
        query: str = "",
        max_results: int = 20,
        catalog_url: Optional[str] = None,
    ) -> dict[str, Any]:
        """Search the NASA Earthdata dataset catalog by short name or title."""
        rows = _load_catalog_rows(catalog_url)
        q = query.strip().lower()
        matches = []
        for row in rows:
            haystack = " ".join(
                [
                    row.get("ShortName", ""),
                    row.get("EntryTitle", ""),
                    row.get("Summary", ""),
                    row.get("Platform", ""),
                    row.get("Instrument", ""),
                ]
            ).lower()
            if not q or q in haystack:
                matches.append(_compact_catalog_row(row))
        count = max(1, int(max_results))
        return {
            "count": len(matches),
            "shown": min(len(matches), count),
            "datasets": matches[:count],
        }

    @geo_tool(
        category="nasa_earthdata",
        name="get_earthdata_dataset_info",
        available_in=("full", "fast"),
    )
    def get_earthdata_dataset_info(
        short_name_or_query: str,
        catalog_url: Optional[str] = None,
    ) -> dict[str, Any]:
        """Return NASA Earthdata catalog metadata for a dataset short name."""
        rows = _load_catalog_rows(catalog_url)
        q = short_name_or_query.strip().lower()
        matches = []
        for row in rows:
            short_name = row.get("ShortName", "").lower()
            title = row.get("EntryTitle", "").lower()
            if q == short_name or q in short_name or q in title:
                matches.append(_compact_catalog_row(row))
        return {"found": bool(matches), "matches": matches[:10]}

    @geo_tool(
        category="nasa_earthdata",
        name="search_earthdata_data",
        requires_packages=("earthaccess",),
    )
    def search_earthdata_data(
        short_name: str,
        bbox: str | list[float] | None = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: int = 50,
        provider: Optional[str] = None,
        version: Optional[str] = None,
        cloud_cover_min: Optional[float] = None,
        cloud_cover_max: Optional[float] = None,
        day_night: Optional[str] = None,
        granule_id: Optional[str] = None,
        orbit_number: Optional[int] = None,
    ) -> dict[str, Any]:
        """Search NASA Earthdata granules by dataset, bbox, and filters."""
        import earthaccess

        parsed_bbox = _parse_bbox(bbox)
        if parsed_bbox is None:
            parsed_bbox = _current_bbox_wgs84(iface)

        _earthdata_login()
        search_params: dict[str, Any] = {
            "short_name": short_name,
            "count": max(1, int(max_results)),
            "bounding_box": parsed_bbox,
        }
        if start_date and end_date:
            search_params["temporal"] = (start_date, end_date)
        elif start_date:
            search_params["temporal"] = (
                start_date,
                datetime.today().strftime("%Y-%m-%d"),
            )
        if provider:
            search_params["provider"] = provider
        if version:
            search_params["version"] = version
        if cloud_cover_min is not None or cloud_cover_max is not None:
            search_params["cloud_cover"] = (
                cloud_cover_min or 0,
                cloud_cover_max if cloud_cover_max is not None else 100,
            )
        if day_night:
            search_params["day_night_flag"] = day_night
        if granule_id:
            search_params["granule_ur"] = granule_id
        if orbit_number is not None:
            search_params["orbit_number"] = orbit_number

        results = list(earthaccess.search_data(**search_params))
        state["last_search_results"] = results
        state["last_search_short_name"] = short_name
        state["last_search_bbox"] = parsed_bbox

        return {
            "count": len(results),
            "short_name": short_name,
            "bbox": parsed_bbox,
            "granules": [_granule_summary(granule) for granule in results],
        }

    @geo_tool(
        category="nasa_earthdata",
        name="display_earthdata_footprints",
        requires_confirmation=True,
    )
    def display_earthdata_footprints(
        layer_name: str = "NASA Earthdata Footprints",
    ) -> dict[str, Any]:
        """Display footprints for the most recent NASA Earthdata search."""

        def _run() -> dict[str, Any]:
            results = state.get("last_search_results")
            if not results:
                return {"error": "No NASA Earthdata search results are available."}

            features: list[dict[str, Any]] = []
            for granule in results:
                geometry = _granule_geometry(granule)
                if geometry is None:
                    continue
                features.append(
                    {
                        "type": "Feature",
                        "geometry": geometry,
                        "properties": _granule_summary(granule, max_links=0),
                    }
                )

            if not features:
                return {"error": "No valid footprint geometries found."}

            path = os.path.join(
                tempfile.gettempdir(),
                f"geoagent_earthdata_footprints_{uuid.uuid4().hex}.geojson",
            )
            geojson = {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
                "features": features,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(geojson, f)

            from qgis.PyQt.QtGui import QColor  # type: ignore[import-not-found]
            from qgis.core import QgsFillSymbol, QgsVectorLayer  # type: ignore

            proj = _project()
            for existing in proj.mapLayersByName(layer_name):
                proj.removeMapLayer(existing.id())

            layer = QgsVectorLayer(path, layer_name, "ogr")
            if not layer.isValid():
                return {"error": "Failed to create NASA Earthdata footprint layer."}

            symbol = QgsFillSymbol.createSimple({})
            fill = symbol.symbolLayer(0)
            fill.setColor(QColor(11, 61, 145, 50))
            fill.setStrokeColor(QColor(11, 61, 145, 220))
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
        category="nasa_earthdata",
        name="load_earthdata_raster",
        requires_confirmation=True,
        long_running=True,
        requires_packages=("earthaccess",),
    )
    def load_earthdata_raster(
        url: str,
        layer_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> dict[str, Any]:
        """Download a NASA Earthdata raster asset URL and load it into QGIS."""
        import earthaccess

        _earthdata_login()
        target_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), "nasa_earthdata_cache"
        )
        os.makedirs(target_dir, exist_ok=True)
        basename = _safe_basename(url, fallback="earthdata_raster")
        local_path = os.path.join(target_dir, basename)
        if not os.path.exists(local_path):
            downloaded = earthaccess.download([url], local_path=target_dir, threads=1)
            if downloaded:
                local_path = str(downloaded[0])
        name = layer_name or os.path.basename(local_path) or "NASA Earthdata Raster"
        result = _add_qgis_raster_layer(iface, _project, local_path, name)
        if result.get("success"):
            result["path"] = local_path
        return result

    @geo_tool(
        category="nasa_earthdata",
        name="open_nasa_earthdata_search_panel",
        requires_confirmation=True,
    )
    def open_nasa_earthdata_search_panel() -> dict[str, Any]:
        """Open or focus the NASA Earthdata plugin search panel."""
        return _ensure_plugin_dock("search")

    @geo_tool(
        category="nasa_earthdata",
        name="open_nasa_earthdata_settings",
        requires_confirmation=True,
    )
    def open_nasa_earthdata_settings() -> dict[str, Any]:
        """Open or focus the NASA Earthdata plugin settings panel."""
        return _ensure_plugin_dock("settings")

    return [
        search_earthdata_catalog,
        get_earthdata_dataset_info,
        search_earthdata_data,
        display_earthdata_footprints,
        load_earthdata_raster,
        open_nasa_earthdata_search_panel,
        open_nasa_earthdata_settings,
    ]


__all__ = ["earthdata_tools"]
