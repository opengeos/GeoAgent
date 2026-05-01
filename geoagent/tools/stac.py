"""STAC catalog tools and optional QGIS raster loading helpers."""

from __future__ import annotations

from typing import Any

from geoagent.core.decorators import geo_tool
from geoagent.tools._qt_marshal import run_on_qt_gui_thread

STAC_HTTP_TIMEOUT = (3.05, 15.0)
STAC_DEFAULT_LIMIT = 5
STAC_MAX_LIMIT = 10
STAC_CLOUD_CANDIDATE_LIMIT = 50
PLANETARY_COMPUTER_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
_ACTIVE_QGIS_STAC_TASKS: list[Any] = []


def _coerce_bbox(value: Any) -> list[float] | None:
    """Return a STAC bbox list from common string/list inputs."""
    if value in (None, ""):
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        try:
            parts = list(value)
        except TypeError:
            parts = []
    if len(parts) != 4:
        raise ValueError("bbox must contain west,south,east,north values.")
    return [float(part) for part in parts]


def _open_catalog(catalog_url: str) -> Any:
    """Open a STAC catalog with pystac-client."""
    from pystac_client import Client

    return Client.open(_resolve_catalog_url(catalog_url), timeout=STAC_HTTP_TIMEOUT)


def _resolve_catalog_url(catalog_url: str | None = None) -> str:
    """Return a concrete STAC catalog URL, defaulting to Planetary Computer."""
    return str(catalog_url or "").strip() or PLANETARY_COMPUTER_STAC_URL


def _bounded_limit(limit: Any, default: int = STAC_DEFAULT_LIMIT) -> int:
    """Return a small positive STAC result limit suitable for interactive QGIS."""
    try:
        value = int(limit or default)
    except (TypeError, ValueError):
        value = default
    return max(1, min(value, STAC_MAX_LIMIT))


def _optional_float(value: Any) -> float | None:
    """Return a float for non-empty numeric values."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cloud_cover_from_properties(properties: dict[str, Any]) -> float | None:
    """Return STAC cloud-cover percentage from common property names."""
    for key in ("eo:cloud_cover", "s2:cloudy_pixel_percentage", "cloud_cover"):
        value = _optional_float(properties.get(key))
        if value is not None:
            return value
    return None


def _is_low_cloud_request(query_text: str) -> bool:
    """Return True when free text asks for cloud-free or low-cloud imagery."""
    text = str(query_text or "").lower()
    return any(
        phrase in text
        for phrase in (
            "cloud free",
            "cloud-free",
            "low cloud",
            "clear sky",
            "least cloudy",
            "minimal cloud",
        )
    )


def _is_cloud_aware_collection(collection: str) -> bool:
    """Return True for collections where cloud-cover ranking is expected."""
    text = str(collection or "").lower()
    return any(token in text for token in ("sentinel-2", "landsat", "modis"))


def _bbox_area(bbox: list[float] | None) -> float:
    """Return bbox area in degree units for ranking only."""
    if not bbox or len(bbox) != 4:
        return 0.0
    west, south, east, north = bbox
    return max(0.0, east - west) * max(0.0, north - south)


def _bbox_intersection_area(a: list[float] | None, b: list[float] | None) -> float:
    """Return intersection area for two [west, south, east, north] bboxes."""
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    west = max(a[0], b[0])
    south = max(a[1], b[1])
    east = min(a[2], b[2])
    north = min(a[3], b[3])
    return max(0.0, east - west) * max(0.0, north - south)


def _bbox_center(bbox: list[float] | None) -> tuple[float, float] | None:
    """Return bbox center point as lon, lat."""
    if not bbox or len(bbox) != 4:
        return None
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _point_in_ring(point: tuple[float, float], ring: list[Any]) -> bool:
    """Return True when a point is inside a polygon ring."""
    if len(ring) < 3:
        return False
    x, y = point
    inside = False
    j = len(ring) - 1
    for i, coord in enumerate(ring):
        xi, yi = float(coord[0]), float(coord[1])
        xj, yj = float(ring[j][0]), float(ring[j][1])
        if (yi > y) != (yj > y):
            x_at_y = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_at_y:
                inside = not inside
        j = i
    return inside


def _geometry_contains_point(
    geometry: dict[str, Any], point: tuple[float, float]
) -> bool:
    """Return True for Polygon/MultiPolygon geometries containing a point."""
    try:
        geom_type = geometry.get("type")
        coords = geometry.get("coordinates") or []
        polygons = [coords] if geom_type == "Polygon" else coords
        if geom_type not in {"Polygon", "MultiPolygon"}:
            return False
        for polygon in polygons:
            if not polygon or not _point_in_ring(point, polygon[0]):
                continue
            if any(_point_in_ring(point, hole) for hole in polygon[1:]):
                continue
            return True
    except Exception:
        return False
    return False


def _preferred_asset_summaries(item_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a small set of likely QGIS-loadable raster assets."""
    assets = item_dict.get("assets") or {}
    preferred = []
    priority = (
        "visual",
        "data",
        "elevation",
        "dem",
        "DSM",
        "dtm",
        "analytic",
        "rendered_preview",
        "B04",
        "B03",
        "B02",
        "red",
        "green",
        "blue",
    )
    keys = [key for key in priority if key in assets]
    keys.extend(key for key in sorted(assets) if key not in keys)
    for key in keys[:5]:
        asset = assets.get(key) or {}
        asset_dict = asset.to_dict() if hasattr(asset, "to_dict") else dict(asset)
        media_type = str(asset_dict.get("type") or "")
        href = str(asset_dict.get("href") or "")
        if not href:
            continue
        if key not in priority and not any(
            token in media_type.lower() for token in ("tiff", "geotiff", "cog")
        ):
            continue
        preferred.append(
            {
                "key": key,
                "href": href,
                "title": asset_dict.get("title"),
                "type": asset_dict.get("type"),
                "roles": asset_dict.get("roles") or [],
            }
        )
    return preferred[:3]


def _is_remote_http_href(href: str) -> bool:
    """Return True for HTTP(S) asset URLs."""
    return str(href or "").strip().lower().startswith(("http://", "https://"))


def _qgis_raster_uri(href: str) -> str:
    """Return a QGIS/GDAL raster URI for local paths and remote COG assets."""
    value = str(href or "").strip()
    lower = value.lower()
    if lower.startswith("/vsi"):
        return value
    if _is_remote_http_href(value):
        return f"/vsicurl/{value}"
    return value


def _configure_gdal_for_remote_cog() -> None:
    """Set GDAL options that avoid expensive directory probing for COG URLs."""
    try:
        from osgeo import gdal  # type: ignore[import-not-found]

        gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    except Exception:
        return


def _qgis_layer_error(layer: Any) -> str:
    """Return a readable QGIS layer error message when available."""
    if layer is None:
        return "QGIS did not create a raster layer."
    try:
        error = layer.error()
    except Exception:
        return "QGIS reported the raster layer is invalid."
    for attr in ("summary", "message"):
        try:
            value = getattr(error, attr)
            text = value() if callable(value) else value
            if text:
                return str(text)
        except Exception:
            continue
    return "QGIS reported the raster layer is invalid."


def _zoom_to_qgis_layer(iface: Any, layer: Any) -> bool:
    """Set the loaded layer active and zoom the QGIS canvas to its extent."""
    if iface is None or layer is None:
        return False
    zoomed = False
    try:
        if hasattr(iface, "setActiveLayer"):
            iface.setActiveLayer(layer)
    except Exception:
        pass
    try:
        canvas = iface.mapCanvas()
        extent = layer.extent() if hasattr(layer, "extent") else None
        if extent is not None and hasattr(canvas, "setExtent"):
            try:
                from geoagent.tools.qgis import _transform_extent_to_canvas_crs

                extent = _transform_extent_to_canvas_crs(layer, canvas, extent)
            except Exception:
                pass
            canvas.setExtent(extent)
            zoomed = True
    except Exception:
        pass
    if not zoomed and hasattr(iface, "zoomToActiveLayer"):
        try:
            iface.zoomToActiveLayer()
            zoomed = True
        except Exception:
            pass
    try:
        canvas = iface.mapCanvas()
        if hasattr(canvas, "refresh"):
            canvas.refresh()
    except Exception:
        pass
    return zoomed


def _schedule_zoom_to_qgis_layer(iface: Any, layer: Any) -> None:
    """Zoom after QGIS has finished wiring the layer into the layer tree."""
    try:
        from qgis.PyQt.QtCore import QTimer  # type: ignore[import-not-found]
    except Exception:
        _zoom_to_qgis_layer(iface, layer)
        return

    QTimer.singleShot(0, lambda: _zoom_to_qgis_layer(iface, layer))
    QTimer.singleShot(500, lambda: _zoom_to_qgis_layer(iface, layer))


def _set_qgis_status_message(iface: Any, message: str, timeout_ms: int = 0) -> None:
    """Show a short bottom status-bar message when QGIS exposes one."""
    try:
        window = iface.mainWindow()
        status_bar = window.statusBar()
        status_bar.showMessage(message, int(timeout_ms or 0))
    except Exception:
        return


def _queue_qgis_raster_load(
    iface: Any,
    uri: str,
    layer_name: str,
) -> dict[str, Any] | None:
    """Queue a QGIS background task to create a raster layer without UI blocking."""
    try:
        from qgis.core import (  # type: ignore[import-not-found]
            Qgis,
            QgsApplication,
            QgsMessageLog,
            QgsProject,
            QgsRasterLayer,
            QgsTask,
        )
    except Exception:
        return None

    _configure_gdal_for_remote_cog()

    class _RasterLoadTask(QgsTask):
        """Load a raster provider in QGIS' task manager."""

        def __init__(self) -> None:
            super().__init__(f"Load STAC raster: {layer_name}")
            self.error = ""
            self.gdal_description = ""

        def run(self) -> bool:
            try:
                self.setProgress(5)
                from osgeo import gdal  # type: ignore[import-not-found]

                _configure_gdal_for_remote_cog()
                dataset = gdal.Open(uri)
                valid = dataset is not None
                if valid:
                    try:
                        self.gdal_description = str(dataset.GetDescription() or "")
                    except Exception:
                        self.gdal_description = ""
                    dataset = None
                else:
                    self.error = (
                        gdal.GetLastErrorMsg()
                        or "GDAL could not open the STAC raster URI."
                    )
                self.setProgress(100 if valid else 0)
                return valid
            except Exception as exc:  # pragma: no cover - QGIS runtime path
                self.error = f"{type(exc).__name__}: {exc}"
                return False

    def _enqueue() -> dict[str, Any]:
        task = _RasterLoadTask()
        callbacks: dict[str, Any] = {}

        def _message(text: str, level: Any) -> None:
            try:
                QgsMessageLog.logMessage(text, "OpenGeoAgent STAC", level)
            except Exception:
                pass

        def _cleanup() -> None:
            try:
                _ACTIVE_QGIS_STAC_TASKS.remove(task)
            except ValueError:
                pass
            callbacks.clear()

        def _on_completed() -> None:
            message_level = getattr(Qgis, "MessageLevel", Qgis)
            layer = QgsRasterLayer(uri, layer_name)
            if layer is not None and layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                _schedule_zoom_to_qgis_layer(iface, layer)
                _set_qgis_status_message(
                    iface,
                    f"Loaded STAC raster layer: {layer_name}",
                    7000,
                )
                _message(
                    f"Loaded STAC raster layer and zoomed to extent: {layer_name}",
                    getattr(message_level, "Info"),
                )
            else:
                reason = task.error or _qgis_layer_error(layer)
                _message(
                    f"Failed to load STAC raster layer {layer_name!r}: {reason}",
                    getattr(message_level, "Critical"),
                )
            _cleanup()

        def _on_terminated() -> None:
            message_level = getattr(Qgis, "MessageLevel", Qgis)
            reason = task.error or "QGIS terminated the STAC raster loading task."
            _set_qgis_status_message(
                iface,
                f"STAC raster load failed: {layer_name}",
                7000,
            )
            _message(
                f"STAC raster loading task terminated for {layer_name!r}: {reason}",
                getattr(message_level, "Critical"),
            )
            _cleanup()

        callbacks["completed"] = _on_completed
        callbacks["terminated"] = _on_terminated
        task._geoagent_callbacks = callbacks  # noqa: SLF001 - retain Qt callbacks
        task.taskCompleted.connect(_on_completed)
        task.taskTerminated.connect(_on_terminated)
        _ACTIVE_QGIS_STAC_TASKS.append(task)
        _message(
            f"Started STAC raster loading task for {layer_name}: {uri}",
            getattr(getattr(Qgis, "MessageLevel", Qgis), "Info"),
        )
        _set_qgis_status_message(iface, f"Loading STAC raster: {layer_name}")
        QgsApplication.taskManager().addTask(task)
        return {
            "success": True,
            "loaded": False,
            "queued": True,
            "load_mode": "qgis_task",
            "qgis_uri": uri,
            "layer_name": layer_name,
            "reason": (
                "Queued a QGIS background task to load the STAC raster. "
                "The layer will be added to the project when QGIS validates it."
            ),
        }

    return run_on_qt_gui_thread(_enqueue)


def _item_summary(item: Any, query_bbox: list[float] | None = None) -> dict[str, Any]:
    """Return a compact JSON-friendly STAC item summary."""
    item_dict = item.to_dict() if hasattr(item, "to_dict") else dict(item)
    geometry = item_dict.get("geometry") or {}
    properties = item_dict.get("properties") or {}
    assets = item_dict.get("assets") or {}
    cloud_cover = _cloud_cover_from_properties(properties)
    item_bbox = item_dict.get("bbox")
    center = _bbox_center(query_bbox)
    overlap_area = _bbox_intersection_area(item_bbox, query_bbox)
    query_area = _bbox_area(query_bbox)
    return {
        "id": item_dict.get("id"),
        "collection": item_dict.get("collection"),
        "bbox": item_bbox,
        "datetime": properties.get("datetime"),
        "cloud_cover": cloud_cover,
        "mgrs_tile": properties.get("s2:mgrs_tile"),
        "contains_query_center": (
            _geometry_contains_point(geometry, center) if center else None
        ),
        "bbox_overlap_ratio": (
            round(overlap_area / query_area, 6) if query_area > 0 else None
        ),
        "geometry_type": geometry.get("type") if isinstance(geometry, dict) else None,
        "asset_keys": sorted(assets.keys()),
        "preferred_assets": _preferred_asset_summaries(item_dict),
    }


def _asset_summary(key: str, asset: Any, *, sign: bool = True) -> dict[str, Any]:
    """Return a compact STAC asset summary."""
    asset_dict = asset.to_dict() if hasattr(asset, "to_dict") else dict(asset)
    href = asset_dict.get("href", "")
    if sign and href:
        href = _sign_href(href)
    return {
        "key": key,
        "href": href,
        "title": asset_dict.get("title"),
        "type": asset_dict.get("type"),
        "roles": asset_dict.get("roles") or [],
    }


def _sign_href(href: str) -> str:
    """Sign a URL with planetary-computer when available."""
    try:
        import planetary_computer

        return planetary_computer.sign(href)
    except Exception:
        return href


def _get_item(catalog_url: str, collection: str, item_id: str) -> Any:
    """Fetch one item from a STAC catalog."""
    catalog = _open_catalog(catalog_url)
    search = catalog.search(collections=[collection], ids=[item_id], max_items=1)
    items = list(search.items())
    if items:
        return items[0]
    if hasattr(catalog, "get_item"):
        try:
            item = catalog.get_item(item_id, recursive=True)
        except Exception:
            item = None
        if item is not None:
            return item
    raise ValueError(f"No STAC item {item_id!r} found in collection {collection!r}.")


def _extent_to_list(extent: Any) -> list[float] | None:
    """Return [west, south, east, north] values from a QGIS-like extent."""
    if extent is None:
        return None
    if isinstance(extent, (list, tuple)) and len(extent) == 4:
        return [float(value) for value in extent]
    accessors = ("xMinimum", "yMinimum", "xMaximum", "yMaximum")
    if all(hasattr(extent, name) for name in accessors):
        return [float(getattr(extent, name)()) for name in accessors]
    try:
        values = list(extent)
        if len(values) == 4:
            return [float(value) for value in values]
    except Exception:
        pass
    return None


def _canvas_extent_to_wgs84_bbox(canvas: Any) -> dict[str, Any]:
    """Return the current canvas extent as a WGS84 STAC bbox."""
    extent = canvas.extent() if hasattr(canvas, "extent") else None
    raw_bbox = _extent_to_list(extent)
    if raw_bbox is None:
        return {
            "success": False,
            "bbox": None,
            "crs": None,
            "reason": "The QGIS map canvas did not expose an extent.",
        }

    source_crs = ""
    if hasattr(canvas, "mapSettings"):
        try:
            crs = canvas.mapSettings().destinationCrs()
            source_crs = crs.authid() if hasattr(crs, "authid") else str(crs)
        except Exception:
            source_crs = ""

    if not source_crs or source_crs.upper() in {"EPSG:4326", "OGC:CRS84"}:
        return {
            "success": True,
            "bbox": raw_bbox,
            "crs": source_crs or "unknown",
            "source_bbox": raw_bbox,
            "source_crs": source_crs or "unknown",
        }

    try:
        from qgis.core import (  # type: ignore[import-not-found]
            QgsCoordinateReferenceSystem,
            QgsCoordinateTransform,
            QgsProject,
            QgsRectangle,
        )

        src = QgsCoordinateReferenceSystem(source_crs)
        dst = QgsCoordinateReferenceSystem("EPSG:4326")
        rect = QgsRectangle(*raw_bbox)
        transform = QgsCoordinateTransform(src, dst, QgsProject.instance())
        transformed = transform.transformBoundingBox(rect)
        bbox = _extent_to_list(transformed)
        if bbox is not None:
            return {
                "success": True,
                "bbox": bbox,
                "crs": "EPSG:4326",
                "source_bbox": raw_bbox,
                "source_crs": source_crs,
            }
    except Exception as exc:
        return {
            "success": False,
            "bbox": None,
            "crs": None,
            "source_bbox": raw_bbox,
            "source_crs": source_crs,
            "reason": (
                "Could not transform the QGIS canvas extent to EPSG:4326: "
                f"{type(exc).__name__}: {exc}"
            ),
        }

    return {
        "success": False,
        "bbox": None,
        "crs": None,
        "source_bbox": raw_bbox,
        "source_crs": source_crs,
        "reason": "Could not transform the QGIS canvas extent to EPSG:4326.",
    }


def stac_tools(iface: Any = None, project: Any = None) -> list[Any]:
    """Return STAC catalog and QGIS loading tools."""

    @geo_tool(
        category="stac",
        available_in=("full", "fast"),
    )
    def get_current_stac_search_extent() -> dict[str, Any]:
        """Return the current QGIS map extent as a STAC WGS84 bbox.

        Use this before ``search_stac_items`` when the user asks to search the
        current map extent. This avoids generated PyQGIS scripts for a common
        read-only operation.
        """
        if iface is None or not hasattr(iface, "mapCanvas"):
            return {
                "success": False,
                "bbox": None,
                "crs": None,
                "reason": "No QGIS interface is bound to this GeoAgent.",
            }

        def _read_extent() -> dict[str, Any]:
            return _canvas_extent_to_wgs84_bbox(iface.mapCanvas())

        return run_on_qt_gui_thread(_read_extent)

    @geo_tool(
        category="stac",
        requires_packages=("pystac_client",),
        available_in=("full", "fast"),
    )
    def list_stac_collections(
        catalog_url: str = PLANETARY_COMPUTER_STAC_URL,
        limit: int = 20,
    ) -> dict[str, Any]:
        """List collections from a STAC catalog.

        Args:
            catalog_url: STAC API or catalog URL.
            limit: Maximum number of collections to return.
        """
        catalog_url = _resolve_catalog_url(catalog_url)
        max_collections = _bounded_limit(limit, default=10)
        catalog = _open_catalog(catalog_url)
        collections = []
        for collection in catalog.get_collections():
            collection_dict = (
                collection.to_dict() if hasattr(collection, "to_dict") else {}
            )
            collections.append(
                {
                    "id": getattr(collection, "id", collection_dict.get("id", "")),
                    "title": collection_dict.get("title"),
                    "description": collection_dict.get("description"),
                    "extent": collection_dict.get("extent"),
                }
            )
            if len(collections) >= max_collections:
                break
        return {
            "success": True,
            "catalog_url": catalog_url,
            "collections": collections,
            "count": len(collections),
        }

    @geo_tool(
        category="stac",
        requires_packages=("pystac_client",),
    )
    def search_stac_items(
        catalog_url: str = PLANETARY_COMPUTER_STAC_URL,
        collection: str = "",
        bbox: Any = None,
        datetime: str = "",
        query_text: str = "",
        limit: int = 10,
        max_cloud_cover: float | None = None,
    ) -> dict[str, Any]:
        """Search a STAC catalog for items.

        Args:
            catalog_url: STAC API or catalog URL.
            collection: Optional collection id.
            bbox: Optional west,south,east,north bbox.
            datetime: Optional STAC datetime or interval.
            query_text: Optional text filter applied to item ids and asset keys.
            limit: Maximum number of items to return.
            max_cloud_cover: Optional maximum ``eo:cloud_cover`` percentage.
        """
        catalog_url = _resolve_catalog_url(catalog_url)
        max_items = _bounded_limit(limit)
        cloud_limit = _optional_float(max_cloud_cover)
        if cloud_limit is None and _is_low_cloud_request(query_text):
            cloud_limit = 10.0
        cloud_filter_supported = bool(
            cloud_limit is not None
            and (_is_cloud_aware_collection(collection) or not collection)
        )
        cloud_aware = cloud_filter_supported or _is_cloud_aware_collection(collection)
        candidate_limit = (
            max(max_items, STAC_CLOUD_CANDIDATE_LIMIT) if cloud_aware else max_items
        )
        catalog = _open_catalog(catalog_url)
        search_kwargs: dict[str, Any] = {"max_items": candidate_limit}
        if collection:
            search_kwargs["collections"] = [collection]
        coerced_bbox = _coerce_bbox(bbox)
        if coerced_bbox:
            search_kwargs["bbox"] = coerced_bbox
        if datetime:
            search_kwargs["datetime"] = datetime
        if cloud_filter_supported:
            search_kwargs["query"] = {"eo:cloud_cover": {"lte": cloud_limit}}
        search = catalog.search(**search_kwargs)
        q = str(query_text or "").strip().lower()
        items = []
        for item in search.items():
            summary = _item_summary(item, coerced_bbox)
            cloud_cover = _optional_float(summary.get("cloud_cover"))
            if cloud_filter_supported and (
                cloud_cover is None or cloud_cover > cloud_limit
            ):
                continue
            haystack = " ".join(
                [
                    str(summary.get("id") or ""),
                    str(summary.get("collection") or ""),
                    " ".join(summary.get("asset_keys") or []),
                ]
            ).lower()
            if q and not _is_low_cloud_request(q) and q not in haystack:
                continue
            items.append(summary)
        if coerced_bbox:
            items.sort(
                key=lambda item: (
                    not bool(item.get("contains_query_center")),
                    -float(item.get("bbox_overlap_ratio") or 0.0),
                )
            )
        if any(item.get("cloud_cover") is not None for item in items):
            items.sort(
                key=lambda item: (
                    (
                        not bool(item.get("contains_query_center"))
                        if coerced_bbox
                        else False
                    ),
                    (
                        -float(item.get("bbox_overlap_ratio") or 0.0)
                        if coerced_bbox
                        else 0.0
                    ),
                    _optional_float(item.get("cloud_cover")) is None,
                    _optional_float(item.get("cloud_cover")) or 999999.0,
                    str(item.get("datetime") or ""),
                )
            )
        items = items[:max_items]
        return {
            "success": True,
            "catalog_url": catalog_url,
            "collection": collection or None,
            "bbox": coerced_bbox,
            "datetime": datetime or None,
            "max_cloud_cover": cloud_limit if cloud_filter_supported else None,
            "ignored_max_cloud_cover": (
                cloud_limit
                if cloud_limit is not None and not cloud_filter_supported
                else None
            ),
            "sorted_by": (
                "spatial_fit,cloud_cover"
                if coerced_bbox and cloud_aware
                else (
                    "cloud_cover"
                    if cloud_aware
                    else "spatial_fit" if coerced_bbox else None
                )
            ),
            "items": items,
            "count": len(items),
        }

    @geo_tool(
        category="stac",
        requires_packages=("pystac_client",),
    )
    def get_stac_item_assets(
        catalog_url: str = PLANETARY_COMPUTER_STAC_URL,
        collection: str = "",
        item_id: str = "",
        sign_assets: bool = True,
    ) -> dict[str, Any]:
        """Inspect one STAC item and list its assets.

        Args:
            catalog_url: STAC API or catalog URL.
            collection: Collection id containing the item.
            item_id: STAC item id.
            sign_assets: Sign asset URLs with planetary-computer when available.
        """
        catalog_url = _resolve_catalog_url(catalog_url)
        if not collection or not item_id:
            raise ValueError("collection and item_id are required.")
        item = _get_item(catalog_url, collection, item_id)
        item_dict = item.to_dict() if hasattr(item, "to_dict") else {}
        assets = item_dict.get("assets") or {}
        return {
            "success": True,
            "item": _item_summary(item),
            "assets": [
                _asset_summary(key, asset, sign=sign_assets)
                for key, asset in sorted(assets.items())
            ],
        }

    @geo_tool(
        category="stac",
        requires_confirmation=True,
    )
    def add_stac_asset_to_qgis(
        asset_href: str,
        layer_name: str = "STAC asset",
        sign_asset: bool = True,
    ) -> dict[str, Any]:
        """Add a STAC raster asset URL to QGIS when a QGIS interface is bound.

        Args:
            asset_href: STAC asset URL or local raster path.
            layer_name: QGIS layer name.
            sign_asset: Sign the URL with planetary-computer when available.
        """
        href = _sign_href(asset_href) if sign_asset else asset_href
        qgis_uri = _qgis_raster_uri(href)
        if iface is None:
            return {
                "success": False,
                "loaded": False,
                "asset_href": href,
                "qgis_uri": qgis_uri,
                "layer_name": layer_name,
                "reason": "No QGIS interface is bound to this GeoAgent.",
            }

        queued = _queue_qgis_raster_load(iface, qgis_uri, layer_name)
        if queued is not None:
            queued["asset_href"] = href
            return queued

        def _load() -> dict[str, Any]:
            _configure_gdal_for_remote_cog()
            try:
                layer = iface.addRasterLayer(qgis_uri, layer_name, "gdal")
            except TypeError:
                layer = iface.addRasterLayer(qgis_uri, layer_name)
            valid = layer is not None and not (
                hasattr(layer, "isValid") and not layer.isValid()
            )
            if valid:
                zoomed = _zoom_to_qgis_layer(iface, layer)
                return {
                    "success": True,
                    "loaded": True,
                    "zoomed": zoomed,
                    "asset_href": href,
                    "qgis_uri": qgis_uri,
                    "layer_name": layer_name,
                }
            return {
                "success": False,
                "loaded": False,
                "asset_href": href,
                "qgis_uri": qgis_uri,
                "layer_name": layer_name,
                "reason": (
                    "QGIS did not accept this STAC asset as a raster layer. "
                    "The asset URL is included so the user can choose another "
                    "asset, provider, or tiling workflow."
                ),
            }

        return run_on_qt_gui_thread(_load)

    return [
        get_current_stac_search_extent,
        list_stac_collections,
        search_stac_items,
        get_stac_item_assets,
        add_stac_asset_to_qgis,
    ]
