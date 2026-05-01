"""Tool adapters for the QGIS Vantor plugin.

The tools read the public Vantor Open Data static STAC catalog directly and
avoid importing QGIS or the Vantor plugin at module import time.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from typing import Any, Optional
from urllib.error import URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from geoagent.core.decorators import geo_tool
from geoagent.tools._qt_marshal import run_on_qt_gui_thread
from geoagent.tools.stac import (
    _bounded_limit,
    _canvas_extent_to_wgs84_bbox,
    _coerce_bbox,
    _configure_gdal_for_remote_cog,
    _qgis_raster_uri,
    _queue_qgis_raster_load,
    _zoom_to_qgis_layer,
)

VANTOR_CATALOG_URL = "https://vantor-opendata.s3.amazonaws.com/events/catalog.json"
VANTOR_HTTP_TIMEOUT = 30


def _fetch_json(url: str) -> dict[str, Any]:
    """Fetch and parse JSON from the Vantor static STAC catalog."""
    if not str(url).lower().startswith("https://"):
        raise ValueError("Vantor catalog URLs must use HTTPS.")
    req = Request(url, headers={"User-Agent": "GeoAgent-Vantor/1.0"})
    with urlopen(
        req, timeout=VANTOR_HTTP_TIMEOUT
    ) as response:  # nosec B310 - HTTPS enforced above
        return json.loads(response.read().decode("utf-8"))


def _resolve_href(base_url: str, href: str) -> str:
    """Resolve a STAC href against its parent document URL.

    Only HTTPS URLs are accepted. Non-HTTPS schemes such as ``http://``,
    ``file://`` or ``ftp://`` are rejected so a malicious or compromised
    catalog cannot trigger local file reads or unexpected protocol access.
    """
    resolved = (
        href if href.startswith(("http://", "https://")) else urljoin(base_url, href)
    )
    if urlparse(resolved).scheme.lower() != "https":
        raise ValueError(f"Refusing non-HTTPS Vantor URL: {resolved!r}")
    return resolved


def _event_id_from_href(href: str) -> str:
    """Return a readable event identifier from a collection URL."""
    parts = [part for part in href.rstrip("/").split("/") if part]
    if len(parts) >= 2 and parts[-1].lower() == "collection.json":
        return parts[-2]
    return parts[-1] if parts else href


def _catalog_events() -> list[dict[str, str]]:
    """Return event collection links from the Vantor root catalog."""
    catalog = _fetch_json(VANTOR_CATALOG_URL)
    events: list[dict[str, str]] = []
    for link in catalog.get("links", []):
        if link.get("rel") != "child" or not link.get("href"):
            continue
        href = _resolve_href(VANTOR_CATALOG_URL, str(link["href"]))
        fallback = _event_id_from_href(href)
        title = str(link.get("title") or fallback)
        events.append({"id": fallback, "title": title, "href": href})
    return events


def _resolve_event_href(event: str) -> str:
    """Resolve an event id, title, or collection URL to a collection href."""
    value = str(event or "").strip()
    if value.startswith(("http://", "https://")):
        return value
    if not value:
        raise ValueError("event is required.")

    normalized = value.lower()
    events = _catalog_events()
    for item in events:
        if normalized in {item["id"].lower(), item["title"].lower()}:
            return item["href"]
    for item in events:
        if normalized in item["id"].lower() or normalized in item["title"].lower():
            return item["href"]
    raise ValueError(f"No Vantor event matched {event!r}.")


def _collection_item_urls(collection_url: str) -> list[str]:
    """Return absolute item URLs linked from a Vantor collection."""
    collection = _fetch_json(collection_url)
    urls = []
    for link in collection.get("links", []):
        if link.get("rel") == "item" and link.get("href"):
            urls.append(_resolve_href(collection_url, str(link["href"])))
    return urls


def _fetch_items(
    collection_url: str,
) -> tuple[list[dict[str, Any]], int, str | None]:
    """Fetch all STAC items for a collection.

    Returns the items list along with the count of item URLs that failed to
    fetch and the last error message so callers can surface partial results.
    """
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    failures = 0
    last_error: str | None = None
    for item_url in _collection_item_urls(collection_url):
        try:
            item = _fetch_json(item_url)
        except (URLError, json.JSONDecodeError, ValueError) as exc:
            failures += 1
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        item_id = str(item.get("id") or item_url)
        if item_id in seen:
            continue
        seen.add(item_id)
        items.append(item)
    return items, failures, last_error


def _bbox_intersects(item_bbox: Any, query_bbox: list[float]) -> bool:
    """Return whether a STAC item bbox intersects a query bbox."""
    if not item_bbox or len(item_bbox) < 4:
        return True
    west, south, east, north = query_bbox
    item_west, item_south, item_east, item_north = [float(v) for v in item_bbox[:4]]
    return (
        item_west <= east
        and item_east >= west
        and item_south <= north
        and item_north >= south
    )


def _phase_matches(item: dict[str, Any], phase: str) -> bool:
    """Return whether an item matches the requested pre/post event phase."""
    key = str(phase or "all").strip().lower().replace("-event", "")
    if key in {"", "all", "any"}:
        return True
    item_phase = (
        str((item.get("properties") or {}).get("phase") or "")
        .strip()
        .lower()
        .replace("-event", "")
    )
    return item_phase == key


def _asset_keys(item: dict[str, Any]) -> list[str]:
    """Return sorted asset keys from a STAC item."""
    assets = item.get("assets") or {}
    return sorted(str(key) for key in assets.keys())


def _asset_url(item: dict[str, Any], asset_key: str | None = None) -> str | None:
    """Return a preferred raster asset URL from a Vantor STAC item."""
    assets = item.get("assets") or {}
    if asset_key and asset_key in assets:
        href = (assets.get(asset_key) or {}).get("href")
        return str(href) if href else None
    if "visual" in assets and (assets["visual"] or {}).get("href"):
        return str(assets["visual"]["href"])
    for asset in assets.values():
        asset_type = str((asset or {}).get("type") or "").lower()
        href = (asset or {}).get("href")
        if href and any(token in asset_type for token in ("geotiff", "tiff", "cog")):
            return str(href)
    return None


def _thumbnail_url(item: dict[str, Any]) -> str | None:
    """Return an item thumbnail URL when available."""
    assets = item.get("assets") or {}
    href = (assets.get("thumbnail") or {}).get("href")
    return str(href) if href else None


def _item_summary(item: dict[str, Any]) -> dict[str, Any]:
    """Return a compact JSON-friendly Vantor STAC item summary."""
    props = item.get("properties") or {}
    return {
        "id": item.get("id"),
        "collection": item.get("collection"),
        "bbox": item.get("bbox"),
        "datetime": props.get("datetime"),
        "phase": props.get("phase"),
        "sensor": props.get("vehicle_name") or props.get("constellation"),
        "cloud_cover": props.get("eo:cloud_cover") or props.get("cloud_cover"),
        "pan_gsd": props.get("pan_gsd") or props.get("panchromatic_gsd"),
        "ms_gsd": props.get("multispectral_gsd"),
        "off_nadir": props.get("view:off_nadir"),
        "geometry_type": (item.get("geometry") or {}).get("type"),
        "asset_keys": _asset_keys(item),
        "cog_url": _asset_url(item),
        "thumbnail_url": _thumbnail_url(item),
    }


def _filter_items(
    items: list[dict[str, Any]],
    *,
    bbox: Any = None,
    phase: str = "all",
    query_text: str = "",
) -> list[dict[str, Any]]:
    """Apply bbox, phase, and text filters to Vantor STAC items."""
    parsed_bbox = _coerce_bbox(bbox)
    query = str(query_text or "").strip().lower()
    out = []
    for item in items:
        if parsed_bbox and not _bbox_intersects(item.get("bbox"), parsed_bbox):
            continue
        if not _phase_matches(item, phase):
            continue
        if query:
            haystack = " ".join(
                [
                    str(item.get("id") or ""),
                    str(item.get("collection") or ""),
                    str((item.get("properties") or {}).get("phase") or ""),
                    str((item.get("properties") or {}).get("vehicle_name") or ""),
                    " ".join(_asset_keys(item)),
                ]
            ).lower()
            if query not in haystack:
                continue
        out.append(item)
    return out


def _project_from_runtime(iface: Any, project: Any | None) -> Any | None:
    """Return the bound project, iface project, or QGIS singleton project."""
    if project is not None:
        return project
    try:
        return iface.project()
    except Exception:
        pass
    try:
        from qgis.core import QgsProject  # type: ignore[import-not-found]

        return QgsProject.instance()
    except Exception:
        return None


def _write_footprints_geojson(
    items: list[dict[str, Any]],
    path: str,
) -> int:
    """Write item footprint geometries to GeoJSON and return feature count."""
    features = []
    for item in items:
        geometry = item.get("geometry")
        if not geometry:
            continue
        summary = _item_summary(item)
        features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "item_id": summary.get("id"),
                    "phase": summary.get("phase"),
                    "datetime": summary.get("datetime"),
                    "sensor": summary.get("sensor"),
                    "cloud_cover": summary.get("cloud_cover"),
                    "cog_url": summary.get("cog_url"),
                },
            }
        )
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": features,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return len(features)


def _find_item_by_id(
    items: list[dict[str, Any]],
    item_id: str,
) -> dict[str, Any] | None:
    """Return a STAC item by exact or substring id match."""
    value = str(item_id or "").strip().lower()
    if not value:
        return None
    for item in items:
        if str(item.get("id") or "").lower() == value:
            return item
    for item in items:
        if value in str(item.get("id") or "").lower():
            return item
    return None


def vantor_tools(
    iface: Any,
    project: Optional[Any] = None,
    plugin: Optional[Any] = None,
) -> list[Any]:
    """Return GeoAgent tools for Vantor Open Data workflows in QGIS."""
    if iface is None:
        return []

    state: dict[str, Any] = {}

    @geo_tool(
        category="vantor",
        name="list_vantor_events",
        available_in=("full", "fast"),
    )
    def list_vantor_events(limit: int = 50) -> dict[str, Any]:
        """List available Vantor Open Data event collections."""
        max_events = max(1, int(limit or 50))
        events = _catalog_events()
        state["events"] = events
        return {
            "success": True,
            "catalog_url": VANTOR_CATALOG_URL,
            "count": len(events),
            "events": events[:max_events],
        }

    @geo_tool(
        category="vantor",
        name="get_vantor_event_info",
        available_in=("full", "fast"),
    )
    def get_vantor_event_info(event: str) -> dict[str, Any]:
        """Return compact metadata for one Vantor event collection."""
        href = _resolve_event_href(event)
        collection = _fetch_json(href)
        item_count = sum(
            1 for link in collection.get("links", []) if link.get("rel") == "item"
        )
        extent = collection.get("extent") or {}
        return {
            "success": True,
            "event": event,
            "href": href,
            "id": collection.get("id") or _event_id_from_href(href),
            "title": collection.get("title"),
            "description": collection.get("description"),
            "item_count": item_count,
            "extent": extent,
            "license": collection.get("license"),
            "keywords": collection.get("keywords") or [],
        }

    @geo_tool(
        category="vantor",
        name="get_current_vantor_search_extent",
        available_in=("full", "fast"),
    )
    def get_current_vantor_search_extent() -> dict[str, Any]:
        """Return the current QGIS map extent as a Vantor/STAC WGS84 bbox."""
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

    @geo_tool(category="vantor", name="search_vantor_items")
    def search_vantor_items(
        event: str,
        bbox: Any = None,
        phase: str = "all",
        query_text: str = "",
        limit: int = 10,
    ) -> dict[str, Any]:
        """Search a Vantor event collection by bbox, phase, and text.

        Args:
            event: Event id, title, or collection URL.
            bbox: Optional west,south,east,north WGS84 bounding box.
            phase: One of all, pre-event, post-event, pre, or post.
            query_text: Optional text filter for item ids, sensors, and assets.
            limit: Maximum number of item summaries to return.
        """
        href = _resolve_event_href(event)
        items, failures, last_error = _fetch_items(href)
        filtered = _filter_items(
            items,
            bbox=bbox,
            phase=phase,
            query_text=query_text,
        )
        max_items = _bounded_limit(limit, default=10)
        state["last_event"] = event
        state["last_event_href"] = href
        state["last_search_items"] = filtered
        return {
            "success": True,
            "event": event,
            "event_href": href,
            "bbox": _coerce_bbox(bbox),
            "phase": phase,
            "query_text": query_text or None,
            "count": len(filtered),
            "items": [_item_summary(item) for item in filtered[:max_items]],
            "fetch_failures": failures,
            "last_fetch_error": last_error,
        }

    @geo_tool(
        category="vantor",
        name="display_vantor_footprints",
        requires_confirmation=True,
    )
    def display_vantor_footprints(
        event: str = "",
        bbox: Any = None,
        phase: str = "all",
        query_text: str = "",
        layer_name: str = "Vantor Footprints",
        limit: int = 100,
    ) -> dict[str, Any]:
        """Display Vantor STAC item footprints in the current QGIS project."""
        if event:
            href = _resolve_event_href(event)
            fetched, _, _ = _fetch_items(href)
            items = _filter_items(
                fetched,
                bbox=bbox,
                phase=phase,
                query_text=query_text,
            )
        else:
            href = state.get("last_event_href")
            items = list(state.get("last_search_items") or [])
        if not items:
            return {"success": False, "error": "No Vantor search results available."}
        items = items[: max(1, int(limit or 100))]

        def _display() -> dict[str, Any]:
            proj = _project_from_runtime(iface, project)
            if proj is not None and hasattr(proj, "mapLayersByName"):
                for existing in proj.mapLayersByName(layer_name):
                    try:
                        proj.removeMapLayer(existing.id())
                    except Exception:
                        try:
                            proj.removeMapLayer(existing)
                        except Exception:
                            pass

            path = os.path.join(
                tempfile.gettempdir(),
                f"geoagent_vantor_footprints_{uuid.uuid4().hex}.geojson",
            )
            feature_count = _write_footprints_geojson(items, path)
            if feature_count == 0:
                return {"success": False, "error": "No footprint geometries found."}

            layer = None
            if hasattr(iface, "addVectorLayer"):
                layer = iface.addVectorLayer(path, layer_name, "ogr")
            if layer is None and proj is not None:
                from qgis.core import QgsVectorLayer  # type: ignore[import-not-found]

                layer = QgsVectorLayer(path, layer_name, "ogr")
                if layer is not None and layer.isValid():
                    proj.addMapLayer(layer)
            valid = layer is not None and not (
                hasattr(layer, "isValid") and not layer.isValid()
            )
            if not valid:
                return {
                    "success": False,
                    "error": "Failed to create Vantor footprint layer.",
                }
            _zoom_to_qgis_layer(iface, layer)
            return {
                "success": True,
                "event_href": href,
                "layer_name": layer_name,
                "feature_count": feature_count,
                "path": path,
            }

        return run_on_qt_gui_thread(_display)

    @geo_tool(
        category="vantor",
        name="load_vantor_cog",
        requires_confirmation=True,
    )
    def load_vantor_cog(
        cog_url: str = "",
        item_id: str = "",
        event: str = "",
        asset_key: str = "visual",
        layer_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Load a Vantor COG asset into QGIS.

        Args:
            cog_url: Direct COG URL. When omitted, ``item_id`` is resolved from
                the most recent search or the supplied event collection.
            item_id: Vantor STAC item id to load.
            event: Optional event id, title, or collection URL used to resolve
                ``item_id`` when it is not in the most recent search.
            asset_key: Preferred STAC asset key. Defaults to visual.
            layer_name: Optional QGIS layer name.
        """
        href = str(cog_url or "").strip()
        item = None
        if not href:
            items = list(state.get("last_search_items") or [])
            if event:
                items, _, _ = _fetch_items(_resolve_event_href(event))
            item = _find_item_by_id(items, item_id)
            if item is None:
                return {
                    "success": False,
                    "loaded": False,
                    "error": "No matching Vantor item found. Search first or provide cog_url.",
                    "item_id": item_id,
                }
            href = _asset_url(item, asset_key=asset_key)
        if not href:
            return {
                "success": False,
                "loaded": False,
                "error": "No COG or GeoTIFF asset URL found for the requested item.",
                "item_id": item_id,
            }
        name = (
            layer_name
            or (item.get("id") if item else None)
            or os.path.basename(href)
            or "Vantor COG"
        )
        qgis_uri = _qgis_raster_uri(href)

        queued = _queue_qgis_raster_load(iface, qgis_uri, name)
        if queued is not None:
            queued["asset_href"] = href
            queued["item_id"] = item.get("id") if item else item_id or None
            return queued

        def _load() -> dict[str, Any]:
            _configure_gdal_for_remote_cog()
            try:
                layer = iface.addRasterLayer(qgis_uri, name, "gdal")
            except TypeError:
                layer = iface.addRasterLayer(qgis_uri, name)
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
                    "layer_name": name,
                    "item_id": item.get("id") if item else item_id or None,
                }
            return {
                "success": False,
                "loaded": False,
                "asset_href": href,
                "qgis_uri": qgis_uri,
                "layer_name": name,
                "reason": "QGIS did not accept this Vantor COG as a raster layer.",
            }

        return run_on_qt_gui_thread(_load)

    @geo_tool(
        category="vantor",
        name="open_vantor_panel",
        available_in=("full", "fast"),
    )
    def open_vantor_panel() -> dict[str, Any]:
        """Open the Vantor plugin panel when a plugin instance is available."""
        if plugin is None:
            return {"success": False, "error": "Plugin instance is not available."}

        def _open() -> dict[str, Any]:
            dock = getattr(plugin, "_main_dock", None)
            if dock is None and hasattr(plugin, "toggle_main_dock"):
                plugin.toggle_main_dock()
                dock = getattr(plugin, "_main_dock", None)
            if dock is not None:
                if hasattr(dock, "show"):
                    dock.show()
                if hasattr(dock, "raise_"):
                    dock.raise_()
                return {"success": True, "opened": True}
            return {"success": False, "error": "Vantor panel could not be opened."}

        return run_on_qt_gui_thread(_open)

    return [
        list_vantor_events,
        get_vantor_event_info,
        get_current_vantor_search_extent,
        search_vantor_items,
        display_vantor_footprints,
        load_vantor_cog,
        open_vantor_panel,
    ]


__all__ = ["VANTOR_CATALOG_URL", "vantor_tools"]
