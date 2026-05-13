"""Tool adapters for the HyperCoast QGIS plugin.

This module is import-safe outside QGIS and outside HyperCoast. Tool bodies
resolve HyperCoast, QGIS, and plugin objects lazily so the GeoAgent package can
be imported in ordinary Python environments.
"""

from __future__ import annotations

import json
import importlib
import importlib.util
import netrc
import os
import shutil
import sys
import tempfile
import types
import uuid
from typing import Any
from urllib.parse import unquote, urlencode, urlparse
from urllib.request import (
    HTTPCookieProcessor,
    HTTPBasicAuthHandler,
    HTTPPasswordMgrWithDefaultRealm,
    Request,
    build_opener,
    urlopen,
)

from geoagent.core.decorators import geo_tool
from geoagent.tools._qt_marshal import run_on_qt_gui_thread

DATA_TYPES: dict[str, dict[str, Any]] = {
    "EMIT": {
        "extensions": [".nc", ".nc4"],
        "description": "NASA EMIT L2A Reflectance",
        "variable": "reflectance",
    },
    "PACE": {
        "extensions": [".nc", ".nc4"],
        "description": "NASA PACE OCI L2 AOP",
        "variable": "Rrs",
    },
    "DESIS": {
        "extensions": [".nc", ".tif", ".tiff"],
        "description": "DESIS Hyperspectral",
        "variable": "reflectance",
    },
    "NEON": {
        "extensions": [".h5"],
        "description": "NEON AOP Hyperspectral",
        "variable": "reflectance",
    },
    "AVIRIS": {
        "extensions": [".nc", ".img", ".bil"],
        "description": "AVIRIS/AVIRIS-NG",
        "variable": "reflectance",
    },
    "PRISMA": {
        "extensions": [".he5", ".nc"],
        "description": "ASI PRISMA",
        "variable": "reflectance",
    },
    "EnMAP": {
        "extensions": [".nc", ".tif"],
        "description": "DLR EnMAP",
        "variable": "reflectance",
    },
    "Tanager": {
        "extensions": [".h5"],
        "description": "Planet Tanager",
        "variable": "toa_radiance",
    },
    "Wyvern": {
        "extensions": [".tif", ".tiff"],
        "description": "Wyvern Hyperspectral",
        "variable": "reflectance",
    },
    "Generic": {
        "extensions": [".tif", ".tiff", ".nc", ".nc4"],
        "description": "Generic Hyperspectral",
        "variable": "data",
    },
}

WAVELENGTH_PRESETS: dict[str, list[float]] = {
    "True Color (RGB)": [650.0, 550.0, 450.0],
    "Color Infrared (CIR)": [850.0, 650.0, 550.0],
    "False Color (Urban)": [2200.0, 850.0, 650.0],
    "Agriculture": [850.0, 680.0, 550.0],
    "Vegetation Analysis": [850.0, 680.0, 450.0],
    "Water Bodies": [550.0, 480.0, 450.0],
    "Geology": [2200.0, 1600.0, 850.0],
    "Chlorophyll-a": [700.0, 680.0, 550.0],
}

DATA_TYPE_ALIASES = {
    "auto": "auto",
    "emit": "EMIT",
    "pace": "PACE",
    "desis": "DESIS",
    "neon": "NEON",
    "aviris": "AVIRIS",
    "aviris-ng": "AVIRIS",
    "aviris_ng": "AVIRIS",
    "prisma": "PRISMA",
    "enmap": "EnMAP",
    "tanager": "Tanager",
    "wyvern": "Wyvern",
    "generic": "Generic",
}


def _on_gui(fn: Any) -> Any:
    """Run a callable on the Qt GUI thread when QGIS is available.

    Args:
        fn: Callable to execute.

    Returns:
        The callable return value.
    """
    return run_on_qt_gui_thread(fn)


def _parse_bbox(
    bbox: str | list[float] | tuple[float, ...] | None,
) -> list[float] | None:
    """Parse west,south,east,north bounds.

    Args:
        bbox: Bounds as a comma-separated string or four numeric values.

    Returns:
        Parsed bounds, or ``None`` when no bounds were supplied.

    Raises:
        ValueError: If the bounds are malformed.
    """
    if bbox is None or bbox == "":
        return None
    values = list(bbox) if isinstance(bbox, (list, tuple)) else str(bbox).split(",")
    if len(values) != 4:
        raise ValueError("bbox must contain west,south,east,north")
    west, south, east, north = [float(value) for value in values]
    if west >= east or south >= north:
        raise ValueError("bbox coordinates must satisfy west < east and south < north")
    return [west, south, east, north]


def _normalize_data_type(data_type: str | None) -> str:
    """Return a HyperCoast data type key with expected capitalization.

    Args:
        data_type: User-supplied data type.

    Returns:
        Canonical HyperCoast data type, or ``auto``.
    """
    text = str(data_type or "auto").strip()
    if not text:
        return "auto"
    return DATA_TYPE_ALIASES.get(text.lower(), text)


def _normalize_cloud_cover_filter(
    cloud_cover_min: float | None = None,
    cloud_cover_max: float | None = None,
    max_cloud_cover: float | None = None,
) -> tuple[float, float] | None:
    """Return a validated CMR cloud-cover filter.

    Args:
        cloud_cover_min: Optional minimum cloud-cover percentage.
        cloud_cover_max: Optional maximum cloud-cover percentage.
        max_cloud_cover: Convenience alias for ``cloud_cover_max``.

    Returns:
        ``(minimum, maximum)`` percentage tuple, or ``None`` when unset.

    Raises:
        ValueError: If the cloud-cover bounds are invalid.
    """
    if cloud_cover_max is None and max_cloud_cover is not None:
        cloud_cover_max = max_cloud_cover
    if cloud_cover_min is None and cloud_cover_max is None:
        return None

    minimum = 0.0 if cloud_cover_min is None else float(cloud_cover_min)
    maximum = 100.0 if cloud_cover_max is None else float(cloud_cover_max)
    if minimum < 0 or maximum > 100:
        raise ValueError("cloud cover must be between 0 and 100 percent")
    if minimum > maximum:
        raise ValueError(
            "cloud_cover_min must be less than or equal to cloud_cover_max"
        )
    return (minimum, maximum)


def _pace_short_name_ignores_cloud_cover(short_name: str | None) -> bool:
    """Return whether a PACE short name lacks per-granule CloudCover metadata.

    PACE L3 mapped composites (for example ``PACE_OCI_L3M_BGC_*``) do not
    carry a per-granule ``CloudCover`` field in CMR, so filtering by
    ``cloud_cover`` silently drops every result. Detect those products so the
    caller can skip the filter.

    Args:
        short_name: Earthdata short name supplied to the search tool.

    Returns:
        True when ``cloud_cover`` filters should be ignored for ``short_name``.
    """
    if not short_name:
        return False
    upper = short_name.upper()
    return "L3M" in upper or "_BGC" in upper or upper.endswith("_BGC")


def _current_bbox_wgs84(iface: Any) -> list[float]:
    """Return the current QGIS canvas extent as WGS84 bounds.

    Args:
        iface: QGIS interface-like object.

    Returns:
        Bounds in west,south,east,north order.
    """

    def _run() -> list[float]:
        canvas = iface.mapCanvas()
        extent = canvas.extent()
        if isinstance(extent, (list, tuple)):
            return [float(value) for value in extent]

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
        return [
            float(extent.xMinimum()),
            float(extent.yMinimum()),
            float(extent.xMaximum()),
            float(extent.yMaximum()),
        ]

    return _on_gui(_run)


def _load_hypercoast_common() -> Any:
    """Import and return ``hypercoast.common`` lazily.

    Returns:
        The ``hypercoast.common`` module.
    """
    from hypercoast import common

    return common


def _resolve_hypercoast_plugin(plugin: Any | None = None) -> Any | None:
    """Resolve the installed HyperCoast plugin instance.

    Args:
        plugin: Optional explicit plugin instance.

    Returns:
        The plugin instance, or ``None`` when it is unavailable.
    """
    if plugin is not None:
        return plugin
    try:
        from qgis.utils import plugins  # type: ignore[import-not-found]

        for name in ("HyperCoast", "hypercoast", "hypercoast_qgis"):
            if name in plugins:
                return plugins[name]
        for candidate in plugins.values():
            if candidate.__class__.__name__ == "HyperCoastPlugin":
                return candidate
    except Exception:
        return None
    return None


def _candidate_hypercoast_plugin_dirs(plugin: Any | None = None) -> list[str]:
    """Return possible HyperCoast QGIS plugin directories.

    Args:
        plugin: Optional HyperCoast plugin instance.

    Returns:
        Existing plugin directories to try, in priority order.
    """
    dirs: list[str] = []

    def _add(path: Any) -> None:
        if not path:
            return
        candidate = os.path.abspath(os.path.expanduser(str(path)))
        if os.path.isdir(candidate) and candidate not in dirs:
            dirs.append(candidate)

    _add(os.environ.get("HYPERCOAST_QGIS_PLUGIN_DIR"))
    _add(getattr(plugin, "plugin_dir", None))
    try:
        from qgis.utils import plugins  # type: ignore[import-not-found]

        for name in ("HyperCoast", "hypercoast", "hypercoast_qgis"):
            _add(getattr(plugins.get(name), "plugin_dir", None))
        for candidate in plugins.values():
            if candidate.__class__.__name__ == "HyperCoastPlugin":
                _add(getattr(candidate, "plugin_dir", None))
    except Exception:
        pass

    home = os.path.expanduser("~")
    for qgis_name in ("QGIS3", "QGIS4"):
        plugins_dir = os.path.join(
            home,
            ".local",
            "share",
            "QGIS",
            qgis_name,
            "profiles",
            "default",
            "python",
            "plugins",
        )
        for plugin_name in ("hypercoast", "hypercoast_qgis"):
            _add(os.path.join(plugins_dir, plugin_name))

    return dirs


def _add_plugin_parent_to_path(plugin_dir: str | None) -> None:
    """Add a HyperCoast plugin parent directory to ``sys.path`` when needed.

    Args:
        plugin_dir: Optional HyperCoast QGIS plugin directory.
    """
    if not plugin_dir:
        return
    parent = os.path.dirname(os.path.abspath(plugin_dir))
    if parent and parent not in sys.path:
        sys.path.insert(0, parent)


def _load_plugin_module_from_dir(plugin_dir: str, module_name: str) -> Any:
    """Import an installed HyperCoast QGIS module from a plugin directory.

    Installed QGIS packages are commonly named ``hypercoast``, which conflicts
    with the PyPI ``hypercoast`` package. Loading the plugin package under a
    private alias avoids that collision while preserving relative imports.

    Args:
        plugin_dir: Directory containing the QGIS plugin ``__init__.py``.
        module_name: Top-level plugin module name to import.

    Returns:
        Imported module.

    Raises:
        ModuleNotFoundError: If the module file cannot be found.
        ImportError: If the module cannot be loaded.
    """
    module_path = os.path.join(
        plugin_dir,
        *module_name.split("."),
    )
    if os.path.isdir(module_path):
        module_path = os.path.join(module_path, "__init__.py")
    else:
        module_path = f"{module_path}.py"
    if not os.path.exists(module_path):
        raise ModuleNotFoundError(module_name)

    alias = "_geoagent_hypercoast_qgis"
    if alias not in sys.modules:
        package = types.ModuleType(alias)
        package.__file__ = os.path.join(plugin_dir, "__init__.py")
        package.__path__ = [plugin_dir]  # type: ignore[attr-defined]
        package.__package__ = alias
        sys.modules[alias] = package

    qualified_name = f"{alias}.{module_name}"
    cached = sys.modules.get(qualified_name)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(qualified_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_name} from {plugin_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


def _load_hyperspectral_runtime(plugin: Any | None = None) -> tuple[Any, Any]:
    """Import HyperCoast QGIS runtime helpers lazily.

    Args:
        plugin: Optional HyperCoast plugin instance used to locate plugin code.

    Returns:
        Tuple of ``HyperspectralDataset`` class and raster path factory.
    """
    plugin = _resolve_hypercoast_plugin(plugin)
    errors: list[str] = []
    for plugin_dir in _candidate_hypercoast_plugin_dirs(plugin):
        _add_plugin_parent_to_path(plugin_dir)
        try:
            cache_module = importlib.import_module("hypercoast_qgis.cache_manager")
            provider_module = importlib.import_module(
                "hypercoast_qgis.hyperspectral_provider"
            )
            return (
                provider_module.HyperspectralDataset,
                cache_module.create_generated_raster_path,
            )
        except Exception as exc:
            errors.append(f"{plugin_dir}: {exc}")

        try:
            cache_module = _load_plugin_module_from_dir(plugin_dir, "cache_manager")
            provider_module = _load_plugin_module_from_dir(
                plugin_dir,
                "hyperspectral_provider",
            )
            return (
                provider_module.HyperspectralDataset,
                cache_module.create_generated_raster_path,
            )
        except Exception as exc:
            errors.append(f"{plugin_dir}: {exc}")

    try:
        from hypercoast_qgis.cache_manager import create_generated_raster_path
        from hypercoast_qgis.hyperspectral_provider import HyperspectralDataset

        return HyperspectralDataset, create_generated_raster_path
    except Exception as exc:
        errors.append(str(exc))

    raise ModuleNotFoundError(
        "No module named 'hypercoast_qgis'. Tried installed HyperCoast plugin "
        f"directories: {'; '.join(errors) or 'none found'}"
    )


HYPERCOAST_SENSOR_HELPERS = {
    "pace": (
        "read_pace",
        "read_pace_aop",
        "read_pace_bgc",
        "read_pace_chla",
        "pace_to_image",
        "pace_chla_to_image",
    ),
    "emit": ("read_emit", "emit_to_image"),
    "desis": ("read_desis", "desis_to_image"),
    "neon": ("read_neon", "neon_to_image"),
    "aviris": ("read_aviris", "aviris_to_image"),
    "prisma": ("read_prisma", "prisma_to_image"),
    "enmap": ("read_enmap", "enmap_to_image"),
    "tanager": ("read_tanager", "tanager_to_image"),
    "wyvern": ("read_wyvern", "wyvern_to_image"),
}


def _patch_hypercoast_module_exports(hypercoast_module: Any) -> None:
    """Attach missing top-level helpers to one HyperCoast module object.

    Args:
        hypercoast_module: Imported HyperCoast Python package.
    """
    module_base = getattr(hypercoast_module, "__name__", "hypercoast")
    if not module_base:
        module_base = "hypercoast"
    for module_key, names in HYPERCOAST_SENSOR_HELPERS.items():
        missing = [name for name in names if not hasattr(hypercoast_module, name)]
        if not missing:
            continue
        try:
            submodule = importlib.import_module(f"{module_base}.{module_key}")
        except Exception:
            continue
        for name in missing:
            value = getattr(submodule, name, None)
            if value is not None:
                setattr(hypercoast_module, name, value)


def _runtime_hypercoast_module(runtime_object: Any | None) -> Any | None:
    """Return the HyperCoast module cached by an installed QGIS runtime module.

    Args:
        runtime_object: Class or object loaded from the HyperCoast QGIS plugin.

    Returns:
        Cached HyperCoast Python package, or ``None``.
    """
    if runtime_object is None:
        return None
    module_name = getattr(runtime_object, "__module__", "")
    provider_module = sys.modules.get(module_name)
    if provider_module is None:
        return None
    cached = getattr(provider_module, "hypercoast", None)
    if cached is not None:
        return cached
    getter = getattr(provider_module, "get_hypercoast", None)
    if callable(getter):
        try:
            cached = getter()
            setattr(provider_module, "hypercoast", cached)
            return cached
        except Exception:
            return None
    return None


def _patch_hypercoast_top_level_exports(runtime_object: Any | None = None) -> None:
    """Attach sensor helper functions missing from some HyperCoast installs.

    Args:
        runtime_object: Optional object from the HyperCoast QGIS runtime. The
            installed QGIS plugin can cache the external library under an alias,
            so this lets GeoAgent patch that exact module reference.
    """
    targets: list[Any] = []
    for module_name in ("hypercoast", "hypercoast_external"):
        try:
            targets.append(importlib.import_module(module_name))
        except Exception:
            pass
    runtime_module = _runtime_hypercoast_module(runtime_object)
    if runtime_module is not None:
        targets.append(runtime_module)

    seen: set[int] = set()
    for target in targets:
        if id(target) in seen:
            continue
        seen.add(id(target))
        _patch_hypercoast_module_exports(target)


def _project_from_iface(iface: Any, project: Any | None = None) -> Any | None:
    """Return the QGIS project from explicit input, iface, or QgsProject.

    Args:
        iface: QGIS interface-like object.
        project: Optional project-like object.

    Returns:
        QGIS project-like object, or ``None``.
    """
    if project is not None:
        return project
    try:
        iface_project = iface.project()
        if iface_project is not None:
            return iface_project
    except Exception:
        pass
    try:
        from qgis.core import QgsProject  # type: ignore[import-not-found]

        return QgsProject.instance()
    except Exception:
        return None


def _compact_granule(granule: Any) -> dict[str, Any]:
    """Return compact JSON-friendly granule metadata.

    Args:
        granule: Earthaccess granule-like object or dictionary.

    Returns:
        A concise granule record.
    """
    if isinstance(granule, dict):
        data = granule
    else:
        try:
            data = dict(granule)
        except Exception:
            data = {}

    umm = data.get("umm", {}) if isinstance(data, dict) else {}
    out: dict[str, Any] = {}
    for key in (
        "id",
        "title",
        "producer_granule_id",
        "short_name",
        "concept_id",
        "cloud_cover",
        "time_start",
        "time_end",
        "bbox",
        "boxes",
        "geometry",
    ):
        value = data.get(key) if isinstance(data, dict) else None
        if value is not None:
            out[key] = value
    for key in ("data_links", "cloud_data_links"):
        value = data.get(key) if isinstance(data, dict) else None
        if value:
            out[key] = list(value)[:10]
    if isinstance(data, dict):
        links = data.get("links")
        if isinstance(links, list):
            out["links"] = links[:10]
        related_urls = data.get("related_urls")
        if isinstance(related_urls, list):
            out["related_urls"] = related_urls[:10]
    if umm:
        if umm.get("GranuleUR"):
            out["granule_ur"] = umm["GranuleUR"]
        if umm.get("TemporalExtent"):
            out["temporal"] = umm["TemporalExtent"]
        if umm.get("SpatialExtent"):
            out["spatial"] = umm["SpatialExtent"]
        related_urls = umm.get("RelatedUrls")
        if isinstance(related_urls, list):
            out["related_urls"] = related_urls[:10]
    for method_name, output_key in (
        ("data_links", "data_links"),
        ("cloud_data_links", "cloud_data_links"),
    ):
        method = getattr(granule, method_name, None)
        if callable(method):
            try:
                links = method()
                if links:
                    out[output_key] = list(links)[:10]
            except Exception:
                pass
    if not out:
        out["repr"] = repr(granule)[:500]
    return out


def _compact_search_results(results: Any) -> dict[str, Any]:
    """Return compact metadata for a HyperCoast search result.

    Args:
        results: Search results returned by HyperCoast.

    Returns:
        Search summary with compact granules.
    """
    if isinstance(results, tuple):
        granules = results[0]
    else:
        granules = results
    granule_list = list(granules or [])
    return {
        "success": True,
        "count": len(granule_list),
        "granules": [_compact_granule(item) for item in granule_list[:25]],
    }


def _search_result_granules(results: Any) -> list[Any]:
    """Return a concrete granule list from a search-result object.

    Args:
        results: Search results returned by HyperCoast, earthaccess, or CMR.

    Returns:
        Concrete list of granules.
    """
    if isinstance(results, tuple):
        granules = results[0]
    else:
        granules = results
    return list(granules or [])


def _umm_geometry(umm: dict[str, Any]) -> dict[str, Any] | None:
    """Extract GeoJSON geometry from UMM granule metadata.

    Args:
        umm: UMM granule metadata dictionary.

    Returns:
        GeoJSON geometry, or ``None`` when unavailable.
    """
    spatial = umm.get("SpatialExtent", {})
    horizontal = spatial.get("HorizontalSpatialDomain", {})
    geometry = horizontal.get("Geometry", {})

    rects = geometry.get("BoundingRectangles", [])
    if rects:
        rect = rects[0]
        west = rect.get("WestBoundingCoordinate")
        south = rect.get("SouthBoundingCoordinate")
        east = rect.get("EastBoundingCoordinate")
        north = rect.get("NorthBoundingCoordinate")
        if None not in (west, south, east, north):
            return _bbox_geometry(
                [float(west), float(south), float(east), float(north)]
            )

    polygons = geometry.get("GPolygons", [])
    if polygons:
        points = polygons[0].get("Boundary", {}).get("Points", [])
        coords = [
            [float(point.get("Longitude", 0)), float(point.get("Latitude", 0))]
            for point in points
        ]
        if coords:
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            return {"type": "Polygon", "coordinates": [coords]}
    return None


def _bbox_geometry(bbox: list[float] | tuple[float, ...]) -> dict[str, Any]:
    """Return a GeoJSON polygon for west,south,east,north bounds.

    Args:
        bbox: Bounds in west,south,east,north order.

    Returns:
        GeoJSON polygon geometry.
    """
    west, south, east, north = [float(value) for value in bbox]
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


def _cmr_box_geometry(box: Any) -> dict[str, Any] | None:
    """Return GeoJSON geometry from a CMR box value.

    CMR JSON boxes are usually strings in ``south west north east`` order.

    Args:
        box: CMR box value.

    Returns:
        GeoJSON polygon geometry, or ``None`` when malformed.
    """
    if isinstance(box, str):
        values = [float(value) for value in box.replace(",", " ").split()]
        if len(values) != 4:
            return None
        south, west, north, east = values
        return _bbox_geometry([west, south, east, north])
    if isinstance(box, (list, tuple)) and len(box) == 4:
        return _bbox_geometry([float(value) for value in box])
    return None


def _granule_geometry(granule: Any) -> dict[str, Any] | None:
    """Extract a GeoJSON footprint geometry from a granule.

    Args:
        granule: Granule dictionary or earthaccess-like object.

    Returns:
        GeoJSON geometry, or ``None`` when unavailable.
    """
    if isinstance(granule, dict):
        data = granule
    else:
        try:
            data = dict(granule)
        except Exception:
            data = {}

    geometry = data.get("geometry") if isinstance(data, dict) else None
    if isinstance(geometry, dict):
        return geometry

    umm = data.get("umm", {}) if isinstance(data, dict) else {}
    if isinstance(umm, dict):
        geometry = _umm_geometry(umm)
        if geometry is not None:
            return geometry

    spatial = data.get("spatial", {}) if isinstance(data, dict) else {}
    if isinstance(spatial, dict):
        geometry = _umm_geometry({"SpatialExtent": spatial})
        if geometry is not None:
            return geometry

    boxes = data.get("boxes") or data.get("bbox") if isinstance(data, dict) else None
    if isinstance(boxes, str):
        return _cmr_box_geometry(boxes)
    if isinstance(boxes, (list, tuple)):
        if len(boxes) == 4 and all(isinstance(value, (int, float)) for value in boxes):
            return _bbox_geometry([float(value) for value in boxes])
        for box in boxes:
            geometry = _cmr_box_geometry(box)
            if geometry is not None:
                return geometry
    return None


def _write_footprints_geojson(granules: list[Any], path: str) -> int:
    """Write granule footprints to GeoJSON.

    Args:
        granules: Granules with CMR or UMM spatial metadata.
        path: Output GeoJSON path.

    Returns:
        Number of written features.
    """
    features: list[dict[str, Any]] = []
    for granule in granules:
        geometry = _granule_geometry(granule)
        if geometry is None:
            continue
        compact = _compact_granule(granule)
        properties = {
            "granule_json": json.dumps(_json_safe(compact)),
            "id": compact.get("id"),
            "title": compact.get("title")
            or compact.get("granule_ur")
            or compact.get("producer_granule_id"),
            "granule_ur": compact.get("granule_ur"),
            "producer_granule_id": compact.get("producer_granule_id"),
            "short_name": compact.get("short_name"),
            "concept_id": compact.get("concept_id"),
            "cloud_cover": compact.get("cloud_cover"),
            "time_start": compact.get("time_start"),
            "time_end": compact.get("time_end"),
            "data_links_json": json.dumps(
                _granule_data_links(compact, 10),
            ),
        }
        properties = {
            key: value for key, value in properties.items() if value is not None
        }
        features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": properties,
            }
        )

    geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": features,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(geojson, file)
    return len(features)


def _search_earthaccess_data(
    *,
    source_key: str,
    bbox: list[float] | None,
    temporal: str | None,
    count: int,
    short_name: str | None,
    provider: str | None,
    cloud_cover: tuple[float, float] | None = None,
) -> Any:
    """Search Earthdata directly when HyperCoast's helper path fails.

    Args:
        source_key: Search source, either ``emit`` or ``pace``.
        bbox: Optional west,south,east,north bounds.
        temporal: Optional temporal range.
        count: Maximum result count.
        short_name: Optional NASA Earthdata short name override.
        provider: Optional NASA Earthdata provider.
        cloud_cover: Optional CMR cloud-cover percentage range.

    Returns:
        Earthaccess granule results.
    """
    import earthaccess

    default_short_names = {
        "emit": "EMITL2ARFL",
        "pace": "PACE_OCI_L2_AOP_NRT",
    }
    kwargs: dict[str, Any] = {
        "count": int(count),
        "short_name": short_name or default_short_names[source_key],
    }
    if bbox:
        kwargs["bounding_box"] = tuple(bbox)
    if temporal:
        kwargs["temporal"] = temporal
    if provider:
        kwargs["provider"] = provider
    if cloud_cover is not None:
        kwargs["cloud_cover"] = cloud_cover
    return earthaccess.search_data(**kwargs)


def _search_cmr_granules(
    *,
    source_key: str,
    bbox: list[float] | None,
    temporal: str | None,
    count: int,
    short_name: str | None,
    provider: str | None,
    cloud_cover: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    """Search NASA CMR directly without earthaccess or leafmap.

    Args:
        source_key: Search source, either ``emit`` or ``pace``.
        bbox: Optional west,south,east,north bounds.
        temporal: Optional temporal range.
        count: Maximum result count.
        short_name: Optional NASA Earthdata short name override.
        provider: Optional NASA Earthdata provider.
        cloud_cover: Optional CMR cloud-cover percentage range.

    Returns:
        Compact granule dictionaries.
    """
    default_short_names = {
        "emit": "EMITL2ARFL",
        "pace": "PACE_OCI_L2_AOP_NRT",
    }
    params: dict[str, Any] = {
        "page_size": max(1, min(int(count), 100)),
        "short_name": short_name or default_short_names[source_key],
    }
    if bbox:
        params["bounding_box"] = ",".join(str(value) for value in bbox)
    if temporal:
        params["temporal"] = temporal
    if provider:
        params["provider"] = provider
    if cloud_cover is not None:
        params["cloud_cover"] = ",".join(str(value) for value in cloud_cover)
    url = f"https://cmr.earthdata.nasa.gov/search/granules.json?{urlencode(params)}"
    with urlopen(url, timeout=30) as response:  # nosec B310
        payload = json.loads(response.read().decode("utf-8"))

    entries = payload.get("feed", {}).get("entry", [])
    granules: list[dict[str, Any]] = []
    for entry in entries:
        links = [
            link.get("href")
            for link in entry.get("links", [])
            if link.get("href")
            and "inherited" not in link
            and _is_data_download_url(link.get("href"))
        ]
        granules.append(
            {
                "id": entry.get("id"),
                "title": entry.get("title"),
                "producer_granule_id": entry.get("producer_granule_id"),
                "concept_id": entry.get("id"),
                "time_start": entry.get("time_start"),
                "time_end": entry.get("time_end"),
                "updated": entry.get("updated"),
                "cloud_cover": entry.get("cloud_cover"),
                "bbox": entry.get("boxes", []),
                "data_links": links[:10],
            }
        )
    return granules


def _is_known_search_dependency_error(exc: Exception) -> bool:
    """Return whether a search exception should use a dependency-free fallback.

    Args:
        exc: Exception raised by HyperCoast or earthaccess.

    Returns:
        True when the exception matches a known dependency issue.
    """
    text = str(exc)
    if isinstance(exc, UnboundLocalError) and "earthaccess" in text:
        return True
    if isinstance(exc, ImportError) and "botocore.compat" in text:
        return True
    if isinstance(exc, ImportError) and "cannot import name 'EC'" in text:
        return True
    return False


def _earthdata_credentials() -> tuple[str, str] | None:
    """Return Earthdata credentials from environment or ``~/.netrc``.

    Returns:
        ``(username, password)`` when credentials are available, otherwise
        ``None``.
    """
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if username and password:
        return username, password
    try:
        auth = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
    except Exception:
        return None
    if not auth:
        return None
    login, _, password = auth
    if login and password:
        return login, password
    return None


def _download_opener() -> Any:
    """Create a urllib opener for Earthdata downloads.

    Returns:
        URL opener with cookies and optional Earthdata basic auth support.
    """
    handlers: list[Any] = [HTTPCookieProcessor()]
    credentials = _earthdata_credentials()
    if credentials:
        username, password = credentials
        password_mgr = HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(
            None,
            "https://urs.earthdata.nasa.gov",
            username,
            password,
        )
        handlers.append(HTTPBasicAuthHandler(password_mgr))
    return build_opener(*handlers)


def _filename_from_download_url(url: str, index: int) -> str:
    """Return a safe local filename for a download URL.

    Args:
        url: Download URL.
        index: Fallback index.

    Returns:
        File basename.
    """
    path = unquote(urlparse(url).path)
    name = os.path.basename(path.rstrip("/"))
    if not name:
        name = f"hypercoast_download_{index}.dat"
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)


def _is_data_download_url(url: str) -> bool:
    """Return whether a URL points to a science data file.

    Only ``https://`` URLs (and ``s3://`` URIs for cloud listings) are
    accepted. ``http://``, ``file://``, and other schemes are rejected so
    that downstream openers never fetch unsafe local or plaintext targets.

    Args:
        url: Candidate CMR, UMM, or Earthaccess URL.

    Returns:
        True when the URL looks like a downloadable data asset.
    """
    parsed = urlparse(str(url))
    scheme = parsed.scheme.lower()
    if scheme not in ("https", "s3"):
        return False
    path = unquote(parsed.path).lower()
    if not path:
        return False
    if "browse_images" in path or "/images/" in path:
        return False
    if path.endswith((".png", ".jpg", ".jpeg", ".gif", ".html", ".htm", ".xml")):
        return False
    if "search.earthdata.nasa.gov" in parsed.netloc:
        return False
    if path.endswith((".nc", ".nc4", ".h5", ".he5", ".tif", ".tiff")):
        return True
    return False


def _json_safe(value: Any) -> Any:
    """Return a JSON-serializable representation of a value.

    Args:
        value: Arbitrary metadata value.

    Returns:
        JSON-compatible value.
    """
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _parse_json_value(value: Any) -> Any:
    """Parse a JSON string when possible.

    Args:
        value: Candidate JSON string or any other value.

    Returns:
        Parsed JSON value, or the original value.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except Exception:
        return value


def _granule_from_feature(feature: Any) -> dict[str, Any] | None:
    """Extract HyperCoast granule metadata from a selected QGIS feature.

    Args:
        feature: QGIS feature object or a selected-feature dictionary.

    Returns:
        Granule metadata dictionary, or ``None`` when unavailable.
    """
    if isinstance(feature, dict):
        if isinstance(feature.get("granule_json"), str):
            parsed = _parse_json_value(feature["granule_json"])
            if isinstance(parsed, dict):
                return parsed
        attrs = feature.get("attributes")
        if isinstance(attrs, dict):
            parsed = _granule_from_feature(attrs)
            if parsed is not None:
                return parsed
        if isinstance(attrs, list):
            for value in attrs:
                parsed = _parse_json_value(value)
                if isinstance(parsed, dict):
                    if "granule_json" in parsed:
                        nested = _parse_json_value(parsed["granule_json"])
                        if isinstance(nested, dict):
                            return nested
                    if any(
                        key in parsed
                        for key in (
                            "data_links",
                            "links",
                            "related_urls",
                            "umm",
                            "granule_ur",
                            "title",
                        )
                    ):
                        return parsed
        if any(
            key in feature
            for key in (
                "data_links",
                "links",
                "related_urls",
                "umm",
                "granule_ur",
                "title",
            )
        ):
            return feature
        return None

    try:
        fields = feature.fields()
        names = [field.name() for field in fields]
        attrs = feature.attributes()
    except Exception:
        return None
    values = {name: _parse_json_value(attrs[index]) for index, name in enumerate(names)}
    granule_json = values.get("granule_json")
    if isinstance(granule_json, dict):
        return granule_json
    if isinstance(granule_json, str):
        parsed = _parse_json_value(granule_json)
        if isinstance(parsed, dict):
            return parsed
    return values


def _normalize_granules_input(granules: list[Any]) -> list[Any]:
    """Normalize selected-feature records into granule dictionaries.

    Args:
        granules: Granule records or selected QGIS feature records.

    Returns:
        Normalized granule records.
    """
    normalized: list[Any] = []
    for item in granules:
        parsed = _granule_from_feature(item)
        normalized.append(parsed if parsed is not None else item)
    return normalized


def _granule_data_links(granule: Any, max_links: int) -> list[str]:
    """Return data download links from a granule-like object.

    Args:
        granule: Granule dictionary or earthaccess-like object.
        max_links: Maximum links to return.

    Returns:
        List of URL strings.
    """
    parsed = _granule_from_feature(granule)
    if parsed is not None and parsed is not granule:
        granule = parsed

    links: list[str] = []
    if isinstance(granule, dict):
        for key in ("data_links", "cloud_data_links"):
            value = granule.get(key)
            if isinstance(value, str):
                links.append(value)
            elif isinstance(value, list):
                links.extend(str(item) for item in value if item)
        raw_links = granule.get("links")
        if isinstance(raw_links, list):
            for item in raw_links:
                if isinstance(item, str):
                    links.append(item)
                elif isinstance(item, dict):
                    href = item.get("href") or item.get("URL")
                    if href:
                        links.append(str(href))
        related_urls = granule.get("umm", {}).get("RelatedUrls", [])
        if isinstance(related_urls, list):
            for item in related_urls:
                if isinstance(item, dict):
                    url = item.get("URL") or item.get("href")
                    if url:
                        links.append(str(url))
        related_urls = granule.get("related_urls", [])
        if isinstance(related_urls, list):
            for item in related_urls:
                if isinstance(item, str):
                    links.append(item)
                elif isinstance(item, dict):
                    url = item.get("URL") or item.get("href")
                    if url:
                        links.append(str(url))
    else:
        for method_name in ("data_links", "cloud_data_links"):
            method = getattr(granule, method_name, None)
            if callable(method):
                try:
                    links.extend(str(item) for item in method() if item)
                except Exception:
                    pass
    out: list[str] = []
    for link in links:
        if not _is_data_download_url(link):
            continue
        if link not in out:
            out.append(link)
        if len(out) >= max_links:
            break
    return out


def _download_data_links(
    granules: list[Any],
    *,
    out_dir: str,
    max_links_per_granule: int = 1,
) -> list[str]:
    """Download CMR/earthaccess data links without using earthaccess.

    Args:
        granules: Granule dictionaries or earthaccess-like objects.
        out_dir: Output directory.
        max_links_per_granule: Maximum links to download per granule.

    Returns:
        Downloaded file paths.

    Raises:
        ValueError: If no downloadable links are present.
    """
    opener = _download_opener()
    token = os.environ.get("EARTHDATA_TOKEN") or os.environ.get("NASA_EARTHDATA_TOKEN")
    paths: list[str] = []
    index = 0
    for granule in granules:
        for url in _granule_data_links(granule, max_links_per_granule):
            if urlparse(url).scheme.lower() != "https":
                continue
            index += 1
            filename = _filename_from_download_url(url, index)
            output_path = os.path.join(out_dir, filename)
            request = Request(url, headers={"User-Agent": "GeoAgent/HyperCoast"})
            if token:
                request.add_header("Authorization", f"Bearer {token}")
            with opener.open(request, timeout=120) as response:  # nosec B310
                with open(output_path, "wb") as dst:
                    shutil.copyfileobj(response, dst)
            paths.append(output_path)
    if not paths:
        raise ValueError(
            "No data_links or cloud_data_links were found. Search again and pass "
            "granules from search_hypercoast_data, or install a working "
            "earthaccess stack for provider-managed downloads."
        )
    return paths


def _dataset_metadata(dataset: Any) -> dict[str, Any]:
    """Return compact metadata for a loaded HyperCoast dataset.

    Args:
        dataset: HyperCoast ``HyperspectralDataset``-like object.

    Returns:
        JSON-friendly metadata.
    """
    wavelengths = getattr(dataset, "wavelengths", None)
    wavelength_values: list[float] = []
    if wavelengths is not None:
        try:
            wavelength_values = [float(value) for value in list(wavelengths)]
        except Exception:
            wavelength_values = []

    xarray_dataset = getattr(dataset, "dataset", None)
    variables: list[str] = []
    dims: dict[str, Any] = {}
    if xarray_dataset is not None:
        try:
            variables = [str(value) for value in list(xarray_dataset.data_vars)]
        except Exception:
            variables = []
        try:
            dims = {
                str(key): int(value)
                for key, value in getattr(xarray_dataset, "dims", {}).items()
            }
        except Exception:
            dims = {}

    bounds = getattr(dataset, "bounds", None)
    return {
        "filepath": getattr(dataset, "filepath", None),
        "data_type": getattr(dataset, "data_type", None),
        "selected_variable": getattr(dataset, "selected_variable", None),
        "band_count": len(wavelength_values),
        "wavelength_min": min(wavelength_values) if wavelength_values else None,
        "wavelength_max": max(wavelength_values) if wavelength_values else None,
        "bounds": list(bounds) if bounds else None,
        "crs": (
            str(getattr(dataset, "crs", "")) if getattr(dataset, "crs", None) else None
        ),
        "variables": variables,
        "dims": dims,
    }


def _ensure_output_path(
    layer_name: str,
    project: Any | None,
    create_generated_raster_path: Any | None,
    output_path: str | None,
    suffix: str = "rgb",
) -> str:
    """Return an output GeoTIFF path for a generated HyperCoast raster.

    Args:
        layer_name: User-facing layer name.
        project: QGIS project-like object.
        create_generated_raster_path: Optional HyperCoast path factory.
        output_path: Optional explicit output path.
        suffix: Output-kind tag (for example ``"rgb"`` or a variable name)
            used to disambiguate RGB exports from variable exports.

    Returns:
        Absolute output path.
    """
    if output_path:
        return os.path.abspath(os.path.expanduser(output_path))
    safe_suffix = (
        "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in (suffix or "rgb"))
        or "rgb"
    )
    if create_generated_raster_path is not None:
        return create_generated_raster_path(layer_name, safe_suffix, project=project)
    cache_dir = os.path.join(os.path.expanduser("~"), ".qgis_hypercoast", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    safe_name = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in layer_name
    )
    fd, path = tempfile.mkstemp(
        prefix=f"{safe_name or 'hypercoast'}_{safe_suffix}_",
        suffix=".tif",
        dir=cache_dir,
    )
    os.close(fd)
    return path


def _add_qgis_raster_layer(
    iface: Any,
    project: Any | None,
    path: str,
    layer_name: str,
) -> Any:
    """Add a raster layer to QGIS on the GUI thread.

    Args:
        iface: QGIS interface-like object.
        project: Optional QGIS project-like object.
        path: Raster path.
        layer_name: QGIS layer name.

    Returns:
        Added layer object.
    """

    def _run() -> Any:
        if hasattr(iface, "addRasterLayer"):
            layer = iface.addRasterLayer(path, layer_name)
        else:
            from qgis.core import QgsRasterLayer  # type: ignore[import-not-found]

            layer = QgsRasterLayer(path, layer_name)
            if project is not None:
                project.addMapLayer(layer)
        if not layer or (hasattr(layer, "isValid") and not layer.isValid()):
            raise RuntimeError(f"Failed to load raster from {path}.")
        if hasattr(iface, "setActiveLayer"):
            iface.setActiveLayer(layer)
        try:
            iface.mapCanvas().refresh()
        except Exception:
            pass
        return layer

    return _on_gui(_run)


def _set_layer_custom_properties(
    layer: Any,
    dataset: Any,
    source_path: str,
    wavelengths: list[float] | None,
    selected_variable: str | None,
) -> None:
    """Persist HyperCoast metadata on a QGIS layer.

    Args:
        layer: QGIS layer-like object.
        dataset: Loaded HyperCoast dataset.
        source_path: Source hyperspectral file path.
        wavelengths: Selected RGB wavelengths, if any.
        selected_variable: Optional selected variable name.
    """
    setter = getattr(layer, "setCustomProperty", None)
    if not callable(setter):
        return
    setter("hypercoast/source_path", source_path)
    setter("hypercoast/data_type", getattr(dataset, "data_type", "auto"))
    if selected_variable:
        setter("hypercoast/selected_variable", selected_variable)
    if wavelengths:
        setter(
            "hypercoast/rgb_wavelengths",
            ",".join(str(value) for value in wavelengths),
        )
    if getattr(dataset, "crs", None):
        setter("hypercoast/crs", str(dataset.crs))


def _set_selected_variable(dataset: Any, variable_name: str | None) -> None:
    """Set the selected variable on a HyperCoast dataset when supported.

    Args:
        dataset: HyperCoast dataset-like object.
        variable_name: Optional variable name.
    """
    if not variable_name:
        return
    setter = getattr(dataset, "set_selected_variable", None)
    if callable(setter):
        setter(variable_name)


def _load_and_export_dataset(
    dataset: Any, output_path: str, wavelengths: list[float] | None
) -> Any:
    """Load and export a HyperCoast dataset to GeoTIFF.

    Args:
        dataset: HyperCoast dataset-like object.
        output_path: Output GeoTIFF path.
        wavelengths: Optional RGB wavelengths.

    Returns:
        The export result returned by HyperCoast.
    """
    load_and_export = getattr(dataset, "load_and_export", None)
    if callable(load_and_export):
        return load_and_export(output_path, wavelengths=wavelengths)
    if not dataset.load():
        return None
    return dataset.export_to_geotiff(output_path, wavelengths=wavelengths)


def hypercoast_tools(
    iface: Any,
    project: Any | None = None,
    *,
    plugin: Any | None = None,
) -> list[Any]:
    """Create HyperCoast QGIS plugin tools.

    Args:
        iface: QGIS interface-like object.
        project: Optional QGIS project-like object.
        plugin: Optional HyperCoast plugin instance.

    Returns:
        List of Strands-compatible GeoAgent tools.
    """
    if iface is None:
        return []

    state: dict[str, Any] = {}

    @geo_tool(category="hypercoast", available_in=("full", "fast"))
    def list_hypercoast_data_types() -> dict[str, Any]:
        """List supported HyperCoast data types and wavelength presets."""
        return {
            "success": True,
            "data_types": DATA_TYPES,
            "wavelength_presets": WAVELENGTH_PRESETS,
            "search_sources": ["emit", "pace"],
        }

    @geo_tool(category="hypercoast")
    def search_hypercoast_data(
        source: str = "emit",
        bbox: str | list[float] | None = None,
        temporal: str | None = None,
        count: int = 10,
        short_name: str | None = None,
        provider: str | None = None,
        cloud_cover_min: float | None = None,
        cloud_cover_max: float | None = None,
        max_cloud_cover: float | None = None,
        use_current_extent: bool = False,
    ) -> dict[str, Any]:
        """Search NASA EMIT or PACE hyperspectral granules with HyperCoast.

        Do not pass ``cloud_cover_min``/``cloud_cover_max``/``max_cloud_cover``
        when ``source="pace"`` with an L3 mapped short name (for example
        ``PACE_OCI_L3M_BGC_*``); those composites have no per-granule
        CloudCover metadata and the filter would silently exclude every
        result. Pass cloud-cover filters only for L2 imaging products when
        the user explicitly asks for cloud-free or low-cloud results.

        Args:
            source: Search source, either ``emit`` or ``pace``.
            bbox: Optional west,south,east,north bounds.
            temporal: Optional Earthdata temporal range string.
            count: Maximum granules to request.
            short_name: Optional NASA Earthdata short name override.
            provider: Optional NASA Earthdata provider.
            cloud_cover_min: Optional minimum CMR cloud-cover percentage.
            cloud_cover_max: Optional maximum CMR cloud-cover percentage.
            max_cloud_cover: Convenience alias for ``cloud_cover_max``.
            use_current_extent: Whether to search the current QGIS map extent.

        Returns:
            Compact granule search results.
        """
        source_key = str(source or "emit").strip().lower()
        if source_key not in {"emit", "pace"}:
            raise ValueError("source must be 'emit' or 'pace'")

        parsed_bbox = (
            _current_bbox_wgs84(iface) if use_current_extent else _parse_bbox(bbox)
        )
        cloud_cover_requested = (
            cloud_cover_min is not None
            or cloud_cover_max is not None
            or max_cloud_cover is not None
        )
        cloud_cover = _normalize_cloud_cover_filter(
            cloud_cover_min=cloud_cover_min,
            cloud_cover_max=cloud_cover_max,
            max_cloud_cover=max_cloud_cover,
        )
        cloud_cover_ignored = False
        if (
            cloud_cover is not None
            and source_key == "pace"
            and _pace_short_name_ignores_cloud_cover(short_name)
        ):
            cloud_cover = None
            cloud_cover_ignored = True
        kwargs: dict[str, Any] = {
            "bbox": parsed_bbox,
            "temporal": temporal,
            "count": int(count),
        }
        if short_name:
            kwargs["short_name"] = short_name
        if provider and source_key == "pace":
            kwargs["provider"] = provider

        if cloud_cover is not None:
            try:
                results = _search_earthaccess_data(
                    source_key=source_key,
                    bbox=parsed_bbox,
                    temporal=temporal,
                    count=int(count),
                    short_name=short_name,
                    provider=provider,
                    cloud_cover=cloud_cover,
                )
                search_backend = "earthaccess"
            except Exception:
                results = _search_cmr_granules(
                    source_key=source_key,
                    bbox=parsed_bbox,
                    temporal=temporal,
                    count=int(count),
                    short_name=short_name,
                    provider=provider,
                    cloud_cover=cloud_cover,
                )
                search_backend = "cmr"
        else:
            common = _load_hypercoast_common()
            try:
                if source_key == "pace":
                    results = common.search_pace(**kwargs)
                else:
                    results = common.search_emit(**kwargs)
                search_backend = "hypercoast"
            except Exception as exc:
                if not _is_known_search_dependency_error(exc):
                    raise
                try:
                    results = _search_earthaccess_data(
                        source_key=source_key,
                        bbox=parsed_bbox,
                        temporal=temporal,
                        count=int(count),
                        short_name=short_name,
                        provider=provider,
                    )
                    search_backend = "earthaccess"
                except Exception as earthaccess_exc:
                    if not _is_known_search_dependency_error(earthaccess_exc):
                        raise
                    results = _search_cmr_granules(
                        source_key=source_key,
                        bbox=parsed_bbox,
                        temporal=temporal,
                        count=int(count),
                        short_name=short_name,
                        provider=provider,
                    )
                    search_backend = "cmr"

        granules = _search_result_granules(results)
        state["last_search_results"] = granules
        compact = _compact_search_results(granules)
        notes: list[str] = []
        if cloud_cover_ignored:
            notes.append(
                "Ignored cloud_cover for PACE L3 mapped product "
                f"'{short_name}' because L3 composites carry no per-granule "
                "CloudCover metadata."
            )
        if (
            compact.get("count", 0) == 0
            and cloud_cover is not None
            and cloud_cover[0] == 0
            and cloud_cover[1] == 0
        ):
            notes.append(
                "cloud_cover_min and cloud_cover_max were both 0, which only "
                "matches granules with exactly 0% cloud cover. Retry without "
                "the cloud_cover filter or widen the range."
            )
        compact.update(
            {
                "source": source_key,
                "search_backend": search_backend,
                "bbox": parsed_bbox,
                "temporal": temporal,
                "short_name": short_name,
                "provider": provider,
                "cloud_cover": list(cloud_cover) if cloud_cover else None,
                "cloud_cover_min": cloud_cover[0] if cloud_cover else None,
                "cloud_cover_max": cloud_cover[1] if cloud_cover else None,
                "cloud_cover_requested": cloud_cover_requested,
                "cloud_cover_ignored": cloud_cover_ignored,
            }
        )
        if notes:
            compact["notes"] = notes
        return compact

    @geo_tool(
        category="hypercoast",
        requires_confirmation=True,
    )
    def display_hypercoast_footprints(
        granules: str | list[Any] | None = None,
        layer_name: str = "HyperCoast Footprints",
    ) -> dict[str, Any]:
        """Display HyperCoast search-result footprints in the QGIS project.

        Args:
            granules: Optional granules from ``search_hypercoast_data``. When
                omitted, the most recent HyperCoast search results are used.
            layer_name: Name for the footprint layer.

        Returns:
            Footprint layer creation summary.
        """
        if isinstance(granules, str):
            parsed_granules = json.loads(granules)
        elif granules is None:
            parsed_granules = state.get("last_search_results")
        else:
            parsed_granules = granules

        if not isinstance(parsed_granules, list) or not parsed_granules:
            raise ValueError(
                "No HyperCoast search results are available. Search first or "
                "pass granules from search_hypercoast_data."
            )

        path = os.path.join(
            tempfile.gettempdir(),
            f"geoagent_hypercoast_footprints_{uuid.uuid4().hex}.geojson",
        )
        feature_count = _write_footprints_geojson(parsed_granules, path)
        if feature_count == 0:
            return {"success": False, "error": "No valid footprint geometries found."}

        def _run() -> dict[str, Any]:
            proj = _project_from_iface(iface, project)
            if proj is not None and hasattr(proj, "mapLayersByName"):
                for existing in proj.mapLayersByName(layer_name):
                    try:
                        proj.removeMapLayer(existing.id())
                    except Exception:
                        proj.removeMapLayer(existing)

            if hasattr(iface, "addVectorLayer"):
                layer = iface.addVectorLayer(path, layer_name, "ogr")
            else:
                from qgis.core import QgsVectorLayer  # type: ignore[import-not-found]

                layer = QgsVectorLayer(path, layer_name, "ogr")
                if proj is not None:
                    proj.addMapLayer(layer)

            if not layer or (hasattr(layer, "isValid") and not layer.isValid()):
                return {
                    "success": False,
                    "error": "Failed to create HyperCoast footprint layer.",
                    "path": path,
                }

            try:
                from qgis.PyQt.QtGui import QColor  # type: ignore[import-not-found]
                from qgis.core import QgsFillSymbol  # type: ignore[import-not-found]

                symbol = QgsFillSymbol.createSimple({})
                fill = symbol.symbolLayer(0)
                fill.setColor(QColor(0, 128, 128, 45))
                fill.setStrokeColor(QColor(0, 128, 128, 220))
                fill.setStrokeWidth(0.5)
                layer.renderer().setSymbol(symbol)
            except Exception:
                pass

            if hasattr(iface, "setActiveLayer"):
                iface.setActiveLayer(layer)
            try:
                iface.mapCanvas().refresh()
            except Exception:
                pass
            return {
                "success": True,
                "layer_name": layer_name,
                "feature_count": feature_count,
                "path": path,
            }

        return _on_gui(_run)

    @geo_tool(category="hypercoast")
    def get_selected_hypercoast_footprints(
        layer_name: str | None = None,
    ) -> dict[str, Any]:
        """Return selected HyperCoast footprint granules with named metadata.

        Args:
            layer_name: Optional footprint layer name. When omitted, the active
                layer is used.

        Returns:
            Selected granule records suitable for ``download_hypercoast_data``.
        """

        def _run() -> dict[str, Any]:
            if layer_name is None:
                layer = iface.activeLayer()
                if layer is None:
                    return {"success": False, "error": "No active layer."}
            else:
                proj = _project_from_iface(iface, project)
                matches = (
                    proj.mapLayersByName(layer_name)
                    if proj is not None and hasattr(proj, "mapLayersByName")
                    else []
                )
                if not matches:
                    return {
                        "success": False,
                        "error": f"Layer not found: {layer_name}",
                    }
                layer = matches[0]

            selected = getattr(layer, "selectedFeatures", lambda: [])()
            granules: list[dict[str, Any]] = []
            for feature in selected:
                granule = _granule_from_feature(feature)
                if granule is not None:
                    granules.append(_compact_granule(granule))
            return {
                "success": True,
                "layer_name": layer_name or getattr(layer, "name", lambda: "")(),
                "count": len(granules),
                "granules": granules,
            }

        return _on_gui(_run)

    @geo_tool(
        category="hypercoast",
        requires_confirmation=True,
        long_running=True,
    )
    def download_hypercoast_data(
        granules: str | list[Any],
        source: str = "emit",
        out_dir: str | None = None,
        provider: str | None = None,
        threads: int = 8,
        max_links_per_granule: int = 1,
    ) -> dict[str, Any]:
        """Download selected EMIT or PACE granules with HyperCoast.

        Args:
            granules: Granules from ``search_hypercoast_data`` or a JSON list.
            source: Download source, either ``emit`` or ``pace``.
            out_dir: Optional output directory.
            provider: Optional provider for PACE downloads.
            threads: Number of download threads.
            max_links_per_granule: Direct-link fallback downloads this many
                data links per granule.

        Returns:
            Downloaded local file paths.
        """
        source_key = str(source or "emit").strip().lower()
        parsed_granules = (
            json.loads(granules) if isinstance(granules, str) else granules
        )
        if not isinstance(parsed_granules, list) or not parsed_granules:
            raise ValueError("granules must be a non-empty list or JSON list")
        parsed_granules = _normalize_granules_input(parsed_granules)
        target_dir = os.path.abspath(
            os.path.expanduser(
                out_dir or os.path.join("~", ".qgis_hypercoast", "downloads")
            )
        )
        os.makedirs(target_dir, exist_ok=True)
        if source_key not in {"emit", "pace"}:
            raise ValueError("source must be 'emit' or 'pace'")
        if any(
            _granule_data_links(item, max_links_per_granule) for item in parsed_granules
        ):
            paths = _download_data_links(
                parsed_granules,
                out_dir=target_dir,
                max_links_per_granule=max_links_per_granule,
            )
            download_backend = "direct_links"
        else:
            try:
                common = _load_hypercoast_common()
                if source_key == "pace":
                    paths = common.download_pace(
                        parsed_granules,
                        out_dir=target_dir,
                        provider=provider,
                        threads=int(threads),
                    )
                else:
                    paths = common.download_emit(
                        parsed_granules,
                        out_dir=target_dir,
                        threads=int(threads),
                    )
                download_backend = "hypercoast"
            except Exception as exc:
                if not _is_known_search_dependency_error(exc):
                    raise
                paths = _download_data_links(
                    parsed_granules,
                    out_dir=target_dir,
                    max_links_per_granule=max_links_per_granule,
                )
                download_backend = "direct_links"
        return {
            "success": True,
            "source": source_key,
            "download_backend": download_backend,
            "out_dir": target_dir,
            "count": len(paths or []),
            "paths": list(paths or []),
        }

    @geo_tool(category="hypercoast")
    def preview_hypercoast_dataset(
        filepath: str,
        data_type: str = "auto",
        variable_name: str | None = None,
    ) -> dict[str, Any]:
        """Preview metadata for a local HyperCoast-compatible dataset.

        Args:
            filepath: Local hyperspectral dataset path.
            data_type: HyperCoast data type or ``auto``.
            variable_name: Optional data variable to inspect.

        Returns:
            Compact dataset metadata.
        """
        path = os.path.abspath(os.path.expanduser(filepath))
        if not os.path.exists(path):
            return {"success": False, "error": f"File does not exist: {path}"}
        normalized_data_type = _normalize_data_type(data_type)
        HyperspectralDataset, _ = _load_hyperspectral_runtime(plugin)
        _patch_hypercoast_top_level_exports(HyperspectralDataset)
        dataset = HyperspectralDataset(path, normalized_data_type)
        _set_selected_variable(dataset, variable_name)
        if not dataset.load():
            return {
                "success": False,
                "error": getattr(dataset, "last_error", None)
                or "Failed to load dataset",
            }
        metadata = _dataset_metadata(dataset)
        metadata["success"] = True
        return metadata

    @geo_tool(
        category="hypercoast",
        requires_confirmation=True,
        long_running=True,
    )
    def load_hypercoast_rgb(
        filepath: str,
        data_type: str = "auto",
        red: float = 650.0,
        green: float = 550.0,
        blue: float = 450.0,
        layer_name: str | None = None,
        variable_name: str | None = None,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Export an RGB hyperspectral GeoTIFF and add it to QGIS.

        Args:
            filepath: Local hyperspectral dataset path.
            data_type: HyperCoast data type or ``auto``.
            red: Red wavelength in nanometers.
            green: Green wavelength in nanometers.
            blue: Blue wavelength in nanometers.
            layer_name: Optional QGIS layer name.
            variable_name: Optional data variable to export.
            output_path: Optional GeoTIFF output path.

        Returns:
            Load result with output path and QGIS layer metadata.
        """
        path = os.path.abspath(os.path.expanduser(filepath))
        if not os.path.exists(path):
            return {"success": False, "error": f"File does not exist: {path}"}
        normalized_data_type = _normalize_data_type(data_type)
        plugin_instance = _resolve_hypercoast_plugin(plugin)
        HyperspectralDataset, create_generated_raster_path = (
            _load_hyperspectral_runtime(plugin_instance)
        )
        _patch_hypercoast_top_level_exports(HyperspectralDataset)
        dataset = HyperspectralDataset(path, normalized_data_type)
        _set_selected_variable(dataset, variable_name)
        name = layer_name or os.path.splitext(os.path.basename(path))[0]
        if variable_name and layer_name is None:
            name = f"{name} - {variable_name}"
        project_obj = _project_from_iface(iface, project)
        wavelengths = [float(red), float(green), float(blue)]
        target_path = _ensure_output_path(
            name,
            project_obj,
            create_generated_raster_path,
            output_path,
        )
        result = _load_and_export_dataset(dataset, target_path, wavelengths)
        if result is None:
            return {
                "success": False,
                "error": getattr(dataset, "last_error", None) or "Failed to export RGB",
            }
        layer = _add_qgis_raster_layer(iface, project_obj, target_path, name)
        _set_layer_custom_properties(layer, dataset, path, wavelengths, variable_name)
        layer_id_getter = getattr(layer, "id", None)
        layer_id = layer_id_getter() if callable(layer_id_getter) else name
        data_info = {
            "dataset": dataset,
            "filepath": path,
            "data_type": getattr(dataset, "data_type", normalized_data_type),
            "wavelengths": getattr(dataset, "wavelengths", None),
            "rgb_wavelengths": wavelengths,
            "selected_variable": variable_name,
            "bounds": getattr(dataset, "bounds", None),
            "crs": getattr(dataset, "crs", None),
        }
        register = getattr(plugin_instance, "register_hyperspectral_layer", None)
        if callable(register):
            register(layer_id, data_info)
        return {
            "success": True,
            "loaded": True,
            "layer_name": name,
            "layer_id": layer_id,
            "output_path": target_path,
            "source_path": path,
            "data_type": getattr(dataset, "data_type", normalized_data_type),
            "rgb_wavelengths": wavelengths,
            "selected_variable": variable_name,
        }

    @geo_tool(
        category="hypercoast",
        requires_confirmation=True,
        long_running=True,
    )
    def load_hypercoast_variable(
        filepath: str,
        data_type: str = "auto",
        variable_name: str | None = None,
        layer_name: str | None = None,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Export a single HyperCoast raster variable and add it to QGIS.

        This is the preferred visualization tool for PACE OCI BGC products
        such as ``chlor_a``. The file must be local; download online granules
        with ``download_hypercoast_data`` before calling this tool.

        Args:
            filepath: Local hyperspectral or ocean-color dataset path.
            data_type: HyperCoast data type or ``auto``.
            variable_name: Dataset variable to export, for example
                ``chlor_a`` for PACE OCI BGC.
            layer_name: Optional QGIS layer name.
            output_path: Optional GeoTIFF output path.

        Returns:
            Load result with output path and QGIS layer metadata.
        """
        path = os.path.abspath(os.path.expanduser(filepath))
        if not os.path.exists(path):
            return {"success": False, "error": f"File does not exist: {path}"}
        normalized_data_type = _normalize_data_type(data_type)
        plugin_instance = _resolve_hypercoast_plugin(plugin)
        HyperspectralDataset, create_generated_raster_path = (
            _load_hyperspectral_runtime(plugin_instance)
        )
        _patch_hypercoast_top_level_exports(HyperspectralDataset)
        dataset = HyperspectralDataset(path, normalized_data_type)
        _set_selected_variable(dataset, variable_name)
        base_name = os.path.splitext(os.path.basename(path))[0]
        name = layer_name or (
            f"{base_name} - {variable_name}" if variable_name else base_name
        )
        project_obj = _project_from_iface(iface, project)
        target_path = _ensure_output_path(
            name,
            project_obj,
            create_generated_raster_path,
            output_path,
            suffix=variable_name or "var",
        )
        result = _load_and_export_dataset(dataset, target_path, None)
        if result is None:
            return {
                "success": False,
                "error": getattr(dataset, "last_error", None)
                or "Failed to export variable raster",
            }
        layer = _add_qgis_raster_layer(iface, project_obj, target_path, name)
        _set_layer_custom_properties(layer, dataset, path, None, variable_name)
        layer_id_getter = getattr(layer, "id", None)
        layer_id = layer_id_getter() if callable(layer_id_getter) else name
        data_info = {
            "dataset": dataset,
            "filepath": path,
            "data_type": getattr(dataset, "data_type", normalized_data_type),
            "wavelengths": getattr(dataset, "wavelengths", None),
            "selected_variable": variable_name,
            "bounds": getattr(dataset, "bounds", None),
            "crs": getattr(dataset, "crs", None),
        }
        register = getattr(plugin_instance, "register_hyperspectral_layer", None)
        if callable(register):
            register(layer_id, data_info)
        return {
            "success": True,
            "loaded": True,
            "layer_name": name,
            "layer_id": layer_id,
            "output_path": target_path,
            "source_path": path,
            "data_type": getattr(dataset, "data_type", normalized_data_type),
            "selected_variable": variable_name,
        }

    @geo_tool(category="hypercoast", requires_confirmation=True)
    def open_hypercoast_panel(panel: str = "load") -> dict[str, Any]:
        """Open a HyperCoast plugin dock or map tool.

        Args:
            panel: Panel name: ``load``, ``band``, ``spectral``, or ``settings``.

        Returns:
            Panel-open result.
        """
        plugin_instance = _resolve_hypercoast_plugin(plugin)
        if plugin_instance is None:
            return {"success": False, "error": "HyperCoast plugin is not available"}
        panel_key = str(panel or "load").strip().lower().replace("-", "_")
        method_name = {
            "load": "show_load_dialog",
            "load_data": "show_load_dialog",
            "band": "show_band_dialog",
            "bands": "show_band_dialog",
            "band_combination": "show_band_dialog",
            "spectral": "toggle_spectral_inspector",
            "spectral_inspector": "toggle_spectral_inspector",
            "settings": "toggle_settings_dock",
            "dependencies": "toggle_settings_dock",
        }.get(panel_key)
        if method_name is None:
            raise ValueError("panel must be load, band, spectral, or settings")
        method = getattr(plugin_instance, method_name, None)
        if not callable(method):
            return {
                "success": False,
                "error": f"HyperCoast plugin does not expose {method_name}",
            }

        def _run() -> None:
            method()

        _on_gui(_run)
        return {"success": True, "panel": panel_key, "method": method_name}

    return [
        list_hypercoast_data_types,
        search_hypercoast_data,
        display_hypercoast_footprints,
        get_selected_hypercoast_footprints,
        download_hypercoast_data,
        preview_hypercoast_dataset,
        load_hypercoast_rgb,
        load_hypercoast_variable,
        open_hypercoast_panel,
    ]
