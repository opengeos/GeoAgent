"""Tool adapters for the QGIS Timelapse plugin.

The module is import-safe outside QGIS and outside the Timelapse plugin. Tool
bodies resolve QGIS, Earth Engine, and ``timelapse`` plugin modules lazily so
ordinary GeoAgent imports do not depend on a QGIS runtime.
"""

from __future__ import annotations

import os
import sys
import tempfile
import uuid
from datetime import datetime
from typing import Any

from geoagent.core.decorators import geo_tool
from geoagent.tools._qt_marshal import run_on_qt_gui_thread

TIMELAPSE_IMAGERY_TYPES: dict[str, dict[str, Any]] = {
    "Landsat": {
        "description": "Long-running optical satellite timelapses from 1984 onward.",
        "default_start_year": 1990,
        "default_bands": ["NIR", "Red", "Green"],
        "default_date_window": ["06-10", "09-20"],
        "supports_frequency": True,
    },
    "Sentinel-2": {
        "description": "Optical Sentinel-2 surface reflectance timelapses.",
        "default_start_year": 2018,
        "default_bands": ["NIR", "Red", "Green"],
        "default_date_window": ["06-10", "09-20"],
        "supports_frequency": True,
    },
    "Sentinel-1": {
        "description": "Synthetic aperture radar timelapses using Sentinel-1.",
        "default_start_year": 2018,
        "default_bands": ["VV"],
        "default_date_window": ["01-01", "12-31"],
        "supports_frequency": True,
    },
    "NAIP": {
        "description": "High-resolution aerial imagery timelapses for the US.",
        "default_start_year": 2010,
        "default_bands": ["R", "G", "B"],
        "supports_frequency": False,
    },
    "MODIS NDVI": {
        "description": "Vegetation index phenology timelapses from MODIS.",
        "default_start_year": 2010,
        "default_bands": ["NDVI"],
        "supports_frequency": False,
    },
    "GOES": {
        "description": "Weather satellite timelapses from GOES imagery.",
        "default_start_date": "2021-10-24T14:00:00",
        "default_end_date": "2021-10-25T01:00:00",
        "default_band_combination": "true_color",
        "supports_frequency": False,
    },
}


def _on_gui(fn: Any) -> Any:
    """Run ``fn`` on the Qt GUI thread when QGIS is available."""
    return run_on_qt_gui_thread(fn)


def _resolve_timelapse_plugin(plugin: Any = None) -> Any:
    """Return the supplied or installed Timelapse plugin instance, if present."""
    if plugin is not None:
        return plugin
    try:
        from qgis.utils import plugins  # type: ignore[import-not-found]

        return plugins.get("timelapse") or plugins.get("Timelapse")
    except Exception:
        return None


def _add_plugin_parent_to_path(plugin: Any = None) -> None:
    """Ensure the plugin package parent is importable when a plugin is loaded."""
    resolved = _resolve_timelapse_plugin(plugin)
    plugin_dir = getattr(resolved, "plugin_dir", None)
    if not plugin_dir:
        return
    parent = os.path.dirname(os.path.abspath(str(plugin_dir)))
    if not parent:
        return
    while parent in sys.path:
        sys.path.remove(parent)
    sys.path.insert(0, parent)


def _ensure_plugin_deps(plugin: Any = None) -> bool:
    """Ask the plugin to prepare dependencies when it exposes that helper."""
    resolved = _resolve_timelapse_plugin(plugin)
    ensure = getattr(resolved, "_ensure_deps", None)
    if callable(ensure):
        return bool(_on_gui(ensure))
    return False


def _ensure_timelapse_runtime(core: Any) -> tuple[bool, str]:
    """Return whether Timelapse core dependencies are importable."""
    try:
        from timelapse.core import venv_manager

        venv_manager.ensure_venv_packages_available()
    except Exception:
        pass

    try:
        reload_deps = getattr(core, "reload_dependencies", None)
        if callable(reload_deps):
            deps = reload_deps()
        else:
            deps = core.check_dependencies()
    except Exception as exc:
        return False, f"Could not check Timelapse dependencies: {exc}"

    missing = [name for name, available in deps.items() if not available]
    if missing:
        return (
            False,
            "Timelapse dependencies are not importable: "
            f"{', '.join(missing)}. Open the Timelapse settings panel and "
            "install dependencies.",
        )
    return True, ""


def _load_timelapse_core(plugin: Any = None) -> Any:
    """Import the Timelapse core module lazily."""
    _add_plugin_parent_to_path(plugin)
    _ensure_plugin_deps(plugin)
    try:
        from timelapse.core import timelapse_core
    except Exception as exc:
        raise RuntimeError(
            "Could not import timelapse.core.timelapse_core. Make sure the "
            "QGIS Timelapse plugin is installed and loaded."
        ) from exc

    runtime_ready, reason = _ensure_timelapse_runtime(timelapse_core)
    if not runtime_ready:
        raise RuntimeError(reason)
    return timelapse_core


def _parse_list(value: Any) -> list[str] | None:
    """Parse list-like or comma-separated tool input."""
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_bbox(bbox: Any) -> list[float] | None:
    """Parse west,south,east,north bbox input."""
    if bbox is None or bbox == "":
        return None
    if isinstance(bbox, dict):
        if {"xmin", "ymin", "xmax", "ymax"}.issubset(bbox):
            values = [bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]]
        else:
            values = [
                bbox.get("west"),
                bbox.get("south"),
                bbox.get("east"),
                bbox.get("north"),
            ]
    elif isinstance(bbox, (list, tuple)):
        values = list(bbox)
    else:
        values = str(bbox).split(",")
    if len(values) != 4:
        raise ValueError("bbox must contain west,south,east,north")
    west, south, east, north = [float(value) for value in values]
    if west >= east or south >= north:
        raise ValueError("bbox coordinates must satisfy west < east and south < north")
    return [west, south, east, north]


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
    """Return the current canvas extent as a WGS84 bbox."""
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


def _normalise_imagery_type(imagery_type: str) -> str:
    """Return a canonical Timelapse imagery type."""
    key = str(imagery_type or "").strip().lower().replace("_", " ")
    key = " ".join(key.replace("-", " ").split())
    aliases = {
        "landsat": "Landsat",
        "sentinel 2": "Sentinel-2",
        "sentinel2": "Sentinel-2",
        "s2": "Sentinel-2",
        "sentinel 1": "Sentinel-1",
        "sentinel1": "Sentinel-1",
        "s1": "Sentinel-1",
        "naip": "NAIP",
        "modis": "MODIS NDVI",
        "modis ndvi": "MODIS NDVI",
        "ndvi": "MODIS NDVI",
        "goes": "GOES",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        allowed = ", ".join(TIMELAPSE_IMAGERY_TYPES)
        raise ValueError(f"Unsupported imagery_type. Choose one of: {allowed}") from exc


def _default_output_path(imagery_type: str) -> str:
    """Return a unique default output GIF path in a dedicated temp directory."""
    output_dir = os.path.join(tempfile.gettempdir(), "open_geoagent_timelapse")
    os.makedirs(output_dir, exist_ok=True)
    name = imagery_type.lower().replace("-", "").replace(" ", "_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return os.path.join(output_dir, f"{name}_timelapse_{stamp}_{suffix}.gif")


def _mp4_path_for(output_path: str, create_mp4: bool) -> str | None:
    """Return the expected MP4 path when MP4 output is requested."""
    if not create_mp4:
        return None
    root, ext = os.path.splitext(output_path)
    return f"{root}.mp4" if ext.lower() == ".gif" else f"{output_path}.mp4"


def _image_artifact_for_output(output_path: str, imagery_type: str) -> dict[str, Any]:
    """Return transcript-renderable image metadata for a generated GIF."""
    ext = output_path.rsplit(".", 1)[-1].lower() if "." in output_path else "gif"
    if ext == "jpeg":
        ext = "jpg"
    mime_type = "image/jpeg" if ext == "jpg" else f"image/{ext}"
    return {
        "path": output_path,
        "format": ext,
        "mime_type": mime_type,
        "alt": f"{imagery_type} timelapse",
    }


class _FallbackBboxLayer:
    """Small QGIS-layer-like object for tests and non-QGIS fallbacks."""

    def __init__(self, name: str, bbox: list[float]) -> None:
        self._name = name
        self._bbox = tuple(bbox)

    def name(self) -> str:
        """Return layer name."""
        return self._name

    def source(self) -> str:
        """Return a compact bbox source marker."""
        return ",".join(str(value) for value in self._bbox)

    def type(self) -> str:
        """Return layer type."""
        return "vector"

    def extent(self) -> tuple[float, float, float, float]:
        """Return layer extent."""
        return self._bbox

    def isValid(self) -> bool:
        """Return layer validity."""
        return True


def _project_for_bbox_layer(iface: Any, project: Any = None) -> Any:
    """Return a QGIS project-like object for adding bbox layers."""
    if project is not None:
        return project
    try:
        candidate = iface.project()
        if candidate is not None:
            return candidate
    except Exception:
        pass
    try:
        from qgis.core import QgsProject  # type: ignore[import-not-found]

        return QgsProject.instance()
    except Exception:
        return None


def _add_timelapse_bbox_layer(
    iface: Any,
    project: Any,
    bbox: list[float],
    layer_name: str,
) -> dict[str, Any]:
    """Add the timelapse bbox as a polygon layer in QGIS."""
    west, south, east, north = bbox

    def _run() -> dict[str, Any]:
        proj = _project_for_bbox_layer(iface, project)
        if proj is None or not hasattr(proj, "addMapLayer"):
            return {
                "success": False,
                "layer_name": layer_name,
                "reason": "No QGIS project is available for adding the bbox layer.",
            }

        try:
            from qgis.core import (  # type: ignore[import-not-found]
                QgsFeature,
                QgsFillSymbol,
                QgsGeometry,
                QgsPointXY,
                QgsVectorLayer,
            )
        except ImportError:
            layer = _FallbackBboxLayer(layer_name, bbox)
            proj.addMapLayer(layer)
        else:
            try:
                layer = QgsVectorLayer("Polygon?crs=EPSG:4326", layer_name, "memory")
                if not layer.isValid():
                    return {
                        "success": False,
                        "layer_name": layer_name,
                        "reason": "Could not create bbox memory layer.",
                    }
                feature = QgsFeature()
                points = [
                    QgsPointXY(west, south),
                    QgsPointXY(east, south),
                    QgsPointXY(east, north),
                    QgsPointXY(west, north),
                    QgsPointXY(west, south),
                ]
                feature.setGeometry(QgsGeometry.fromPolygonXY([points]))
                provider = layer.dataProvider()
                provider.addFeatures([feature])
                layer.updateExtents()

                try:
                    symbol = QgsFillSymbol.createSimple(
                        {
                            "color": "255,0,0,20",
                            "outline_color": "255,0,0,255",
                            "outline_width": "0.8",
                        }
                    )
                    layer.renderer().setSymbol(symbol)
                except Exception:
                    pass

                proj.addMapLayer(layer)
            except Exception as exc:
                return {
                    "success": False,
                    "layer_name": layer_name,
                    "reason": str(exc),
                }

        try:
            iface.mapCanvas().refresh()
        except Exception:
            pass
        try:
            canvas = iface.mapCanvas()
            try:
                from geoagent.tools.qgis import _transform_bbox_to_canvas_crs

                extent = _transform_bbox_to_canvas_crs(
                    canvas,
                    west,
                    south,
                    east,
                    north,
                    "EPSG:4326",
                )
            except Exception:
                extent = layer.extent()
            if extent is not None:
                canvas.setExtent(extent)
                canvas.refresh()
        except Exception:
            pass
        return {"success": True, "layer_name": layer_name, "bbox": bbox}

    return _on_gui(_run)


def _open_plugin_panel(plugin: Any, method_name: str, dock_attr: str) -> dict[str, Any]:
    """Open a Timelapse plugin dock by delegating to its public toggle method."""
    if plugin is None:
        return {
            "success": False,
            "opened": False,
            "reason": "The QGIS Timelapse plugin instance is not available.",
        }
    method = getattr(plugin, method_name, None)
    if not callable(method):
        return {
            "success": False,
            "opened": False,
            "reason": f"The plugin does not expose {method_name}().",
        }

    def _run() -> dict[str, Any]:
        method()
        dock = getattr(plugin, dock_attr, None)
        if dock is not None:
            try:
                dock.show()
            except Exception:
                pass
            try:
                dock.raise_()
            except Exception:
                pass
        return {"success": True, "opened": True}

    return _on_gui(_run)


def timelapse_tools(
    iface: Any = None,
    project: Any = None,
    *,
    plugin: Any | None = None,
) -> list[Any]:
    """Return Timelapse plugin tools bound to a QGIS interface."""
    if iface is None:
        return []

    @geo_tool(
        category="timelapse",
        name="list_timelapse_imagery_types",
        available_in=("full", "fast"),
    )
    def list_timelapse_imagery_types() -> dict[str, Any]:
        """List Timelapse imagery types and their default options."""
        current_year = datetime.now().year
        types = []
        for name, info in TIMELAPSE_IMAGERY_TYPES.items():
            item = {"name": name, **info}
            item.setdefault("default_end_year", current_year)
            types.append(item)
        return {"success": True, "count": len(types), "imagery_types": types}

    @geo_tool(
        category="timelapse",
        name="get_current_timelapse_extent",
        available_in=("full", "fast"),
    )
    def get_current_timelapse_extent() -> dict[str, Any]:
        """Return the current QGIS map extent as a WGS84 Timelapse bbox."""
        if not hasattr(iface, "mapCanvas"):
            return {
                "success": False,
                "bbox": None,
                "crs": None,
                "reason": "No QGIS map canvas is bound to this GeoAgent.",
            }

        def _read_extent() -> dict[str, Any]:
            return _canvas_extent_to_wgs84_bbox(iface.mapCanvas())

        return _on_gui(_read_extent)

    @geo_tool(
        category="timelapse",
        name="initialize_timelapse_earth_engine",
        requires_confirmation=True,
    )
    def initialize_timelapse_earth_engine(
        project_id: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Initialize Earth Engine for Timelapse generation.

        Args:
            project_id: Optional Google Cloud project id. When omitted, the
                Timelapse plugin falls back to QSettings or EE_PROJECT_ID.
            force: Reinitialize even when the plugin reports EE is initialized.
        """
        try:
            core = _load_timelapse_core(plugin)
            ok = bool(core.initialize_ee(project=project_id, force=bool(force)))
            return {
                "success": ok,
                "initialized": ok,
                "project_id": project_id
                or getattr(core, "get_ee_project", lambda: None)(),
                "reason": None if ok else "Earth Engine initialization failed.",
            }
        except Exception as exc:
            return {
                "success": False,
                "initialized": False,
                "project_id": project_id,
                "error": f"{type(exc).__name__}: {exc}",
            }

    @geo_tool(
        category="timelapse",
        name="create_timelapse",
        requires_confirmation=True,
        long_running=True,
    )
    def create_timelapse(
        imagery_type: str = "Landsat",
        bbox: Any = None,
        output_path: str | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        frequency: str = "year",
        step: int = 1,
        dimensions: int = 768,
        fps: int = 5,
        bands: Any = None,
        cloud_pct: int = 30,
        apply_fmask: bool = True,
        orbit: Any = None,
        title: str | None = None,
        add_text: bool = True,
        font_size: int = 20,
        font_color: str = "white",
        add_progress_bar: bool = True,
        progress_bar_color: str = "white",
        progress_bar_height: int = 5,
        create_mp4: bool = False,
        add_bbox_to_map: bool = True,
        bbox_layer_name: str | None = None,
        gee_project: str | None = None,
        modis_satellite: str = "Terra",
        modis_band: str = "NDVI",
        goes_satellite: str = "GOES-19",
        goes_scan: str = "full_disk",
        goes_band_combination: str = "true_color",
        goes_custom_bands: Any = None,
    ) -> dict[str, Any]:
        """Create a Timelapse GIF from a bbox or the current QGIS extent.

        Args:
            imagery_type: Landsat, Sentinel-2, Sentinel-1, NAIP, MODIS NDVI, or GOES.
            bbox: Optional west,south,east,north WGS84 bbox. Uses current map
                extent when omitted.
            output_path: Optional output GIF path.
            start_year: Optional start year for annual imagery types.
            end_year: Optional end year for annual imagery types.
            start_date: MM-DD date window, or ISO datetime for GOES/MODIS.
            end_date: MM-DD date window, or ISO datetime for GOES/MODIS.
            frequency: year, quarter, month, or day for supported imagery types.
            step: Temporal step.
            dimensions: GIF frame dimensions.
            fps: Frames per second.
            bands: Optional band list or comma-separated band names.
            cloud_pct: Sentinel-2 cloud percentage threshold.
            apply_fmask: Whether to mask clouds for Landsat/Sentinel-2.
            orbit: Sentinel-1 orbit list or comma-separated values.
            create_mp4: Also create MP4 when supported by the plugin core.
            add_bbox_to_map: Add the timelapse bbox as a QGIS polygon layer.
            bbox_layer_name: Optional QGIS layer name for the bbox layer.
        """
        try:
            imagery = _normalise_imagery_type(imagery_type)
            parsed_bbox = _parse_bbox(bbox)
            if parsed_bbox is None:
                extent_result = get_current_timelapse_extent.__wrapped__()
                if not extent_result.get("success"):
                    return extent_result
                parsed_bbox = list(extent_result["bbox"])

            core = _load_timelapse_core(plugin)
            if not core.is_ee_initialized():
                if not core.initialize_ee(project=gee_project):
                    return {
                        "success": False,
                        "error": "Failed to initialize Earth Engine.",
                        "imagery_type": imagery,
                        "bbox": parsed_bbox,
                    }

            west, south, east, north = parsed_bbox
            roi = core.bbox_to_ee_geometry(west, south, east, north)
            output = os.path.abspath(
                os.path.expanduser(output_path or _default_output_path(imagery))
            )
            os.makedirs(os.path.dirname(output), exist_ok=True)
            current_year = datetime.now().year
            parsed_bands = _parse_list(bands)
            common = {
                "roi": roi,
                "out_gif": output,
                "dimensions": int(dimensions),
                "frames_per_second": int(fps),
                "title": title,
                "add_text": bool(add_text),
                "font_size": int(font_size),
                "font_color": font_color,
                "add_progress_bar": bool(add_progress_bar),
                "progress_bar_color": progress_bar_color,
                "progress_bar_height": int(progress_bar_height),
                "mp4": bool(create_mp4),
            }

            if imagery == "NAIP":
                result = core.create_naip_timelapse(
                    **common,
                    start_year=start_year or 2010,
                    end_year=end_year or current_year,
                    bands=parsed_bands or ["R", "G", "B"],
                    step=int(step),
                )
            elif imagery == "Sentinel-2":
                result = core.create_sentinel2_timelapse(
                    **common,
                    start_year=start_year or 2018,
                    end_year=end_year or current_year,
                    start_date=start_date or "06-10",
                    end_date=end_date or "09-20",
                    bands=parsed_bands or ["NIR", "Red", "Green"],
                    apply_fmask=bool(apply_fmask),
                    cloud_pct=int(cloud_pct),
                    frequency=frequency,
                    step=int(step),
                )
            elif imagery == "Sentinel-1":
                result = core.create_sentinel1_timelapse(
                    **common,
                    start_year=start_year or 2018,
                    end_year=end_year or current_year,
                    start_date=start_date or "01-01",
                    end_date=end_date or "12-31",
                    bands=parsed_bands or ["VV"],
                    orbit=_parse_list(orbit) or ["ascending", "descending"],
                    frequency=frequency,
                    step=int(step),
                )
            elif imagery == "Landsat":
                result = core.create_landsat_timelapse(
                    **common,
                    start_year=start_year or 1990,
                    end_year=end_year or current_year,
                    start_date=start_date or "06-10",
                    end_date=end_date or "09-20",
                    bands=parsed_bands or ["NIR", "Red", "Green"],
                    apply_fmask=bool(apply_fmask),
                    frequency=frequency,
                    step=int(step),
                )
            elif imagery == "MODIS NDVI":
                result = core.create_modis_ndvi_timelapse(
                    **common,
                    data=modis_satellite,
                    band=modis_band,
                    start_date=start_date or f"{start_year or 2010}-01-01",
                    end_date=end_date or f"{end_year or current_year}-12-31",
                )
            else:
                result = core.create_goes_timelapse(
                    **common,
                    start_date=start_date or "2021-10-24T14:00:00",
                    end_date=end_date or "2021-10-25T01:00:00",
                    data=goes_satellite,
                    scan=goes_scan,
                    band_combination=goes_band_combination,
                    custom_bands=_parse_list(goes_custom_bands),
                )

            result_path = os.path.abspath(os.path.expanduser(str(result or output)))
            bbox_layer = None
            if add_bbox_to_map:
                layer_name = (
                    bbox_layer_name
                    or f"Timelapse BBOX - {imagery} - {datetime.now():%H%M%S}"
                )
                bbox_layer = _add_timelapse_bbox_layer(
                    iface,
                    project,
                    parsed_bbox,
                    layer_name,
                )
            return {
                "success": True,
                "imagery_type": imagery,
                "output_path": result_path,
                "mp4_path": _mp4_path_for(result_path, bool(create_mp4)),
                "images": [_image_artifact_for_output(result_path, imagery)],
                "bbox": parsed_bbox,
                "bbox_layer": bbox_layer,
                "start_year": start_year,
                "end_year": end_year,
                "start_date": start_date,
                "end_date": end_date,
            }
        except Exception as exc:
            return {
                "success": False,
                "imagery_type": imagery_type,
                "error": f"{type(exc).__name__}: {exc}",
            }

    @geo_tool(
        category="timelapse",
        name="open_timelapse_panel",
        requires_confirmation=True,
    )
    def open_timelapse_panel() -> dict[str, Any]:
        """Open the QGIS Timelapse plugin creation panel."""
        resolved = _resolve_timelapse_plugin(plugin)
        return _open_plugin_panel(
            resolved,
            "toggle_timelapse_dock",
            "_timelapse_dock",
        )

    @geo_tool(
        category="timelapse",
        name="open_timelapse_settings",
        requires_confirmation=True,
    )
    def open_timelapse_settings() -> dict[str, Any]:
        """Open the QGIS Timelapse plugin settings panel."""
        resolved = _resolve_timelapse_plugin(plugin)
        return _open_plugin_panel(
            resolved,
            "toggle_settings_dock",
            "_settings_dock",
        )

    return [
        list_timelapse_imagery_types,
        get_current_timelapse_extent,
        initialize_timelapse_earth_engine,
        create_timelapse,
        open_timelapse_panel,
        open_timelapse_settings,
    ]
