"""Tool adapters for the GeoAI QGIS plugin.

This module is import-safe outside QGIS and outside the GeoAI plugin. Tool
bodies resolve QGIS and GeoAI plugin modules lazily so ordinary GeoAgent
imports do not import PyTorch, SamGeo, or the external ``geoai`` library.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from typing import Any, Callable

from geoagent.core.decorators import geo_tool
from geoagent.tools._qt_marshal import run_on_qt_gui_thread


def _on_gui(fn: Any) -> Any:
    """Run ``fn`` on the Qt GUI thread when QGIS is available."""
    return run_on_qt_gui_thread(fn)


def _project_getter(project: Any | None) -> Callable[[], Any]:
    """Return a callable that yields the active QGIS project."""

    def _get_project() -> Any:
        if project is not None:
            return project
        try:
            from qgis.core import QgsProject  # type: ignore[import-not-found]

            return QgsProject.instance()
        except Exception:
            return None

    return _get_project


def _resolve_geoai_plugin(plugin: Any = None) -> Any:
    """Return the supplied or installed GeoAI plugin instance, if present."""
    if plugin is not None:
        return plugin
    try:
        from qgis.utils import plugins  # type: ignore[import-not-found]

        return plugins.get("geoai") or plugins.get("GeoAI")
    except Exception:
        return None


def _add_plugin_parent_to_path(plugin: Any) -> None:
    """Ensure the GeoAI plugin package parent is importable."""
    plugin_dir = getattr(plugin, "plugin_dir", None)
    if not plugin_dir:
        return
    parent = os.path.dirname(os.path.abspath(str(plugin_dir)))
    if not parent:
        return
    while parent in sys.path:
        sys.path.remove(parent)
    sys.path.insert(0, parent)


def _geoai_package_name(plugin: Any) -> str:
    """Return the loaded GeoAI plugin package name."""
    module_name = getattr(plugin.__class__, "__module__", "") or ""
    package_name = module_name.split(".", 1)[0]
    return package_name or "geoai"


def _ensure_geoai_dependencies(plugin: Any) -> tuple[bool, str | None]:
    """Ask the GeoAI plugin to prepare its managed dependencies."""
    ensure = getattr(plugin, "_ensure_dependencies", None)
    if callable(ensure):
        try:
            ready = bool(_on_gui(lambda: ensure("samgeo")))
        except Exception as exc:
            return False, f"GeoAI dependency check failed: {exc}"
        if not ready:
            return (
                False,
                "GeoAI SamGeo dependencies are not ready. The GeoAI plugin "
                "opened its dependency installer; finish that installation and "
                "retry the segmentation request.",
            )
    return True, None


def _import_samgeo_client(plugin: Any) -> Any:
    """Import GeoAI plugin's subprocess-backed SamGeo client lazily."""
    _add_plugin_parent_to_path(plugin)
    package_name = _geoai_package_name(plugin)
    module = importlib.import_module(f"{package_name}.core.samgeo_subprocess")
    return module.SamGeoSubprocessClient


def _run_geoai_task(plugin: Any, action: str, params: dict[str, Any]) -> Any:
    """Run a GeoAI plugin task in its managed venv subprocess."""
    _add_plugin_parent_to_path(plugin)
    package_name = _geoai_package_name(plugin)
    module = importlib.import_module(f"{package_name}.core.geoai_task_subprocess")
    return module.run_geoai_task(action, params)


def _get_samgeo_client(plugin: Any) -> Any:
    """Return a cached SamGeo subprocess client from the GeoAI plugin runtime."""
    dock = getattr(plugin, "_samgeo_dock", None)
    existing = getattr(dock, "sam", None)
    if existing is not None:
        return existing

    cached = getattr(plugin, "_geoagent_samgeo_client", None)
    if cached is not None:
        return cached

    client_cls = _import_samgeo_client(plugin)
    backend = "transformers" if sys.platform == "darwin" else "meta"
    client = client_cls(
        model_version="SamGeo3 (SAM3)",
        backend=backend,
        device=None,
        confidence=0.5,
        enable_interactive=True,
    )
    client.initialize()
    setattr(plugin, "_geoagent_samgeo_client", client)
    return client


def _is_raster_layer(layer: Any) -> bool:
    """Return whether ``layer`` looks like a QGIS raster layer."""
    if layer is None:
        return False
    try:
        layer_type = layer.type()
        if isinstance(layer_type, str):
            return layer_type.lower() == "raster"
        try:
            from qgis.core import QgsMapLayer  # type: ignore[import-not-found]

            return layer_type == QgsMapLayer.RasterLayer
        except Exception:
            return int(layer_type) == 1
    except Exception:
        return False


def _layer_name(layer: Any) -> str:
    """Return a display name for a QGIS layer-like object."""
    try:
        name = layer.name()
        if name:
            return str(name)
    except Exception:
        pass
    return "raster"


def _layer_source(layer: Any) -> str:
    """Return the source URI for a QGIS layer-like object."""
    try:
        return str(layer.source() or "")
    except Exception:
        return ""


def _resolve_raster_source(
    iface: Any,
    project_getter: Callable[[], Any],
    input_layer_name: str = "",
    image_path: str = "",
) -> dict[str, Any]:
    """Resolve explicit path, named raster layer, or active raster layer."""
    path = str(image_path or "").strip()
    if path:
        return {"source": path, "source_type": "image_path", "layer_name": ""}

    layer_name = str(input_layer_name or "").strip()

    def _run() -> dict[str, Any]:
        if layer_name:
            project = project_getter()
            layers = []
            if project is not None:
                try:
                    layers = list(project.mapLayersByName(layer_name))
                except Exception:
                    layers = []
            for layer in layers:
                if _is_raster_layer(layer):
                    source = _layer_source(layer)
                    if source:
                        return {
                            "source": source,
                            "source_type": "qgis_layer",
                            "layer_name": _layer_name(layer),
                        }
            raise ValueError(f"No raster layer named {layer_name!r} was found.")

        active = None
        try:
            active = iface.activeLayer()
        except Exception:
            active = None
        if active is None:
            raise ValueError(
                "No image path, input layer, or active raster layer found."
            )
        if not _is_raster_layer(active):
            raise ValueError(f"Active layer {_layer_name(active)!r} is not a raster.")
        source = _layer_source(active)
        if not source:
            raise ValueError(f"Raster layer {_layer_name(active)!r} has no source URI.")
        return {
            "source": source,
            "source_type": "active_qgis_layer",
            "layer_name": _layer_name(active),
        }

    return _on_gui(_run)


def _parse_bands(bands: str | list[int] | tuple[int, ...] | None) -> list[int] | None:
    """Parse optional RGB band selection for SamGeo ``set_image``."""
    if bands is None or bands == "":
        return None
    if isinstance(bands, str):
        parts = [part.strip() for part in bands.replace(";", ",").split(",")]
        parsed = [int(part) for part in parts if part]
    else:
        parsed = [int(part) for part in bands]
    if len(parsed) != 3:
        raise ValueError(
            "bands must contain exactly three band numbers: red,green,blue"
        )
    if len(set(parsed)) != 3:
        raise ValueError("bands must contain three different band numbers")
    return parsed


def _mask_count(client: Any) -> int:
    """Return SamGeo mask count from the client result state."""
    masks = getattr(client, "masks", None)
    if masks is None:
        return 0
    try:
        return int(len(masks))
    except Exception:
        try:
            return int(getattr(masks, "count", 0) or 0)
        except Exception:
            return 0


def _add_raster_to_qgis(
    iface: Any,
    project_getter: Callable[[], Any],
    output_path: str,
    layer_name: str,
) -> dict[str, Any]:
    """Add a raster mask to QGIS on the GUI thread."""

    def _run() -> dict[str, Any]:
        add_raster = getattr(iface, "addRasterLayer", None)
        if callable(add_raster):
            layer = add_raster(output_path, layer_name)
            if layer is None or not getattr(layer, "isValid", lambda: True)():
                return {"success": False, "error": f"Failed to load {output_path}."}
        else:
            try:
                from qgis.core import QgsRasterLayer  # type: ignore[import-not-found]
            except Exception as exc:
                return {
                    "success": False,
                    "error": f"QGIS raster API unavailable: {exc}",
                }
            layer = QgsRasterLayer(output_path, layer_name)
            if not layer.isValid():
                return {"success": False, "error": f"Failed to load {output_path}."}
            project = project_getter()
            if project is not None:
                project.addMapLayer(layer)
        try:
            iface.mapCanvas().refresh()
        except Exception:
            pass
        return {
            "success": True,
            "layer_name": layer_name,
        }

    return _on_gui(_run)


def _add_vector_to_qgis(
    iface: Any,
    project_getter: Callable[[], Any],
    output_path: str,
    layer_name: str,
) -> dict[str, Any]:
    """Add a vector output to QGIS on the GUI thread."""

    def _run() -> dict[str, Any]:
        add_vector = getattr(iface, "addVectorLayer", None)
        if callable(add_vector):
            layer = add_vector(output_path, layer_name, "ogr")
            if layer is None or not getattr(layer, "isValid", lambda: True)():
                return {"success": False, "error": f"Failed to load {output_path}."}
        else:
            try:
                from qgis.core import QgsVectorLayer  # type: ignore[import-not-found]
            except Exception as exc:
                return {
                    "success": False,
                    "error": f"QGIS vector API unavailable: {exc}",
                }
            layer = QgsVectorLayer(output_path, layer_name, "ogr")
            if not layer.isValid():
                return {"success": False, "error": f"Failed to load {output_path}."}
            project = project_getter()
            if project is not None:
                project.addMapLayer(layer)
        try:
            iface.mapCanvas().refresh()
        except Exception:
            pass
        return {"success": True, "layer_name": layer_name}

    return _on_gui(_run)


def _vector_format_from_path(output_path: str, fallback: str = "geojson") -> str:
    """Return GeoAI worker vector format value for an output path."""
    lower = output_path.lower()
    if lower.endswith(".gpkg"):
        return "gpkg"
    if lower.endswith(".shp"):
        return "shapefile"
    if lower.endswith(".geojson") or lower.endswith(".json"):
        return "geojson"
    return fallback


def _normalize_output_format(value: str, output_path: str = "") -> tuple[str, str]:
    """Return ``(kind, vector_format)`` for a requested output format."""
    fmt = str(value or "").strip().lower()
    if not fmt and output_path:
        fmt = os.path.splitext(output_path)[1].lstrip(".").lower()
    if fmt in {"", "raster", "tif", "tiff", "geotiff"}:
        return "raster", ""
    if fmt == "vector":
        return "vector", "gpkg"
    if fmt in {"geojson", "json"}:
        return "vector", "geojson"
    if fmt in {"gpkg", "geopackage"}:
        return "vector", "gpkg"
    if fmt in {"shp", "shapefile"}:
        return "vector", "shapefile"
    raise ValueError(
        "output_format must be raster, geojson, gpkg, shapefile, or vector"
    )


def _default_output_path(kind: str, vector_format: str = "") -> str:
    """Create a temporary output path for a raster or vector result."""
    if kind == "raster":
        handle = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        path = handle.name
        handle.close()
        return path
    if vector_format == "gpkg":
        handle = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
        path = handle.name
        handle.close()
        return path
    if vector_format == "shapefile":
        return os.path.join(tempfile.mkdtemp(), "masks.shp")
    handle = tempfile.NamedTemporaryFile(suffix=".geojson", delete=False)
    path = handle.name
    handle.close()
    return path


def _vectorize_mask(
    plugin: Any,
    mask_path: str,
    output_path: str,
    *,
    mode: str = "simple",
    output_format: str = "geojson",
    min_area: float = 0,
    epsilon: float = 2.0,
    smooth_iterations: int = 3,
    simplify_tolerance: float = 0,
) -> dict[str, Any]:
    """Convert a raster mask to vector using GeoAI plugin task helpers."""
    cleaned_mode = str(mode or "simple").strip().lower()
    params: dict[str, Any] = {
        "mask_path": mask_path,
        "output_path": output_path,
        "output_format": output_format,
    }
    min_area_value = float(min_area or 0)
    simplify_value = float(simplify_tolerance or 0)
    if cleaned_mode in {"simple", "plain", "vector"}:
        params["min_area"] = min_area_value
        if simplify_value > 0:
            params["simplify_tolerance"] = simplify_value
        result = _run_geoai_task(plugin, "raster_to_vector", params)
    elif cleaned_mode in {"regularize", "regularized", "orthogonalize", "buildings"}:
        params["epsilon"] = float(epsilon)
        params["min_area"] = min_area_value if min_area_value > 0 else None
        result = _run_geoai_task(plugin, "vectorize_mask", params)
    elif cleaned_mode in {"smooth", "smoothed", "natural"}:
        params["smooth_iterations"] = int(smooth_iterations)
        params["min_area"] = min_area_value if min_area_value > 0 else None
        if simplify_value > 0:
            params["simplify_tolerance"] = simplify_value
        result = _run_geoai_task(plugin, "smooth_vector", params)
    else:
        raise ValueError("vector_mode must be simple, regularize, or smooth")
    return dict(result or {}, mode=cleaned_mode, output_path=output_path)


def _postprocess_mask_to_vector_result(
    *,
    plugin: Any,
    iface: Any,
    project_getter: Callable[[], Any],
    mask_path: str,
    output_path: str,
    output_format: str,
    vector_mode: str,
    min_area: float,
    epsilon: float,
    smooth_iterations: int,
    simplify_tolerance: float,
    add_to_qgis: bool,
    output_layer_name: str,
) -> dict[str, Any]:
    """Run vector post-processing and optionally add the result to QGIS."""
    _, vector_format = _normalize_output_format(output_format, output_path)
    if not vector_format:
        vector_format = _vector_format_from_path(output_path)
    vector_result = _vectorize_mask(
        plugin,
        mask_path,
        output_path,
        mode=vector_mode,
        output_format=vector_format,
        min_area=min_area,
        epsilon=epsilon,
        smooth_iterations=smooth_iterations,
        simplify_tolerance=simplify_tolerance,
    )
    layer_name = str(output_layer_name or "").strip()
    if not layer_name:
        layer_name = os.path.basename(output_path) or "samgeo_masks"
    add_result: dict[str, Any] | None = None
    if add_to_qgis:
        add_result = _add_vector_to_qgis(iface, project_getter, output_path, layer_name)
    return {
        "output_path": output_path,
        "output_layer_name": layer_name if add_to_qgis else "",
        "added_to_qgis": bool(add_result and add_result.get("success")),
        "qgis_add_result": add_result,
        "vector_result": vector_result,
    }


def geoai_tools(
    iface: Any,
    project: Any | None = None,
    plugin: Any | None = None,
) -> list[Any]:
    """Build GeoAI QGIS plugin tools bound to ``iface``."""
    if iface is None:
        return []

    get_project = _project_getter(project)
    recent_segmentations: dict[
        tuple[str, str, str, str, str, str, str], dict[str, Any]
    ] = {}

    @geo_tool(
        category="geoai",
        requires_confirmation=True,
        long_running=True,
    )
    def segment_image_with_text_prompt(
        prompt: str,
        input_layer_name: str = "",
        image_path: str = "",
        output_path: str = "",
        output_format: str = "raster",
        vector_mode: str = "simple",
        min_size: int = 0,
        max_size: int = 0,
        min_area: float = 0,
        epsilon: float = 2.0,
        smooth_iterations: int = 3,
        simplify_tolerance: float = 0,
        bands: str = "",
        unique: bool = False,
        add_to_qgis: bool = True,
        output_layer_name: str = "",
    ) -> dict[str, Any]:
        """Segment a raster image using GeoAI SamGeo text-prompt segmentation.

        Provide either a QGIS raster layer name, an image file path, or neither
        to use the active raster layer. Set output_format to geojson, gpkg, or
        shapefile to save a vector result instead of a raster mask.
        """
        text_prompt = str(prompt or "").strip()
        if not text_prompt:
            return {
                "success": False,
                "prompt": "",
                "error": "prompt is required.",
            }

        resolved_plugin = _resolve_geoai_plugin(plugin)
        if resolved_plugin is None:
            return {
                "success": False,
                "prompt": text_prompt,
                "error": (
                    "GeoAI QGIS plugin is not loaded. Install and enable the "
                    "GeoAI plugin, then retry."
                ),
            }

        ready, message = _ensure_geoai_dependencies(resolved_plugin)
        if not ready:
            return {"success": False, "prompt": text_prompt, "error": message}

        try:
            source_info = _resolve_raster_source(
                iface,
                get_project,
                input_layer_name=input_layer_name,
                image_path=image_path,
            )
            rgb_bands = _parse_bands(bands)

            requested_output_path = str(output_path or "").strip()
            out_path = requested_output_path
            output_kind, vector_format = _normalize_output_format(
                output_format, out_path
            )
            if output_kind == "vector" and out_path:
                vector_format = _vector_format_from_path(out_path, vector_format)
            if not out_path:
                out_path = _default_output_path(output_kind, vector_format)

            layer_name = str(output_layer_name or "").strip()
            if not layer_name:
                layer_name = os.path.basename(out_path) or "samgeo_masks.tif"
            cache_key = (
                text_prompt.lower(),
                source_info["source"],
                output_kind,
                vector_format,
                str(vector_mode or "simple").strip().lower(),
                layer_name,
                requested_output_path,
            )
            if cache_key in recent_segmentations:
                cached = dict(recent_segmentations[cache_key])
                cached["skipped_duplicate"] = True
                cached["duplicate_reason"] = (
                    "A matching GeoAI segmentation tool call already succeeded "
                    "during this chat turn."
                )
                return cached

            client = _get_samgeo_client(resolved_plugin)
            client.set_image(source_info["source"], bands=rgb_bands)
            min_size_value = int(min_size or 0)
            max_size_value = int(max_size or 0)
            client.generate_masks(
                text_prompt,
                min_size=min_size_value,
                max_size=max_size_value if max_size_value > 0 else None,
            )
            count = _mask_count(client)
            temp_mask_path = out_path
            if output_kind == "vector":
                handle = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
                temp_mask_path = handle.name
                handle.close()
            client.save_masks(output=temp_mask_path, unique=bool(unique))

            if output_kind == "raster":
                add_result: dict[str, Any] | None = None
                if add_to_qgis:
                    add_result = _add_raster_to_qgis(
                        iface, get_project, out_path, layer_name
                    )
                output_payload = {
                    "output_path": out_path,
                    "output_layer_name": layer_name if add_to_qgis else "",
                    "added_to_qgis": bool(add_result and add_result.get("success")),
                    "qgis_add_result": add_result,
                }
            else:
                try:
                    output_payload = _postprocess_mask_to_vector_result(
                        plugin=resolved_plugin,
                        iface=iface,
                        project_getter=get_project,
                        mask_path=temp_mask_path,
                        output_path=out_path,
                        output_format=vector_format,
                        vector_mode=vector_mode,
                        min_area=min_area,
                        epsilon=epsilon,
                        smooth_iterations=smooth_iterations,
                        simplify_tolerance=simplify_tolerance,
                        add_to_qgis=add_to_qgis,
                        output_layer_name=layer_name,
                    )
                finally:
                    if temp_mask_path != out_path and os.path.exists(temp_mask_path):
                        os.remove(temp_mask_path)

            result = {
                "success": True,
                "prompt": text_prompt,
                "source": source_info["source"],
                "source_type": source_info["source_type"],
                "input_layer_name": source_info.get("layer_name", ""),
                "output_kind": output_kind,
                "output_format": output_format,
                "mask_count": count,
                **output_payload,
            }
            recent_segmentations[cache_key] = dict(result)
            return result
        except Exception as exc:
            return {
                "success": False,
                "prompt": text_prompt,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }

    @geo_tool(
        category="geoai",
        requires_confirmation=True,
        long_running=True,
    )
    def regularize_segmentation_mask_to_vector(
        mask_path: str,
        output_path: str = "",
        output_format: str = "gpkg",
        epsilon: float = 2.0,
        min_area: float = 0,
        add_to_qgis: bool = True,
        output_layer_name: str = "",
    ) -> dict[str, Any]:
        """Convert a raster mask to a regularized vector layer for buildings."""
        resolved_plugin = _resolve_geoai_plugin(plugin)
        if resolved_plugin is None:
            return {
                "success": False,
                "error": "GeoAI QGIS plugin is not loaded.",
            }
        ready, message = _ensure_geoai_dependencies(resolved_plugin)
        if not ready:
            return {"success": False, "error": message}
        try:
            source = str(mask_path or "").strip()
            if not source:
                raise ValueError("mask_path is required.")
            _, vector_format = _normalize_output_format(output_format, output_path)
            out_path = str(output_path or "").strip() or _default_output_path(
                "vector", vector_format
            )
            payload = _postprocess_mask_to_vector_result(
                plugin=resolved_plugin,
                iface=iface,
                project_getter=get_project,
                mask_path=source,
                output_path=out_path,
                output_format=vector_format,
                vector_mode="regularize",
                min_area=min_area,
                epsilon=epsilon,
                smooth_iterations=3,
                simplify_tolerance=0,
                add_to_qgis=add_to_qgis,
                output_layer_name=output_layer_name,
            )
            return {
                "success": True,
                "mask_path": source,
                "output_kind": "vector",
                "output_format": vector_format,
                **payload,
            }
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }

    @geo_tool(
        category="geoai",
        requires_confirmation=True,
        long_running=True,
    )
    def smooth_segmentation_mask_to_vector(
        mask_path: str,
        output_path: str = "",
        output_format: str = "gpkg",
        smooth_iterations: int = 3,
        min_area: float = 0,
        simplify_tolerance: float = 0,
        add_to_qgis: bool = True,
        output_layer_name: str = "",
    ) -> dict[str, Any]:
        """Convert a raster mask to a smoothed vector layer for natural features."""
        resolved_plugin = _resolve_geoai_plugin(plugin)
        if resolved_plugin is None:
            return {
                "success": False,
                "error": "GeoAI QGIS plugin is not loaded.",
            }
        ready, message = _ensure_geoai_dependencies(resolved_plugin)
        if not ready:
            return {"success": False, "error": message}
        try:
            source = str(mask_path or "").strip()
            if not source:
                raise ValueError("mask_path is required.")
            _, vector_format = _normalize_output_format(output_format, output_path)
            out_path = str(output_path or "").strip() or _default_output_path(
                "vector", vector_format
            )
            payload = _postprocess_mask_to_vector_result(
                plugin=resolved_plugin,
                iface=iface,
                project_getter=get_project,
                mask_path=source,
                output_path=out_path,
                output_format=vector_format,
                vector_mode="smooth",
                min_area=min_area,
                epsilon=2.0,
                smooth_iterations=smooth_iterations,
                simplify_tolerance=simplify_tolerance,
                add_to_qgis=add_to_qgis,
                output_layer_name=output_layer_name,
            )
            return {
                "success": True,
                "mask_path": source,
                "output_kind": "vector",
                "output_format": vector_format,
                **payload,
            }
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }

    return [
        segment_image_with_text_prompt,
        regularize_segmentation_mask_to_vector,
        smooth_segmentation_mask_to_vector,
    ]
