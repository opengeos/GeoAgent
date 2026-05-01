"""Tool adapters for QGIS via the ``QgisInterface`` (typically ``iface``).

This module is **import-safe** outside a QGIS Python environment: the
top-level imports do NOT pull in the ``qgis`` package, and the
:func:`qgis_tools` factory returns an empty list when no ``iface`` is
provided. Tool bodies that need QGIS classes (``QgsProject``,
``QgsRectangle``, ``QgsVectorLayer``, ``QgsRasterLayer``) import them
lazily so the module can still be imported in CI without QGIS.

Typical usage from inside a QGIS plugin's Python console::

    from qgis.utils import iface
    from geoagent import for_qgis
    agent = for_qgis(iface)
    agent.chat("Zoom to the active layer.")
"""

from __future__ import annotations

import ast
import contextlib
import io
import os
import tempfile
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from geoagent.core.decorators import geo_tool
from geoagent.tools._qt_marshal import run_on_qt_gui_thread

_PYQGIS_ALLOWED_TOP_LEVEL = frozenset({"qgis", "math"})
_PYQGIS_BLOCKED_CALLS = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "input",
    "open",
}


def _is_allowed_pyqgis_module(name: str) -> bool:
    """Return True only for the qgis/math packages and their submodules.

    Matching on the top-level package (split on ``.``) avoids prefix
    collisions with unrelated modules such as ``mathutils`` or ``qgisx``.
    """
    if not name:
        return False
    top, _, _ = name.partition(".")
    return top in _PYQGIS_ALLOWED_TOP_LEVEL


def _validate_pyqgis_script(code: str) -> None:
    """Reject generated code that is outside the intended PyQGIS scope."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Invalid Python syntax: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _is_allowed_pyqgis_module(alias.name):
                    raise ValueError(
                        "Only qgis/PyQt and math imports are allowed in "
                        "run_pyqgis_script."
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level or not _is_allowed_pyqgis_module(module):
                raise ValueError(
                    "Only absolute qgis/PyQt and math imports are allowed in "
                    "run_pyqgis_script."
                )
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _PYQGIS_BLOCKED_CALLS:
                raise ValueError(
                    f"Calling {node.func.id!r} is not allowed in run_pyqgis_script."
                )
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Dunder attribute access is not allowed.")
        elif isinstance(node, ast.Name) and node.id in {
            "__builtins__",
            "__dict__",
            "__globals__",
            "__subclasses__",
        }:
            raise ValueError(f"Access to {node.id!r} is not allowed.")


def _limited_pyqgis_import(
    name: str,
    globals: dict[str, Any] | None = None,
    locals: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    """Allow imports needed for PyQGIS snippets and reject everything else."""
    if level or not _is_allowed_pyqgis_module(name):
        raise ImportError(
            f"Import {name!r} is not allowed in run_pyqgis_script. "
            "Use QGIS/PyQt API imports only."
        )
    return __import__(name, globals, locals, fromlist, level)


def _pyqgis_builtins() -> dict[str, Any]:
    """Return a small builtins namespace for generated PyQGIS snippets."""
    return {
        "__import__": _limited_pyqgis_import,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "print": print,
        "range": range,
        "round": round,
        "set": set,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }


def _qcolor(value: Any) -> Any:
    """Return a QColor-like object for real QGIS, or the raw value in tests."""
    if value is None:
        return None
    if not str(value).strip():
        return None
    try:
        from qgis.PyQt.QtGui import QColor  # type: ignore[import-not-found]
    except Exception:
        return value

    if isinstance(value, (list, tuple)):
        channels = [int(float(channel)) for channel in value]
        color = QColor(*channels)
    else:
        text = str(value).strip()
        if "," in text and not text.startswith("#"):
            channels = [int(float(part.strip())) for part in text.split(",")]
            color = QColor(*channels)
        else:
            color = QColor(text)
    if hasattr(color, "isValid") and not color.isValid():
        raise ValueError(f"Invalid color value: {value!r}")
    return color


def _set_symbol_layer_attr(
    symbol_layer: Any, names: tuple[str, ...], value: Any
) -> bool:
    """Set the first available symbol-layer setter."""
    for name in names:
        method = getattr(symbol_layer, name, None)
        if callable(method):
            method(value)
            return True
    return False


def _set_symbol_width(symbol: Any, width: float) -> bool:
    """Set line/stroke width on a QGIS symbol when supported."""
    for obj in (symbol, getattr(symbol, "symbolLayer", lambda *_: None)(0)):
        if obj is None:
            continue
        if _set_symbol_layer_attr(obj, ("setWidth", "setStrokeWidth"), width):
            return True
    return False


def _set_symbol_outline_color(symbol: Any, color: Any) -> bool:
    """Set an outline/stroke color when the symbol layer supports it."""
    symbol_layer = getattr(symbol, "symbolLayer", lambda *_: None)(0)
    if symbol_layer is None:
        return False
    return _set_symbol_layer_attr(
        symbol_layer,
        ("setStrokeColor", "setBorderColor", "setOutlineColor"),
        color,
    )


_ACTIVE_QGIS_HILLSHADE_TASKS: list[Any] = []


def _is_raster_layer(layer: Any) -> bool:
    """Return True when a layer looks like a QGIS raster layer."""
    try:
        from qgis.core import QgsMapLayer  # type: ignore[import-not-found]

        if hasattr(layer, "type") and layer.type() == QgsMapLayer.RasterLayer:
            return True
    except Exception:
        pass
    try:
        layer_type = str(layer.type()).lower() if hasattr(layer, "type") else ""
        if layer_type == "raster" or "raster" in layer_type:
            return True
    except Exception:
        pass
    return hasattr(layer, "bandCount") and callable(getattr(layer, "bandCount"))


def _is_remote_raster_source(layer: Any) -> bool:
    """Return True when a raster source is remote and stats reads may block."""
    try:
        source = str(layer.source() if hasattr(layer, "source") else "")
    except Exception:
        source = ""
    source = source.strip().lower()
    return source.startswith(
        ("http://", "https://", "/vsicurl/http://", "/vsicurl/https://")
    )


def _raster_value_range(
    layer: Any,
    min_value: float | None = None,
    max_value: float | None = None,
) -> tuple[float, float, bool]:
    """Return a practical raster value range for color-ramp rendering."""
    if min_value is not None and max_value is not None and max_value > min_value:
        return float(min_value), float(max_value), False
    if _is_remote_raster_source(layer):
        # Avoid provider.bandStatistics() on remote COGs; QGIS/GDAL can block the
        # GUI while reading over HTTP. These defaults work well for most DEMs.
        return (
            float(min_value if min_value is not None else 0.0),
            float(max_value if max_value is not None else 3000.0),
            True,
        )
    try:
        provider = layer.dataProvider()
        stats = provider.bandStatistics(1)
        low = float(getattr(stats, "minimumValue"))
        high = float(getattr(stats, "maximumValue"))
        if high > low:
            return low, high, False
    except Exception:
        pass
    for low_name, high_name in (
        ("minimumValue", "maximumValue"),
        ("min", "max"),
    ):
        try:
            low_method = getattr(layer, low_name)
            high_method = getattr(layer, high_name)
            low = float(low_method(1) if callable(low_method) else low_method)
            high = float(high_method(1) if callable(high_method) else high_method)
            if high > low:
                return low, high, False
        except Exception:
            continue
    return (
        float(min_value if min_value is not None else 0.0),
        float(max_value if max_value is not None else 3000.0),
        True,
    )


def _palette_colors(
    name: str | None, fallback_color: Any = None
) -> list[tuple[float, Any, str]]:
    """Return normalized color stops for common raster palettes."""
    palette = str(name or "").strip().lower()
    if palette in {"terrain", "dem", "elevation", "earth", ""}:
        return [
            (0.0, _qcolor("#1a9850"), "low"),
            (0.35, _qcolor("#91cf60"), "lower"),
            (0.55, _qcolor("#fee08b"), "mid"),
            (0.75, _qcolor("#d08b39"), "high"),
            (1.0, _qcolor("#f5f5f5"), "highest"),
        ]
    if palette in {"viridis"}:
        return [
            (0.0, _qcolor("#440154"), "low"),
            (0.33, _qcolor("#31688e"), "mid-low"),
            (0.66, _qcolor("#35b779"), "mid-high"),
            (1.0, _qcolor("#fde725"), "high"),
        ]
    if palette in {"grayscale", "grey", "gray"}:
        return [
            (0.0, _qcolor("#000000"), "low"),
            (1.0, _qcolor("#ffffff"), "high"),
        ]
    color = fallback_color or _qcolor("#8c510a")
    return [(0.0, _qcolor("#ffffff"), "low"), (1.0, color, "high")]


def _apply_raster_symbology(
    layer: Any,
    *,
    raster_palette: str | None = None,
    color: Any = None,
    min_value: float | None = None,
    max_value: float | None = None,
) -> dict[str, Any]:
    """Apply a QGIS single-band pseudocolor renderer when available."""
    applied: dict[str, Any] = {}
    if not _is_raster_layer(layer):
        return applied

    low, high, estimated = _raster_value_range(layer, min_value, max_value)
    palette = raster_palette or "terrain"
    stops = _palette_colors(palette, color)

    try:
        from qgis.core import (  # type: ignore[import-not-found]
            QgsColorRampShader,
            QgsRasterShader,
            QgsSingleBandPseudoColorRenderer,
        )

        shader = QgsRasterShader()
        color_ramp = QgsColorRampShader()
        ramp_type = getattr(QgsColorRampShader, "Interpolated", None)
        if ramp_type is None:
            ramp_type = getattr(
                getattr(QgsColorRampShader, "Type", object), "Interpolated"
            )
        color_ramp.setColorRampType(ramp_type)
        color_ramp.setColorRampItemList(
            [
                QgsColorRampShader.ColorRampItem(
                    low + ratio * (high - low),
                    qcolor,
                    label,
                )
                for ratio, qcolor, label in stops
            ]
        )
        shader.setRasterShaderFunction(color_ramp)
        provider = layer.dataProvider()
        renderer = QgsSingleBandPseudoColorRenderer(provider, 1, shader)
        layer.setRenderer(renderer)
        applied.update(
            {
                "raster_palette": palette,
                "raster_band": 1,
                "raster_min": low,
                "raster_max": high,
                "raster_range_estimated": estimated,
            }
        )
        return applied
    except Exception:
        style = getattr(layer, "symbology", None)
        if not isinstance(style, dict):
            style = {}
            try:
                layer.symbology = style
            except Exception:
                pass
        style.update(
            {
                "raster_palette": palette,
                "raster_band": 1,
                "raster_min": low,
                "raster_max": high,
                "raster_range_estimated": estimated,
            }
        )
        applied.update(style)
        return applied


def _transform_extent_to_canvas_crs(layer: Any, canvas: Any, extent: Any) -> Any:
    """Re-project a layer's extent into the canvas / project CRS.

    ``QgsMapLayer.extent()`` returns the extent in the layer's native
    CRS. ``QgsMapCanvas.setExtent()`` expects coordinates in the
    canvas's destination CRS. When a layer is loaded as EPSG:4326
    (typical for GeoJSON) but the project canvas is EPSG:3857 (Web
    Mercator), passing the layer's lat/lon extent directly to
    ``setExtent`` zooms the canvas to a sliver near (0, 0) in
    Mercator, which renders as a blank view. Transforming the bbox
    through ``QgsCoordinateTransform`` first puts it in the right
    CRS so the canvas zooms to the actual layer.

    The helper degrades gracefully when called with mocks (no
    ``crs()`` / ``mapSettings()`` methods) or outside QGIS (no
    ``QgsCoordinateTransform`` import) — in those cases it returns
    the original extent so existing tests keep passing.

    Args:
        layer: The source layer whose ``extent`` was just fetched.
        canvas: The map canvas the extent will be applied to.
        extent: The layer-CRS extent (a ``QgsRectangle`` in real QGIS;
            anything else from a mock).

    Returns:
        The extent re-projected into the canvas's destination CRS,
        or the original extent when the transform cannot be set up.
    """
    if not (hasattr(layer, "crs") and hasattr(canvas, "mapSettings")):
        return extent
    try:
        from qgis.core import (  # type: ignore[import-not-found]
            QgsCoordinateTransform,
            QgsProject,
        )
    except ImportError:
        return extent
    try:
        src_crs = layer.crs()
        dst_crs = canvas.mapSettings().destinationCrs()
    except Exception:
        return extent
    if src_crs is None or dst_crs is None:
        return extent
    # ``QgsCoordinateReferenceSystem`` defines ``__eq__`` so this works
    # for both authority-id and proj-string-defined CRSes.
    try:
        if src_crs == dst_crs:
            return extent
    except Exception:
        pass
    try:
        transform = QgsCoordinateTransform(src_crs, dst_crs, QgsProject.instance())
        return transform.transformBoundingBox(extent)
    except Exception:
        return extent


def _transform_bbox_to_canvas_crs(
    canvas: Any,
    west: float,
    south: float,
    east: float,
    north: float,
    src_crs: str,
) -> Any:
    """Re-project a [west, south, east, north] bbox into the canvas CRS.

    LLMs naturally produce place-name extents in lat/lon (EPSG:4326)
    even when the project canvas is Web Mercator (EPSG:3857). Without a
    transform, ``canvas.setExtent`` interprets the lat/lon coordinates
    as the canvas's metres, zooming to a sliver near (0, 0) and
    rendering blank.

    Returns a ``QgsRectangle`` in the canvas CRS when the transform
    succeeds, falls back to a ``QgsRectangle`` (or 4-tuple, outside
    QGIS) in the source CRS when any step is unavailable.

    Args:
        canvas: The map canvas the bbox will be applied to.
        west: Western coordinate in ``src_crs``.
        south: Southern coordinate in ``src_crs``.
        east: Eastern coordinate in ``src_crs``.
        north: Northern coordinate in ``src_crs``.
        src_crs: Authority ID of the bbox's CRS, e.g. ``"EPSG:4326"``.

    Returns:
        A ``QgsRectangle`` in the canvas's destination CRS when both
        QGIS and the canvas are available; otherwise a ``QgsRectangle``
        (or 4-tuple if QGIS is missing entirely) in the original CRS.
    """
    try:
        from qgis.core import (  # type: ignore[import-not-found]
            QgsCoordinateReferenceSystem,
            QgsCoordinateTransform,
            QgsProject,
            QgsRectangle,
        )
    except ImportError:
        return (west, south, east, north)

    rect = QgsRectangle(west, south, east, north)
    if not hasattr(canvas, "mapSettings"):
        return rect
    try:
        src = QgsCoordinateReferenceSystem(src_crs)
        dst = canvas.mapSettings().destinationCrs()
    except Exception:
        return rect
    if dst is None:
        return rect
    try:
        if src == dst:
            return rect
    except Exception:
        pass
    try:
        transform = QgsCoordinateTransform(src, dst, QgsProject.instance())
        return transform.transformBoundingBox(rect)
    except Exception:
        return rect


def _transform_point_to_canvas_crs(
    canvas: Any,
    lon: float,
    lat: float,
    src_crs: str,
) -> Any:
    """Re-project a WGS84-like point into the canvas CRS when QGIS exists."""
    try:
        from qgis.core import (  # type: ignore[import-not-found]
            QgsCoordinateReferenceSystem,
            QgsCoordinateTransform,
            QgsPointXY,
            QgsProject,
        )
    except ImportError:
        return (lon, lat)

    point = QgsPointXY(lon, lat)
    if not hasattr(canvas, "mapSettings"):
        return point
    try:
        src = QgsCoordinateReferenceSystem(src_crs)
        dst = canvas.mapSettings().destinationCrs()
    except Exception:
        return point
    if dst is None:
        return point
    try:
        if src == dst:
            return point
    except Exception:
        pass
    try:
        transform = QgsCoordinateTransform(src, dst, QgsProject.instance())
        return transform.transform(point)
    except Exception:
        return point


def _resolve_layer(project: Any, layer_name: str) -> Any:
    """Resolve a layer by name from a project.

    Args:
        project: A ``QgsProject`` (or :class:`MockQGISProject`) instance.
        layer_name: Layer name to look up.

    Returns:
        The first matching layer.

    Raises:
        LookupError: If no layer with that name exists in the project.
    """
    layers = project.mapLayersByName(layer_name)
    if not layers:
        raise LookupError(f"No layer named {layer_name!r} in the project.")
    return layers[0]


def _safe_stem(value: str, fallback: str = "qgis_output") -> str:
    """Return a filesystem-safe output stem."""
    import re

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned or fallback


def _project_output_dir(project: Any) -> str:
    """Choose a stable writable directory for generated QGIS outputs."""
    for attr in ("homePath", "absolutePath"):
        method = getattr(project, attr, None)
        if callable(method):
            try:
                value = str(method()).strip()
                if value:
                    return value
            except Exception:
                pass
    file_name = getattr(project, "fileName", None)
    if callable(file_name):
        try:
            value = str(file_name()).strip()
            if value:
                return str(Path(value).parent)
        except Exception:
            pass
    out_dir = Path(tempfile.gettempdir()) / "geoagent_qgis"
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir)


def _default_processing_output_path(project: Any, layer_name: str, suffix: str) -> str:
    """Create a non-conflicting path for a generated Processing output."""
    out_dir = Path(_project_output_dir(project))
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate = out_dir / f"{_safe_stem(layer_name)}{suffix}"
    index = 2
    while candidate.exists():
        candidate = out_dir / f"{_safe_stem(layer_name)}_{index}{suffix}"
        index += 1
    return str(candidate)


def _extent_payload(extent: Any) -> Any:
    """Return a JSON-friendly representation of a QGIS extent."""
    if extent is None:
        return None
    if isinstance(extent, (list, tuple)) and len(extent) == 4:
        return [float(v) for v in extent]
    accessors = ("xMinimum", "yMinimum", "xMaximum", "yMaximum")
    if all(hasattr(extent, name) for name in accessors):
        try:
            return [float(getattr(extent, name)()) for name in accessors]
        except Exception:
            pass
    return str(extent)


def _crs_payload(obj: Any) -> Optional[str]:
    """Return a readable CRS identifier for a QGIS object."""
    if obj is None or not hasattr(obj, "crs"):
        return None
    try:
        crs = obj.crs()
    except Exception:
        return None
    if crs is None:
        return None
    for attr in ("authid", "description"):
        value = getattr(crs, attr, None)
        if callable(value):
            try:
                result = value()
                if result:
                    return str(result)
            except Exception:
                pass
    return str(crs)


def _call_int(obj: Any, method_name: str) -> Optional[int]:
    """Call a no-argument method and coerce its result to an integer."""
    method = getattr(obj, method_name, None)
    if not callable(method):
        return None
    try:
        return int(method())
    except Exception:
        return None


def _layer_visibility(project: Any, layer: Any) -> Optional[bool]:
    """Return the layer tree visibility state for a layer."""
    try:
        root = project.layerTreeRoot()
        tree_layer = root.findLayer(layer.id())
        return bool(tree_layer.isVisible())
    except Exception:
        pass
    value = getattr(layer, "visible", None)
    if value is not None:
        try:
            return bool(value() if callable(value) else value)
        except Exception:
            return None
    return None


def _layer_opacity(layer: Any) -> Optional[float]:
    """Return layer opacity when the object exposes it."""
    value = getattr(layer, "opacity", None)
    if callable(value):
        try:
            return float(value())
        except Exception:
            return None
    if value is not None:
        try:
            return float(value)
        except Exception:
            return None
    return None


def _layer_metadata(layer: Any, project: Any | None = None) -> dict[str, Any]:
    """Collect serializable metadata for a QGIS layer."""
    record: dict[str, Any] = {
        "id": str(layer.id()) if hasattr(layer, "id") else None,
        "name": layer.name() if hasattr(layer, "name") else str(layer),
        "type": str(layer.type()) if hasattr(layer, "type") else type(layer).__name__,
        "source": layer.source() if hasattr(layer, "source") else None,
        "crs": _crs_payload(layer),
        "extent": _extent_payload(layer.extent() if hasattr(layer, "extent") else None),
        "feature_count": _call_int(layer, "featureCount"),
        "selected_count": _call_int(layer, "selectedFeatureCount"),
        "geometry_type": (
            str(layer.geometryType()) if hasattr(layer, "geometryType") else None
        ),
        "opacity": _layer_opacity(layer),
    }
    if project is not None:
        record["visible"] = _layer_visibility(project, layer)
    return {k: v for k, v in record.items() if v is not None}


def qgis_tools(iface: Any, project: Optional[Any] = None) -> list[Any]:
    """Build the QGIS tool set bound to a live ``QgisInterface``.

    Args:
        iface: The QGIS ``QgisInterface`` (``qgis.utils.iface``) or a mock.
            Passing ``None`` returns an empty list, so callers may safely do
            ``qgis_tools(getattr(some_ctx, 'qgis_iface', None))`` outside
            QGIS.
        project: Optional ``QgsProject`` instance. If omitted, the tools fall
            back to ``QgsProject.instance()``.

    Returns:
        A list of Strands tool objects. Empty when ``iface`` is
        ``None``.
    """
    if iface is None:
        return []

    def _on_gui(func: Any) -> Any:
        """Run a callable on the Qt GUI thread."""
        return run_on_qt_gui_thread(func)

    def _project() -> Any:
        """Return the configured project or resolve it from the QGIS iface."""
        if project is not None:
            return project
        if hasattr(iface, "project"):
            try:
                proj = iface.project()
                if proj is not None:
                    return proj
            except Exception:
                pass
        try:
            from qgis.core import QgsProject  # type: ignore[import-not-found]

            return QgsProject.instance()
        except Exception as exc:  # pragma: no cover - QGIS-only path
            raise RuntimeError(
                "QGIS is not available; cannot resolve QgsProject."
            ) from exc

    @geo_tool(
        category="qgis",
    )
    def list_project_layers() -> list[dict[str, Any]]:
        """List all layers in the active QGIS project.

        Returns:
            A list of layer metadata dictionaries including name, type,
            source, CRS, extent, feature counts, visibility, and opacity
            when QGIS exposes those values.
        """

        def _run() -> list[dict[str, Any]]:
            """Run the worker body."""
            proj = _project()
            return [_layer_metadata(layer, proj) for layer in proj.mapLayers().values()]

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def get_active_layer() -> dict[str, Any]:
        """Return metadata for the currently active layer.

        Returns:
            A dict with ``name``, ``type``, and ``source`` keys, or a single
            ``{"active_layer": None}`` if no layer is active.
        """

        def _run() -> dict[str, Any]:
            """Run the worker body."""
            layer = iface.activeLayer()
            if layer is None:
                return {"active_layer": None}
            return {
                **_layer_metadata(layer, _project()),
            }

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def get_project_state() -> dict[str, Any]:
        """Return the QGIS project layer list and canvas camera state."""

        def _run() -> dict[str, Any]:
            """Run the worker body."""
            proj = _project()
            canvas = iface.mapCanvas()
            active = iface.activeLayer()
            destination_crs = None
            try:
                destination_crs = canvas.mapSettings().destinationCrs().authid()
            except Exception:
                pass
            return {
                "layers": [
                    _layer_metadata(layer, proj) for layer in proj.mapLayers().values()
                ],
                "active_layer": (
                    _layer_metadata(active, proj) if active is not None else None
                ),
                "canvas": {
                    "extent": _extent_payload(
                        canvas.extent() if hasattr(canvas, "extent") else None
                    ),
                    "scale": canvas.scale() if hasattr(canvas, "scale") else None,
                    "destination_crs": destination_crs,
                },
            }

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def zoom_in() -> str:
        """Zoom the QGIS map canvas in by one step.

        Returns:
            A status string.
        """

        def _run() -> str:
            """Run the worker body."""
            iface.mapCanvas().zoomIn()
            return "Zoomed in."

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def zoom_out() -> str:
        """Zoom the QGIS map canvas out by one step.

        Returns:
            A status string.
        """

        def _run() -> str:
            """Run the worker body."""
            iface.mapCanvas().zoomOut()
            return "Zoomed out."

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def zoom_to_layer(layer_name: str) -> str:
        """Zoom the canvas to the extent of a named layer.

        Args:
            layer_name: Display name of the layer.

        Returns:
            A status string.
        """

        def _run() -> str:
            """Run the worker body."""
            layer = _resolve_layer(_project(), layer_name)
            iface.setActiveLayer(layer)
            canvas = iface.mapCanvas()
            # Prefer the explicit ``setExtent`` + ``refresh`` path over
            # ``iface.zoomToActiveLayer()``. The latter updates the extent
            # but does not always trigger XYZ tile providers (Google
            # Satellite, OSM, ESRI) to refetch tiles at the new zoom-pyramid
            # level, leaving basemaps stuck on upscaled lower-resolution
            # tiles. ``setExtent`` + ``refresh`` mirrors the path QGIS uses
            # for user-driven zoom and resolves the tile pyramid correctly.
            # ``iface.zoomToActiveLayer`` stays as a fallback for canvas
            # types where ``setExtent`` is unavailable (e.g. test mocks
            # without the method) — call it AFTER setExtent so the iface
            # path runs only when needed.
            extent = layer.extent() if hasattr(layer, "extent") else None
            if extent is not None and hasattr(canvas, "setExtent"):
                # Re-project from the layer's CRS to the canvas / project
                # CRS before applying the extent. A GeoJSON loaded as
                # EPSG:4326 has a lat/lon extent that ``setExtent`` would
                # otherwise interpret as Web-Mercator metres, zooming to
                # ~(0, 0) and rendering as a blank canvas.
                extent = _transform_extent_to_canvas_crs(layer, canvas, extent)
                canvas.setExtent(extent)
                if hasattr(canvas, "refresh"):
                    canvas.refresh()
            elif hasattr(iface, "zoomToActiveLayer"):
                iface.zoomToActiveLayer()
            return f"Zoomed to layer {layer_name!r}."

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def zoom_to_extent(
        west: float,
        south: float,
        east: float,
        north: float,
        crs: str = "EPSG:4326",
    ) -> str:
        """Zoom the canvas to a geographic extent.

        The bbox is interpreted in ``crs`` (lat/lon by default — the
        natural CRS for resolving place names). The tool re-projects
        into the canvas / project CRS before applying ``setExtent``,
        so passing ``[-122.5, 47.5, -122.2, 47.7]`` for Seattle works
        regardless of whether the project canvas is EPSG:3857 (Web
        Mercator), EPSG:4326, or anything else.

        Args:
            west: Western coordinate (in ``crs``).
            south: Southern coordinate (in ``crs``).
            east: Eastern coordinate (in ``crs``).
            north: Northern coordinate (in ``crs``).
            crs: Authority ID of the bbox's CRS. Defaults to
                ``"EPSG:4326"`` (WGS84 lat/lon). Use ``"EPSG:3857"``
                if you already have Web-Mercator metres, or any other
                authority ID supported by ``QgsCoordinateReferenceSystem``.

        Returns:
            A status string.
        """

        def _run() -> str:
            """Run the worker body."""
            canvas = iface.mapCanvas()
            rect = _transform_bbox_to_canvas_crs(canvas, west, south, east, north, crs)
            canvas.setExtent(rect)
            canvas.refresh()
            return f"Zoomed to extent [{west}, {south}, {east}, {north}] ({crs})."

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def set_center(
        lat: float,
        lon: float,
        scale: Optional[float] = None,
        crs: str = "EPSG:4326",
    ) -> str:
        """Center the QGIS canvas on a coordinate.

        Args:
            lat: Latitude in ``crs``.
            lon: Longitude in ``crs``.
            scale: Optional map scale denominator to apply after centering.
            crs: Coordinate reference system for ``lat``/``lon``.

        Returns:
            A status string.
        """

        def _run() -> str:
            """Run the worker body."""
            canvas = iface.mapCanvas()
            point = _transform_point_to_canvas_crs(canvas, lon, lat, crs)
            if hasattr(canvas, "setCenter"):
                canvas.setCenter(point)
            elif hasattr(canvas, "setExtent") and hasattr(canvas, "extent"):
                west, south, east, north = canvas.extent()
                width = east - west
                height = north - south
                try:
                    x = point.x()
                    y = point.y()
                except Exception:
                    x, y = point
                canvas.setExtent(
                    (
                        x - width / 2,
                        y - height / 2,
                        x + width / 2,
                        y + height / 2,
                    )
                )
            if scale is not None:
                if hasattr(canvas, "zoomScale"):
                    canvas.zoomScale(float(scale))
                elif hasattr(canvas, "setScale"):
                    canvas.setScale(float(scale))
            if hasattr(canvas, "refresh"):
                canvas.refresh()
            return f"Centred canvas on ({lat}, {lon}) ({crs})."

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def set_scale(scale: float) -> str:
        """Set the QGIS canvas map scale."""

        def _run() -> str:
            """Run the worker body."""
            canvas = iface.mapCanvas()
            if hasattr(canvas, "zoomScale"):
                canvas.zoomScale(float(scale))
            elif hasattr(canvas, "setScale"):
                canvas.setScale(float(scale))
            else:
                return "Canvas does not expose a scale setter."
            if hasattr(canvas, "refresh"):
                canvas.refresh()
            return f"Canvas scale set to 1:{scale}."

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def add_vector_layer(
        path_or_uri: str,
        name: str,
        provider: str = "ogr",
    ) -> str:
        """Add a vector layer to the project.

        Args:
            path_or_uri: Path or provider URI for the data source.
            name: Display name.
            provider: QGIS data provider key (default ``"ogr"``).

        Returns:
            A status string.
        """

        def _run() -> str:
            """Run the worker body."""
            layer = iface.addVectorLayer(path_or_uri, name, provider)
            if layer is None or (hasattr(layer, "isValid") and not layer.isValid()):
                return f"Failed to load vector layer from {path_or_uri!r}."
            return f"Added vector layer {name!r}."

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def add_raster_layer(path_or_uri: str, name: str) -> str:
        """Add a raster layer to the project.

        Args:
            path_or_uri: Path or provider URI for the raster.
            name: Display name.

        Returns:
            A status string.
        """

        def _run() -> str:
            """Run the worker body."""
            layer = iface.addRasterLayer(path_or_uri, name)
            if layer is None or (hasattr(layer, "isValid") and not layer.isValid()):
                return f"Failed to load raster layer from {path_or_uri!r}."
            return f"Added raster layer {name!r}."

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def add_xyz_tile_layer(
        url: str,
        name: str,
        zmin: Optional[int] = None,
        zmax: Optional[int] = None,
        attribution: Optional[str] = None,
    ) -> str:
        """Add an XYZ tile service as a QGIS raster layer.

        Args:
            url: XYZ URL template, e.g. ``https://.../{z}/{x}/{y}.png``.
            name: Display name.
            zmin: Optional minimum zoom.
            zmax: Optional maximum zoom.
            attribution: Optional provider attribution.

        Returns:
            A status string.
        """

        def _run() -> str:
            """Run the worker body."""
            parts = ["type=xyz", f"url={quote(url, safe='')}"]
            if zmin is not None:
                parts.append(f"zmin={int(zmin)}")
            if zmax is not None:
                parts.append(f"zmax={int(zmax)}")
            if attribution:
                parts.append(f"referer={quote(attribution, safe='')}")
            uri = "&".join(parts)

            layer: Any | None = None
            try:
                from qgis.core import QgsRasterLayer  # type: ignore[import-not-found]

                candidate = QgsRasterLayer(uri, name, "wms")
                if candidate is not None and (
                    not hasattr(candidate, "isValid") or candidate.isValid()
                ):
                    _project().addMapLayer(candidate)
                    layer = candidate
            except ImportError:
                pass

            if layer is None:
                try:
                    layer = iface.addRasterLayer(uri, name, "wms")
                except TypeError:
                    layer = iface.addRasterLayer(uri, name)

            if layer is None or (hasattr(layer, "isValid") and not layer.isValid()):
                return f"Failed to load XYZ tile layer from {url!r}."
            return f"Added XYZ tile layer {name!r}."

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
        requires_confirmation=True,
    )
    def remove_layer(layer_name: str) -> str:
        """Remove a layer from the project.

        Args:
            layer_name: Display name of the layer to remove.

        Returns:
            A status string.
        """

        def _run() -> str:
            """Run the worker body."""
            proj = _project()
            layers = proj.mapLayersByName(layer_name)
            if not layers:
                return f"Layer {layer_name!r} not found."
            for layer in layers:
                try:
                    proj.removeMapLayer(layer.id())
                except Exception:
                    proj.removeMapLayer(layer)
            return f"Removed layer {layer_name!r}."

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def set_layer_visibility(layer_name: str, visible: bool) -> str:
        """Show or hide a layer in the layer panel.

        Args:
            layer_name: Display name of the layer.
            visible: ``True`` to show, ``False`` to hide.

        Returns:
            A status string.
        """

        def _run() -> str:
            """Run the worker body."""
            proj = _project()
            layer = _resolve_layer(proj, layer_name)
            # Try the layer-tree-based path first; fall back to a simple attribute.
            try:
                root = proj.layerTreeRoot()
                tree_layer = root.findLayer(layer.id())
                tree_layer.setItemVisibilityChecked(bool(visible))
            except Exception:
                try:
                    layer.visible = bool(visible)
                except Exception as exc:
                    return f"Could not set visibility on {layer_name!r}: {exc}"
            return f"Layer {layer_name!r} visibility set to {visible}."

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def set_layer_opacity(layer_name: str, opacity: float) -> str:
        """Set a layer opacity from 0.0 (transparent) to 1.0 (opaque)."""

        def _run() -> str:
            """Run the worker body."""
            value = min(1.0, max(0.0, float(opacity)))
            layer = _resolve_layer(_project(), layer_name)
            if hasattr(layer, "setOpacity"):
                layer.setOpacity(value)
            elif hasattr(layer, "renderer") and layer.renderer() is not None:
                renderer = layer.renderer()
                if hasattr(renderer, "setOpacity"):
                    renderer.setOpacity(value)
                else:
                    return f"Layer {layer_name!r} does not expose opacity controls."
            else:
                try:
                    layer.opacity = value
                except Exception:
                    return f"Layer {layer_name!r} does not expose opacity controls."
            if hasattr(layer, "triggerRepaint"):
                layer.triggerRepaint()
            if hasattr(iface.mapCanvas(), "refresh"):
                iface.mapCanvas().refresh()
            return f"Layer {layer_name!r} opacity set to {value}."

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def set_layer_symbology(
        layer_name: str,
        color: Optional[str] = None,
        line_width: Optional[float] = None,
        fill_color: Optional[str] = None,
        outline_color: Optional[str] = None,
        opacity: Optional[float] = None,
        raster_palette: Optional[str] = None,
        raster_min: Optional[float] = None,
        raster_max: Optional[float] = None,
    ) -> dict[str, Any]:
        """Change simple layer symbology such as color and line width.

        Args:
            layer_name: Display name of the target QGIS layer.
            color: Main symbol color, e.g. ``"blue"`` or ``"#0066ff"``.
            line_width: Line or outline width in QGIS symbol units.
            fill_color: Polygon fill color. When omitted, ``color`` is used.
            outline_color: Polygon outline/stroke color.
            opacity: Optional layer opacity from 0.0 to 1.0.
            raster_palette: Optional raster palette name, such as ``"terrain"``,
                ``"viridis"``, or ``"grayscale"``.
            raster_min: Optional minimum raster value for color-ramp rendering.
            raster_max: Optional maximum raster value for color-ramp rendering.

        Returns:
            A dict describing the applied style changes.
        """

        def _run() -> dict[str, Any]:
            """Run the worker body."""
            layer = _resolve_layer(_project(), layer_name)
            applied: dict[str, Any] = {"layer_name": layer_name}

            main_color = _qcolor(color)
            fill = _qcolor(fill_color) or main_color
            outline = _qcolor(outline_color)

            if _is_raster_layer(layer) and (
                raster_palette or fill is not None or color is not None
            ):
                applied.update(
                    _apply_raster_symbology(
                        layer,
                        raster_palette=raster_palette,
                        color=fill or main_color,
                        min_value=raster_min,
                        max_value=raster_max,
                    )
                )
                if color or fill_color:
                    applied["color"] = str(color or fill_color)
            else:
                try:
                    renderer = layer.renderer() if hasattr(layer, "renderer") else None
                except Exception:
                    renderer = None
                symbol = (
                    renderer.symbol()
                    if renderer is not None and hasattr(renderer, "symbol")
                    else None
                )
                if symbol is not None:
                    if fill is not None and hasattr(symbol, "setColor"):
                        symbol.setColor(fill)
                        applied["color"] = str(color or fill_color)
                    if outline is not None and _set_symbol_outline_color(
                        symbol, outline
                    ):
                        applied["outline_color"] = str(outline_color)
                    if line_width is not None:
                        width = max(0.0, float(line_width))
                        if _set_symbol_width(symbol, width):
                            applied["line_width"] = width
                else:
                    style = getattr(layer, "symbology", None)
                    if not isinstance(style, dict):
                        style = {}
                        try:
                            layer.symbology = style
                        except Exception:
                            pass
                    if fill is not None:
                        style["color"] = str(color or fill_color)
                        applied["color"] = str(color or fill_color)
                    if outline is not None:
                        style["outline_color"] = str(outline_color)
                        applied["outline_color"] = str(outline_color)
                    if line_width is not None:
                        style["line_width"] = max(0.0, float(line_width))
                        applied["line_width"] = style["line_width"]

            try:
                renderer = layer.renderer() if hasattr(layer, "renderer") else None
            except Exception:
                renderer = None

            if opacity is not None:
                value = min(1.0, max(0.0, float(opacity)))
                if hasattr(layer, "setOpacity"):
                    layer.setOpacity(value)
                    applied["opacity"] = value
                elif renderer is not None and hasattr(renderer, "setOpacity"):
                    renderer.setOpacity(value)
                    applied["opacity"] = value

            if hasattr(layer, "triggerRepaint"):
                layer.triggerRepaint()
            canvas = iface.mapCanvas()
            if hasattr(canvas, "refresh"):
                canvas.refresh()
            if len(applied) == 1:
                applied["message"] = "No supported symbology changes were applied."
            else:
                applied["message"] = f"Updated symbology for layer {layer_name!r}."
            return applied

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
        requires_confirmation=True,
        long_running=True,
    )
    def create_hillshade_layer(
        layer_name: Optional[str] = None,
        output_layer_name: Optional[str] = None,
        azimuth: float = 315.0,
        altitude: float = 45.0,
        z_factor: float = 1.0,
        output_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a hillshade raster from a DEM using a QGIS background task.

        This avoids freezing QGIS when the DEM source is a remote COG. The task
        uses GDAL in the worker thread, then creates and styles the output layer
        on the QGIS GUI thread.

        Args:
            layer_name: DEM layer name. Defaults to the active layer.
            output_layer_name: Display name for the hillshade layer.
            azimuth: Sun azimuth in degrees.
            altitude: Sun elevation angle in degrees.
            z_factor: Vertical exaggeration factor.
            output_path: Optional GeoTIFF output path.

        Returns:
            A dict describing the queued or completed hillshade task.
        """

        def _resolve() -> dict[str, Any]:
            layer = (
                iface.activeLayer()
                if not layer_name
                else _resolve_layer(_project(), layer_name)
            )
            if layer is None:
                raise ValueError("No active DEM layer is available.")
            source = str(layer.source() if hasattr(layer, "source") else "")
            if not source:
                raise ValueError(
                    f"Layer {layer.name()!r} does not expose a source URI."
                )
            name = output_layer_name or f"{layer.name()} Hillshade"
            path = output_path
            if not path:
                out_dir = tempfile.mkdtemp(prefix="geoagent_hillshade_")
                safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
                path = os.path.join(out_dir, f"{safe or 'hillshade'}.tif")
            return {"source": source, "name": name, "path": path}

        resolved = _on_gui(_resolve)

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
            # Test/mock fallback: create a lightweight raster layer immediately.
            iface.addRasterLayer(resolved["path"], resolved["name"])
            return {
                "success": True,
                "queued": False,
                "output_path": resolved["path"],
                "output_layer": resolved["name"],
                "source": resolved["source"],
                "message": f"Created hillshade layer {resolved['name']!r}.",
            }

        class _HillshadeTask(QgsTask):
            def __init__(self) -> None:
                super().__init__(f"Create hillshade: {resolved['name']}")
                self.error = ""

            def run(self) -> bool:
                try:
                    from osgeo import gdal  # type: ignore[import-not-found]

                    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
                    self.setProgress(5)
                    options = gdal.DEMProcessingOptions(
                        azimuth=float(azimuth),
                        altitude=float(altitude),
                        zFactor=float(z_factor),
                    )
                    result = gdal.DEMProcessing(
                        resolved["path"],
                        resolved["source"],
                        "hillshade",
                        options=options,
                    )
                    self.setProgress(90)
                    if result is None:
                        self.error = (
                            gdal.GetLastErrorMsg()
                            or "GDAL DEMProcessing did not produce an output."
                        )
                        return False
                    result = None
                    return os.path.exists(resolved["path"])
                except Exception as exc:  # pragma: no cover - QGIS runtime path
                    self.error = f"{type(exc).__name__}: {exc}"
                    return False

        def _message(text: str, level: Any) -> None:
            try:
                QgsMessageLog.logMessage(text, "OpenGeoAgent Hillshade", level)
            except Exception:
                pass

        def _status(text: str, timeout_ms: int = 0) -> None:
            try:
                window = iface.mainWindow()
                window.statusBar().showMessage(text, int(timeout_ms or 0))
            except Exception:
                pass

        def _enqueue() -> dict[str, Any]:
            task = _HillshadeTask()
            callbacks: dict[str, Any] = {}

            def _cleanup() -> None:
                try:
                    _ACTIVE_QGIS_HILLSHADE_TASKS.remove(task)
                except ValueError:
                    pass
                callbacks.clear()

            def _on_completed() -> None:
                message_level = getattr(Qgis, "MessageLevel", Qgis)
                layer = QgsRasterLayer(resolved["path"], resolved["name"])
                if layer is not None and layer.isValid():
                    QgsProject.instance().addMapLayer(layer)
                    try:
                        set_layer_symbology(
                            resolved["name"],
                            raster_palette="grayscale",
                            raster_min=0,
                            raster_max=255,
                            opacity=0.6,
                        )
                    except Exception:
                        pass
                    _status(f"Created hillshade layer: {resolved['name']}", 7000)
                    _message(
                        f"Created hillshade layer {resolved['name']}: {resolved['path']}",
                        getattr(message_level, "Info"),
                    )
                else:
                    reason = task.error or "QGIS could not load the hillshade output."
                    _status(f"Hillshade failed: {resolved['name']}", 7000)
                    _message(
                        f"Failed to create hillshade {resolved['name']!r}: {reason}",
                        getattr(message_level, "Critical"),
                    )
                _cleanup()

            def _on_terminated() -> None:
                message_level = getattr(Qgis, "MessageLevel", Qgis)
                reason = task.error or "QGIS terminated the hillshade task."
                _status(f"Hillshade failed: {resolved['name']}", 7000)
                _message(
                    f"Hillshade task terminated for {resolved['name']!r}: {reason}",
                    getattr(message_level, "Critical"),
                )
                _cleanup()

            callbacks["completed"] = _on_completed
            callbacks["terminated"] = _on_terminated
            task._geoagent_callbacks = callbacks  # noqa: SLF001 - retain Qt callbacks
            task.taskCompleted.connect(_on_completed)
            task.taskTerminated.connect(_on_terminated)
            _ACTIVE_QGIS_HILLSHADE_TASKS.append(task)
            _status(f"Creating hillshade: {resolved['name']}")
            QgsApplication.taskManager().addTask(task)
            return {
                "success": True,
                "queued": True,
                "source": resolved["source"],
                "output_path": resolved["path"],
                "output_layer": resolved["name"],
                "azimuth": float(azimuth),
                "altitude": float(altitude),
                "z_factor": float(z_factor),
                "message": (
                    "Queued a QGIS background task to create the hillshade. "
                    "The output layer will be added when the task completes."
                ),
            }

        return _on_gui(_enqueue)

    @geo_tool(
        category="qgis",
    )
    def inspect_layer_fields(layer_name: str) -> list[dict[str, Any]]:
        """List fields and types for a vector layer.

        Args:
            layer_name: Display name of the layer.

        Returns:
            A list of ``{"name": ..., "type": ...}`` per field.
        """

        def _run() -> list[dict[str, Any]]:
            """Run the worker body."""
            layer = _resolve_layer(_project(), layer_name)
            out: list[dict[str, Any]] = []
            for field in layer.fields():
                if isinstance(field, dict):
                    out.append(
                        {"name": field.get("name", ""), "type": field.get("type", "")}
                    )
                else:
                    out.append(
                        {
                            "name": getattr(field, "name", lambda: "")(),
                            "type": str(
                                getattr(
                                    field, "typeName", lambda: type(field).__name__
                                )()
                            ),
                        }
                    )
            return out

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def get_selected_features(
        layer_name: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Return selected features from a layer (or the active layer).

        Args:
            layer_name: Display name of the layer; if omitted, uses the
                active layer.

        Returns:
            A list of feature dicts (mock-friendly; the QGIS path returns a
            simple ``{"id": ..., "attributes": ...}``).
        """

        def _run() -> list[dict[str, Any]]:
            """Run the worker body."""
            if layer_name is None:
                layer = iface.activeLayer()
                if layer is None:
                    return []
            else:
                layer = _resolve_layer(_project(), layer_name)
            features = layer.selectedFeatures()
            out: list[dict[str, Any]] = []
            for feature in features:
                if isinstance(feature, dict):
                    out.append(feature)
                else:
                    attrs = getattr(feature, "attributes", lambda: [])()
                    fid = getattr(feature, "id", lambda: None)()
                    out.append({"id": fid, "attributes": list(attrs)})
            return out

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def select_features_by_expression(
        layer_name: str,
        expression: str,
        behavior: str = "set",
    ) -> str:
        """Select features in a vector layer using a QGIS expression.

        Args:
            layer_name: Display name of the layer.
            expression: QGIS expression, e.g. ``"population" > 10000``.
            behavior: ``"set"``, ``"add"``, ``"remove"``, or
                ``"intersect"``.

        Returns:
            A status string with the selected feature count when available.
        """

        def _run() -> str:
            """Run the worker body."""
            layer = _resolve_layer(_project(), layer_name)
            if not hasattr(layer, "selectByExpression"):
                return f"Layer {layer_name!r} does not support expression selection."
            try:
                from qgis.core import QgsVectorLayer  # type: ignore[import-not-found]

                behavior_map = {
                    "set": getattr(QgsVectorLayer, "SetSelection", 0),
                    "add": getattr(QgsVectorLayer, "AddToSelection", 1),
                    "remove": getattr(QgsVectorLayer, "RemoveFromSelection", 3),
                    "intersect": getattr(QgsVectorLayer, "IntersectSelection", 2),
                }
            except Exception:
                behavior_map = {"set": 0, "add": 1, "intersect": 2, "remove": 3}
            mode = behavior_map.get(behavior.lower())
            if mode is None:
                return f"Unknown selection behavior {behavior!r}."
            layer.selectByExpression(expression, mode)
            count = _call_int(layer, "selectedFeatureCount")
            suffix = f" ({count} selected)." if count is not None else "."
            return f"Selected features on {layer_name!r}{suffix}"

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def clear_selection(layer_name: Optional[str] = None) -> str:
        """Clear selected features on one layer, or every project layer."""

        def _run() -> str:
            """Run the worker body."""
            layers = (
                [_resolve_layer(_project(), layer_name)]
                if layer_name is not None
                else list(_project().mapLayers().values())
            )
            cleared = 0
            for layer in layers:
                if hasattr(layer, "removeSelection"):
                    layer.removeSelection()
                    cleared += 1
            return f"Cleared selection on {cleared} layer(s)."

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def zoom_to_selected(layer_name: Optional[str] = None) -> str:
        """Zoom to the selected features on a layer or the active layer."""

        def _run() -> str:
            """Run the worker body."""
            layer = (
                iface.activeLayer()
                if layer_name is None
                else _resolve_layer(_project(), layer_name)
            )
            if layer is None:
                return "No active layer."
            if not hasattr(layer, "boundingBoxOfSelected"):
                return (
                    f"Layer {layer.name()!r} does not expose selected-feature bounds."
                )
            extent = layer.boundingBoxOfSelected()
            if extent is None:
                return f"Layer {layer.name()!r} has no selected feature bounds."
            canvas = iface.mapCanvas()
            extent = _transform_extent_to_canvas_crs(layer, canvas, extent)
            canvas.setExtent(extent)
            if hasattr(canvas, "refresh"):
                canvas.refresh()
            return f"Zoomed to selected features on {layer.name()!r}."

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def get_layer_summary(layer_name: str) -> dict[str, Any]:
        """Return detailed metadata for a single project layer."""

        def _run() -> dict[str, Any]:
            """Run the worker body."""
            layer = _resolve_layer(_project(), layer_name)
            summary = _layer_metadata(layer, _project())
            if hasattr(layer, "fields"):
                fields: list[dict[str, Any]] = []
                for field in layer.fields():
                    if isinstance(field, dict):
                        fields.append(
                            {
                                "name": field.get("name", ""),
                                "type": field.get("type", ""),
                            }
                        )
                    else:
                        fields.append(
                            {
                                "name": getattr(field, "name", lambda: "")(),
                                "type": str(
                                    getattr(
                                        field, "typeName", lambda: type(field).__name__
                                    )()
                                ),
                            }
                        )
                summary["fields"] = fields
            return summary

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
        requires_confirmation=True,
    )
    def run_processing_algorithm(
        algorithm_id: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Run a QGIS processing algorithm.

        Args:
            algorithm_id: Algorithm identifier (e.g. ``"native:buffer"``).
            parameters: Algorithm parameters as a dict.

        Returns:
            The algorithm's result dictionary.
        """

        def _run() -> dict[str, Any]:
            """Run the worker body."""
            try:
                import processing  # type: ignore[import-not-found]
            except Exception as exc:  # pragma: no cover - QGIS-only path
                raise RuntimeError(
                    "QGIS Processing framework is not available."
                ) from exc
            return processing.run(algorithm_id, parameters)

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
        requires_confirmation=True,
        long_running=True,
    )
    def buffer_active_layer(
        distance_meters: float,
        output_layer_name: Optional[str] = None,
        output_path: Optional[str] = None,
        segments: int = 8,
        dissolve: bool = False,
    ) -> dict[str, Any]:
        """Buffer the active vector layer and add the result to the project.

        Args:
            distance_meters: Buffer distance. QGIS Processing interprets this
                in the active layer's CRS units; projected CRS layers commonly
                use metres.
            output_layer_name: Optional display name for the output layer.
            output_path: Optional output file path. When omitted, GeoAgent
                writes a GeoPackage into the project directory or temp dir.
            segments: Number of segments used to approximate quarter circles.
            dissolve: Whether to dissolve buffered features.

        Returns:
            A compact dict describing the Processing result and loaded layer.
        """

        def _run() -> dict[str, Any]:
            """Run the worker body."""
            try:
                import processing  # type: ignore[import-not-found]
            except Exception as exc:  # pragma: no cover - QGIS-only path
                raise RuntimeError(
                    "QGIS Processing framework is not available."
                ) from exc

            layer = iface.activeLayer()
            if layer is None:
                return {"success": False, "reason": "No active layer."}

            name = output_layer_name or f"{layer.name()} buffer {distance_meters:g}m"
            target_path = output_path or _default_processing_output_path(
                _project(),
                name,
                ".gpkg",
            )
            params = {
                "INPUT": layer,
                "DISTANCE": float(distance_meters),
                "SEGMENTS": int(segments),
                "END_CAP_STYLE": 0,
                "JOIN_STYLE": 0,
                "MITER_LIMIT": 2,
                "DISSOLVE": bool(dissolve),
                "OUTPUT": target_path,
            }
            result = processing.run("native:buffer", params)
            output = (
                result.get("OUTPUT", target_path)
                if isinstance(result, dict)
                else target_path
            )
            added = None
            if output:
                added = iface.addVectorLayer(str(output), name, "ogr")
            success = added is not None and (
                not hasattr(added, "isValid") or added.isValid()
            )
            if hasattr(iface.mapCanvas(), "refresh"):
                iface.mapCanvas().refresh()
            return {
                "success": bool(success),
                "algorithm_id": "native:buffer",
                "input_layer": layer.name(),
                "distance_meters": float(distance_meters),
                "output": str(output) if output else None,
                "layer_name": name if success else None,
                "processing_result": (
                    {key: str(value) for key, value in result.items()}
                    if isinstance(result, dict)
                    else {}
                ),
            }

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def open_attribute_table(layer_name: str) -> str:
        """Open the attribute table for a layer.

        Args:
            layer_name: Display name of the layer.

        Returns:
            A status string.
        """

        def _run() -> str:
            """Run the worker body."""
            layer = _resolve_layer(_project(), layer_name)
            iface.setActiveLayer(layer)
            if hasattr(iface, "showAttributeTable"):
                iface.showAttributeTable(layer)  # type: ignore[attr-defined]
            return f"Attribute table requested for {layer_name!r}."

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def refresh_canvas() -> str:
        """Refresh the QGIS map canvas.

        Returns:
            A status string.
        """

        def _run() -> str:
            """Run the worker body."""
            iface.mapCanvas().refresh()
            return "Canvas refreshed."

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
        requires_confirmation=True,
        destructive=True,
    )
    def run_pyqgis_script(code: str, description: str = "") -> dict[str, Any]:
        """Run a short PyQGIS script when no dedicated GeoAgent tool exists.

        Use this for QGIS API operations that are not covered by a named
        GeoAgent tool, such as raster renderer/band styling, labeling, layer
        tree tweaks, or other project/canvas changes. The script runs on the
        QGIS GUI thread with ``iface``, ``project``, ``canvas``, and
        ``active_layer`` already defined.

        Args:
            code: Python code using the QGIS/PyQt API. Imports are limited to
                qgis/PyQt and math modules. Do not use this for shell,
                network, filesystem, or secret-handling operations.
            description: One-sentence explanation of the intended QGIS change.

        Returns:
            A dict with success status, printed output, changed variable names,
            and the executed code.
        """
        code = (code or "").strip()
        if not code:
            return {
                "success": False,
                "error": "No PyQGIS code was provided.",
                "pyqgis_script": "",
            }
        _validate_pyqgis_script(code)

        def _run() -> dict[str, Any]:
            """Run the generated PyQGIS script on the GUI thread."""
            proj = _project()
            canvas = iface.mapCanvas() if hasattr(iface, "mapCanvas") else None
            active_layer = (
                iface.activeLayer() if hasattr(iface, "activeLayer") else None
            )
            namespace: dict[str, Any] = {
                "__builtins__": _pyqgis_builtins(),
                "iface": iface,
                "project": proj,
                "canvas": canvas,
                "active_layer": active_layer,
            }
            stdout = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout):
                    exec(  # nosec B102 - confirmation-gated PyQGIS tool.
                        compile(code, "<geoagent_pyqgis_script>", "exec"),
                        namespace,
                    )
            except Exception as exc:
                return {
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "stdout": stdout.getvalue().strip(),
                    "description": description,
                    "pyqgis_script": code,
                }

            ignored = {"__builtins__", "iface", "project", "canvas", "active_layer"}
            variables = sorted(k for k in namespace if k not in ignored)
            return {
                "success": True,
                "message": description or "PyQGIS script executed.",
                "stdout": stdout.getvalue().strip(),
                "variables": variables,
                "pyqgis_script": code,
            }

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
        requires_confirmation=True,
    )
    def save_project(path: Optional[str] = None) -> str:
        """Save the current QGIS project, optionally to a new file path."""

        def _run() -> str:
            """Run the worker body."""
            proj = _project()
            if path:
                out = Path(path).expanduser().resolve()
                if hasattr(proj, "write"):
                    ok = proj.write(str(out))
                else:
                    return "Project object does not expose write()."
                if ok is False:
                    return f"Failed to save project to {str(out)!r}."
                return str(out)
            if hasattr(proj, "write"):
                ok = proj.write()
                if ok is False:
                    return "Failed to save project."
                return "Project saved."
            return "Project object does not expose write()."

        return _on_gui(_run)

    return [
        list_project_layers,
        get_active_layer,
        get_project_state,
        zoom_in,
        zoom_out,
        zoom_to_layer,
        zoom_to_extent,
        set_center,
        set_scale,
        add_vector_layer,
        add_raster_layer,
        add_xyz_tile_layer,
        remove_layer,
        set_layer_visibility,
        set_layer_opacity,
        set_layer_symbology,
        create_hillshade_layer,
        inspect_layer_fields,
        get_selected_features,
        select_features_by_expression,
        clear_selection,
        zoom_to_selected,
        get_layer_summary,
        run_processing_algorithm,
        buffer_active_layer,
        open_attribute_table,
        refresh_canvas,
        run_pyqgis_script,
        save_project,
    ]


__all__ = ["qgis_tools"]
