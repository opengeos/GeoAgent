"""Tool adapters for the QGIS GEE Data Catalogs plugin.

The module is import-safe outside QGIS and outside the plugin. Tool bodies
resolve ``gee_data_catalogs`` lazily so GeoAgent can be imported in ordinary
Python environments.
"""

from __future__ import annotations

import ast
import contextlib
import io
import os
import types
from typing import Any, Optional
from urllib.parse import quote

from geoagent.core.decorators import geo_tool
from geoagent.tools._qt_marshal import run_on_qt_gui_thread


def _on_gui(fn: Any) -> Any:
    """Run ``fn`` on the Qt GUI thread when QGIS is available."""
    return run_on_qt_gui_thread(fn)


def _project_id_from_settings() -> str | None:
    """Return the configured Earth Engine project id, if any."""
    try:
        from qgis.PyQt.QtCore import QSettings  # type: ignore[import-not-found]

        project_id = QSettings().value("GeeDataCatalogs/ee_project", "", type=str)
        project_id = project_id.strip() if project_id else ""
        if project_id:
            return project_id
    except Exception:
        pass
    return os.environ.get("EE_PROJECT_ID") or None


def _compact_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    """Return a concise, JSON-friendly catalog record."""
    keys = [
        "id",
        "name",
        "title",
        "type",
        "category",
        "provider",
        "start_date",
        "end_date",
        "source",
        "url",
        "license",
        "bigquery_table",
    ]
    out = {key: dataset.get(key) for key in keys if dataset.get(key)}
    description = str(dataset.get("description", "")).strip()
    if description:
        out["description"] = description[:500]
    keywords = dataset.get("keywords")
    if keywords:
        out["keywords"] = keywords[:15] if isinstance(keywords, list) else keywords
    return out


def _parse_list(value: str | list[str] | None) -> list[str] | None:
    """Parse comma-separated UI-style values into a list."""
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_bbox(
    bbox: str | list[float] | tuple[float, ...] | None,
) -> list[float] | None:
    """Parse west,south,east,north bounds."""
    if bbox is None or bbox == "":
        return None
    values = list(bbox) if isinstance(bbox, (list, tuple)) else str(bbox).split(",")
    if len(values) != 4:
        raise ValueError("bbox must contain west,south,east,north")
    west, south, east, north = [float(value) for value in values]
    if west >= east or south >= north:
        raise ValueError("bbox coordinates must satisfy west < east and south < north")
    return [west, south, east, north]


def _build_vis_params(
    *,
    bands: str | list[str] | None = None,
    min_value: float | int | None = None,
    max_value: float | int | None = None,
    palette: str | list[str] | None = None,
    for_feature_collection: bool = False,
) -> dict[str, Any]:
    """Build Earth Engine visualization parameters from scalar inputs."""
    vis_params: dict[str, Any] = {}
    parsed_bands = _parse_list(bands)
    if parsed_bands and not for_feature_collection:
        vis_params["bands"] = parsed_bands
    if min_value is not None and not for_feature_collection:
        vis_params["min"] = float(min_value)
    if max_value is not None and not for_feature_collection:
        vis_params["max"] = float(max_value)
    parsed_palette = _parse_list(palette)
    if parsed_palette:
        if for_feature_collection:
            vis_params["color"] = parsed_palette[0]
        else:
            vis_params["palette"] = parsed_palette
    return vis_params


def _coerce_filter_value(value: str) -> Any:
    """Convert simple string filter values to scalar Earth Engine values."""
    text = str(value).strip()
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    try:
        if "." not in text:
            return int(text)
        return float(text)
    except ValueError:
        return text


def _default_index_palette(index_name: str) -> list[str]:
    """Return a useful visualization palette for common normalized indexes."""
    key = index_name.strip().upper()
    if key in {"NDVI", "SAVI", "EVI"}:
        return ["8c510a", "f6e8c3", "f5f5f5", "c7eae5", "01665e"]
    if key in {"NDWI", "MNDWI", "NDMI"}:
        return ["8c510a", "f7f7f7", "2166ac"]
    if key in {"NBR", "NBR2"}:
        return ["d7191c", "fdae61", "ffffbf", "a6d96a", "1a9641"]
    return ["d73027", "f7f7f7", "1a9850"]


def _normalize_composite_method(
    method: str | None,
    default: str = "mosaic",
    *,
    allow_mode: bool = False,
) -> str:
    """Return a supported ImageCollection compositing method."""
    method = str(method or "").lower().strip()
    allowed = {"mosaic", "median", "mean", "min", "max", "first"}
    if allow_mode:
        allowed.add("mode")
    if method in allowed:
        return method
    return default


def _composite_image_collection(collection: Any, method: str) -> Any:
    """Convert an ImageCollection to an Image using a supported method."""
    if method == "mosaic":
        return collection.mosaic()
    if method == "median":
        return collection.median()
    if method == "mean":
        return collection.mean()
    if method == "min":
        return collection.min()
    if method == "max":
        return collection.max()
    if method == "first":
        return collection.first()
    if method == "mode":
        try:
            import ee

            return collection.reduce(ee.Reducer.mode())
        except Exception:
            return collection.mosaic()
    return collection.mosaic()


def _is_opera_dswx(asset_id: str) -> bool:
    """Return True for supported OPERA DSWx Earth Engine collections."""
    return str(asset_id).strip() in {
        "OPERA/DSWX/L3_V1/HLS",
        "OPERA/DSWX/L3_V1/S1",
    }


def _opera_dswx_default_composite_method(asset_id: str) -> str:
    """Return the catalog-recommended DSWx composite method."""
    if str(asset_id).strip() == "OPERA/DSWX/L3_V1/S1":
        return "max"
    return "mode"


def _opera_dswx_band_alias(bands: str | list[str] | None) -> str:
    """Return a valid OPERA DSWx-HLS display band from common aliases."""
    parsed = _parse_list(bands)
    if not parsed:
        return "WTR_Water_classification"
    first = parsed[0].strip()
    aliases = {
        "WTR": "WTR_Water_classification",
        "B01_WTR": "WTR_Water_classification",
        "WTR_Water_classification": "WTR_Water_classification",
        "BWTR": "BWTR_Binary_water",
        "BINARY_WATER": "BWTR_Binary_water",
        "BWTR_Binary_water": "BWTR_Binary_water",
    }
    return aliases.get(first, first)


def _opera_dswx_render_image(
    collection: Any,
    band_name: str,
    method: str = "mode",
) -> tuple[Any, dict[str, Any]]:
    """Build a renderable OPERA DSWx classification composite."""
    import ee

    if band_name == "BWTR_Binary_water":
        class_values = [0, 1, 252, 253, 254]
        palette = ["ffffff", "0000ff", "f2f2f2", "dfdfdf", "da00ff"]
    else:
        band_name = "WTR_Water_classification"
        class_values = [0, 1, 2, 252, 253, 254]
        palette = ["ffffff", "0000ff", "0088ff", "f2f2f2", "dfdfdf", "da00ff"]

    masked_collection = collection.map(
        lambda image: image.select(band_name).updateMask(
            image.select(band_name).lt(252)
        )
    )
    reducer = ee.Reducer.mode() if method == "mode" else ee.Reducer.max()
    composite = masked_collection.reduce(reducer).rename(band_name)
    remapped = composite.select(band_name).remap(
        class_values, list(range(len(class_values)))
    )
    remapped = remapped.updateMask(remapped.neq(0))
    return remapped, {
        "min": 0.0,
        "max": float(len(class_values) - 1),
        "palette": palette,
    }


def _format_python_snippet_value(value: Any) -> str:
    """Return a compact Python literal for snippet generation."""
    return repr(value)


_ALLOWED_GEE_SNIPPET_IMPORTS = {"ee", "geemap"}
_FORBIDDEN_GEE_SNIPPET_NAMES = {
    "__builtins__",
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}
_FORBIDDEN_GEE_SNIPPET_ROOTS = {
    "builtins",
    "http",
    "importlib",
    "marshal",
    "os",
    "pathlib",
    "pickle",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "urllib",
}
_FORBIDDEN_GEE_SNIPPET_ATTRIBUTES = {
    "computeValue",
    "getInfo",
    "reduceRegion",
    "reduceRegions",
}
_FORBIDDEN_GEE_SNIPPET_NODES = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.For,
    ast.FunctionDef,
    ast.Global,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.While,
    ast.With,
)


def _attribute_root(node: ast.AST) -> str | None:
    """Return the leftmost name in an attribute expression."""
    current = node
    while isinstance(current, ast.Attribute):
        current = current.value
    if isinstance(current, ast.Name):
        return current.id
    return None


def _validate_gee_python_snippet(code: str) -> None:
    """Reject generated snippets outside the constrained EE layer surface."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"Invalid Python snippet: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_GEE_SNIPPET_NODES):
            raise ValueError(
                f"Unsupported Python construct in generated snippet: "
                f"{type(node).__name__}"
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in _ALLOWED_GEE_SNIPPET_IMPORTS:
                    raise ValueError(
                        f"Import is not allowed in generated Earth Engine "
                        f"snippets: {alias.name}"
                    )
        if isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".", 1)[0]
            if module not in _ALLOWED_GEE_SNIPPET_IMPORTS:
                raise ValueError(
                    f"Import is not allowed in generated Earth Engine snippets: "
                    f"from {node.module or ''}"
                )
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in _FORBIDDEN_GEE_SNIPPET_NAMES:
                raise ValueError(
                    f"Name is not allowed in generated Earth Engine snippets: "
                    f"{node.id}"
                )
            if node.id in _FORBIDDEN_GEE_SNIPPET_ROOTS:
                raise ValueError(
                    f"Module is not allowed in generated Earth Engine snippets: "
                    f"{node.id}"
                )
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise ValueError(
                    "Dunder attributes are not allowed in generated Earth "
                    "Engine snippets."
                )
            if node.attr in _FORBIDDEN_GEE_SNIPPET_ATTRIBUTES:
                raise ValueError(
                    f"{node.attr} is not allowed in generated Earth Engine "
                    "layer snippets. Use a dedicated analysis/statistics tool "
                    "for scalar Earth Engine results."
                )
            root = _attribute_root(node)
            if root in _FORBIDDEN_GEE_SNIPPET_ROOTS:
                raise ValueError(
                    f"Module is not allowed in generated Earth Engine snippets: "
                    f"{root}"
                )
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_GEE_SNIPPET_NAMES:
                raise ValueError(
                    f"Call is not allowed in generated Earth Engine snippets: "
                    f"{func.id}"
                )
            root = _attribute_root(func)
            if root in _FORBIDDEN_GEE_SNIPPET_ROOTS:
                raise ValueError(
                    f"Call is not allowed in generated Earth Engine snippets: "
                    f"{root}"
                )


def _safe_builtins_for_gee_snippet(
    ee_module: Any, geemap_module: Any
) -> dict[str, Any]:
    """Return a minimal builtins dict for generated EE snippets."""

    def _safe_import(
        name: str,
        globals: dict[str, Any] | None = None,  # noqa: A002
        locals: dict[str, Any] | None = None,  # noqa: A002
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if level != 0:
            raise ImportError("Relative imports are not allowed in EE snippets.")
        root = name.split(".", 1)[0]
        if root == "ee":
            return ee_module
        if root == "geemap":
            return geemap_module
        raise ImportError(f"Import is not allowed in EE snippets: {name}")

    return {
        "__import__": _safe_import,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "print": print,
        "range": range,
        "repr": repr,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }


def _loaded_gee_layer_records() -> list[dict[str, Any]]:
    """Return compact metadata for registered plugin Earth Engine layers."""
    from gee_data_catalogs.core.ee_utils import get_ee_layers

    records: list[dict[str, Any]] = []
    for name, payload in get_ee_layers().items():
        ee_object = payload[0] if isinstance(payload, tuple) and payload else payload
        vis_params = (
            payload[1] if isinstance(payload, tuple) and len(payload) > 1 else {}
        )
        type_name = type(ee_object).__name__
        records.append(
            {
                "name": str(name),
                "object_type": type_name,
                "vis_params": vis_params or {},
            }
        )
    return records


def _get_loaded_gee_object(layer_name: str) -> Any:
    """Return a registered Earth Engine object by exact or case-insensitive name."""
    from gee_data_catalogs.core.ee_utils import get_ee_layers

    layers = get_ee_layers()
    if layer_name in layers:
        payload = layers[layer_name]
        return payload[0] if isinstance(payload, tuple) else payload

    requested = layer_name.strip().lower()
    matches = [
        (name, payload)
        for name, payload in layers.items()
        if str(name).strip().lower() == requested
    ]
    if not matches:
        matches = [
            (name, payload)
            for name, payload in layers.items()
            if requested and requested in str(name).strip().lower()
        ]
    if not matches:
        available = ", ".join(str(name) for name in layers) or "(none)"
        raise KeyError(
            f"Earth Engine layer not found: {layer_name}. Available layers: {available}"
        )
    if len(matches) > 1:
        names = ", ".join(str(name) for name, _payload in matches)
        raise KeyError(f"Earth Engine layer name is ambiguous: {layer_name}. {names}")
    payload = matches[0][1]
    return payload[0] if isinstance(payload, tuple) else payload


@contextlib.contextmanager
def _temporary_ee_deadline(timeout_seconds: int | float):
    """Temporarily set the Earth Engine API deadline when supported."""
    try:
        import ee

        data = getattr(ee, "data", None)
        set_deadline = getattr(data, "setDeadline", None)
        get_deadline = getattr(data, "getDeadline", None)
        old_deadline = get_deadline() if callable(get_deadline) else None
        has_old_deadline = callable(get_deadline)
        if callable(set_deadline):
            set_deadline(max(1, int(float(timeout_seconds) * 1000)))
        try:
            yield
        finally:
            if callable(set_deadline) and has_old_deadline:
                set_deadline(old_deadline)
    except ImportError:
        yield


def _build_ee_statistics_reducer(
    statistics: str | list[str] | None,
) -> tuple[Any, list[str]]:
    """Build a combined Earth Engine reducer for requested statistics."""
    import ee

    requested = _parse_list(statistics) or ["mean"]
    aliases = {
        "average": "mean",
        "avg": "mean",
        "std": "stdDev",
        "stdev": "stdDev",
        "stddev": "stdDev",
    }
    supported = {
        "mean",
        "min",
        "max",
        "median",
        "stdDev",
        "sum",
        "count",
    }
    normalized: list[str] = []
    for item in requested:
        key = aliases.get(str(item).strip().lower(), str(item).strip())
        if key not in supported:
            raise ValueError(
                f"Unsupported statistic: {item}. Supported statistics: "
                f"{', '.join(sorted(supported))}"
            )
        normalized.append(key)

    reducer = None
    for stat in normalized:
        reducer_fn = getattr(ee.Reducer, stat)
        next_reducer = reducer_fn()
        if reducer is None:
            reducer = next_reducer
        else:
            reducer = reducer.combine(reducer2=next_reducer, sharedInputs=True)
    return reducer, normalized


def _calculate_gee_layer_statistics_impl(
    *,
    layer_name: str,
    band: str | None,
    statistics: str | list[str] | None,
    scale: int | float,
    max_pixels: int | float,
    best_effort: bool,
    tile_scale: int | float,
    timeout_seconds: int | float,
    region_collection_asset_id: str | None,
    region_filter_property: str | None,
    region_filter_value: str | None,
) -> dict[str, Any]:
    """Calculate bounded Earth Engine statistics for a registered image layer."""
    import ee

    source = _get_loaded_gee_object(layer_name)
    image = ee.Image(source)
    if band:
        image = image.select(str(band).strip())

    region = None
    region_info = None
    if region_collection_asset_id:
        fc = ee.FeatureCollection(region_collection_asset_id)
        if region_filter_property and region_filter_value is not None:
            fc = fc.filter(
                ee.Filter.eq(
                    region_filter_property,
                    _coerce_filter_value(region_filter_value),
                )
            )
        region = fc.geometry()
        region_info = {
            "collection_asset_id": region_collection_asset_id,
            "filter_property": region_filter_property,
            "filter_value": region_filter_value,
        }
    else:
        region = image.geometry()

    reducer, normalized_statistics = _build_ee_statistics_reducer(statistics)
    scale_value = float(scale)
    max_pixels_value = int(float(max_pixels))
    tile_scale_value = float(tile_scale)

    reduction = image.reduceRegion(
        reducer=reducer,
        geometry=region,
        scale=scale_value,
        maxPixels=max_pixels_value,
        bestEffort=bool(best_effort),
        tileScale=tile_scale_value,
    )
    with _temporary_ee_deadline(timeout_seconds):
        values = reduction.getInfo()

    if not isinstance(values, dict):
        values = {"value": values}

    numeric_values = [
        value for value in values.values() if isinstance(value, (int, float))
    ]
    return {
        "success": True,
        "layer_name": layer_name,
        "band": band,
        "statistics": normalized_statistics,
        "values": values,
        "mean": (
            numeric_values[0]
            if normalized_statistics == ["mean"] and numeric_values
            else None
        ),
        "scale": scale_value,
        "max_pixels": max_pixels_value,
        "best_effort": bool(best_effort),
        "tile_scale": tile_scale_value,
        "timeout_seconds": float(timeout_seconds),
        "region": region_info,
        "approximate": bool(best_effort),
    }


def _run_gee_python_snippet_impl(
    *,
    iface: Any,
    code: str,
) -> dict[str, Any]:
    """Execute a constrained EE/geemap snippet and add requested layers."""
    import ee

    _validate_gee_python_snippet(code)

    layers_added: list[dict[str, Any]] = []

    def _add_layer(
        ee_object: Any,
        vis_params: dict[str, Any] | None = None,
        name: str = "Earth Engine Layer",
        shown: bool = True,  # noqa: ARG001 - retained for geemap compatibility
        opacity: float = 1.0,  # noqa: ARG001 - retained for geemap compatibility
    ) -> Any:
        display_name = str(name or "Earth Engine Layer")[:50]
        layer = _add_ee_layer_nonblocking(
            iface=iface,
            ee_object=ee_object,
            vis_params=vis_params or {},
            name=display_name,
        )
        _validate_added_layer(layer, display_name)
        layers_added.append(
            {
                "name": display_name,
                "vis_params": vis_params or {},
                "object_type": type(ee_object).__name__,
            }
        )
        return layer

    class QGISMap:
        """Small geemap.Map-compatible adapter for generated snippets."""

        def add_layer(
            self,
            ee_object: Any,
            vis_params: dict[str, Any] | None = None,
            name: str = "Earth Engine Layer",
            shown: bool = True,
            opacity: float = 1.0,
        ) -> Any:
            return _add_layer(ee_object, vis_params, name, shown, opacity)

        def addLayer(
            self,
            ee_object: Any,
            vis_params: dict[str, Any] | None = None,
            name: str = "Earth Engine Layer",
            shown: bool = True,
            opacity: float = 1.0,
        ) -> Any:
            return self.add_layer(ee_object, vis_params, name, shown, opacity)

        def add_ee_layer(
            self,
            ee_object: Any,
            vis_params: dict[str, Any] | None = None,
            name: str = "Earth Engine Layer",
            shown: bool = True,
            opacity: float = 1.0,
        ) -> Any:
            return self.add_layer(ee_object, vis_params, name, shown, opacity)

    geemap_module = types.ModuleType("geemap")
    geemap_module.Map = QGISMap

    stdout = io.StringIO()
    stderr = io.StringIO()
    namespace: dict[str, Any] = {
        "__builtins__": _safe_builtins_for_gee_snippet(ee, geemap_module),
        "Map": QGISMap,
        "add_layer": _add_layer,
        "ee": ee,
        "geemap": geemap_module,
        "get_ee_layer": _get_loaded_gee_object,
        "list_ee_layers": _loaded_gee_layer_records,
        "m": QGISMap(),
    }

    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        exec(compile(code, "<generated_gee_snippet>", "exec"), namespace)  # nosec B102

    if not layers_added:
        return {
            "success": False,
            "error": (
                "The generated Earth Engine snippet ran but did not add a layer. "
                "Call m.add_layer(...), m.addLayer(...), or add_layer(...)."
            ),
            "layers_added": [],
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "earth_engine_python_snippet": code,
        }

    try:
        _on_gui(lambda: iface.mapCanvas().refresh())
    except Exception:
        pass

    return {
        "success": True,
        "layers_added": layers_added,
        "stdout": stdout.getvalue(),
        "stderr": stderr.getvalue(),
        "earth_engine_python_snippet": code,
    }


def _build_load_gee_dataset_snippet(
    *,
    asset_id: str,
    resolved_type: str,
    start_date: str | None,
    end_date: str | None,
    bbox: list[float] | None,
    cloud_cover: int | None,
    cloud_property: str | None,
    bounds_collection_asset_id: str | None,
    bounds_filter_property: str | None,
    bounds_filter_value: str | None,
    method: str | None,
    bands: str | list[str] | None,
    min_value: float | None,
    max_value: float | None,
    palette: str | list[str] | None,
    vis_params: dict[str, Any],
    layer_name: str,
    rendered_band: str | None,
    clip_collection_asset_id: str | None,
    clip_filter_property: str | None,
    clip_filter_value: str | None,
) -> str:
    """Build a reproducible Earth Engine Python snippet for a loaded layer."""
    lines = [
        "import ee",
        "import geemap",
        "",
        "m = geemap.Map()",
        "",
        f"asset_id = {_format_python_snippet_value(asset_id)}",
    ]

    if resolved_type == "ImageCollection":
        lines.append("collection = ee.ImageCollection(asset_id)")
        if start_date and end_date:
            lines.append(
                "collection = collection.filterDate("
                f"{_format_python_snippet_value(start_date)}, "
                f"{_format_python_snippet_value(end_date)}"
                ")"
            )
        elif start_date:
            lines.append(
                "collection = collection.filterDate("
                f"{_format_python_snippet_value(start_date)}"
                ")"
            )
        elif end_date:
            lines.append(
                "collection = collection.filterDate("
                f"{_format_python_snippet_value(end_date)}"
                ")"
            )
        if bounds_collection_asset_id:
            lines.append(
                "bounds_fc = ee.FeatureCollection("
                f"{_format_python_snippet_value(bounds_collection_asset_id)})"
            )
            if bounds_filter_property and bounds_filter_value is not None:
                lines.append(
                    "bounds_fc = bounds_fc.filter(ee.Filter.eq("
                    f"{_format_python_snippet_value(bounds_filter_property)}, "
                    f"{_format_python_snippet_value(_coerce_filter_value(bounds_filter_value))}"
                    "))"
                )
            lines.append("collection = collection.filterBounds(bounds_fc)")
        elif bbox is not None:
            lines.append(f"bbox = {_format_python_snippet_value(bbox)}")
            lines.append(
                "collection = collection.filterBounds(ee.Geometry.Rectangle(bbox))"
            )

        if _is_opera_dswx(asset_id):
            band = rendered_band or _opera_dswx_band_alias(bands)
            reducer_name = "mode" if method == "mode" else "max"
            if band == "BWTR_Binary_water":
                class_values = [0, 1, 252, 253, 254]
            else:
                class_values = [0, 1, 2, 252, 253, 254]
            to_values = list(range(len(class_values)))
            lines.extend(
                [
                    f"band = {_format_python_snippet_value(band)}",
                    "masked = collection.map(",
                    "    lambda image: image.select(band).updateMask(",
                    "        image.select(band).lt(252)",
                    "    )",
                    ")",
                    f"image = masked.reduce(ee.Reducer.{reducer_name}()).rename(band)",
                    f"class_values = {_format_python_snippet_value(class_values)}",
                    f"to_values = {_format_python_snippet_value(to_values)}",
                    "image = image.select(band).remap(class_values, to_values)",
                    "image = image.updateMask(image.neq(0))  # make non-water transparent",
                ]
            )
        else:
            if cloud_cover is not None:
                prop = cloud_property or "CLOUDY_PIXEL_PERCENTAGE"
                lines.append(
                    "collection = collection.filter("
                    f"ee.Filter.lt({_format_python_snippet_value(prop)}, "
                    f"{_format_python_snippet_value(cloud_cover)}))"
                )
            composite = method or "mosaic"
            if composite == "mosaic":
                lines.append("image = collection.mosaic()")
            elif composite == "mode":
                lines.append("image = collection.reduce(ee.Reducer.mode())")
            else:
                lines.append(f"image = collection.{composite}()")
    elif resolved_type == "FeatureCollection":
        lines.append("image = ee.FeatureCollection(asset_id)")
    else:
        lines.append("image = ee.Image(asset_id)")

    if clip_collection_asset_id and resolved_type != "FeatureCollection":
        lines.append(
            "clip_fc = ee.FeatureCollection("
            f"{_format_python_snippet_value(clip_collection_asset_id)})"
        )
        if clip_filter_property and clip_filter_value is not None:
            lines.append(
                "clip_fc = clip_fc.filter(ee.Filter.eq("
                f"{_format_python_snippet_value(clip_filter_property)}, "
                f"{_format_python_snippet_value(_coerce_filter_value(clip_filter_value))}"
                "))"
            )
        lines.append("image = image.clipToCollection(clip_fc)")

    lines.append(f"vis_params = {_format_python_snippet_value(vis_params)}")
    lines.append("")
    lines.append(
        "m.add_layer("
        f"image, vis_params, {_format_python_snippet_value(layer_name)}"
        ")"
    )
    return "\n".join(lines)


def _build_normalized_difference_snippet(
    *,
    asset_id: str,
    resolved_type: str,
    start_date: str | None,
    end_date: str | None,
    bbox: list[float] | None,
    cloud_cover: int | None,
    cloud_property: str | None,
    bounds_collection_asset_id: str | None,
    bounds_filter_property: str | None,
    bounds_filter_value: str | None,
    method: str | None,
    positive_band: str,
    negative_band: str,
    output_name: str,
    vis_params: dict[str, Any],
    layer_name: str,
    clip_collection_asset_id: str | None,
    clip_filter_property: str | None,
    clip_filter_value: str | None,
) -> str:
    """Build a reproducible Earth Engine Python snippet for an index layer."""
    lines = [
        "import ee",
        "import geemap",
        "",
        "m = geemap.Map()",
        "",
        f"asset_id = {_format_python_snippet_value(asset_id)}",
    ]
    if resolved_type == "ImageCollection":
        lines.append("collection = ee.ImageCollection(asset_id)")
        if start_date and end_date:
            lines.append(
                "collection = collection.filterDate("
                f"{_format_python_snippet_value(start_date)}, "
                f"{_format_python_snippet_value(end_date)}"
                ")"
            )
        elif start_date:
            lines.append(
                "collection = collection.filterDate("
                f"{_format_python_snippet_value(start_date)}"
                ")"
            )
        elif end_date:
            lines.append(
                "collection = collection.filterDate("
                f"{_format_python_snippet_value(end_date)}"
                ")"
            )
        if bounds_collection_asset_id:
            lines.append(
                "bounds_fc = ee.FeatureCollection("
                f"{_format_python_snippet_value(bounds_collection_asset_id)})"
            )
            if bounds_filter_property and bounds_filter_value is not None:
                lines.append(
                    "bounds_fc = bounds_fc.filter(ee.Filter.eq("
                    f"{_format_python_snippet_value(bounds_filter_property)}, "
                    f"{_format_python_snippet_value(_coerce_filter_value(bounds_filter_value))}"
                    "))"
                )
            lines.append("collection = collection.filterBounds(bounds_fc)")
        elif bbox is not None:
            lines.append(f"bbox = {_format_python_snippet_value(bbox)}")
            lines.append(
                "collection = collection.filterBounds(ee.Geometry.Rectangle(bbox))"
            )
        if cloud_cover is not None:
            prop = cloud_property or "CLOUDY_PIXEL_PERCENTAGE"
            lines.append(
                "collection = collection.filter("
                f"ee.Filter.lt({_format_python_snippet_value(prop)}, "
                f"{_format_python_snippet_value(cloud_cover)}))"
            )
        composite = method or "median"
        if composite == "mosaic":
            lines.append("source = collection.mosaic()")
        elif composite == "mode":
            lines.append("source = collection.reduce(ee.Reducer.mode())")
        else:
            lines.append(f"source = collection.{composite}()")
    else:
        lines.append("source = ee.Image(asset_id)")

    lines.append(
        "image = source.normalizedDifference("
        f"[{_format_python_snippet_value(positive_band)}, "
        f"{_format_python_snippet_value(negative_band)}]"
        f").rename({_format_python_snippet_value(output_name)})"
    )
    if clip_collection_asset_id:
        lines.append(
            "clip_fc = ee.FeatureCollection("
            f"{_format_python_snippet_value(clip_collection_asset_id)})"
        )
        if clip_filter_property and clip_filter_value is not None:
            lines.append(
                "clip_fc = clip_fc.filter(ee.Filter.eq("
                f"{_format_python_snippet_value(clip_filter_property)}, "
                f"{_format_python_snippet_value(_coerce_filter_value(clip_filter_value))}"
                "))"
            )
        lines.append("image = image.clipToCollection(clip_fc)")

    lines.append(f"vis_params = {_format_python_snippet_value(vis_params)}")
    lines.append("")
    lines.append(
        "m.add_layer("
        f"image, vis_params, {_format_python_snippet_value(layer_name)}"
        ")"
    )
    return "\n".join(lines)


def _safe_get_info(value: Any) -> Any:
    """Best-effort ``getInfo`` for Earth Engine diagnostics."""
    try:
        get_info = getattr(value, "getInfo", None)
        if callable(get_info):
            return get_info()
    except Exception as exc:
        return {"error": str(exc)}
    return None


def _image_band_names(image: Any) -> list[str] | dict[str, str]:
    """Return output image band names or an error diagnostic."""
    try:
        import ee

        info = _safe_get_info(ee.Image(image).bandNames())
        if isinstance(info, list):
            return [str(item) for item in info]
        if isinstance(info, dict) and "error" in info:
            return {"error": info["error"]}
        return []
    except Exception as exc:
        return {"error": str(exc)}


def _first_image_band_names(collection: Any) -> list[str] | dict[str, str]:
    """Return band names from the first image without counting the collection."""
    try:
        import ee

        return _image_band_names(ee.Image(collection.first()))
    except Exception as exc:
        return {"error": str(exc)}


def _raise_layer_error(
    exc: Exception,
    *,
    asset_id: str,
    layer_name: str,
    diagnostics: dict[str, Any],
) -> None:
    """Raise a layer-loading error with Earth Engine diagnostics attached."""
    raise RuntimeError(
        "Failed to add Earth Engine layer to QGIS.\n"
        f"Asset: {asset_id}\n"
        f"Layer: {layer_name}\n"
        f"Diagnostics: {diagnostics}\n"
        f"Error: {exc}"
    ) from exc


def _xyz_uri_from_tile_url(tile_url: str) -> str:
    """Return a QGIS XYZ datasource URI with the nested tile URL encoded."""
    encoded_url = quote(tile_url, safe=":/{}")
    return f"type=xyz&url={encoded_url}&zmax=24&zmin=0"


def _ee_tile_url(ee_object: Any, vis_params: dict[str, Any]) -> str:
    """Generate an Earth Engine XYZ tile URL off the QGIS GUI thread."""
    try:
        import ee

        if isinstance(ee_object, ee.FeatureCollection):
            style_keys = {
                "color",
                "pointSize",
                "pointShape",
                "width",
                "fillColor",
                "styleProperty",
                "neighborhood",
                "lineType",
            }
            style_params = {k: v for k, v in vis_params.items() if k in style_keys}
            if style_params:
                ee_object = ee_object.style(**style_params)
            vis_params = {}
        elif isinstance(ee_object, ee.ImageCollection):
            ee_object = ee_object.mosaic()

        map_id = ee_object.getMapId(vis_params)
        return map_id.get("tile_fetcher").url_format
    except Exception as exc:
        raise RuntimeError(f"Failed to create Earth Engine tile URL: {exc}") from exc


def _add_xyz_layer_on_gui(
    *,
    iface: Any,
    ee_object: Any,
    vis_params: dict[str, Any],
    name: str,
    tile_url: str,
    shown: bool = True,
    opacity: float = 1.0,
) -> Any:
    """Add a precomputed Earth Engine XYZ URL to QGIS on the GUI thread."""

    def _run() -> Any:
        from qgis.core import QgsProject, QgsRasterLayer

        from gee_data_catalogs.core.ee_utils import (
            _store_ee_layer_metadata,
            add_ee_layer_to_registry,
            remove_ee_layer_from_registry,
        )

        project = QgsProject.instance()
        for existing_layer in project.mapLayersByName(name):
            project.removeMapLayer(existing_layer.id())
        remove_ee_layer_from_registry(name)

        current_extent = None
        try:
            canvas = iface.mapCanvas()
            current_extent = canvas.extent() if canvas is not None else None
        except Exception:  # nosec B110
            pass

        uri = _xyz_uri_from_tile_url(tile_url)
        layer = QgsRasterLayer(uri, name, "wms")
        if not layer.isValid():
            raise ValueError(f"Failed to create valid Earth Engine XYZ layer: {name}")

        if hasattr(layer, "renderer") and layer.renderer():
            layer.renderer().setOpacity(opacity)

        project.addMapLayer(layer, False)
        root = project.layerTreeRoot()
        root.insertLayer(0, layer)
        layer_tree = root.findLayer(layer.id())
        if layer_tree:
            layer_tree.setItemVisibilityChecked(shown)

        if current_extent is not None:
            try:
                canvas = iface.mapCanvas()
                canvas.setExtent(current_extent)
                canvas.refresh()
            except Exception:  # nosec B110
                pass

        add_ee_layer_to_registry(name, ee_object, vis_params)
        _store_ee_layer_metadata(layer, ee_object, vis_params)
        return layer

    return _on_gui(_run)


def _add_ee_layer_nonblocking(
    *,
    iface: Any,
    ee_object: Any,
    vis_params: dict[str, Any],
    name: str,
) -> Any:
    """Add an EE layer while keeping slow EE calls off the QGIS GUI thread."""
    try:
        import qgis  # noqa: F401
    except Exception:
        from gee_data_catalogs.core.ee_utils import add_ee_layer

        return add_ee_layer(ee_object, vis_params, name)

    tile_url = _ee_tile_url(ee_object, vis_params)
    return _add_xyz_layer_on_gui(
        iface=iface,
        ee_object=ee_object,
        vis_params=vis_params,
        name=name,
        tile_url=tile_url,
    )


def _validate_added_layer(layer: Any, name: str) -> None:
    """Raise when the plugin returns an invalid QGIS layer."""
    if layer is None:
        raise ValueError(f"Failed to add Earth Engine layer to QGIS: {name}")
    is_valid = getattr(layer, "isValid", None)
    if callable(is_valid) and not is_valid():
        raise ValueError(f"Earth Engine layer was added but is invalid: {name}")


def _zoom_to_layer_area(
    *,
    iface: Any,
    layer: Any,
    bbox: list[float] | None = None,
) -> dict[str, Any]:
    """Best-effort zoom to an explicit bbox or the added layer extent."""

    def _run() -> dict[str, Any]:
        try:
            canvas = iface.mapCanvas()
        except Exception as exc:
            return {"success": False, "error": f"Map canvas unavailable: {exc}"}

        if canvas is None:
            return {"success": False, "error": "Map canvas unavailable."}

        try:
            if bbox is not None:
                west, south, east, north = bbox
                try:
                    from qgis.core import (  # type: ignore[import-not-found]
                        QgsCoordinateReferenceSystem,
                        QgsCoordinateTransform,
                        QgsProject,
                        QgsRectangle,
                    )

                    extent = QgsRectangle(west, south, east, north)
                    destination_crs = canvas.mapSettings().destinationCrs()
                    if destination_crs and destination_crs.authid() != "EPSG:4326":
                        transform = QgsCoordinateTransform(
                            QgsCoordinateReferenceSystem("EPSG:4326"),
                            destination_crs,
                            QgsProject.instance(),
                        )
                        extent = transform.transformBoundingBox(extent)
                except Exception:
                    extent = (west, south, east, north)
                canvas.setExtent(extent)
                canvas.refresh()
                return {"success": True, "target": "bbox", "bbox": bbox}

            extent_fn = getattr(layer, "extent", None)
            if callable(extent_fn):
                extent = extent_fn()
                is_empty = getattr(extent, "isEmpty", None)
                if not callable(is_empty) or not is_empty():
                    canvas.setExtent(extent)
                    canvas.refresh()
                    return {"success": True, "target": "layer_extent"}

            return {"success": False, "error": "No zoomable bbox or layer extent."}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    return _on_gui(_run)


def gee_data_catalogs_tools(
    iface: Any,
    plugin: Optional[Any] = None,
) -> list[Any]:
    """Return GeoAgent tools for the QGIS GEE Data Catalogs plugin."""
    if iface is None:
        return []

    def _ensure_plugin_catalog_dock() -> Any | None:
        """Return a live plugin catalog dock when a plugin instance was supplied."""
        if plugin is None:
            return None

        def _run() -> Any | None:
            dock = getattr(plugin, "_catalog_dock", None)
            if dock is None and hasattr(plugin, "_create_catalog_dock"):
                plugin._create_catalog_dock()
                dock = getattr(plugin, "_catalog_dock", None)
            if dock is not None:
                dock.show()
                dock.raise_()
            return dock

        return _on_gui(_run)

    @geo_tool(
        category="gee_data_catalogs",
        name="search_gee_datasets",
        available_in=("full", "fast"),
        requires_packages=("gee_data_catalogs",),
    )
    def search_gee_datasets(
        query: str = "",
        category: Optional[str] = None,
        data_type: Optional[str] = None,
        source: Optional[str] = None,
        max_results: int = 20,
        include_community: bool = True,
    ) -> dict[str, Any]:
        """Search official and community Google Earth Engine data catalogs."""
        from gee_data_catalogs.core.catalog_data import search_datasets

        results = search_datasets(
            query=query,
            category=category,
            data_type=data_type,
            source=source,
            include_community=include_community,
        )
        count = max(1, int(max_results))
        return {
            "count": len(results),
            "shown": min(len(results), count),
            "datasets": [_compact_dataset(item) for item in results[:count]],
        }

    @geo_tool(
        category="gee_data_catalogs",
        name="get_gee_dataset_info",
        available_in=("full", "fast"),
        requires_packages=("gee_data_catalogs",),
    )
    def get_gee_dataset_info(
        asset_id: str,
        include_community: bool = True,
    ) -> dict[str, Any]:
        """Return catalog metadata for a Google Earth Engine asset id."""
        from gee_data_catalogs.core.catalog_data import get_dataset_info

        dataset = get_dataset_info(asset_id, include_community=include_community)
        if dataset is None:
            return {"asset_id": asset_id, "found": False}
        out = _compact_dataset(dataset)
        out["found"] = True
        return out

    @geo_tool(
        category="gee_data_catalogs",
        name="summarize_gee_catalog",
        available_in=("full", "fast"),
        requires_packages=("gee_data_catalogs",),
    )
    def summarize_gee_catalog(include_community: bool = True) -> dict[str, Any]:
        """Summarize dataset counts by catalog category."""
        from gee_data_catalogs.core.catalog_data import get_catalog_data

        catalog = get_catalog_data(include_community=include_community)
        categories = []
        total = 0
        for category, payload in catalog.items():
            datasets = payload.get("datasets", [])
            total += len(datasets)
            categories.append({"category": category, "count": len(datasets)})
        return {"total_count": total, "categories": categories}

    @geo_tool(
        category="gee_data_catalogs",
        name="initialize_earth_engine",
        requires_confirmation=True,
        requires_packages=("gee_data_catalogs",),
    )
    def initialize_earth_engine(project_id: Optional[str] = None) -> dict[str, Any]:
        """Initialize Earth Engine using the plugin utility and saved project id."""
        from gee_data_catalogs.core.ee_utils import initialize_ee, is_ee_initialized

        project_to_use = project_id or _project_id_from_settings()
        initialize_ee(project=project_to_use)
        return {
            "success": is_ee_initialized(),
            "project_id": project_to_use,
        }

    @geo_tool(
        category="gee_data_catalogs",
        name="load_gee_dataset",
        requires_confirmation=True,
        long_running=True,
        requires_packages=("gee_data_catalogs", "ee"),
    )
    def load_gee_dataset(
        asset_id: str,
        layer_name: Optional[str] = None,
        asset_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        bbox: Optional[str] = None,
        bounds_collection_asset_id: Optional[str] = None,
        bounds_filter_property: Optional[str] = None,
        bounds_filter_value: Optional[str] = None,
        cloud_cover: Optional[int] = None,
        cloud_property: Optional[str] = None,
        reducer: str = "mosaic",
        bands: Optional[str] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        palette: Optional[str] = None,
        clip_collection_asset_id: Optional[str] = None,
        clip_filter_property: Optional[str] = None,
        clip_filter_value: Optional[str] = None,
        diagnose: bool = False,
    ) -> dict[str, Any]:
        """Load a GEE Image, ImageCollection, or FeatureCollection into QGIS.

        Args:
            asset_id: Earth Engine asset id.
            layer_name: Optional QGIS layer name.
            asset_type: Optional explicit type: Image, ImageCollection, or
                FeatureCollection. When omitted, the plugin detects the type.
            start_date: Optional ImageCollection start date in YYYY-MM-DD.
            end_date: Optional ImageCollection end date in YYYY-MM-DD.
            bbox: Optional WGS84 west,south,east,north region filter. For
                ImageCollections, this uses ``filterBounds`` when no
                ``bounds_collection_asset_id`` is provided. When a bounds
                FeatureCollection is provided, bbox is still used to zoom QGIS
                after loading if it is already known.
            bounds_collection_asset_id: Optional FeatureCollection asset id
                used with ImageCollection ``filterBounds`` for regional
                display. Prefer this over bbox coordinates when a specific
                FeatureCollection exists for the requested region.
                Example: TIGER/2018/States.
            bounds_filter_property: Optional property to filter the bounds
                FeatureCollection before calling ``filterBounds``. Example:
                NAME.
            bounds_filter_value: Optional value for ``bounds_filter_property``.
                Example: Tennessee.
            cloud_cover: Optional maximum cloud-cover value.
            cloud_property: Optional cloud-cover property name.
            reducer: ImageCollection compositing method: mosaic, median, mean,
                min, max, first, mode. ``mosaic`` is not an ee.Reducer; it is
                applied with ``ee.ImageCollection.mosaic()``.
            bands: Optional comma-separated visualization bands.
            min_value: Optional visualization minimum.
            max_value: Optional visualization maximum.
            palette: Optional comma-separated colors or palette entries.
            clip_collection_asset_id: Optional FeatureCollection asset id used
                to clip raster outputs with ``ee.Image.clipToCollection``.
                Clipping is computationally intensive; only use this when the
                user specifically asks to clip, crop, or mask data to the
                region. For ordinary regional display, use
                ``bounds_collection_asset_id`` when a specific
                FeatureCollection exists, otherwise use ``bbox``.
                Example: TIGER/2018/States.
            clip_filter_property: Optional property to filter the clipping
                FeatureCollection before clipping. Example: NAME.
            clip_filter_value: Optional value for ``clip_filter_property``.
                Example: Tennessee.
            diagnose: When true, evaluate lightweight Earth Engine diagnostics
                such as first-image and output band names. Leave false for
                normal layer loading to avoid blocking QGIS on ``getInfo``.
        """

        def _run() -> dict[str, Any]:
            import ee

            from gee_data_catalogs.core.ee_utils import (
                detect_asset_type,
                filter_image_collection,
                initialize_ee,
                is_ee_initialized,
            )

            if not is_ee_initialized():
                initialize_ee(project=_project_id_from_settings())

            resolved_type = asset_type or detect_asset_type(asset_id)
            name = layer_name or asset_id.split("/")[-1]
            parsed_bbox = _parse_bbox(bbox)
            bounds_collection = None
            clip_collection = None
            diagnostics: dict[str, Any] = {}
            if bounds_collection_asset_id:
                bounds_collection = ee.FeatureCollection(bounds_collection_asset_id)
                if bounds_filter_property and bounds_filter_value is not None:
                    bounds_collection = bounds_collection.filter(
                        ee.Filter.eq(
                            bounds_filter_property,
                            _coerce_filter_value(bounds_filter_value),
                        )
                    )
            if clip_collection_asset_id:
                clip_collection = ee.FeatureCollection(clip_collection_asset_id)
                if clip_filter_property and clip_filter_value is not None:
                    clip_collection = clip_collection.filter(
                        ee.Filter.eq(
                            clip_filter_property,
                            _coerce_filter_value(clip_filter_value),
                        )
                    )

            if resolved_type == "ImageCollection":
                collection = ee.ImageCollection(asset_id)
                is_dswx = _is_opera_dswx(asset_id)
                if parsed_bbox is not None:
                    diagnostics["bbox"] = parsed_bbox
                if bounds_collection_asset_id:
                    diagnostics["bounds"] = {
                        "collection_asset_id": bounds_collection_asset_id,
                        "filter_property": bounds_filter_property,
                        "filter_value": (
                            _coerce_filter_value(bounds_filter_value)
                            if bounds_filter_value is not None
                            else None
                        ),
                        "method": "ImageCollection.filterBounds",
                    }
                if is_dswx and cloud_cover is not None:
                    diagnostics["ignored_cloud_cover"] = cloud_cover
                    diagnostics["cloud_filter_reason"] = (
                        "OPERA DSWx examples mask WTR values >= 252 instead "
                        "of filtering by a generic cloud-cover property."
                    )

                if is_dswx:
                    diagnostics["filtering"] = "direct_dswx_filterDate_filterBounds"
                    if start_date or end_date:
                        collection = collection.filterDate(start_date, end_date)
                    if bounds_collection is not None:
                        collection = collection.filterBounds(bounds_collection)
                    elif parsed_bbox is not None:
                        collection = collection.filterBounds(
                            ee.Geometry.Rectangle(parsed_bbox)
                        )
                elif (
                    start_date
                    or end_date
                    or (parsed_bbox and bounds_collection is None)
                    or cloud_cover is not None
                ):
                    collection = filter_image_collection(
                        collection,
                        start_date=start_date,
                        end_date=end_date,
                        bbox=None if bounds_collection is not None else parsed_bbox,
                        cloud_cover=cloud_cover,
                        cloud_property=cloud_property or "CLOUDY_PIXEL_PERCENTAGE",
                    )
                if bounds_collection is not None and not is_dswx:
                    collection = collection.filterBounds(bounds_collection)
                if diagnose:
                    diagnostics["first_image_bands"] = _first_image_band_names(
                        collection
                    )
                if is_dswx:
                    default_method = _opera_dswx_default_composite_method(asset_id)
                    requested_method = _normalize_composite_method(
                        reducer, default=default_method, allow_mode=True
                    )
                    method = (
                        requested_method
                        if requested_method in {"max", "mode"}
                        else default_method
                    )
                    band_name = _opera_dswx_band_alias(bands)
                    ee_object, vis_params = _opera_dswx_render_image(
                        collection, band_name, method
                    )
                else:
                    method = _normalize_composite_method(
                        reducer, default="mosaic", allow_mode=True
                    )
                    ee_object = _composite_image_collection(collection, method)
                    vis_params = _build_vis_params(
                        bands=bands,
                        min_value=min_value,
                        max_value=max_value,
                        palette=palette,
                    )
            elif resolved_type == "FeatureCollection":
                ee_object = ee.FeatureCollection(asset_id)
                vis_params = _build_vis_params(
                    palette=palette,
                    for_feature_collection=True,
                )
            else:
                ee_object = ee.Image(asset_id)
                resolved_type = "Image"
                vis_params = _build_vis_params(
                    bands=bands,
                    min_value=min_value,
                    max_value=max_value,
                    palette=palette,
                )

            bbox_filter = (
                parsed_bbox
                if resolved_type == "ImageCollection" and parsed_bbox
                else None
            )
            bounds_filter = (
                {
                    "collection_asset_id": bounds_collection_asset_id,
                    "filter_property": bounds_filter_property,
                    "filter_value": (
                        _coerce_filter_value(bounds_filter_value)
                        if bounds_filter_value is not None
                        else None
                    ),
                    "method": "ImageCollection.filterBounds",
                }
                if (resolved_type == "ImageCollection" and bounds_collection_asset_id)
                else None
            )
            if clip_collection is not None and resolved_type != "FeatureCollection":
                ee_object = ee.Image(ee_object).clipToCollection(clip_collection)

            display_name = name[:50]
            rendered_band = (
                band_name
                if resolved_type == "ImageCollection" and _is_opera_dswx(asset_id)
                else None
            )
            earth_engine_python_snippet = _build_load_gee_dataset_snippet(
                asset_id=asset_id,
                resolved_type=resolved_type,
                start_date=start_date,
                end_date=end_date,
                bbox=parsed_bbox,
                cloud_cover=None if _is_opera_dswx(asset_id) else cloud_cover,
                cloud_property=cloud_property,
                bounds_collection_asset_id=bounds_collection_asset_id,
                bounds_filter_property=bounds_filter_property,
                bounds_filter_value=bounds_filter_value,
                method=method if resolved_type == "ImageCollection" else None,
                bands=bands,
                min_value=min_value,
                max_value=max_value,
                palette=palette,
                vis_params=vis_params,
                layer_name=display_name,
                rendered_band=rendered_band,
                clip_collection_asset_id=clip_collection_asset_id,
                clip_filter_property=clip_filter_property,
                clip_filter_value=clip_filter_value,
            )
            if diagnose and resolved_type != "FeatureCollection":
                diagnostics["output_bands"] = _image_band_names(ee_object)
                if diagnostics["output_bands"] == []:
                    return {
                        "success": False,
                        "asset_id": asset_id,
                        "asset_type": resolved_type,
                        "layer_name": display_name,
                        "error": (
                            "Earth Engine output image has no bands after "
                            "filtering and compositing."
                        ),
                        "diagnostics": diagnostics,
                    }
            try:
                layer = _add_ee_layer_nonblocking(
                    iface=iface,
                    ee_object=ee_object,
                    vis_params=vis_params,
                    name=display_name,
                )
                _validate_added_layer(layer, display_name)
                zoom_result = _zoom_to_layer_area(
                    iface=iface,
                    layer=layer,
                    bbox=bbox_filter,
                )
            except Exception as exc:
                try:
                    _raise_layer_error(
                        exc,
                        asset_id=asset_id,
                        layer_name=display_name,
                        diagnostics=diagnostics,
                    )
                except RuntimeError as layer_error:
                    return {
                        "success": False,
                        "asset_id": asset_id,
                        "asset_type": resolved_type,
                        "layer_name": display_name,
                        "error": str(layer_error),
                        "failure_stage": "qgis_layer_add",
                        "diagnostics": diagnostics,
                        "bbox": bbox_filter,
                        "bounds": bounds_filter,
                        "earth_engine_python_snippet": earth_engine_python_snippet,
                        "composite_method": (
                            method if resolved_type == "ImageCollection" else None
                        ),
                        "requested_reducer": (
                            reducer if resolved_type == "ImageCollection" else None
                        ),
                        "rendered_band": rendered_band,
                        "vis_params": vis_params,
                    }
            try:
                _on_gui(lambda: iface.mapCanvas().refresh())
            except Exception:
                pass
            return {
                "success": True,
                "asset_id": asset_id,
                "asset_type": resolved_type,
                "layer_name": display_name,
                "composite_method": (
                    method if resolved_type == "ImageCollection" else None
                ),
                "requested_reducer": (
                    reducer if resolved_type == "ImageCollection" else None
                ),
                "rendered_band": rendered_band,
                "diagnostics": diagnostics,
                "bbox": bbox_filter,
                "bounds": bounds_filter,
                "zoom": zoom_result,
                "earth_engine_python_snippet": earth_engine_python_snippet,
                "vis_params": vis_params,
                "clip": (
                    {
                        "collection_asset_id": clip_collection_asset_id,
                        "filter_property": clip_filter_property,
                        "filter_value": clip_filter_value,
                        "method": "ee.Image.clipToCollection",
                    }
                    if clip_collection_asset_id
                    else None
                ),
            }

        return _run()

    @geo_tool(
        category="gee_data_catalogs",
        name="calculate_gee_normalized_difference",
        requires_confirmation=True,
        long_running=True,
        requires_packages=("gee_data_catalogs", "ee"),
    )
    def calculate_gee_normalized_difference(
        asset_id: str,
        positive_band: str,
        negative_band: str,
        index_name: str = "NDVI",
        layer_name: Optional[str] = None,
        asset_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        bbox: Optional[str] = None,
        bounds_collection_asset_id: Optional[str] = None,
        bounds_filter_property: Optional[str] = None,
        bounds_filter_value: Optional[str] = None,
        cloud_cover: Optional[int] = None,
        cloud_property: Optional[str] = None,
        reducer: str = "median",
        min_value: float = -1.0,
        max_value: float = 1.0,
        palette: Optional[str] = None,
        clip_collection_asset_id: Optional[str] = None,
        clip_filter_property: Optional[str] = None,
        clip_filter_value: Optional[str] = None,
    ) -> dict[str, Any]:
        """Calculate and load a normalized difference index image.

        The index is computed server-side as
        ``ee.Image.normalizedDifference([positive_band, negative_band])`` and
        renamed to ``index_name`` before display.

        Args:
            asset_id: Earth Engine Image or ImageCollection asset id.
            positive_band: Numerator-positive band. NDVI example: NIR band.
            negative_band: Numerator-negative band. NDVI example: red band.
            index_name: Output band/index name, such as NDVI, NDWI, MNDWI, NBR.
            layer_name: Optional QGIS layer name.
            asset_type: Optional explicit type: Image or ImageCollection.
            start_date: Optional ImageCollection start date in YYYY-MM-DD.
            end_date: Optional ImageCollection end date in YYYY-MM-DD.
            bbox: Optional WGS84 west,south,east,north ImageCollection
                ``filterBounds`` region filter used when no
                ``bounds_collection_asset_id`` is provided. When a bounds
                FeatureCollection is provided, bbox is still used to zoom QGIS
                after loading if it is already known.
            bounds_collection_asset_id: Optional FeatureCollection asset id
                used with ImageCollection ``filterBounds`` for regional index
                display. Prefer this over bbox coordinates when a specific
                FeatureCollection exists for the requested region.
                Example: TIGER/2018/States.
            bounds_filter_property: Optional property to filter the bounds
                FeatureCollection before calling ``filterBounds``. Example:
                NAME.
            bounds_filter_value: Optional value for ``bounds_filter_property``.
                Example: Tennessee.
            cloud_cover: Optional maximum cloud-cover value.
            cloud_property: Optional cloud-cover property name.
            reducer: ImageCollection compositing method: mosaic, median, mean,
                min, max, first, mode. ``mosaic`` is not an ee.Reducer; it is
                applied with ``ee.ImageCollection.mosaic()``.
            min_value: Visualization minimum. Defaults to -1.
            max_value: Visualization maximum. Defaults to 1.
            palette: Optional comma-separated colors. Defaults by index type.
            clip_collection_asset_id: Optional FeatureCollection asset id used
                to clip the index image with ``ee.Image.clipToCollection``.
                Clipping is computationally intensive; only use this when the
                user specifically asks to clip, crop, or mask data to the
                region. For ordinary regional display, use
                ``bounds_collection_asset_id`` when a specific
                FeatureCollection exists, otherwise use ``bbox``.
            clip_filter_property: Optional property to filter the clipping
                FeatureCollection before clipping. Example: NAME.
            clip_filter_value: Optional value for ``clip_filter_property``.
                Example: Tennessee.

        Common band examples:
            Sentinel-2 or HLS S30 NDVI: positive_band=B8, negative_band=B4.
            Landsat 8/9 NDVI: positive_band=SR_B5, negative_band=SR_B4.
            Sentinel-2 NDWI: positive_band=B3, negative_band=B8.
            Sentinel-2 MNDWI: positive_band=B3, negative_band=B11.
            Sentinel-2 NBR: positive_band=B8, negative_band=B12.
        """

        def _run() -> dict[str, Any]:
            import ee

            from gee_data_catalogs.core.ee_utils import (
                detect_asset_type,
                filter_image_collection,
                initialize_ee,
                is_ee_initialized,
            )

            if not is_ee_initialized():
                initialize_ee(project=_project_id_from_settings())

            resolved_type = asset_type or detect_asset_type(asset_id)
            if resolved_type == "FeatureCollection":
                raise ValueError(
                    "Normalized difference indexes require an Image or ImageCollection."
                )

            parsed_bbox = _parse_bbox(bbox)
            bounds_collection = None
            clip_collection = None
            if bounds_collection_asset_id:
                bounds_collection = ee.FeatureCollection(bounds_collection_asset_id)
                if bounds_filter_property and bounds_filter_value is not None:
                    bounds_collection = bounds_collection.filter(
                        ee.Filter.eq(
                            bounds_filter_property,
                            _coerce_filter_value(bounds_filter_value),
                        )
                    )
            if clip_collection_asset_id:
                clip_collection = ee.FeatureCollection(clip_collection_asset_id)
                if clip_filter_property and clip_filter_value is not None:
                    clip_collection = clip_collection.filter(
                        ee.Filter.eq(
                            clip_filter_property,
                            _coerce_filter_value(clip_filter_value),
                        )
                    )

            if resolved_type == "ImageCollection":
                collection = ee.ImageCollection(asset_id)
                if (
                    start_date
                    or end_date
                    or (parsed_bbox and bounds_collection is None)
                    or cloud_cover is not None
                ):
                    collection = filter_image_collection(
                        collection,
                        start_date=start_date,
                        end_date=end_date,
                        bbox=None if bounds_collection is not None else parsed_bbox,
                        cloud_cover=cloud_cover,
                        cloud_property=cloud_property or "CLOUDY_PIXEL_PERCENTAGE",
                    )
                if bounds_collection is not None:
                    collection = collection.filterBounds(bounds_collection)
                method = _normalize_composite_method(reducer, default="median")
                source_image = _composite_image_collection(collection, method)
                resolved_type = "ImageCollection"
            else:
                source_image = ee.Image(asset_id)
                resolved_type = "Image"

            output_name = index_name.strip() or "ND"
            index_image = source_image.normalizedDifference(
                [positive_band, negative_band]
            ).rename(output_name)
            if clip_collection is not None:
                index_image = ee.Image(index_image).clipToCollection(clip_collection)

            vis_params = _build_vis_params(
                bands=output_name,
                min_value=min_value,
                max_value=max_value,
                palette=palette or _default_index_palette(output_name),
            )
            name = layer_name or f"{asset_id.split('/')[-1]} {output_name}"
            display_name = name[:50]
            earth_engine_python_snippet = _build_normalized_difference_snippet(
                asset_id=asset_id,
                resolved_type=resolved_type,
                start_date=start_date,
                end_date=end_date,
                bbox=parsed_bbox,
                cloud_cover=cloud_cover,
                cloud_property=cloud_property,
                bounds_collection_asset_id=bounds_collection_asset_id,
                bounds_filter_property=bounds_filter_property,
                bounds_filter_value=bounds_filter_value,
                method=method if resolved_type == "ImageCollection" else None,
                positive_band=positive_band,
                negative_band=negative_band,
                output_name=output_name,
                vis_params=vis_params,
                layer_name=display_name,
                clip_collection_asset_id=clip_collection_asset_id,
                clip_filter_property=clip_filter_property,
                clip_filter_value=clip_filter_value,
            )
            layer = _add_ee_layer_nonblocking(
                iface=iface,
                ee_object=index_image,
                vis_params=vis_params,
                name=display_name,
            )
            _validate_added_layer(layer, display_name)
            zoom_result = _zoom_to_layer_area(
                iface=iface,
                layer=layer,
                bbox=parsed_bbox,
            )
            try:
                _on_gui(lambda: iface.mapCanvas().refresh())
            except Exception:
                pass
            return {
                "success": True,
                "asset_id": asset_id,
                "asset_type": resolved_type,
                "layer_name": display_name,
                "index_name": output_name,
                "formula": f"({positive_band} - {negative_band}) / "
                f"({positive_band} + {negative_band})",
                "bands": [positive_band, negative_band],
                "composite_method": (
                    method if resolved_type == "ImageCollection" else None
                ),
                "requested_reducer": (
                    reducer if resolved_type == "ImageCollection" else None
                ),
                "vis_params": vis_params,
                "bbox": parsed_bbox,
                "bounds": (
                    {
                        "collection_asset_id": bounds_collection_asset_id,
                        "filter_property": bounds_filter_property,
                        "filter_value": (
                            _coerce_filter_value(bounds_filter_value)
                            if bounds_filter_value is not None
                            else None
                        ),
                        "method": "ImageCollection.filterBounds",
                    }
                    if (
                        resolved_type == "ImageCollection"
                        and bounds_collection_asset_id
                    )
                    else None
                ),
                "zoom": zoom_result,
                "earth_engine_python_snippet": earth_engine_python_snippet,
                "clip": (
                    {
                        "collection_asset_id": clip_collection_asset_id,
                        "filter_property": clip_filter_property,
                        "filter_value": clip_filter_value,
                        "method": "ee.Image.clipToCollection",
                    }
                    if clip_collection_asset_id
                    else None
                ),
            }

        return _run()

    @geo_tool(
        category="gee_data_catalogs",
        name="list_loaded_gee_layers",
        available_in=("full", "fast"),
        requires_packages=("gee_data_catalogs",),
    )
    def list_loaded_gee_layers() -> dict[str, Any]:
        """List currently registered Earth Engine layers in the QGIS project."""
        layers = _loaded_gee_layer_records()
        return {"count": len(layers), "layers": layers}

    @geo_tool(
        category="gee_data_catalogs",
        name="run_gee_python_snippet",
        requires_confirmation=True,
        long_running=True,
        requires_packages=("gee_data_catalogs", "ee"),
    )
    def run_gee_python_snippet(
        code: str,
        description: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run a constrained Earth Engine Python snippet and add QGIS layers.

        Use this when the user asks for an Earth Engine operation that is not
        covered by a dedicated GeoAgent tool. The snippet may use ``ee`` and a
        small geemap-compatible map surface: ``m.add_layer(...)``,
        ``m.addLayer(...)``, ``add_layer(...)``, ``get_ee_layer(name)``, and
        ``list_ee_layers()``. It must add at least one layer.

        Args:
            code: Short Earth Engine Python snippet to run.
            description: Optional plain-language summary for confirmation.
        """
        if not code or not code.strip():
            return {
                "success": False,
                "error": "No Earth Engine Python snippet was provided.",
                "earth_engine_python_snippet": code or "",
            }

        try:
            result = _run_gee_python_snippet_impl(iface=iface, code=code.strip())
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "description": description,
                "earth_engine_python_snippet": code.strip(),
            }
        result["description"] = description
        return result

    @geo_tool(
        category="gee_data_catalogs",
        name="calculate_gee_layer_statistics",
        requires_confirmation=True,
        long_running=True,
        requires_packages=("gee_data_catalogs", "ee"),
    )
    def calculate_gee_layer_statistics(
        layer_name: str,
        band: Optional[str] = None,
        statistics: Optional[str] = "mean",
        scale: float = 1000.0,
        max_pixels: float = 100000000,
        best_effort: bool = True,
        tile_scale: float = 4.0,
        timeout_seconds: float = 60.0,
        region_collection_asset_id: Optional[str] = None,
        region_filter_property: Optional[str] = None,
        region_filter_value: Optional[str] = None,
    ) -> dict[str, Any]:
        """Calculate bounded server-side statistics for a loaded EE image layer.

        Use this for scalar questions such as mean elevation, min/max slope, or
        summary statistics of a registered Earth Engine raster. Defaults are
        intentionally coarse and best-effort so large regions such as the
        continental United States return instead of blocking QGIS indefinitely.

        Args:
            layer_name: Registered Earth Engine layer name to analyze.
            band: Optional band to select before reducing.
            statistics: Comma-separated statistics: mean, min, max, median,
                stdDev, sum, count.
            scale: Reduction scale in meters. Defaults to 1000 m for large
                area responsiveness.
            max_pixels: Earth Engine maxPixels value.
            best_effort: Whether Earth Engine may coarsen scale to fit limits.
            tile_scale: Earth Engine reduceRegion tileScale.
            timeout_seconds: Request deadline for the blocking EE evaluation.
            region_collection_asset_id: Optional FeatureCollection region. If
                omitted, the loaded image geometry/mask is used.
            region_filter_property: Optional filter property for the region FC.
            region_filter_value: Optional filter value for the region FC.
        """
        try:
            return _calculate_gee_layer_statistics_impl(
                layer_name=layer_name,
                band=band,
                statistics=statistics,
                scale=scale,
                max_pixels=max_pixels,
                best_effort=best_effort,
                tile_scale=tile_scale,
                timeout_seconds=timeout_seconds,
                region_collection_asset_id=region_collection_asset_id,
                region_filter_property=region_filter_property,
                region_filter_value=region_filter_value,
            )
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "layer_name": layer_name,
                "band": band,
                "statistics": statistics,
                "scale": scale,
                "max_pixels": max_pixels,
                "best_effort": best_effort,
                "tile_scale": tile_scale,
                "timeout_seconds": timeout_seconds,
                "hint": (
                    "For large regions, retry with a coarser scale such as "
                    "2000 or 5000 meters, or provide a smaller region."
                ),
            }

    @geo_tool(
        category="gee_data_catalogs",
        name="open_gee_catalog_panel",
        available_in=("full", "fast"),
    )
    def open_gee_catalog_panel(tab: str = "Search") -> dict[str, Any]:
        """Open the GEE Data Catalogs panel and optionally select a tab."""
        dock = _ensure_plugin_catalog_dock()
        if dock is None:
            return {"success": False, "error": "Plugin instance is not available."}

        def _run() -> dict[str, Any]:
            names = []
            tab_widget = getattr(dock, "tab_widget", None)
            if tab_widget is not None:
                for i in range(tab_widget.count()):
                    names.append(tab_widget.tabText(i))
                    if tab_widget.tabText(i).lower() == tab.lower():
                        tab_widget.setCurrentIndex(i)
            return {"success": True, "tabs": names}

        return _on_gui(_run)

    @geo_tool(
        category="gee_data_catalogs",
        name="configure_gee_dataset_load",
        available_in=("full", "fast"),
    )
    def configure_gee_dataset_load(
        asset_id: str,
        layer_name: Optional[str] = None,
        bands: Optional[str] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        palette: Optional[str] = None,
    ) -> dict[str, Any]:
        """Populate the plugin Load tab for a dataset without loading it."""
        dock = _ensure_plugin_catalog_dock()
        if dock is None:
            return {"success": False, "error": "Plugin instance is not available."}

        def _run() -> dict[str, Any]:
            dock.dataset_id_input.setText(asset_id)
            dock.layer_name_input.setText(layer_name or asset_id.split("/")[-1])
            if bands is not None:
                dock.bands_input.setText(bands)
            if min_value is not None:
                dock.vis_min_input.setText(str(min_value))
            if max_value is not None:
                dock.vis_max_input.setText(str(max_value))
            if palette is not None:
                dock.palette_input.setText(palette)
            dock.tab_widget.setCurrentIndex(3)
            return {"success": True, "asset_id": asset_id}

        return _on_gui(_run)

    return [
        search_gee_datasets,
        get_gee_dataset_info,
        summarize_gee_catalog,
        initialize_earth_engine,
        load_gee_dataset,
        calculate_gee_normalized_difference,
        list_loaded_gee_layers,
        run_gee_python_snippet,
        calculate_gee_layer_statistics,
        open_gee_catalog_panel,
        configure_gee_dataset_load,
    ]


__all__ = ["gee_data_catalogs_tools"]
