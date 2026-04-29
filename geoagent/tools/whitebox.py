"""Tool adapter for WhiteboxTools in QGIS.

The adapter intentionally exposes a small broker surface instead of one
GeoAgent tool per WhiteboxTools command. WhiteboxTools is resolved lazily so
GeoAgent remains import-safe outside environments where ``whitebox`` or QGIS
are installed.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

from geoagent.core.decorators import geo_tool
from geoagent.tools._qt_marshal import run_on_qt_gui_thread

RASTER_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".dep",
    ".flt",
    ".sdat",
    ".rdc",
    ".asc",
}
VECTOR_EXTENSIONS = {".shp", ".geojson", ".gpkg", ".json", ".kml"}


def _on_gui(fn: Any) -> Any:
    """Run ``fn`` on the Qt GUI thread when QGIS is available."""
    return run_on_qt_gui_thread(fn)


def _normalize_key(value: Any) -> str:
    """Return a stable lookup key for parameter names and flags."""
    text = str(value).strip().lstrip("-").lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _safe_stem(value: str, fallback: str = "whitebox_output") -> str:
    """Return a filesystem-safe output stem."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned or fallback


def _project_from_iface(iface: Any, project: Optional[Any]) -> Any:
    """Return the configured project or resolve it from the QGIS iface.

    QGIS APIs (``iface.project()`` and ``QgsProject.instance()``) are not
    thread-safe, so the lookup is marshalled onto the Qt GUI thread.
    """
    if project is not None:
        return project

    def _resolve() -> Any:
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

    return run_on_qt_gui_thread(_resolve)


def _project_output_dir(project: Any) -> str:
    """Choose a stable directory for generated Whitebox outputs.

    Project path getters (``homePath``, ``absolutePath``, ``fileName``) are
    QGIS API calls and are marshalled onto the Qt GUI thread.
    """

    def _read_paths() -> str | None:
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
        return None

    resolved = run_on_qt_gui_thread(_read_paths)
    if resolved:
        return resolved
    out_dir = Path(tempfile.gettempdir()) / "geoagent_whitebox"
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir)


def _parameter_type(param: dict[str, Any]) -> Any:
    """Return the Whitebox parameter type payload."""
    return param.get("parameter_type") or param.get("type")


def _type_mapping(param: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return ``(kind, data_type)`` for Whitebox parameter metadata."""
    ptype = _parameter_type(param)
    if isinstance(ptype, dict) and ptype:
        key = next(iter(ptype.keys()))
        value = ptype[key]
        return str(key), str(value) if value is not None else None
    if isinstance(ptype, str):
        return ptype, None
    return None, None


def _is_existing_file_param(param: dict[str, Any]) -> bool:
    """Return True when the parameter expects an existing file input."""
    kind, _data_type = _type_mapping(param)
    return kind == "ExistingFile"


def _is_output_file_param(param: dict[str, Any]) -> bool:
    """Return True when the parameter expects a new output file."""
    kind, _data_type = _type_mapping(param)
    return kind == "NewFile"


def _is_raster_param(param: dict[str, Any]) -> bool:
    """Return True when metadata indicates a raster parameter."""
    _kind, data_type = _type_mapping(param)
    return str(data_type or "").lower() == "raster"


def _is_vector_param(param: dict[str, Any]) -> bool:
    """Return True when metadata indicates a vector parameter."""
    _kind, data_type = _type_mapping(param)
    return str(data_type or "").lower() == "vector"


def _required(param: dict[str, Any]) -> bool:
    """Return True when a Whitebox parameter is required."""
    return not bool(param.get("optional", False))


def _flags(param: dict[str, Any]) -> list[str]:
    """Return parameter CLI flags."""
    flags = param.get("flags") or []
    if isinstance(flags, str):
        return [flags]
    return [str(flag) for flag in flags if str(flag).strip()]


def _preferred_flag(param: dict[str, Any]) -> str:
    """Choose the long CLI flag when Whitebox exposes one."""
    flags = _flags(param)
    for flag in flags:
        if flag.startswith("--"):
            return flag
    if flags:
        return flags[0]
    name = _normalize_key(param.get("name", "parameter"))
    return f"--{name}"


def _param_keys(param: dict[str, Any]) -> set[str]:
    """Return accepted lookup keys for a Whitebox parameter."""
    keys = {_normalize_key(param.get("name", ""))}
    keys.update(_normalize_key(flag) for flag in _flags(param))
    return {key for key in keys if key}


def _canonical_param_name(param: dict[str, Any]) -> str:
    """Return a compact parameter name for output payloads."""
    for flag in _flags(param):
        if flag.startswith("--"):
            return _normalize_key(flag)
    keys = sorted(_param_keys(param))
    return keys[0] if keys else "parameter"


def _load_tool_parameters(wbt: Any, tool_name: str) -> dict[str, Any]:
    """Return parsed Whitebox parameter metadata for a tool."""
    text = wbt.tool_parameters(tool_name)
    if isinstance(text, dict):
        payload = text
    else:
        raw = str(text)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            preview = raw[:240] if raw else "<empty response>"
            raise ValueError(
                "WhiteboxTools returned invalid parameter metadata JSON for "
                f"{tool_name!r}. Try searching the tool again, verify the tool "
                "name, or update/reinstall WhiteboxTools. "
                f"Metadata preview: {preview}"
            ) from exc
    params = payload.get("parameters", [])
    if not isinstance(params, list):
        params = []
    payload["parameters"] = params
    return payload


def _toolbox(wbt: Any, tool_name: str) -> str | None:
    """Return a compact toolbox/category name for a Whitebox tool."""
    try:
        value = str(wbt.toolbox(tool_name)).strip()
    except Exception:
        return None
    return value or None


def _layer_source_from_name(project: Any, layer_name: str) -> str | None:
    """Resolve a QGIS layer display name to a file-backed data source.

    QGIS layer/project APIs are not thread-safe, so the actual reads are
    marshalled onto the Qt GUI thread. Outside QGIS this degrades to a direct
    call.
    """

    def _read_layer_source_path() -> str | None:
        try:
            matches = project.mapLayersByName(layer_name)
        except Exception:
            matches = []
        if not matches:
            return None
        layer = matches[0]
        try:
            source = str(layer.source())
        except Exception:
            source = ""
        return source.split("|", 1)[0]

    source_path = run_on_qt_gui_thread(_read_layer_source_path)
    if source_path and os.path.exists(source_path):
        return source_path
    if source_path:
        raise ValueError(
            f"QGIS layer '{layer_name}' is not backed by a local file. "
            "Export the layer to a file or provide a file path for WhiteboxTools."
        )
    return None


def _active_layer_name(iface: Any) -> str:
    """Return the active QGIS layer name."""

    def _read_active_layer_name() -> str:
        layer = iface.activeLayer()
        if layer is None:
            raise ValueError("No active QGIS layer is selected.")
        if hasattr(layer, "name"):
            return str(layer.name())
        return str(layer)

    return run_on_qt_gui_thread(_read_active_layer_name)


def _resolve_existing_file(value: Any, project: Any) -> str:
    """Resolve a file input, accepting either a path or QGIS layer name."""
    text = str(value).strip()
    if not text:
        raise ValueError("Existing file parameter cannot be empty.")
    source_path = text.split("|", 1)[0]
    if os.path.exists(source_path):
        return source_path
    layer_path = _layer_source_from_name(project, text)
    if layer_path is not None:
        return layer_path
    raise ValueError(
        f"Could not resolve existing file '{text}'. Provide a valid existing "
        "file path or the name of a QGIS layer backed by a local file."
    )


def _default_output_path(tool_name: str, param: dict[str, Any], project: Any) -> str:
    """Create a default output path for a missing required output."""
    out_dir = Path(_project_output_dir(project))
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".shp" if _is_vector_param(param) else ".tif"
    stem = _safe_stem(f"{tool_name}_{_canonical_param_name(param)}")
    candidate = out_dir / f"{stem}{suffix}"
    index = 2
    while candidate.exists():
        candidate = out_dir / f"{stem}_{index}{suffix}"
        index += 1
    return str(candidate)


def _coerce_cli_value(value: Any) -> str:
    """Convert Python values to WhiteboxTools CLI argument text."""
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _blank_to_none(value: Any) -> Any:
    """Normalize model-supplied blank optional values to ``None``."""
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _optional_nonzero_float(value: Any) -> float | None:
    """Return a float only when an optional numeric value is meaningful."""
    value = _blank_to_none(value)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number == 0:
        return None
    return number


def _format_arg(flag: str, value: Any) -> str | None:
    """Format a WhiteboxTools CLI argument."""
    if isinstance(value, bool):
        if value:
            return flag
        return f"{flag}=false"
    if value is None or value == "":
        return None
    return f"{flag}={_coerce_cli_value(value)}"


def _build_run_args(
    *,
    wbt: Any,
    tool_name: str,
    parameters: dict[str, Any],
    project: Any,
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """Build WhiteboxTools CLI args from JSON parameter metadata."""
    metadata = _load_tool_parameters(wbt, tool_name)
    params = metadata.get("parameters", [])
    normalized_inputs = {
        _normalize_key(key): value for key, value in parameters.items()
    }
    consumed: set[str] = set()
    args: list[str] = []
    outputs: list[dict[str, Any]] = []

    if not params:
        for key, value in parameters.items():
            arg = _format_arg(f"--{_normalize_key(key)}", value)
            if arg is not None:
                args.append(arg)
        return args, outputs, []

    known_keys: set[str] = set()
    for param in params:
        keys = _param_keys(param)
        known_keys.update(keys)
        found_key = next((key for key in keys if key in normalized_inputs), None)
        value = normalized_inputs.get(found_key) if found_key else None
        if found_key:
            consumed.add(found_key)
        if (value is None or value == "") and _is_output_file_param(param):
            if _required(param):
                value = _default_output_path(tool_name, param, project)
            else:
                continue
        elif value is None or value == "":
            if _required(param):
                raise ValueError(
                    f"Missing required WhiteboxTools parameter: "
                    f"{param.get('name') or _preferred_flag(param)}"
                )
            continue

        if _is_existing_file_param(param):
            value = _resolve_existing_file(value, project)

        arg = _format_arg(_preferred_flag(param), value)
        if arg is not None:
            args.append(arg)
        if _is_output_file_param(param):
            outputs.append(
                {
                    "parameter": _canonical_param_name(param),
                    "path": str(value),
                    "data_type": _type_mapping(param)[1],
                }
            )

    unknown = sorted(set(normalized_inputs) - consumed - known_keys)
    return args, outputs, unknown


def _add_output_layer(
    iface: Any,
    project_getter: Any,
    output_path: str,
    layer_name: str,
) -> dict[str, Any]:
    """Add a generated Whitebox output to QGIS."""

    def _run() -> dict[str, Any]:
        suffix = Path(output_path).suffix.lower()
        if suffix in RASTER_EXTENSIONS:
            if hasattr(iface, "addRasterLayer"):
                layer = iface.addRasterLayer(output_path, layer_name)
            else:  # pragma: no cover - QGIS-only fallback
                from qgis.core import QgsRasterLayer  # type: ignore[import-not-found]

                layer = QgsRasterLayer(output_path, layer_name)
                project_getter().addMapLayer(layer)
        elif suffix in VECTOR_EXTENSIONS:
            if hasattr(iface, "addVectorLayer"):
                layer = iface.addVectorLayer(output_path, layer_name, "ogr")
            else:  # pragma: no cover - QGIS-only fallback
                from qgis.core import QgsVectorLayer  # type: ignore[import-not-found]

                layer = QgsVectorLayer(output_path, layer_name, "ogr")
                project_getter().addMapLayer(layer)
        else:
            return {
                "path": output_path,
                "added": False,
                "reason": "Unsupported output extension for automatic QGIS loading.",
            }
        if hasattr(layer, "isValid") and not layer.isValid():
            return {"path": output_path, "added": False, "reason": "Layer is invalid."}
        try:
            iface.mapCanvas().refresh()
        except Exception:
            pass
        return {"path": output_path, "added": True, "layer_name": layer_name}

    return _on_gui(_run)


def _compact_tool(tool_name: str, description: str, toolbox: str | None = None) -> dict:
    """Return compact Whitebox tool search metadata."""
    payload = {"name": tool_name, "description": str(description)[:500]}
    if toolbox:
        payload["category"] = toolbox
    return payload


def whitebox_tools(iface: Any, project: Optional[Any] = None) -> list[Any]:
    """Return GeoAgent tools for running WhiteboxTools from QGIS."""
    if iface is None:
        return []

    def _wbt() -> Any:
        import whitebox

        instance = whitebox.WhiteboxTools()
        instance.verbose = False
        return instance

    def _project() -> Any:
        return _project_from_iface(iface, project)

    @geo_tool(
        category="whitebox",
        name="summarize_whitebox_tools",
        available_in=("full", "fast"),
        requires_packages=("whitebox",),
    )
    def summarize_whitebox_tools() -> dict[str, Any]:
        """Summarize the available WhiteboxTools backend."""
        wbt = _wbt()
        tools = wbt.list_tools()
        category_counts: dict[str, int] = {}
        for name in tools:
            category = _toolbox(wbt, name) or "Unknown"
            category_counts[category] = category_counts.get(category, 0) + 1
        return {
            "version": str(wbt.version()).strip(),
            "working_dir": str(wbt.get_working_dir()),
            "tool_count": len(tools),
            "category_counts": category_counts,
        }

    @geo_tool(
        category="whitebox",
        name="search_whitebox_tools",
        available_in=("full", "fast"),
        requires_packages=("whitebox",),
    )
    def search_whitebox_tools(
        query: str = "",
        category: Optional[str] = None,
        max_results: int = 20,
    ) -> dict[str, Any]:
        """Search WhiteboxTools by keyword and optional category/toolbox."""
        wbt = _wbt()
        keywords = [part for part in str(query).split() if part]
        tools = wbt.list_tools(keywords=keywords)
        category_filter = str(category).lower() if category else ""
        matches = []
        for name, description in tools.items():
            toolbox = _toolbox(wbt, name) if category_filter else None
            if category_filter and category_filter not in str(toolbox or "").lower():
                continue
            matches.append(_compact_tool(name, description, toolbox))
        limit = max(1, int(max_results))
        return {
            "count": len(matches),
            "shown": min(len(matches), limit),
            "tools": matches[:limit],
        }

    @geo_tool(
        category="whitebox",
        name="get_whitebox_tool_info",
        available_in=("full", "fast"),
        requires_packages=("whitebox",),
    )
    def get_whitebox_tool_info(tool_name: str) -> dict[str, Any]:
        """Return parameter metadata and help text for a WhiteboxTools command."""
        wbt = _wbt()
        metadata = _load_tool_parameters(wbt, tool_name)
        parameters = []
        for param in metadata.get("parameters", []):
            parameters.append(
                {
                    "name": param.get("name"),
                    "keys": sorted(_param_keys(param)),
                    "flags": _flags(param),
                    "description": param.get("description"),
                    "parameter_type": _parameter_type(param),
                    "default_value": param.get("default_value"),
                    "optional": bool(param.get("optional", False)),
                }
            )
        return {
            "tool_name": tool_name,
            "category": _toolbox(wbt, tool_name),
            "parameters": parameters,
            "help": str(wbt.tool_help(tool_name)).strip(),
        }

    @geo_tool(
        category="whitebox",
        name="run_whitebox_tool",
        requires_confirmation=True,
        long_running=True,
        requires_packages=("whitebox",),
    )
    def run_whitebox_tool(
        tool_name: str,
        parameters: dict[str, Any],
        working_dir: Optional[str] = None,
        add_outputs_to_qgis: bool = True,
        output_layer_name: Optional[str] = None,
        verbose: bool = False,
        compress_rasters: Optional[bool] = None,
        max_procs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Run a WhiteboxTools command and optionally load outputs into QGIS."""
        if not isinstance(parameters, dict):
            raise ValueError("parameters must be a dictionary.")
        wbt = _wbt()
        messages: list[str] = []
        if working_dir:
            wbt.set_working_dir(str(working_dir))
        wbt.verbose = bool(verbose)
        if compress_rasters is not None:
            wbt.set_compress_rasters(bool(compress_rasters))
        if max_procs is not None:
            wbt.set_max_procs(int(max_procs))

        def _resolve_args() -> tuple[list[str], list[dict[str, Any]], list[str]]:
            proj = _project()
            return _build_run_args(
                wbt=wbt,
                tool_name=tool_name,
                parameters=parameters,
                project=proj,
            )

        args, outputs, unknown = run_on_qt_gui_thread(_resolve_args)
        if unknown:
            raise ValueError(
                "Unknown WhiteboxTools parameters: "
                + ", ".join(unknown)
                + ". Call get_whitebox_tool_info first and use one of the listed keys."
            )

        def _callback(message: Any) -> None:
            text = str(message).strip()
            if text:
                messages.append(text)

        return_code = int(wbt.run_tool(tool_name, args, callback=_callback))
        layers_added = []
        if return_code == 0 and add_outputs_to_qgis:
            for index, output in enumerate(outputs, start=1):
                path = output["path"]
                if not os.path.exists(path):
                    layers_added.append(
                        {
                            "path": path,
                            "added": False,
                            "reason": "Output file was not found after tool run.",
                        }
                    )
                    continue
                name = output_layer_name or Path(path).stem
                if len(outputs) > 1 and output_layer_name:
                    name = f"{output_layer_name} {index}"
                layers_added.append(_add_output_layer(iface, _project, path, name))
        return {
            "success": return_code == 0,
            "return_code": return_code,
            "tool_name": tool_name,
            "args": args,
            "outputs": outputs,
            "layers_added": layers_added,
            "messages": messages[-40:],
        }

    @geo_tool(
        category="whitebox",
        name="run_whitebox_flow_accumulation",
        requires_confirmation=True,
        long_running=True,
        requires_packages=("whitebox",),
    )
    def run_whitebox_flow_accumulation(
        layer_name: Optional[str] = None,
        output_path: Optional[str] = None,
        output_layer_name: Optional[str] = None,
        tool_name: str = "d8_flow_accumulation",
        add_outputs_to_qgis: bool = True,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Run D8 flow accumulation on a DEM layer and load the raster output.

        Args:
            layer_name: QGIS DEM layer name. When omitted, uses the active
                layer.
            output_path: Optional raster output path.
            output_layer_name: Optional display name for the output layer.
            tool_name: Whitebox flow accumulation tool. Defaults to
                ``d8_flow_accumulation``.
            add_outputs_to_qgis: Whether to add the generated raster to QGIS.
            verbose: Whether WhiteboxTools should emit verbose progress.

        Returns:
            A compact dict with the Whitebox return code, output path, and
            QGIS load status.
        """
        selected_layer = layer_name or _active_layer_name(iface)
        name = output_layer_name or f"{selected_layer} flow accumulation"
        params: dict[str, Any] = {
            "input": selected_layer,
            "output": output_path,
        }
        if output_path is None:
            params.pop("output")
        return run_whitebox_tool.__wrapped__(
            tool_name,
            params,
            add_outputs_to_qgis=add_outputs_to_qgis,
            output_layer_name=name,
            verbose=verbose,
        )

    @geo_tool(
        category="whitebox",
        name="run_whitebox_fill_sinks",
        requires_confirmation=True,
        long_running=True,
        requires_packages=("whitebox",),
    )
    def run_whitebox_fill_sinks(
        layer_name: Optional[str] = None,
        output_path: Optional[str] = None,
        output_layer_name: Optional[str] = None,
        method: str = "fill_depressions",
        max_depth: Optional[float] = None,
        max_length: Optional[float] = None,
        flat_increment: Optional[float] = None,
        fill_pits: Optional[bool] = None,
        fix_flats: Optional[bool] = None,
        add_outputs_to_qgis: bool = True,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Fill sinks/depressions in a DEM layer and load the raster output.

        The default method is ``fill_depressions``, which is the appropriate
        preprocessing step for flow direction and flow accumulation workflows.
        Use ``breach_depressions`` only when the user explicitly asks to breach
        or carve depressions.

        Args:
            layer_name: QGIS DEM layer name. When omitted, uses the active
                layer.
            output_path: Optional raster output path.
            output_layer_name: Optional display name for the output layer.
            method: Whitebox method. Supported values are
                ``fill_depressions`` (default),
                ``fill_depressions_wang_and_liu``, and
                ``breach_depressions``.
            max_depth: Optional maximum breach/fill depth in z units.
            max_length: Optional maximum breach channel length in grid cells.
            flat_increment: Optional elevation increment for flat areas.
            fill_pits: Optional single-cell pit filling flag for breaching.
            fix_flats: Optional flat-area gradient flag for fill methods.
            add_outputs_to_qgis: Whether to add the generated raster to QGIS.
            verbose: Whether WhiteboxTools should emit verbose progress.

        Returns:
            A compact dict with the Whitebox return code, output path, and
            QGIS load status.
        """
        aliases = {
            "breach": "breach_depressions",
            "breach_depressions": "breach_depressions",
            "fill": "fill_depressions",
            "fill_sinks": "fill_depressions",
            "fill_depressions": "fill_depressions",
            "wang_and_liu": "fill_depressions_wang_and_liu",
            "fill_depressions_wang_and_liu": "fill_depressions_wang_and_liu",
        }
        tool_name = aliases.get(str(method).strip().lower())
        if tool_name is None:
            raise ValueError(
                "Unknown sink-filling method. Use 'breach_depressions', "
                "'fill_depressions', or 'fill_depressions_wang_and_liu'."
            )

        layer_name = _blank_to_none(layer_name)
        output_path = _blank_to_none(output_path)
        output_layer_name = _blank_to_none(output_layer_name)
        selected_layer = layer_name or _active_layer_name(iface)
        name = output_layer_name or f"{selected_layer} filled sinks"
        params: dict[str, Any] = {
            "dem": selected_layer,
            "output": output_path,
        }
        if output_path is None:
            params.pop("output")
        max_depth_value = _optional_nonzero_float(max_depth)
        flat_increment_value = _optional_nonzero_float(flat_increment)
        max_length_value = _optional_nonzero_float(max_length)
        if max_depth_value is not None:
            params["max_depth"] = max_depth_value
        if flat_increment_value is not None:
            params["flat_increment"] = flat_increment_value
        if tool_name == "breach_depressions":
            if max_length_value is not None:
                params["max_length"] = max_length_value
            if bool(fill_pits):
                params["fill_pits"] = True
        elif bool(fix_flats):
            params["fix_flats"] = True

        return run_whitebox_tool.__wrapped__(
            tool_name,
            params,
            add_outputs_to_qgis=add_outputs_to_qgis,
            output_layer_name=name,
            verbose=verbose,
        )

    @geo_tool(
        category="whitebox",
        name="run_whitebox_color_shaded_relief",
        requires_confirmation=True,
        long_running=True,
        requires_packages=("whitebox",),
    )
    def run_whitebox_color_shaded_relief(
        layer_name: Optional[str] = None,
        output_path: Optional[str] = None,
        output_layer_name: Optional[str] = None,
        palette: str = "atlas",
        altitude: float = 45.0,
        hillshade_weight: float = 0.5,
        brightness: float = 0.5,
        atmospheric: float = 0.0,
        reverse_palette: bool = False,
        zfactor: Optional[float] = None,
        full_mode: bool = False,
        add_outputs_to_qgis: bool = True,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Create color shaded relief from a DEM layer and load the raster.

        Args:
            layer_name: QGIS DEM layer name. When omitted, uses the active
                layer.
            output_path: Optional raster output path.
            output_layer_name: Optional display name for the output layer.
            palette: Whitebox palette name, e.g. ``atlas`` or ``viridis``.
            altitude: Illumination source altitude in degrees.
            hillshade_weight: Weight given to hillshade relative to relief.
            brightness: Brightness factor from 0.0 to 1.0.
            atmospheric: Atmospheric effects weight from 0.0 to 1.0.
            reverse_palette: Whether to reverse the palette.
            zfactor: Optional vertical unit conversion factor.
            full_mode: Whether to use full 360-degree hillshade mode.
            add_outputs_to_qgis: Whether to add the generated raster to QGIS.
            verbose: Whether WhiteboxTools should emit verbose progress.

        Returns:
            A compact dict with the Whitebox return code, output path, and
            QGIS load status.
        """
        selected_layer = layer_name or _active_layer_name(iface)
        name = output_layer_name or f"{selected_layer} color shaded relief"
        params: dict[str, Any] = {
            "dem": selected_layer,
            "output": output_path,
            "altitude": float(altitude),
            "hs_weight": float(hillshade_weight),
            "brightness": float(brightness),
            "atmospheric": float(atmospheric),
            "palette": palette,
            "reverse": bool(reverse_palette),
            "full_mode": bool(full_mode),
        }
        if output_path is None:
            params.pop("output")
        if zfactor is not None:
            params["zfactor"] = float(zfactor)
        return run_whitebox_tool.__wrapped__(
            "hypsometrically_tinted_hillshade",
            params,
            add_outputs_to_qgis=add_outputs_to_qgis,
            output_layer_name=name,
            verbose=verbose,
        )

    return [
        summarize_whitebox_tools,
        search_whitebox_tools,
        get_whitebox_tool_info,
        run_whitebox_tool,
        run_whitebox_flow_accumulation,
        run_whitebox_fill_sinks,
        run_whitebox_color_shaded_relief,
    ]


__all__ = ["whitebox_tools"]
