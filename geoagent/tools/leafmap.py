"""Tool adapters for ``leafmap.Map`` instances.

The :func:`leafmap_tools` factory returns a list of GeoAgent-decorated tools
bound to a single live ``leafmap.Map`` (or compatible mock) via closure. The
map object never crosses the LLM boundary — only its methods are invoked
under the hood.

These tools target the ``leafmap.maplibregl.Map`` API surface, but degrade
gracefully when methods are absent (e.g. older leafmap versions, or the
:class:`MockLeafmap` test stub) by trying the closest equivalent method name.
"""

from __future__ import annotations

from typing import Any, Optional

from geoagent.core.decorators import geo_tool
from geoagent.tools._map_state import map_state_from_widget

_NAME_KW_ALIASES = ("name", "layer_name")


def _layer_names(m: Any) -> list[str]:
    """Return the active layer names on ``m`` in display order.

    Handles three shapes:

    - leafmap.maplibregl.Map exposes ``layer_names`` (list[str]).
    - leafmap.Map / older leafmap exposes ``layers`` as a list of layer
      objects with a ``.name`` attribute.
    - :class:`MockLeafmap` (test stub) stores ``layers`` as a list of
      dicts with a ``"name"`` key.

    Args:
        m: A live map widget or compatible mock.

    Returns:
        The list of layer names in their existing order. Empty when the
        map exposes none of the recognised attributes.
    """
    names_attr = getattr(m, "layer_names", None)
    if isinstance(names_attr, list):
        return [str(n) for n in names_attr if n]
    layers = getattr(m, "layers", None)
    if isinstance(layers, list):
        out: list[str] = []
        for layer in layers:
            if isinstance(layer, dict):
                name = layer.get("name")
            else:
                name = getattr(layer, "name", None)
            if name:
                out.append(str(name))
        return out
    return []


def _resolve_layer_name(m: Any, query: str) -> tuple[Optional[str], list[str]]:
    """Resolve a user-supplied layer reference to an exact map layer name.

    Tries an exact match first. If the query does not match any layer
    exactly, falls back to a case-insensitive substring match against
    the live layer index. This lets the LLM say "Sentinel-2" without
    having had to call ``list_layers`` first to learn the full name
    ``Sentinel-2 RGB Knoxville 2024-07-15``.

    Args:
        m: A live map widget or compatible mock.
        query: The user / LLM's reference to a layer (full name,
            substring, or keyword).

    Returns:
        ``(resolved, candidates)``:

        - If exactly one layer matches, ``resolved`` is its full name
          and ``candidates`` is ``[resolved]``.
        - If multiple layers match, ``resolved`` is ``None`` and
          ``candidates`` lists every matching layer name so the caller
          can ask for disambiguation.
        - If no layer matches, ``resolved`` is ``None`` and
          ``candidates`` is empty.
    """
    names = _layer_names(m)
    if not names or not query:
        return None, []
    if query in names:
        return query, [query]
    needle = query.casefold()
    matches = [n for n in names if needle in n.casefold()]
    if len(matches) == 1:
        return matches[0], matches
    return None, matches


def _is_planetary_computer_url(url: str) -> bool:
    """Return ``True`` when ``url`` points at Planetary Computer blob storage.

    PC stores all hosted assets under ``*.blob.core.windows.net``. These
    URLs are SAS-protected and must be tiled via PC's hosted TiTiler
    (``add_stac_layer(..., titiler_endpoint="pc")``) rather than handed
    raw to the public TiTiler that leafmap's ``add_cog_layer`` defaults
    to.

    Args:
        url: A candidate raster URL.

    Returns:
        ``True`` when the URL host is a PC blob endpoint.
    """
    return isinstance(url, str) and "blob.core.windows.net" in url


def _safe_call(obj: Any, names: list[str], *args: Any, **kwargs: Any) -> Any:
    """Call the first available method on ``obj`` from ``names``.

    If the chosen method rejects a ``name=`` / ``layer_name=`` kwarg with a
    ``TypeError``, retry with the other alias before giving up. leafmap and
    anymap occasionally disagree on which spelling each helper expects, and
    the fallback methods we degrade to may use the alternate alias.

    Args:
        obj: Target object (a leafmap Map or similar).
        names: Candidate method names, tried in order.
        *args: Positional arguments forwarded to the chosen method.
        **kwargs: Keyword arguments forwarded to the chosen method.

    Returns:
        The method's return value.

    Raises:
        AttributeError: If none of ``names`` exists on ``obj``.
        TypeError: If the chosen method rejects the kwargs even after
            trying the alternative ``name`` / ``layer_name`` alias.
    """
    for n in names:
        if not hasattr(obj, n):
            continue
        method = getattr(obj, n)
        try:
            return method(*args, **kwargs)
        except TypeError:
            # Try the alternate spelling of the layer-name kwarg, if any.
            for primary, alternate in (
                ("name", "layer_name"),
                ("layer_name", "name"),
            ):
                if primary in kwargs and alternate not in kwargs:
                    alt = dict(kwargs)
                    alt[alternate] = alt.pop(primary)
                    try:
                        return method(*args, **alt)
                    except TypeError:
                        continue
            raise
    raise AttributeError(
        f"None of {names!r} found on {type(obj).__name__}; cannot perform action."
    )


def _layer_name(layer: Any) -> Optional[str]:
    """Return a layer name from dict or object-style layer records."""
    if isinstance(layer, dict):
        name = layer.get("name") or layer.get("layer_name") or layer.get("id")
    else:
        name = getattr(layer, "name", None) or getattr(layer, "layer_name", None)
        if callable(name):
            name = name()
    return str(name) if name else None


def _layer_attr(layer: Any, *keys: str, default: Any = None) -> Any:
    """Return the first matching layer attribute or mapping key."""
    for key in keys:
        if isinstance(layer, dict) and key in layer:
            return layer[key]
        value = getattr(layer, key, None)
        if value is not None:
            return value() if callable(value) else value
    return default


def _layer_bounds(layer: Any) -> Optional[list[list[float]]]:
    """Normalize layer bounds into southwest and northeast corners."""
    raw = _layer_attr(layer, "bounds", "bbox", "extent")
    if raw is None:
        return None
    if isinstance(raw, dict):
        if "bbox" in raw:
            raw = raw["bbox"]
        elif all(k in raw for k in ("west", "south", "east", "north")):
            raw = [raw["west"], raw["south"], raw["east"], raw["north"]]
        elif "_sw" in raw and "_ne" in raw:
            sw = raw["_sw"]
            ne = raw["_ne"]
            if isinstance(sw, dict) and isinstance(ne, dict):
                return [[sw["lng"], sw["lat"]], [ne["lng"], ne["lat"]]]
    if isinstance(raw, (list, tuple)):
        if len(raw) == 4 and all(isinstance(v, (int, float)) for v in raw):
            west, south, east, north = raw
            return [[float(west), float(south)], [float(east), float(north)]]
        if len(raw) == 2:
            return [list(raw[0]), list(raw[1])]  # type: ignore[index]
    return None


def _find_layer_entry(m: Any, name: str) -> Optional[Any]:
    """Find a named layer entry on a map object."""
    for layer in getattr(m, "layers", None) or []:
        if _layer_name(layer) == name:
            return layer
    return None


def _set_layer_attr(layer: Any, key: str, value: Any) -> bool:
    """Set a layer attribute through mapping, setter, or attribute access."""
    if isinstance(layer, dict):
        layer[key] = value
        return True
    setter = getattr(layer, f"set{key[:1].upper()}{key[1:]}", None)
    if callable(setter):
        setter(value)
        return True
    try:
        setattr(layer, key, value)
        return True
    except Exception:
        return False


def _layer_record(layer: Any) -> dict[str, Any]:
    """Convert a layer entry into a serializable summary record."""
    name = _layer_name(layer) or "Unnamed"
    record = {
        "name": name,
        "type": str(_layer_attr(layer, "type", default=type(layer).__name__)),
    }
    for key, aliases in {
        "visible": ("visible", "visibility", "shown"),
        "opacity": ("opacity",),
        "source": ("source", "data", "url"),
    }.items():
        value = _layer_attr(layer, *aliases)
        if value is not None:
            record[key] = value
    bounds = _layer_bounds(layer)
    if bounds is not None:
        record["bounds"] = bounds
    return record


def leafmap_tools(m: Any) -> list[Any]:
    """Build the leafmap tool set bound to a live map instance.

    Args:
        m: A ``leafmap.Map`` (or ``leafmap.maplibregl.Map``) instance, or any
            object exposing the same minimal API. Passing ``None`` returns an
            empty list.

    Returns:
        A list of Strands tool objects. Each tool captures ``m`` via closure.
    """
    if m is None:
        return []

    @geo_tool(
        category="map",
    )
    def list_layers() -> list[dict[str, Any]]:
        """List the layers currently on the map.

        Returns:
            A list of layer metadata dictionaries. Each record includes
            at least ``name`` and ``type`` and may include ``source``,
            ``visible``, ``opacity``, and ``bounds`` when the map exposes
            them.
        """
        layers = getattr(m, "layers", None) or []
        return [_layer_record(layer) for layer in layers]

    @geo_tool(
        category="map",
    )
    def add_layer(url: str, name: str, layer_type: str = "auto") -> str:
        """Add a layer to the map by URL.

        Args:
            url: The data URL (HTTP, S3, file path).
            name: Display name for the layer.
            layer_type: One of ``"raster"``, ``"vector"``, ``"cog"``,
                ``"pmtiles"``, ``"xyz"``, or ``"auto"``. ``"auto"`` infers
                from the URL extension.

        Returns:
            A status string describing the action taken.
        """
        kind = layer_type.lower()
        if kind == "auto":
            lower = url.lower()
            if any(lower.endswith(ext) for ext in (".geojson", ".json", ".shp")):
                kind = "vector"
            elif lower.endswith(".pmtiles"):
                kind = "pmtiles"
            elif lower.endswith(".tif") or lower.endswith(".tiff"):
                kind = "cog"
            else:
                kind = "raster"
        if kind in ("cog", "raster"):
            _safe_call(m, ["add_cog_layer", "add_raster"], url, name=name)
        elif kind == "vector":
            _safe_call(m, ["add_vector", "add_geojson"], url, layer_name=name)
        elif kind == "pmtiles":
            _safe_call(m, ["add_pmtiles", "add_pmtiles_layer"], url, name=name)
        elif kind == "xyz":
            _safe_call(m, ["add_xyz_tile_layer", "add_tile_layer"], url, name=name)
        else:
            return f"Unknown layer_type {layer_type!r}."
        return f"Added {kind} layer {name!r} from {url}."

    @geo_tool(
        category="map",
        requires_confirmation=True,
    )
    def remove_layer(name: str) -> str:
        """Remove a layer from the map by display name.

        Accepts either the layer's full name or a unique substring of
        it (case-insensitive). For example, ``remove_layer("Sentinel-2")``
        will remove a layer named ``"Sentinel-2 RGB Knoxville 2024-07-15"``
        as long as no other layer's name also contains "sentinel-2".
        This lets the agent skip a round-trip through ``list_layers``
        when the user says "remove the Sentinel-2 layer".

        Args:
            name: The display name (or a unique substring) of the layer
                to remove.

        Returns:
            A status string. On success: ``Removed layer 'X'.``
            On ambiguous match: ``Layer 'X' is ambiguous; matched: …``
            so the caller can disambiguate without listing layers.
            On miss: ``Layer 'X' not found.``
        """
        resolved, candidates = _resolve_layer_name(m, name)
        if resolved is None and len(candidates) > 1:
            joined = ", ".join(repr(c) for c in candidates)
            return (
                f"Layer {name!r} is ambiguous; matched multiple layers: "
                f"{joined}. Call remove_layer with the full name to pick one."
            )
        target = resolved or name
        if hasattr(m, "remove_layer"):
            removed = m.remove_layer(target)
            if removed in (None, True):
                return f"Removed layer {target!r}."
            return f"Layer {name!r} not found."
        layers = getattr(m, "layers", None)
        if isinstance(layers, list):
            before = len(layers)
            m.layers = [
                layer
                for layer in layers
                if (
                    layer.get("name")
                    if isinstance(layer, dict)
                    else getattr(layer, "name", None)
                )
                != target
            ]
            if len(m.layers) < before:
                return f"Removed layer {target!r}."
        return f"Layer {name!r} not found."

    @geo_tool(
        category="map",
        requires_confirmation=True,
    )
    def clear_layers() -> str:
        """Remove all layers currently tracked by the map.

        Returns:
            A status string with the number of removed layers.
        """
        names = _layer_names(m)
        if hasattr(m, "clear_layers"):
            m.clear_layers()
        elif hasattr(m, "layers") and isinstance(m.layers, list):
            m.layers = []
        elif hasattr(m, "remove_layer"):
            for layer_name in names:
                m.remove_layer(layer_name)
        else:
            return "No supported layer-clearing method is available."
        return f"Cleared {len(names)} layer(s)."

    @geo_tool(
        category="map",
    )
    def set_layer_visibility(name: str, visible: bool) -> str:
        """Show or hide a map layer by name or unique substring.

        Args:
            name: Layer name, or a unique case-insensitive substring.
            visible: ``True`` to show the layer, ``False`` to hide it.

        Returns:
            A status string.
        """
        resolved, candidates = _resolve_layer_name(m, name)
        if resolved is None and len(candidates) > 1:
            return f"Layer {name!r} is ambiguous; matched: {', '.join(candidates)}."
        target = resolved or name
        if hasattr(m, "set_layer_visibility"):
            changed = m.set_layer_visibility(target, bool(visible))
            if changed in (None, True):
                return f"Layer {target!r} visibility set to {visible}."
        layer = _find_layer_entry(m, target)
        if layer is not None and _set_layer_attr(layer, "visible", bool(visible)):
            return f"Layer {target!r} visibility set to {visible}."
        return f"Layer {name!r} not found."

    @geo_tool(
        category="map",
    )
    def set_layer_opacity(name: str, opacity: float) -> str:
        """Set a layer opacity from 0.0 (transparent) to 1.0 (opaque).

        Args:
            name: Layer name, or a unique case-insensitive substring.
            opacity: Desired opacity. Values outside 0..1 are clamped.

        Returns:
            A status string.
        """
        value = min(1.0, max(0.0, float(opacity)))
        resolved, candidates = _resolve_layer_name(m, name)
        if resolved is None and len(candidates) > 1:
            return f"Layer {name!r} is ambiguous; matched: {', '.join(candidates)}."
        target = resolved or name
        if hasattr(m, "set_layer_opacity"):
            changed = m.set_layer_opacity(target, value)
            if changed in (None, True):
                return f"Layer {target!r} opacity set to {value}."
        layer = _find_layer_entry(m, target)
        if layer is not None and _set_layer_attr(layer, "opacity", value):
            return f"Layer {target!r} opacity set to {value}."
        return f"Layer {name!r} not found."

    @geo_tool(
        category="map",
    )
    def set_center(lat: float, lon: float, zoom: Optional[int] = None) -> str:
        """Centre the map on a coordinate.

        Args:
            lat: Latitude in WGS84.
            lon: Longitude in WGS84.
            zoom: Optional new zoom level.

        Returns:
            A status string.
        """
        _safe_call(m, ["set_center"], lon, lat, zoom)
        return f"Centred on ({lat}, {lon})" + (
            f" at zoom {zoom}." if zoom is not None else "."
        )

    @geo_tool(
        category="map",
    )
    def fly_to(lat: float, lon: float, zoom: Optional[int] = None) -> str:
        """Animate or move the map to a coordinate when supported.

        Args:
            lat: Latitude in WGS84.
            lon: Longitude in WGS84.
            zoom: Optional target zoom.

        Returns:
            A status string.
        """
        try:
            _safe_call(m, ["fly_to"], lon, lat, zoom)
        except AttributeError:
            _safe_call(m, ["set_center"], lon, lat, zoom)
        return f"Moved to ({lat}, {lon})" + (
            f" at zoom {zoom}." if zoom is not None else "."
        )

    @geo_tool(
        category="map",
    )
    def set_zoom(zoom: int) -> str:
        """Set the zoom level.

        Args:
            zoom: Zoom level (typically 0–22).

        Returns:
            A status string.
        """
        if hasattr(m, "set_zoom"):
            m.set_zoom(zoom)
        else:
            m.zoom = zoom
        return f"Zoom set to {zoom}."

    @geo_tool(
        category="map",
    )
    def zoom_in(steps: int = 1) -> str:
        """Zoom in by a number of steps.

        Args:
            steps: Number of zoom levels to add. Default 1.

        Returns:
            A status string.
        """
        current = getattr(m, "zoom", 0) or 0
        new_zoom = int(current) + int(steps)
        if hasattr(m, "set_zoom"):
            m.set_zoom(new_zoom)
        else:
            m.zoom = new_zoom
        return f"Zoomed in to {new_zoom}."

    @geo_tool(
        category="map",
    )
    def zoom_out(steps: int = 1) -> str:
        """Zoom out by a number of steps.

        Args:
            steps: Number of zoom levels to subtract. Default 1.

        Returns:
            A status string.
        """
        current = getattr(m, "zoom", 0) or 0
        new_zoom = max(0, int(current) - int(steps))
        if hasattr(m, "set_zoom"):
            m.set_zoom(new_zoom)
        else:
            m.zoom = new_zoom
        return f"Zoomed out to {new_zoom}."

    @geo_tool(
        category="map",
    )
    def zoom_to_bounds(west: float, south: float, east: float, north: float) -> str:
        """Zoom to a bounding box.

        Args:
            west: Western longitude.
            south: Southern latitude.
            east: Eastern longitude.
            north: Northern latitude.

        Returns:
            A status string.
        """
        bounds = [[west, south], [east, north]]
        _safe_call(m, ["fit_bounds", "zoom_to_bounds"], bounds)
        return f"Zoomed to bounds [{west}, {south}, {east}, {north}]."

    @geo_tool(
        category="map",
    )
    def zoom_to_layer(name: str) -> str:
        """Zoom to a layer's recorded bounds, if available.

        Args:
            name: Layer name, or a unique case-insensitive substring.

        Returns:
            A status string.
        """
        resolved, candidates = _resolve_layer_name(m, name)
        if resolved is None and len(candidates) > 1:
            return f"Layer {name!r} is ambiguous; matched: {', '.join(candidates)}."
        target = resolved or name
        layer = _find_layer_entry(m, target)
        if layer is None:
            return f"Layer {name!r} not found."
        bounds = _layer_bounds(layer)
        if bounds is None:
            return f"Layer {target!r} has no recorded bounds."
        _safe_call(m, ["fit_bounds", "zoom_to_bounds"], bounds)
        return f"Zoomed to layer {target!r}."

    @geo_tool(
        category="map",
    )
    def change_basemap(basemap: str) -> str:
        """Change the basemap style.

        Args:
            basemap: A leafmap basemap identifier (e.g. ``"OpenStreetMap"``,
                ``"CartoDB.Positron"``, ``"Esri.WorldImagery"``).

        Returns:
            A status string.
        """
        _safe_call(m, ["add_basemap", "set_basemap"], basemap)
        return f"Basemap set to {basemap!r}."

    @geo_tool(
        category="map",
    )
    def add_vector_data(
        path_or_url: str,
        name: str,
        style: Optional[dict[str, Any]] = None,
    ) -> str:
        """Add a vector dataset (GeoJSON, Shapefile, GeoPackage).

        Args:
            path_or_url: Source path or URL.
            name: Display name.
            style: Optional style dict (e.g. ``{"color": "red"}``).

        Returns:
            A status string.
        """
        _safe_call(
            m,
            ["add_vector", "add_geojson"],
            path_or_url,
            layer_name=name,
            style=style or {},
        )
        return f"Added vector layer {name!r}."

    @geo_tool(
        category="map",
    )
    def add_geojson_data(
        data: dict[str, Any],
        name: str,
        style: Optional[dict[str, Any]] = None,
    ) -> str:
        """Add an in-memory GeoJSON FeatureCollection or geometry.

        Args:
            data: GeoJSON dictionary.
            name: Display name.
            style: Optional style dict.

        Returns:
            A status string.
        """
        _safe_call(
            m,
            ["add_geojson", "add_vector"],
            data,
            layer_name=name,
            style=style or {},
        )
        return f"Added GeoJSON layer {name!r}."

    @geo_tool(
        category="map",
    )
    def add_marker(
        lat: float,
        lon: float,
        popup: Optional[str] = None,
        tooltip: Optional[str] = None,
        name: Optional[str] = None,
    ) -> str:
        """Add a point marker to the map.

        Args:
            lat: Marker latitude.
            lon: Marker longitude.
            popup: Optional popup text.
            tooltip: Optional hover tooltip.
            name: Optional layer/display name.

        Returns:
            A status string.
        """
        marker_name = name or popup or f"Marker ({lat}, {lon})"
        if hasattr(m, "add_marker"):
            m.add_marker(
                location=(lat, lon),
                popup=popup,
                tooltip=tooltip,
                name=marker_name,
            )
        elif hasattr(m, "add_layer"):
            m.add_layer(
                {
                    "type": "marker",
                    "name": marker_name,
                    "location": [lat, lon],
                    "popup": popup,
                    "tooltip": tooltip,
                }
            )
        elif hasattr(m, "layers") and isinstance(m.layers, list):
            m.layers.append(
                {
                    "type": "marker",
                    "name": marker_name,
                    "location": [lat, lon],
                    "popup": popup,
                    "tooltip": tooltip,
                }
            )
        else:
            raise AttributeError(f"{type(m).__name__} cannot add marker layers.")
        return f"Added marker {marker_name!r} at ({lat}, {lon})."

    @geo_tool(
        category="map",
    )
    def add_raster_data(
        path_or_url: str,
        name: str,
        colormap: str = "viridis",
    ) -> str:
        """Add a raster dataset (GeoTIFF, COG, NetCDF).

        Args:
            path_or_url: Source path or URL.
            name: Display name.
            colormap: Colormap name (default ``"viridis"``).

        Returns:
            A status string.
        """
        if _is_planetary_computer_url(path_or_url):
            return (
                "Refusing to call add_raster_data with a Planetary Computer "
                "asset URL: PC's public TiTiler cannot tile raw blob hrefs. "
                "Call add_stac_layer(collection=..., item=..., assets=[...], "
                "titiler_endpoint='pc') instead. See "
                "https://leafmap.org/maplibre/stac/ for the canonical pattern."
            )
        try:
            _safe_call(
                m,
                ["add_raster", "add_cog_layer"],
                path_or_url,
                layer_name=name,
                colormap=colormap,
            )
        except Exception as exc:
            return f"add_raster_data failed: {type(exc).__name__}: {exc}"
        return f"Added raster layer {name!r}."

    @geo_tool(
        category="map",
    )
    def add_stac_layer(
        collection: str,
        item: Optional[str] = None,
        assets: Optional[list[str]] = None,
        name: Optional[str] = None,
        titiler_endpoint: Optional[str] = None,
    ) -> str:
        """Add a STAC layer to the map.

        For items returned by ``search_stac(catalog="microsoft-pc", ...)``,
        pass ``titiler_endpoint="pc"`` so leafmap renders the layer via
        Planetary Computer's hosted TiTiler, which signs SAS-protected
        asset URLs internally. For other catalogs (e.g. ``earth-search``),
        leave ``titiler_endpoint`` unset to use leafmap's default.

        Args:
            collection: STAC collection identifier.
            item: STAC item identifier (optional).
            assets: List of asset keys to render.
            name: Display name (defaults to the collection or item id).
            titiler_endpoint: TiTiler service to use. ``"pc"`` selects
                Planetary Computer's TiTiler (handles SAS signing for
                Microsoft PC items). ``None`` falls back to leafmap's
                default public TiTiler.

        Returns:
            A status string.
        """
        kwargs: dict[str, Any] = {
            "collection": collection,
            "item": item,
            "assets": assets or [],
        }
        if titiler_endpoint is not None:
            kwargs["titiler_endpoint"] = titiler_endpoint
        layer_name = name or item or collection
        kwargs["name"] = layer_name
        try:
            _safe_call(m, ["add_stac_layer"], **kwargs)
        except Exception as exc:
            return f"add_stac_layer failed: {type(exc).__name__}: {exc}"
        return f"Added STAC layer {layer_name!r}."

    @geo_tool(
        category="map",
    )
    def add_cog_layer(
        url: str,
        name: str,
        colormap: str = "viridis",
        titiler_endpoint: Optional[str] = None,
    ) -> str:
        """Add a Cloud Optimized GeoTIFF layer.

        For Planetary Computer asset URLs (Sentinel-2 ``visual``,
        Landsat assets, etc.), prefer ``add_stac_layer`` with
        ``titiler_endpoint="pc"`` instead of this tool: PC's hosted
        TiTiler handles SAS signing for you, whereas a public TiTiler
        called against a raw, unsigned PC href will fail with a missing
        ``tiles`` key.

        Args:
            url: COG URL. For Planetary Computer assets, the URL must
                already be SAS-signed (or use ``add_stac_layer`` instead).
            name: Display name.
            colormap: Colormap name.
            titiler_endpoint: Optional TiTiler service. ``None`` uses
                leafmap's default public TiTiler.

        Returns:
            A status string.
        """
        if _is_planetary_computer_url(url):
            return (
                "Refusing to call add_cog_layer with a Planetary Computer "
                "asset URL: PC's public TiTiler cannot tile raw blob hrefs. "
                "Call add_stac_layer(collection=..., item=..., assets=[...], "
                "titiler_endpoint='pc') instead — Microsoft's hosted TiTiler "
                "handles SAS signing for STAC items internally. See "
                "https://leafmap.org/maplibre/stac/ for the canonical pattern."
            )
        kwargs: dict[str, Any] = {"name": name, "colormap": colormap}
        if titiler_endpoint is not None:
            kwargs["titiler_endpoint"] = titiler_endpoint
        try:
            _safe_call(m, ["add_cog_layer"], url, **kwargs)
        except Exception as exc:
            return f"add_cog_layer failed: {type(exc).__name__}: {exc}"
        return f"Added COG layer {name!r}."

    @geo_tool(
        category="map",
    )
    def add_xyz_tile_layer(
        url: str,
        name: str,
        attribution: str = "",
    ) -> str:
        """Add an XYZ tile layer.

        Args:
            url: XYZ tile URL template.
            name: Display name.
            attribution: Map attribution string.

        Returns:
            A status string.
        """
        _safe_call(
            m,
            ["add_xyz_tile_layer", "add_tile_layer"],
            url,
            name=name,
            attribution=attribution,
        )
        return f"Added XYZ layer {name!r}."

    @geo_tool(
        category="map",
    )
    def add_pmtiles_layer(
        url: str,
        name: str,
        style: Optional[dict[str, Any]] = None,
    ) -> str:
        """Add a PMTiles vector layer.

        Args:
            url: PMTiles URL.
            name: Display name.
            style: Optional MapLibre style dict.

        Returns:
            A status string.
        """
        kwargs: dict[str, Any] = {"name": name}
        if style is not None:
            kwargs["style"] = style
        _safe_call(m, ["add_pmtiles", "add_pmtiles_layer"], url, **kwargs)
        return f"Added PMTiles layer {name!r}."

    @geo_tool(
        category="map",
    )
    def get_map_state() -> dict[str, Any]:
        """Read the current map camera and layers.

        For **MapLibre** maps (``leafmap.maplibregl``), state comes from
        ``m.view_state`` — center ``lng``/``lat``, ``zoom``, ``bounds`` with
        ``_sw`` / ``_ne``, ``bearing``, ``pitch``. Results include a normalized
        ``view_state`` dict plus flattened keys for prompts.

        Older leafmap/ipyleaflet-like objects fall back to ``center``, ``zoom``,
        ``_bounds``, ``_style``.
        """
        return map_state_from_widget(m)

    @geo_tool(
        category="map",
        requires_confirmation=True,
    )
    def save_map(path: str) -> str:
        """Export the map to a standalone HTML file.

        Args:
            path: Destination file path (overwritten if it exists).

        Returns:
            The absolute path of the saved file.
        """
        from pathlib import Path

        out = Path(path).expanduser().resolve()
        if hasattr(m, "to_html"):
            m.to_html(str(out))
        elif hasattr(m, "save"):
            m.save(str(out))
        else:
            raise AttributeError(
                f"{type(m).__name__} has no to_html() or save() method."
            )
        return str(out)

    return [
        list_layers,
        add_layer,
        remove_layer,
        clear_layers,
        set_layer_visibility,
        set_layer_opacity,
        set_center,
        fly_to,
        set_zoom,
        zoom_in,
        zoom_out,
        zoom_to_bounds,
        zoom_to_layer,
        change_basemap,
        add_vector_data,
        add_geojson_data,
        add_marker,
        add_raster_data,
        add_stac_layer,
        add_cog_layer,
        add_xyz_tile_layer,
        add_pmtiles_layer,
        get_map_state,
        save_map,
    ]


__all__ = ["leafmap_tools"]
