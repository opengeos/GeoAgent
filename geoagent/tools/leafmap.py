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

from langchain_core.tools import BaseTool

from geoagent.core.decorators import geo_tool

_NAME_KW_ALIASES = ("name", "layer_name")


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


def leafmap_tools(m: Any) -> list[BaseTool]:
    """Build the leafmap tool set bound to a live map instance.

    Args:
        m: A ``leafmap.Map`` (or ``leafmap.maplibregl.Map``) instance, or any
            object exposing the same minimal API. Passing ``None`` returns an
            empty list.

    Returns:
        A list of LangChain ``BaseTool`` instances. Each tool captures
        ``m`` via closure; the LLM never sees the map object.
    """
    if m is None:
        return []

    @geo_tool(
        category="map",
        requires_packages=("leafmap",),
        context_keys=("map_obj",),
    )
    def list_layers() -> list[dict[str, Any]]:
        """List the layers currently on the map.

        Returns:
            A list of dicts ``{"name": ..., "type": ...}`` for each layer.
        """
        layers = getattr(m, "layers", None) or []
        out: list[dict[str, Any]] = []
        for layer in layers:
            if isinstance(layer, dict):
                out.append(
                    {
                        "name": layer.get("name", "Unnamed"),
                        "type": layer.get("type", "unknown"),
                    }
                )
            else:
                out.append(
                    {
                        "name": getattr(layer, "name", str(layer)),
                        "type": type(layer).__name__,
                    }
                )
        return out

    @geo_tool(
        category="map",
        requires_packages=("leafmap",),
        context_keys=("map_obj",),
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
        requires_packages=("leafmap",),
        context_keys=("map_obj",),
    )
    def remove_layer(name: str) -> str:
        """Remove a layer from the map by display name.

        Args:
            name: The display name of the layer to remove.

        Returns:
            A status string indicating whether the layer was removed.
        """
        if hasattr(m, "remove_layer"):
            removed = m.remove_layer(name)
            if removed in (None, True):
                return f"Removed layer {name!r}."
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
                != name
            ]
            if len(m.layers) < before:
                return f"Removed layer {name!r}."
        return f"Layer {name!r} not found."

    @geo_tool(
        category="map",
        requires_packages=("leafmap",),
        context_keys=("map_obj",),
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
        requires_packages=("leafmap",),
        context_keys=("map_obj",),
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
        requires_packages=("leafmap",),
        context_keys=("map_obj",),
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
        requires_packages=("leafmap",),
        context_keys=("map_obj",),
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
        requires_packages=("leafmap",),
        context_keys=("map_obj",),
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
        requires_packages=("leafmap",),
        context_keys=("map_obj",),
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
        requires_packages=("leafmap",),
        context_keys=("map_obj",),
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
        requires_packages=("leafmap",),
        context_keys=("map_obj",),
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
        _safe_call(
            m,
            ["add_raster", "add_cog_layer"],
            path_or_url,
            layer_name=name,
            colormap=colormap,
        )
        return f"Added raster layer {name!r}."

    @geo_tool(
        category="map",
        requires_packages=("leafmap",),
        context_keys=("map_obj",),
    )
    def add_stac_layer(
        collection: str,
        item: Optional[str] = None,
        assets: Optional[list[str]] = None,
        name: Optional[str] = None,
    ) -> str:
        """Add a STAC layer to the map.

        Args:
            collection: STAC collection identifier.
            item: STAC item identifier (optional).
            assets: List of asset keys to render.
            name: Display name (defaults to the collection or item id).

        Returns:
            A status string.
        """
        kwargs = {"collection": collection, "item": item, "assets": assets or []}
        layer_name = name or item or collection
        kwargs["name"] = layer_name
        _safe_call(m, ["add_stac_layer"], **kwargs)
        return f"Added STAC layer {layer_name!r}."

    @geo_tool(
        category="map",
        requires_packages=("leafmap",),
        context_keys=("map_obj",),
    )
    def add_cog_layer(
        url: str,
        name: str,
        colormap: str = "viridis",
    ) -> str:
        """Add a Cloud Optimized GeoTIFF layer.

        Args:
            url: COG URL.
            name: Display name.
            colormap: Colormap name.

        Returns:
            A status string.
        """
        _safe_call(m, ["add_cog_layer"], url, name=name, colormap=colormap)
        return f"Added COG layer {name!r}."

    @geo_tool(
        category="map",
        requires_packages=("leafmap",),
        context_keys=("map_obj",),
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
        requires_packages=("leafmap",),
        context_keys=("map_obj",),
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
        requires_packages=("leafmap",),
        context_keys=("map_obj",),
    )
    def get_map_state() -> dict[str, Any]:
        """Read the current map view (centre, zoom, bounds, basemap).

        Returns:
            A dict with ``"center"``, ``"zoom"``, ``"bounds"``, and
            ``"basemap"`` keys, populated where available.
        """
        return {
            "center": list(getattr(m, "center", []) or []),
            "zoom": getattr(m, "zoom", None),
            "bounds": getattr(m, "_bounds", None),
            "basemap": getattr(m, "_style", None),
            "layer_count": len(getattr(m, "layers", []) or []),
        }

    @geo_tool(
        category="map",
        requires_confirmation=True,
        requires_packages=("leafmap",),
        context_keys=("map_obj", "workdir"),
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
        set_center,
        set_zoom,
        zoom_in,
        zoom_out,
        zoom_to_bounds,
        change_basemap,
        add_vector_data,
        add_raster_data,
        add_stac_layer,
        add_cog_layer,
        add_xyz_tile_layer,
        add_pmtiles_layer,
        get_map_state,
        save_map,
    ]


__all__ = ["leafmap_tools"]
