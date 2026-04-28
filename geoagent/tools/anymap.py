"""Tool adapters for ``anymap.Map`` instances.

The :func:`anymap_tools` factory returns a list of GeoAgent-decorated tools
bound to a single live ``anymap.Map`` (or compatible mock) via closure.

anymap is a newer, lighter-weight alternative to leafmap with a similar
top-level API. Where method names diverge, the tool bodies use ``_safe_call``
to try the closest equivalents.
"""

from __future__ import annotations

from typing import Any, Optional

from geoagent.core.decorators import geo_tool
from geoagent.tools._map_state import map_state_from_widget
from geoagent.tools.leafmap import (
    _find_layer_entry,
    _is_planetary_computer_url,
    _layer_bounds,
    _layer_names,
    _layer_record,
    _resolve_layer_name,
    _safe_call,
    _set_layer_attr,
)


def anymap_tools(m: Any) -> list[Any]:
    """Build the anymap tool set bound to a live map instance.

    Args:
        m: An ``anymap.Map`` instance, or a compatible mock. Passing
            ``None`` returns an empty list.

    Returns:
        A list of Strands tool objects.
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
            ``visible``, ``opacity``, and ``bounds`` when available.
        """
        layers = getattr(m, "layers", None) or []
        return [_layer_record(layer) for layer in layers]

    @geo_tool(
        category="map",
    )
    def add_layer(url: str, name: str, layer_type: str = "auto") -> str:
        """Add a layer to the map by URL.

        Args:
            url: The data URL.
            name: Display name.
            layer_type: One of ``"raster"``, ``"vector"``, ``"cog"``,
                ``"pmtiles"``, ``"xyz"``, or ``"auto"``.

        Returns:
            A status string.
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
        if kind == "vector":
            _safe_call(m, ["add_geojson", "add_vector"], url, layer_name=name)
        elif kind in ("cog", "raster"):
            _safe_call(m, ["add_raster", "add_cog_layer"], url, layer_name=name)
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
        """Remove a named layer from the map.

        Args:
            name: Display name of the layer, or a unique substring.

        Returns:
            A status string.
        """
        resolved, candidates = _resolve_layer_name(m, name)
        if resolved is None and len(candidates) > 1:
            return f"Layer {name!r} is ambiguous; matched: {', '.join(candidates)}."
        target = resolved or name
        if hasattr(m, "remove_layer"):
            removed = m.remove_layer(target)
            if removed in (None, True):
                return f"Removed layer {target!r}."
            return f"Layer {name!r} not found."
        return f"Layer {name!r} could not be removed (no remove_layer method)."

    @geo_tool(
        category="map",
        requires_confirmation=True,
    )
    def clear_layers() -> str:
        """Remove all layers currently tracked by the map."""
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
        """Show or hide a map layer by name or unique substring."""
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
        """Set a layer opacity from 0.0 (transparent) to 1.0 (opaque)."""
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
            lat: Latitude.
            lon: Longitude.
            zoom: Optional new zoom level.

        Returns:
            A status string.
        """
        _safe_call(m, ["set_center"], lon, lat, zoom)
        return f"Centred on ({lat}, {lon})."

    @geo_tool(
        category="map",
    )
    def fly_to(lat: float, lon: float, zoom: Optional[int] = None) -> str:
        """Animate or move the map to a coordinate when supported."""
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
            zoom: New zoom level.

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
        """Zoom in by a number of steps."""
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
        """Zoom out by a number of steps."""
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
        """Zoom the map to a bounding box.

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
        """Zoom to a layer's recorded bounds, if available."""
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
            basemap: Basemap identifier (e.g. ``"CartoDB.Positron"``).

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
        """Add a vector dataset.

        Args:
            path_or_url: Source path or URL.
            name: Display name.
            style: Optional style dict.

        Returns:
            A status string.
        """
        _safe_call(
            m,
            ["add_geojson", "add_vector"],
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
        """Add an in-memory GeoJSON FeatureCollection or geometry."""
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
        """Add a point marker to the map."""
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
        """Add a raster dataset.

        Args:
            path_or_url: Source path or URL.
            name: Display name.
            colormap: Colormap name.

        Returns:
            A status string.
        """
        if _is_planetary_computer_url(path_or_url):
            return (
                "Refusing to call add_raster_data with a Planetary Computer "
                "asset URL. Call add_stac_layer(..., titiler_endpoint='pc') "
                "instead so Microsoft's hosted TiTiler can sign the assets."
            )
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
    )
    def add_stac_layer(
        collection: str,
        item: Optional[str] = None,
        assets: Optional[list[str]] = None,
        name: Optional[str] = None,
        titiler_endpoint: Optional[str] = None,
    ) -> str:
        """Add a STAC layer to the map."""
        kwargs: dict[str, Any] = {
            "collection": collection,
            "item": item,
            "assets": assets or [],
            "name": name or item or collection,
        }
        if titiler_endpoint is not None:
            kwargs["titiler_endpoint"] = titiler_endpoint
        try:
            _safe_call(m, ["add_stac_layer"], **kwargs)
        except Exception as exc:
            return f"add_stac_layer failed: {type(exc).__name__}: {exc}"
        return f"Added STAC layer {kwargs['name']!r}."

    @geo_tool(
        category="map",
    )
    def add_cog_layer(
        url: str,
        name: str,
        colormap: str = "viridis",
        titiler_endpoint: Optional[str] = None,
    ) -> str:
        """Add a Cloud Optimized GeoTIFF layer."""
        if _is_planetary_computer_url(url):
            return (
                "Refusing to call add_cog_layer with a Planetary Computer "
                "asset URL. Call add_stac_layer(..., titiler_endpoint='pc') "
                "instead so Microsoft's hosted TiTiler can sign the assets."
            )
        kwargs: dict[str, Any] = {"name": name, "colormap": colormap}
        if titiler_endpoint is not None:
            kwargs["titiler_endpoint"] = titiler_endpoint
        try:
            _safe_call(m, ["add_cog_layer", "add_raster"], url, **kwargs)
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
        """Add an XYZ tile layer."""
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
        """Add a PMTiles vector layer."""
        kwargs: dict[str, Any] = {"name": name}
        if style is not None:
            kwargs["style"] = style
        _safe_call(m, ["add_pmtiles", "add_pmtiles_layer"], url, **kwargs)
        return f"Added PMTiles layer {name!r}."

    @geo_tool(
        category="map",
    )
    def get_map_state() -> dict[str, Any]:
        """Return map camera state; MapLibre-like maps use ``view_state``."""
        return map_state_from_widget(m)

    @geo_tool(
        category="map",
        requires_confirmation=True,
    )
    def save_map(path: str) -> str:
        """Export the map to a standalone HTML file.

        Args:
            path: Destination file path.

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


__all__ = ["anymap_tools"]
