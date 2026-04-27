"""Tool adapters for ``anymap.Map`` instances.

The :func:`anymap_tools` factory returns a list of GeoAgent-decorated tools
bound to a single live ``anymap.Map`` (or compatible mock) via closure.

anymap is a newer, lighter-weight alternative to leafmap with a similar
top-level API. Where method names diverge, the tool bodies use ``_safe_call``
to try the closest equivalents.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import BaseTool

from geoagent.core.decorators import geo_tool
from geoagent.tools.leafmap import _safe_call


def anymap_tools(m: Any) -> list[BaseTool]:
    """Build the anymap tool set bound to a live map instance.

    Args:
        m: An ``anymap.Map`` instance, or a compatible mock. Passing
            ``None`` returns an empty list.

    Returns:
        A list of LangChain ``BaseTool`` instances.
    """
    if m is None:
        return []

    @geo_tool(
        category="map",
        requires_packages=("anymap",),
        context_keys=("map_obj",),
    )
    def list_layers() -> list[dict[str, Any]]:
        """List the layers currently on the map.

        Returns:
            A list of dicts ``{"name": ..., "type": ...}``.
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
        requires_packages=("anymap",),
        context_keys=("map_obj",),
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
        requires_packages=("anymap",),
        context_keys=("map_obj",),
    )
    def remove_layer(name: str) -> str:
        """Remove a named layer from the map.

        Args:
            name: Display name of the layer.

        Returns:
            A status string.
        """
        if hasattr(m, "remove_layer"):
            removed = m.remove_layer(name)
            if removed in (None, True):
                return f"Removed layer {name!r}."
            return f"Layer {name!r} not found."
        return f"Layer {name!r} could not be removed (no remove_layer method)."

    @geo_tool(
        category="map",
        requires_packages=("anymap",),
        context_keys=("map_obj",),
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
        requires_packages=("anymap",),
        context_keys=("map_obj",),
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
        requires_packages=("anymap",),
        context_keys=("map_obj",),
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
        requires_packages=("anymap",),
        context_keys=("map_obj",),
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
        requires_packages=("anymap",),
        context_keys=("map_obj",),
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
        requires_packages=("anymap",),
        context_keys=("map_obj",),
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
        requires_packages=("anymap",),
        context_keys=("map_obj",),
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
        requires_packages=("anymap",),
        context_keys=("map_obj",),
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
        requires_packages=("anymap",),
        context_keys=("map_obj",),
    )
    def get_map_state() -> dict[str, Any]:
        """Return centre, zoom, bounds, basemap, and layer count."""
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
        requires_packages=("anymap",),
        context_keys=("map_obj", "workdir"),
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
        set_center,
        set_zoom,
        zoom_in,
        zoom_out,
        zoom_to_bounds,
        change_basemap,
        add_vector_data,
        add_raster_data,
        get_map_state,
        save_map,
    ]


__all__ = ["anymap_tools"]
