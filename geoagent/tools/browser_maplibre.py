"""GeoAgent tools for MapLibre maps running in a browser web app."""

from __future__ import annotations

from typing import Any, Optional

from geoagent.browser.session import BrowserMapSession
from geoagent.core.decorators import geo_tool


def browser_maplibre_tools(
    session: BrowserMapSession | None,
    *,
    allow_code: bool = False,
) -> list[Any]:
    """Build browser MapLibre tools bound to a WebSocket map session."""
    if session is None:
        return []

    @geo_tool(category="browser_map", available_in=("full", "fast"))
    def list_layers() -> list[dict[str, Any]]:
        """List layers currently present in the browser MapLibre map."""
        result = session.call("list_layers")
        return result if isinstance(result, list) else []

    @geo_tool(category="browser_map", available_in=("full", "fast"))
    def get_map_state() -> dict[str, Any]:
        """Return the browser map camera state and current bounds."""
        result = session.call("get_map_state")
        return result if isinstance(result, dict) else {}

    @geo_tool(category="browser_map", available_in=("full", "fast"))
    def set_center(lat: float, lon: float, zoom: Optional[float] = None) -> str:
        """Center the browser map on a latitude/longitude coordinate."""
        result = session.call(
            "set_center",
            {"lat": float(lat), "lon": float(lon), "zoom": zoom},
        )
        return str(result or f"Centered map on ({lat}, {lon}).")

    @geo_tool(category="browser_map", available_in=("full", "fast"))
    def fly_to(lat: float, lon: float, zoom: Optional[float] = None) -> str:
        """Animate the browser map to a latitude/longitude coordinate."""
        result = session.call(
            "fly_to",
            {"lat": float(lat), "lon": float(lon), "zoom": zoom},
        )
        return str(result or f"Moved map to ({lat}, {lon}).")

    @geo_tool(category="browser_map", available_in=("full", "fast"))
    def set_zoom(zoom: float) -> str:
        """Set the browser map zoom level."""
        result = session.call("set_zoom", {"zoom": float(zoom)})
        return str(result or f"Zoom set to {zoom}.")

    @geo_tool(category="browser_map", available_in=("full", "fast"))
    def zoom_to_bounds(west: float, south: float, east: float, north: float) -> str:
        """Zoom the browser map to a bounding box."""
        result = session.call(
            "zoom_to_bounds",
            {
                "west": float(west),
                "south": float(south),
                "east": float(east),
                "north": float(north),
            },
        )
        return str(result or f"Zoomed to [{west}, {south}, {east}, {north}].")

    @geo_tool(category="browser_map", available_in=("full", "fast"))
    def change_basemap(style: str) -> str:
        """Change the browser MapLibre style URL or style identifier."""
        result = session.call("change_basemap", {"style": style})
        return str(result or f"Basemap changed to {style!r}.")

    @geo_tool(category="browser_map")
    def add_marker(
        lat: float,
        lon: float,
        popup: Optional[str] = None,
        tooltip: Optional[str] = None,
        name: Optional[str] = None,
        color: str = "#3388ff",
    ) -> str:
        """Add a marker to the browser map."""
        result = session.call(
            "add_marker",
            {
                "lat": float(lat),
                "lon": float(lon),
                "popup": popup,
                "tooltip": tooltip,
                "name": name,
                "color": color,
            },
        )
        return str(result or f"Added marker at ({lat}, {lon}).")

    @geo_tool(category="browser_map")
    def add_geojson_data(
        data: dict[str, Any],
        name: str,
        style: Optional[dict[str, Any]] = None,
    ) -> str:
        """Add an in-memory GeoJSON object to the browser map."""
        result = session.call(
            "add_geojson_data",
            {"data": data, "name": name, "style": style or {}},
        )
        return str(result or f"Added GeoJSON layer {name!r}.")

    @geo_tool(category="browser_map")
    def add_vector_data(
        url: str,
        name: str,
        style: Optional[dict[str, Any]] = None,
    ) -> str:
        """Add vector data from a URL to the browser map."""
        result = session.call(
            "add_vector_data",
            {"url": url, "name": name, "style": style or {}},
        )
        return str(result or f"Added vector layer {name!r}.")

    @geo_tool(category="browser_map")
    def add_xyz_tile_layer(
        url: str,
        name: str,
        attribution: str = "",
    ) -> str:
        """Add an XYZ raster tile layer to the browser map."""
        result = session.call(
            "add_xyz_tile_layer",
            {"url": url, "name": name, "attribution": attribution},
        )
        return str(result or f"Added XYZ layer {name!r}.")

    @geo_tool(category="browser_map", available_in=("full", "fast"))
    def set_layer_visibility(name: str, visible: bool) -> str:
        """Show or hide a browser map layer."""
        result = session.call(
            "set_layer_visibility",
            {"name": name, "visible": bool(visible)},
        )
        return str(result or f"Layer {name!r} visibility set to {visible}.")

    @geo_tool(category="browser_map", available_in=("full", "fast"))
    def set_layer_opacity(name: str, opacity: float) -> str:
        """Set browser map layer opacity between 0 and 1."""
        value = max(0.0, min(1.0, float(opacity)))
        result = session.call(
            "set_layer_opacity",
            {"name": name, "opacity": value},
        )
        return str(result or f"Layer {name!r} opacity set to {value}.")

    @geo_tool(category="browser_map")
    def query_rendered_features(
        layers: Optional[list[str]] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """Query rendered features from the browser map."""
        result = session.call(
            "query_rendered_features",
            {"layers": layers or [], "x": x, "y": y},
        )
        return result if isinstance(result, list) else []

    @geo_tool(category="browser_map")
    def screenshot_map() -> dict[str, Any]:
        """Capture the browser map canvas as a PNG data URL."""
        result = session.call("screenshot_map")
        return result if isinstance(result, dict) else {"data_url": result}

    @geo_tool(category="browser_map", requires_confirmation=True, destructive=True)
    def remove_layer(name: str) -> str:
        """Remove a layer from the browser map."""
        result = session.call("remove_layer", {"name": name})
        return str(result or f"Removed layer {name!r}.")

    @geo_tool(category="browser_map", requires_confirmation=True, destructive=True)
    def clear_layers() -> str:
        """Remove user-added layers from the browser map."""
        result = session.call("clear_layers")
        return str(result or "Cleared browser map layers.")

    tools = [
        list_layers,
        get_map_state,
        set_center,
        fly_to,
        set_zoom,
        zoom_to_bounds,
        change_basemap,
        add_marker,
        add_geojson_data,
        add_vector_data,
        add_xyz_tile_layer,
        set_layer_visibility,
        set_layer_opacity,
        query_rendered_features,
        screenshot_map,
        remove_layer,
        clear_layers,
    ]

    if allow_code:

        @geo_tool(
            category="browser_map",
            requires_confirmation=True,
            destructive=True,
        )
        def run_maplibre_script(code: str, description: str = "") -> dict[str, Any]:
            """Run a short JavaScript snippet against the live browser MapLibre map.

            Use this only when no dedicated browser map tool can perform the
            requested operation. The browser page executes the code with ``map``,
            ``maplibregl``, and a small ``helpers`` object in scope. Do not use
            this for credential handling, storage access, broad DOM access, or
            unrelated network operations.

            Args:
                code: JavaScript code to execute in the browser. It may use
                    ``await`` and may return a JSON-serializable value.
                description: One-sentence explanation of the intended map change.

            Returns:
                A dict with success status, message, returned value, and the
                executed code.
            """
            code = (code or "").strip()
            if not code:
                return {
                    "success": False,
                    "error": "No MapLibre JavaScript code was provided.",
                    "maplibre_script": "",
                }
            result = session.call(
                "run_maplibre_script",
                {"code": code, "description": description},
            )
            if isinstance(result, dict):
                return result
            return {
                "success": True,
                "message": str(result or description or "MapLibre script executed."),
                "description": description,
                "maplibre_script": code,
            }

        tools.append(run_maplibre_script)

    return tools


__all__ = ["browser_maplibre_tools"]
