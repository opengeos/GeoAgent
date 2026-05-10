"""Tests for browser MapLibre GeoAgent tools."""

from __future__ import annotations

from geoagent.core.decorators import needs_confirmation
from geoagent.tools.browser_maplibre import browser_maplibre_tools


class FakeBrowserMapSession:
    """Test double for browser map command sessions."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def call(self, command: str, args: dict | None = None, **_kwargs):
        self.calls.append((command, args or {}))
        if command == "list_layers":
            return [{"id": "cities", "type": "circle", "visible": True}]
        if command == "get_map_state":
            return {"center": [-83.92, 35.96], "zoom": 10}
        if command == "query_rendered_features":
            return [{"type": "Feature", "properties": {"name": "Knoxville"}}]
        if command == "screenshot_map":
            return {"data_url": "data:image/png;base64,abc"}
        return f"ok:{command}"


def _tools(session: FakeBrowserMapSession) -> dict[str, object]:
    return {tool.tool_name: tool for tool in browser_maplibre_tools(session)}


def test_browser_maplibre_tool_surface() -> None:
    """Verify the browser MapLibre tool surface."""
    names = {tool.tool_name for tool in browser_maplibre_tools(FakeBrowserMapSession())}
    assert {
        "list_layers",
        "get_map_state",
        "set_center",
        "fly_to",
        "set_zoom",
        "zoom_to_bounds",
        "change_basemap",
        "add_marker",
        "add_geojson_data",
        "add_vector_data",
        "add_xyz_tile_layer",
        "set_layer_visibility",
        "set_layer_opacity",
        "query_rendered_features",
        "screenshot_map",
        "remove_layer",
        "clear_layers",
    }.issubset(names)
    assert "run_maplibre_script" not in names


def test_browser_maplibre_code_tool_is_opt_in() -> None:
    """Verify arbitrary browser code execution is opt-in and confirmation gated."""
    tools = {
        tool.tool_name: tool
        for tool in browser_maplibre_tools(FakeBrowserMapSession(), allow_code=True)
    }

    assert "run_maplibre_script" in tools
    assert needs_confirmation(tools["run_maplibre_script"]) is True
    assert tools["run_maplibre_script"](
        code="return map.getZoom();",
        description="Read the zoom level.",
    ) == {
        "success": True,
        "message": "ok:run_maplibre_script",
        "description": "Read the zoom level.",
        "maplibre_script": "return map.getZoom();",
    }


def test_browser_maplibre_tools_send_expected_payloads() -> None:
    """Verify tools translate Python arguments into browser commands."""
    session = FakeBrowserMapSession()
    tools = _tools(session)

    assert tools["set_center"](lat=35.96, lon=-83.92, zoom=10) == "ok:set_center"
    assert tools["add_marker"](lat=35.96, lon=-83.92, name="Knoxville") == (
        "ok:add_marker"
    )
    assert tools["set_layer_opacity"](name="cities", opacity=2.0) == (
        "ok:set_layer_opacity"
    )

    assert session.calls[0] == (
        "set_center",
        {"lat": 35.96, "lon": -83.92, "zoom": 10},
    )
    assert session.calls[1][0] == "add_marker"
    assert session.calls[1][1]["name"] == "Knoxville"
    assert session.calls[2] == (
        "set_layer_opacity",
        {"name": "cities", "opacity": 1.0},
    )


def test_browser_maplibre_query_tools_return_structured_results() -> None:
    """Verify browser query tools preserve structured results."""
    session = FakeBrowserMapSession()
    tools = _tools(session)

    assert tools["list_layers"]() == [
        {"id": "cities", "type": "circle", "visible": True}
    ]
    assert tools["get_map_state"]()["zoom"] == 10
    assert (
        tools["query_rendered_features"](layers=["cities"])[0]["properties"]["name"]
        == "Knoxville"
    )
    assert tools["screenshot_map"]()["data_url"].startswith("data:image/png")


def test_browser_maplibre_destructive_tools_require_confirmation() -> None:
    """Verify destructive browser tools are confirmation gated."""
    tools = _tools(FakeBrowserMapSession())
    assert needs_confirmation(tools["remove_layer"]) is True
    assert needs_confirmation(tools["clear_layers"]) is True
