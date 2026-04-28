"""Tests for the leafmap tool factory (Strands tools + :class:`MockLeafmap`)."""

from __future__ import annotations

from pathlib import Path

from geoagent.core.decorators import needs_confirmation
from geoagent.core.factory import register_all_tools
from geoagent.core.registry import GeoToolRegistry, collect_tools_for_context
from geoagent.testing import MockLeafmap
from geoagent.tools.leafmap import leafmap_tools


def _by_name(m: MockLeafmap) -> dict[str, object]:
    return {t.tool_name: t for t in leafmap_tools(m)}


def test_factory_returns_full_tool_list() -> None:
    m = MockLeafmap()
    names = {t.tool_name for t in leafmap_tools(m)}
    expected = {
        "list_layers",
        "add_layer",
        "remove_layer",
        "clear_layers",
        "set_layer_visibility",
        "set_layer_opacity",
        "set_center",
        "fly_to",
        "set_zoom",
        "zoom_in",
        "zoom_out",
        "zoom_to_bounds",
        "zoom_to_layer",
        "change_basemap",
        "add_vector_data",
        "add_geojson_data",
        "add_marker",
        "add_raster_data",
        "add_stac_layer",
        "add_cog_layer",
        "add_xyz_tile_layer",
        "add_pmtiles_layer",
        "get_map_state",
        "save_map",
    }
    assert expected.issubset(names)


def test_factory_returns_empty_for_none() -> None:
    assert leafmap_tools(None) == []


def test_remove_and_save_require_confirmation() -> None:
    tools = _by_name(MockLeafmap())
    assert needs_confirmation(tools["remove_layer"]) is True
    assert needs_confirmation(tools["clear_layers"]) is True
    assert needs_confirmation(tools["save_map"]) is True
    assert needs_confirmation(tools["list_layers"]) is False
    assert needs_confirmation(tools["zoom_in"]) is False


def test_add_layer_mutates_map() -> None:
    m = MockLeafmap()
    tools = _by_name(m)
    tools["add_layer"](
        url="https://example.com/data.tif",
        name="DEM",
        layer_type="cog",
    )
    assert any(layer.get("name") == "DEM" for layer in m.layers)


def test_remove_layer_mutates_map() -> None:
    m = MockLeafmap()
    tools = _by_name(m)
    tools["add_layer"](
        url="https://example.com/buildings.geojson",
        name="Buildings",
        layer_type="vector",
    )
    assert len(m.layers) == 1
    tools["remove_layer"](name="Buildings")
    assert len(m.layers) == 0


def test_layer_visibility_opacity_and_zoom_to_layer() -> None:
    m = MockLeafmap()
    tools = _by_name(m)
    m.layers.append(
        {
            "name": "Sentinel-2 Knoxville",
            "type": "raster",
            "bounds": [-84.1, 35.8, -83.7, 36.1],
        }
    )
    tools["set_layer_visibility"](name="sentinel", visible=False)
    tools["set_layer_opacity"](name="sentinel", opacity=0.35)
    tools["zoom_to_layer"](name="sentinel")

    assert m.layers[0]["visible"] is False
    assert m.layers[0]["opacity"] == 0.35
    assert m.get_bounds() == [[-84.1, 35.8], [-83.7, 36.1]]
    layer = tools["list_layers"]()[0]
    assert layer["visible"] is False
    assert layer["opacity"] == 0.35


def test_add_marker_and_geojson_data() -> None:
    m = MockLeafmap()
    tools = _by_name(m)
    tools["add_marker"](lat=35.96, lon=-83.92, popup="Knoxville")
    tools["add_geojson_data"](
        data={"type": "FeatureCollection", "features": []},
        name="Empty GeoJSON",
    )
    assert any(layer["type"] == "marker" for layer in m.layers)
    assert any(layer["name"] == "Empty GeoJSON" for layer in m.layers)


def test_clear_layers_mutates_map() -> None:
    m = MockLeafmap()
    tools = _by_name(m)
    tools["add_marker"](lat=0, lon=0, name="Zero")
    assert m.layers
    tools["clear_layers"]()
    assert m.layers == []


def test_fast_mode_filters_tools() -> None:
    m = MockLeafmap()
    items = leafmap_tools(m)
    reg = GeoToolRegistry()
    register_all_tools(reg, items)
    fast = collect_tools_for_context(items, fast=True, registry=reg)
    fast_names = {t.tool_name for t in fast}
    assert "list_layers" in fast_names
    assert "add_stac_layer" not in fast_names


def test_get_map_state_and_viewport() -> None:
    m = MockLeafmap()
    tools = _by_name(m)
    tools["set_center"](lat=35.96, lon=-83.92, zoom=10)
    tools["zoom_in"](steps=1)
    state = tools["get_map_state"]()
    assert isinstance(state, dict)
    assert "zoom" in state


def test_get_map_state_prefers_view_state_maplibre() -> None:
    """MapLibre leafmap stores camera in ``view_state``, not ipyleaflet center/zoom."""

    class MapLibreStub:
        def __init__(self) -> None:
            self.layers: list = []
            self._style = "dark-matter"
            self.view_state = {
                "center": {"lng": -83.92, "lat": 35.96},
                "zoom": 9,
                "bounds": {
                    "_sw": {"lng": -84.82, "lat": 35.67},
                    "_ne": {"lng": -83.02, "lat": 36.25},
                },
                "bearing": 0,
                "pitch": 0,
            }

    tools = _by_name(MapLibreStub())
    state = tools["get_map_state"]()
    assert state["zoom"] == 9
    assert state["view_state"]["center"]["lng"] == -83.92
    assert "_sw" in state["bounds"]


def test_get_map_state_falls_back_to_map_options_when_view_state_empty() -> None:
    """MapLibre can expose empty view_state until frontend sync."""

    class MapLibreStub:
        def __init__(self) -> None:
            self.layers: list = []
            self._style = None
            self.view_state = {}
            self.map_options = {
                "center": (-83.92, 35.96),
                "zoom": 9,
                "bearing": 0,
                "pitch": 0,
                "style": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
            }

    tools = _by_name(MapLibreStub())
    state = tools["get_map_state"]()
    assert state["center"] == {"lng": -83.92, "lat": 35.96}
    assert state["zoom"] == 9
    assert (
        state["basemap"]
        == "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
    )


def test_save_map_writes(tmp_path: Path) -> None:
    m = MockLeafmap()
    tools = _by_name(m)
    out = tmp_path / "x.html"
    result = tools["save_map"](path=str(out))
    assert isinstance(result, str)
    assert Path(result).exists()
