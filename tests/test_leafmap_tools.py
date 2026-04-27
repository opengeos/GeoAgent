"""Tests for the leafmap tool factory.

These tests use :class:`MockLeafmap` so they run on systems without leafmap
installed.
"""

from __future__ import annotations

from geoagent.core.decorators import needs_confirmation
from geoagent.testing import MockLeafmap
from geoagent.tools.leafmap import leafmap_tools


def test_factory_returns_full_tool_list() -> None:
    m = MockLeafmap()
    tools = leafmap_tools(m)
    names = {t.name for t in tools}
    expected = {
        "list_layers",
        "add_layer",
        "remove_layer",
        "set_center",
        "set_zoom",
        "zoom_in",
        "zoom_out",
        "zoom_to_bounds",
        "change_basemap",
        "add_vector_data",
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
    tools = {t.name: t for t in leafmap_tools(MockLeafmap())}
    assert needs_confirmation(tools["remove_layer"]) is True
    assert needs_confirmation(tools["save_map"]) is True
    assert needs_confirmation(tools["list_layers"]) is False
    assert needs_confirmation(tools["zoom_in"]) is False


def test_add_layer_mutates_map() -> None:
    m = MockLeafmap()
    tools = {t.name: t for t in leafmap_tools(m)}

    tools["add_layer"].invoke(
        {"url": "https://example.com/data.tif", "name": "DEM", "layer_type": "cog"}
    )
    assert any(layer.get("name") == "DEM" for layer in m.layers)


def test_remove_layer_mutates_map() -> None:
    m = MockLeafmap()
    tools = {t.name: t for t in leafmap_tools(m)}
    tools["add_layer"].invoke(
        {"url": "https://example.com/buildings.geojson", "name": "Buildings"}
    )
    assert len(m.layers) == 1
    tools["remove_layer"].invoke({"name": "Buildings"})
    assert len(m.layers) == 0


def test_set_center_and_zoom_in() -> None:
    m = MockLeafmap()
    tools = {t.name: t for t in leafmap_tools(m)}
    tools["set_center"].invoke({"lat": 37.7, "lon": -122.4, "zoom": 10})
    assert m.center == [-122.4, 37.7]
    assert m.zoom == 10
    tools["zoom_in"].invoke({"steps": 2})
    assert m.zoom == 12


def test_zoom_to_bounds() -> None:
    m = MockLeafmap()
    tools = {t.name: t for t in leafmap_tools(m)}
    tools["zoom_to_bounds"].invoke(
        {"west": -125.0, "south": 24.0, "east": -66.0, "north": 49.0}
    )
    assert m._bounds == [[-125.0, 24.0], [-66.0, 49.0]]


def test_change_basemap() -> None:
    m = MockLeafmap()
    tools = {t.name: t for t in leafmap_tools(m)}
    tools["change_basemap"].invoke({"basemap": "CartoDB.Positron"})
    assert m._style == "CartoDB.Positron"


def test_get_map_state() -> None:
    m = MockLeafmap()
    tools = {t.name: t for t in leafmap_tools(m)}
    tools["set_center"].invoke({"lat": 35.0, "lon": -83.9, "zoom": 8})
    state = tools["get_map_state"].invoke({})
    assert state["zoom"] == 8
    assert state["center"] == [-83.9, 35.0]


def test_save_map_writes_file(tmp_path) -> None:
    m = MockLeafmap()
    tools = {t.name: t for t in leafmap_tools(m)}
    out = tmp_path / "map.html"
    result = tools["save_map"].invoke({"path": str(out)})
    assert out.exists()
    assert str(out.resolve()) == result


def test_list_layers_returns_dicts() -> None:
    m = MockLeafmap()
    tools = {t.name: t for t in leafmap_tools(m)}
    tools["add_layer"].invoke(
        {"url": "https://example.com/buildings.geojson", "name": "Buildings"}
    )
    layers = tools["list_layers"].invoke({})
    assert layers == [{"name": "Buildings", "type": "geojson"}]
