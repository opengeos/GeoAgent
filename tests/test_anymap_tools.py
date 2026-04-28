"""Tests for the anymap tool factory."""

from __future__ import annotations

from geoagent.core.decorators import needs_confirmation
from geoagent.testing import MockAnymap
from geoagent.tools.anymap import anymap_tools


def _by_name(m: MockAnymap) -> dict[str, object]:
    """Index anymap tools by Strands tool name."""
    return {t.tool_name: t for t in anymap_tools(m)}


def test_anymap_tool_surface_has_leafmap_parity() -> None:
    """Verify that anymap tool surface has leafmap parity."""
    names = {t.tool_name for t in anymap_tools(MockAnymap())}
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


def test_anymap_destructive_tools_require_confirmation() -> None:
    """Verify that anymap destructive tools require confirmation."""
    tools = _by_name(MockAnymap())
    assert needs_confirmation(tools["remove_layer"]) is True
    assert needs_confirmation(tools["clear_layers"]) is True
    assert needs_confirmation(tools["save_map"]) is True


def test_anymap_layer_controls_and_data_helpers() -> None:
    """Verify that anymap layer controls and data helpers."""
    m = MockAnymap()
    tools = _by_name(m)
    tools["add_marker"](lat=35.96, lon=-83.92, name="Knoxville")
    tools["add_geojson_data"](
        data={"type": "FeatureCollection", "features": []},
        name="Empty GeoJSON",
    )
    m.layers[0]["bounds"] = [-84.1, 35.8, -83.7, 36.1]

    tools["set_layer_visibility"](name="knox", visible=False)
    tools["set_layer_opacity"](name="knox", opacity=2)
    tools["zoom_to_layer"](name="knox")

    assert m.layers[0]["visible"] is False
    assert m.layers[0]["opacity"] == 1.0
    assert m.get_bounds() == [[-84.1, 35.8], [-83.7, 36.1]]
    assert any(layer["name"] == "Empty GeoJSON" for layer in m.layers)
