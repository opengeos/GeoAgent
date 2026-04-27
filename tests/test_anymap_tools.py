"""Tests for the anymap tool factory."""

from __future__ import annotations

from geoagent.core.decorators import needs_confirmation
from geoagent.testing import MockAnymap
from geoagent.tools.anymap import anymap_tools


def test_factory_returns_tools() -> None:
    m = MockAnymap()
    tools = anymap_tools(m)
    names = {t.name for t in tools}
    assert {"list_layers", "add_layer", "remove_layer", "save_map"}.issubset(names)


def test_factory_returns_empty_for_none() -> None:
    assert anymap_tools(None) == []


def test_remove_and_save_require_confirmation() -> None:
    tools = {t.name: t for t in anymap_tools(MockAnymap())}
    assert needs_confirmation(tools["remove_layer"]) is True
    assert needs_confirmation(tools["save_map"]) is True


def test_add_layer_mutates_map() -> None:
    m = MockAnymap()
    tools = {t.name: t for t in anymap_tools(m)}
    tools["add_layer"].invoke(
        {"url": "https://example.com/grid.geojson", "name": "Grid"}
    )
    assert any(layer.get("name") == "Grid" for layer in m.layers)


def test_zoom_to_bounds() -> None:
    m = MockAnymap()
    tools = {t.name: t for t in anymap_tools(m)}
    tools["zoom_to_bounds"].invoke(
        {"west": 0.0, "south": 0.0, "east": 10.0, "north": 10.0}
    )
    assert m._bounds == [[0.0, 0.0], [10.0, 10.0]]
