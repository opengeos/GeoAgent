"""Tests for the Vantor GeoAgent tool factory."""

from __future__ import annotations

import sys

import pytest

from geoagent import for_vantor
from geoagent.testing import MockQGISIface, MockQGISProject
from geoagent.tools import vantor_tools
import geoagent.tools.vantor as vantor


class _MockModel:
    """Tiny model stand-in for GeoAgent factory tests."""

    stateful = False


EVENT_URL = "https://example.com/events/flood-2026/collection.json"
ITEM_1_URL = "https://example.com/events/flood-2026/pre.json"
ITEM_2_URL = "https://example.com/events/flood-2026/post.json"


ROOT_CATALOG = {
    "links": [
        {
            "rel": "child",
            "href": "https://example.com/events/flood-2026/collection.json",
            "title": "Flood 2026",
        }
    ]
}


COLLECTION = {
    "id": "flood-2026",
    "title": "Flood 2026",
    "description": "Example event",
    "license": "public-domain",
    "extent": {
        "spatial": {"bbox": [[-85.0, 34.0, -83.0, 36.0]]},
        "temporal": {"interval": [["2026-01-01T00:00:00Z", None]]},
    },
    "links": [
        {"rel": "item", "href": "pre.json"},
        {"rel": "item", "href": "post.json"},
    ],
}


ITEM_PRE = {
    "type": "Feature",
    "id": "flood-pre",
    "collection": "flood-2026",
    "bbox": [-84.9, 34.1, -84.0, 35.0],
    "geometry": {
        "type": "Polygon",
        "coordinates": [
            [
                [-84.9, 34.1],
                [-84.0, 34.1],
                [-84.0, 35.0],
                [-84.9, 35.0],
                [-84.9, 34.1],
            ]
        ],
    },
    "properties": {
        "datetime": "2026-01-01T00:00:00Z",
        "phase": "pre",
        "vehicle_name": "WorldView",
        "eo:cloud_cover": 2.5,
        "pan_gsd": 0.31,
    },
    "assets": {
        "visual": {
            "href": "https://example.com/pre.tif",
            "type": "image/tiff; application=geotiff",
        },
        "thumbnail": {"href": "https://example.com/pre.png", "type": "image/png"},
    },
}


ITEM_POST = {
    "type": "Feature",
    "id": "flood-post",
    "collection": "flood-2026",
    "bbox": [-83.9, 34.2, -83.1, 35.1],
    "geometry": {
        "type": "Polygon",
        "coordinates": [
            [
                [-83.9, 34.2],
                [-83.1, 34.2],
                [-83.1, 35.1],
                [-83.9, 35.1],
                [-83.9, 34.2],
            ]
        ],
    },
    "properties": {
        "datetime": "2026-01-02T00:00:00Z",
        "phase": "post",
        "vehicle_name": "WorldView",
        "eo:cloud_cover": 8.0,
    },
    "assets": {
        "visual": {
            "href": "https://example.com/post.tif",
            "type": "image/tiff; application=geotiff",
        }
    },
}


def _install_fake_vantor_fetch(monkeypatch) -> None:
    """Mock Vantor STAC JSON responses."""
    payloads = {
        vantor.VANTOR_CATALOG_URL: ROOT_CATALOG,
        EVENT_URL: COLLECTION,
        ITEM_1_URL: ITEM_PRE,
        ITEM_2_URL: ITEM_POST,
    }

    def _fake_fetch(url: str):
        return payloads[url]

    monkeypatch.setattr(vantor, "_fetch_json", _fake_fetch)


def test_vantor_module_imports_without_qgis_or_plugin() -> None:
    """Verify Vantor tools are import-safe outside QGIS."""
    assert "geoagent.tools.vantor" in sys.modules
    if "qgis" in sys.modules:
        pytest.skip("qgis is already imported in this environment.")
    assert "qgis" not in sys.modules
    assert "vantor" not in sys.modules


def test_vantor_tools_returns_empty_for_none_iface() -> None:
    """Verify the Vantor factory returns no tools without iface."""
    assert vantor_tools(None) == []


def test_vantor_tools_expose_expected_surface() -> None:
    """Verify Vantor tool names and confirmation metadata."""
    tools = {tool.tool_name: tool for tool in vantor_tools(object())}

    assert set(tools) == {
        "list_vantor_events",
        "get_vantor_event_info",
        "get_current_vantor_search_extent",
        "search_vantor_items",
        "display_vantor_footprints",
        "load_vantor_cog",
        "open_vantor_panel",
    }
    assert tools["display_vantor_footprints"]._geoagent_meta.requires_confirmation
    assert tools["load_vantor_cog"]._geoagent_meta.requires_confirmation


def test_vantor_event_listing_and_info(monkeypatch) -> None:
    """Verify event listing and metadata inspection use static STAC JSON."""
    _install_fake_vantor_fetch(monkeypatch)
    tools = {tool.tool_name: tool for tool in vantor_tools(object())}

    events = tools["list_vantor_events"].__wrapped__()
    info = tools["get_vantor_event_info"].__wrapped__("Flood 2026")

    assert events["success"] is True
    assert events["count"] == 1
    assert events["events"][0]["id"] == "flood-2026"
    assert info["success"] is True
    assert info["id"] == "flood-2026"
    assert info["item_count"] == 2


def test_search_vantor_items_filters_bbox_phase_and_returns_cog(monkeypatch) -> None:
    """Verify item search filters bbox/phase and returns compact COG metadata."""
    _install_fake_vantor_fetch(monkeypatch)
    tools = {tool.tool_name: tool for tool in vantor_tools(object())}

    result = tools["search_vantor_items"].__wrapped__(
        "flood-2026",
        bbox="-85,34,-84,35.5",
        phase="pre-event",
    )

    assert result["success"] is True
    assert result["count"] == 1
    assert result["items"][0]["id"] == "flood-pre"
    assert result["items"][0]["cog_url"] == "https://example.com/pre.tif"
    assert result["items"][0]["thumbnail_url"] == "https://example.com/pre.png"


def test_load_vantor_cog_uses_vsicurl_uri(monkeypatch) -> None:
    """Verify Vantor COG loading forms a QGIS /vsicurl raster URI."""
    _install_fake_vantor_fetch(monkeypatch)
    project = MockQGISProject()
    iface = MockQGISIface(project)
    tools = {tool.tool_name: tool for tool in vantor_tools(iface, project)}

    tools["search_vantor_items"].__wrapped__("flood-2026", phase="post")
    result = tools["load_vantor_cog"].__wrapped__(item_id="flood-post")

    assert result["success"] is True
    assert result["loaded"] is True
    assert result["qgis_uri"] == "/vsicurl/https://example.com/post.tif"
    assert project.mapLayers()["flood-post"].source() == result["qgis_uri"]


def test_display_vantor_footprints_adds_geojson_layer(monkeypatch) -> None:
    """Verify footprint display writes GeoJSON and adds a vector layer."""
    _install_fake_vantor_fetch(monkeypatch)
    project = MockQGISProject()
    iface = MockQGISIface(project)
    tools = {tool.tool_name: tool for tool in vantor_tools(iface, project)}

    result = tools["display_vantor_footprints"].__wrapped__(
        event="flood-2026",
        phase="all",
        layer_name="Vantor Test Footprints",
    )

    assert result["success"] is True
    assert result["feature_count"] == 2
    assert "Vantor Test Footprints" in project.mapLayers()


def test_open_vantor_panel_uses_plugin_instance() -> None:
    """Verify panel opening delegates to the supplied plugin instance."""

    class _Dock:
        def __init__(self) -> None:
            self.shown = False
            self.raised = False

        def show(self) -> None:
            self.shown = True

        def raise_(self) -> None:
            self.raised = True

    class _Plugin:
        def __init__(self) -> None:
            self._main_dock = None
            self.toggled = False

        def toggle_main_dock(self) -> None:
            self.toggled = True
            self._main_dock = _Dock()

    plugin = _Plugin()
    tools = {tool.tool_name: tool for tool in vantor_tools(object(), plugin=plugin)}

    result = tools["open_vantor_panel"].__wrapped__()

    assert result == {"success": True, "opened": True}
    assert plugin.toggled is True
    assert plugin._main_dock.shown is True
    assert plugin._main_dock.raised is True


def test_for_vantor_registers_vantor_and_qgis_tools() -> None:
    """Verify the Vantor factory combines Vantor and QGIS tools."""
    agent = for_vantor(
        MockQGISIface(),
        MockQGISProject(),
        model=_MockModel(),
    )
    names = set(agent.strands_agent.tool_names)

    assert "list_vantor_events" in names
    assert "search_vantor_items" in names
    assert "load_vantor_cog" in names
    assert "list_project_layers" in names
    assert agent.context.metadata["integration"] == "vantor"
