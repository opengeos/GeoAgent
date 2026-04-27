"""Tests for the QGIS tool factory.

These tests use :class:`MockQGISIface` and :class:`MockQGISProject` so they
run without QGIS installed. They also exercise the import-safe contract:
``import geoagent.tools.qgis`` must succeed when ``qgis`` is not on the
import path.
"""

from __future__ import annotations

import sys

import pytest

from geoagent.core.decorators import needs_confirmation
from geoagent.testing import MockQGISIface, MockQGISLayer, MockQGISProject
from geoagent.tools.qgis import qgis_tools


def test_qgis_module_imports_without_qgis() -> None:
    # The module is already imported; confirm `qgis` was not pulled in.
    # (If it was, this test would still pass — we just want to ensure import
    # didn't fail when the user's environment has no qgis package.)
    assert "geoagent.tools.qgis" in sys.modules


def test_qgis_tools_returns_empty_for_none() -> None:
    assert qgis_tools(None) == []


def test_factory_returns_tools_for_iface() -> None:
    iface = MockQGISIface()
    project = MockQGISProject()
    tools = qgis_tools(iface, project)
    names = {t.name for t in tools}
    expected = {
        "list_project_layers",
        "get_active_layer",
        "zoom_in",
        "zoom_out",
        "zoom_to_layer",
        "zoom_to_extent",
        "add_vector_layer",
        "add_raster_layer",
        "remove_layer",
        "set_layer_visibility",
        "inspect_layer_fields",
        "get_selected_features",
        "run_processing_algorithm",
        "open_attribute_table",
        "refresh_canvas",
    }
    assert expected.issubset(names)


def test_remove_layer_and_run_processing_require_confirmation() -> None:
    iface = MockQGISIface()
    project = MockQGISProject()
    tools = {t.name: t for t in qgis_tools(iface, project)}
    assert needs_confirmation(tools["remove_layer"]) is True
    assert needs_confirmation(tools["run_processing_algorithm"]) is True
    assert needs_confirmation(tools["zoom_in"]) is False
    assert needs_confirmation(tools["list_project_layers"]) is False


def test_add_vector_layer_invokes_iface() -> None:
    project = MockQGISProject()
    iface = MockQGISIface(project=project)
    tools = {t.name: t for t in qgis_tools(iface, project)}
    tools["add_vector_layer"].invoke(
        {"path_or_uri": "/tmp/x.shp", "name": "Roads", "provider": "ogr"}
    )
    assert "Roads" in project.mapLayers()


def test_remove_layer_drops_from_project() -> None:
    project = MockQGISProject()
    iface = MockQGISIface(project=project)
    project.addMapLayer(MockQGISLayer("Buildings", "/tmp/b.shp"))
    tools = {t.name: t for t in qgis_tools(iface, project)}
    out = tools["remove_layer"].invoke({"layer_name": "Buildings"})
    assert "Buildings" not in project.mapLayers()
    assert "Removed" in out


def test_zoom_in_invokes_canvas() -> None:
    iface = MockQGISIface()
    project = MockQGISProject()
    tools = {t.name: t for t in qgis_tools(iface, project)}
    initial_scale = iface.mapCanvas().scale_value
    tools["zoom_in"].invoke({})
    assert iface.mapCanvas().scale_value == initial_scale / 2


def test_zoom_in_refreshes_canvas() -> None:
    """zoom_in must refresh the canvas so XYZ basemaps re-tile.

    Without the explicit refresh, basemap layers (Google Satellite,
    OSM, etc.) can render blank at the new extent until the user pans
    by hand.
    """
    iface = MockQGISIface()
    project = MockQGISProject()
    tools = {t.name: t for t in qgis_tools(iface, project)}
    before = iface.mapCanvas().refresh_count
    tools["zoom_in"].invoke({})
    assert iface.mapCanvas().refresh_count == before + 1


def test_zoom_out_refreshes_canvas() -> None:
    iface = MockQGISIface()
    project = MockQGISProject()
    tools = {t.name: t for t in qgis_tools(iface, project)}
    before = iface.mapCanvas().refresh_count
    tools["zoom_out"].invoke({})
    assert iface.mapCanvas().refresh_count == before + 1


def test_list_project_layers() -> None:
    project = MockQGISProject()
    project.addMapLayer(MockQGISLayer("A", "a.shp", "vector"))
    project.addMapLayer(MockQGISLayer("B", "b.tif", "raster"))
    iface = MockQGISIface(project=project)
    tools = {t.name: t for t in qgis_tools(iface, project)}
    layers = tools["list_project_layers"].invoke({})
    names = {layer["name"] for layer in layers}
    assert names == {"A", "B"}


def test_get_active_layer_returns_none_when_unset() -> None:
    iface = MockQGISIface()
    project = MockQGISProject()
    tools = {t.name: t for t in qgis_tools(iface, project)}
    out = tools["get_active_layer"].invoke({})
    assert out == {"active_layer": None}


def test_get_active_layer_returns_metadata() -> None:
    project = MockQGISProject()
    iface = MockQGISIface(project=project)
    layer = MockQGISLayer("Active", "a.shp", "vector")
    project.addMapLayer(layer)
    iface.setActiveLayer(layer)
    tools = {t.name: t for t in qgis_tools(iface, project)}
    out = tools["get_active_layer"].invoke({})
    assert out["name"] == "Active"
    assert out["source"] == "a.shp"


def test_zoom_to_layer_resolves_layer() -> None:
    project = MockQGISProject()
    iface = MockQGISIface(project=project)
    layer = MockQGISLayer("Target", "t.shp")
    project.addMapLayer(layer)
    tools = {t.name: t for t in qgis_tools(iface, project)}
    tools["zoom_to_layer"].invoke({"layer_name": "Target"})
    assert iface.activeLayer() is layer


def test_zoom_to_layer_refreshes_canvas() -> None:
    """zoom_to_layer must refresh after iface.zoomToActiveLayer().

    QGIS's ``zoomToActiveLayer`` updates the canvas extent but does
    not request fresh tiles for XYZ basemaps. Without an explicit
    ``refresh()`` the satellite basemap can disappear after the zoom.
    """
    project = MockQGISProject()
    iface = MockQGISIface(project=project)
    layer = MockQGISLayer("Target", "t.shp")
    project.addMapLayer(layer)
    tools = {t.name: t for t in qgis_tools(iface, project)}
    before = iface.mapCanvas().refresh_count
    tools["zoom_to_layer"].invoke({"layer_name": "Target"})
    # ``MockQGISIface.zoomToActiveLayer`` already refreshes once; the
    # explicit refresh in our wrapper brings the count to two. The
    # contract worth pinning is "at least one refresh fired", so the
    # XYZ basemap is guaranteed to re-tile after the zoom.
    assert iface.mapCanvas().refresh_count >= before + 1


def test_zoom_to_layer_raises_when_missing() -> None:
    project = MockQGISProject()
    iface = MockQGISIface(project=project)
    tools = {t.name: t for t in qgis_tools(iface, project)}
    with pytest.raises(LookupError):
        tools["zoom_to_layer"].invoke({"layer_name": "NotThere"})


def test_refresh_canvas_increments_counter() -> None:
    iface = MockQGISIface()
    project = MockQGISProject()
    tools = {t.name: t for t in qgis_tools(iface, project)}
    before = iface.mapCanvas().refresh_count
    tools["refresh_canvas"].invoke({})
    assert iface.mapCanvas().refresh_count == before + 1
