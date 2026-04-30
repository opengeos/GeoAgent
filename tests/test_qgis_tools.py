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
    """Verify the qgis tools module imports without the qgis package.

    The module is already imported; confirm ``qgis`` was not pulled in.
    (If it was, this test would still pass; we just want to ensure import
    did not fail when the user's environment has no qgis package.)
    """
    assert "geoagent.tools.qgis" in sys.modules


def test_qgis_tools_returns_empty_for_none() -> None:
    """Verify that qgis tools returns empty for none."""
    assert qgis_tools(None) == []


def test_factory_returns_tools_for_iface() -> None:
    """Verify that factory returns tools for iface."""
    iface = MockQGISIface()
    project = MockQGISProject()
    tools = qgis_tools(iface, project)
    names = {t.tool_name for t in tools}
    expected = {
        "list_project_layers",
        "get_active_layer",
        "get_project_state",
        "zoom_in",
        "zoom_out",
        "zoom_to_layer",
        "zoom_to_extent",
        "set_center",
        "set_scale",
        "add_vector_layer",
        "add_raster_layer",
        "add_xyz_tile_layer",
        "remove_layer",
        "set_layer_visibility",
        "set_layer_opacity",
        "set_layer_symbology",
        "inspect_layer_fields",
        "get_selected_features",
        "select_features_by_expression",
        "clear_selection",
        "zoom_to_selected",
        "get_layer_summary",
        "run_processing_algorithm",
        "buffer_active_layer",
        "open_attribute_table",
        "refresh_canvas",
        "save_project",
    }
    assert expected.issubset(names)


def test_remove_layer_and_run_processing_require_confirmation() -> None:
    """Verify that remove layer and run processing require confirmation."""
    iface = MockQGISIface()
    project = MockQGISProject()
    tools = {t.tool_name: t for t in qgis_tools(iface, project)}
    assert needs_confirmation(tools["remove_layer"]) is True
    assert needs_confirmation(tools["run_processing_algorithm"]) is True
    assert needs_confirmation(tools["buffer_active_layer"]) is True
    assert needs_confirmation(tools["save_project"]) is True
    assert needs_confirmation(tools["zoom_in"]) is False
    assert needs_confirmation(tools["list_project_layers"]) is False


def test_add_vector_layer_invokes_iface() -> None:
    """Verify that add vector layer invokes iface."""
    project = MockQGISProject()
    iface = MockQGISIface(project=project)
    tools = {t.tool_name: t for t in qgis_tools(iface, project)}
    tools["add_vector_layer"](
        path_or_uri="/tmp/x.shp",
        name="Roads",
        provider="ogr",
    )
    assert "Roads" in project.mapLayers()


def test_remove_layer_drops_from_project() -> None:
    """Verify that remove layer drops from project."""
    project = MockQGISProject()
    iface = MockQGISIface(project=project)
    project.addMapLayer(MockQGISLayer("Buildings", "/tmp/b.shp"))
    tools = {t.tool_name: t for t in qgis_tools(iface, project)}
    out = tools["remove_layer"](layer_name="Buildings")
    assert "Buildings" not in project.mapLayers()
    assert "Removed" in out


def test_zoom_in_invokes_canvas() -> None:
    """Verify that zoom in invokes canvas."""
    iface = MockQGISIface()
    project = MockQGISProject()
    tools = {t.tool_name: t for t in qgis_tools(iface, project)}
    initial_scale = iface.mapCanvas().scale_value
    tools["zoom_in"]()
    assert iface.mapCanvas().scale_value == initial_scale / 2


def test_list_project_layers() -> None:
    """Verify that list project layers."""
    project = MockQGISProject()
    project.addMapLayer(MockQGISLayer("A", "a.shp", "vector"))
    project.addMapLayer(MockQGISLayer("B", "b.tif", "raster"))
    iface = MockQGISIface(project=project)
    tools = {t.tool_name: t for t in qgis_tools(iface, project)}
    layers = tools["list_project_layers"]()
    names = {layer["name"] for layer in layers}
    assert names == {"A", "B"}
    assert all("id" in layer for layer in layers)


def test_get_active_layer_returns_none_when_unset() -> None:
    """Verify that get active layer returns none when unset."""
    iface = MockQGISIface()
    project = MockQGISProject()
    tools = {t.tool_name: t for t in qgis_tools(iface, project)}
    out = tools["get_active_layer"]()
    assert out == {"active_layer": None}


def test_get_active_layer_returns_metadata() -> None:
    """Verify that get active layer returns metadata."""
    project = MockQGISProject()
    iface = MockQGISIface(project=project)
    layer = MockQGISLayer("Active", "a.shp", "vector")
    project.addMapLayer(layer)
    iface.setActiveLayer(layer)
    tools = {t.tool_name: t for t in qgis_tools(iface, project)}
    out = tools["get_active_layer"]()
    assert out["name"] == "Active"
    assert out["source"] == "a.shp"


def test_get_project_state_includes_canvas_and_layers() -> None:
    """Verify that get project state includes canvas and layers."""
    project = MockQGISProject()
    project.addMapLayer(MockQGISLayer("A", "a.shp", "vector"))
    iface = MockQGISIface(project=project)
    tools = {t.tool_name: t for t in qgis_tools(iface, project)}
    state = tools["get_project_state"]()
    assert state["canvas"]["extent"] == [-180.0, -90.0, 180.0, 90.0]
    assert state["layers"][0]["name"] == "A"


def test_zoom_to_layer_resolves_layer() -> None:
    """Verify that zoom to layer resolves layer."""
    project = MockQGISProject()
    iface = MockQGISIface(project=project)
    layer = MockQGISLayer("Target", "t.shp")
    project.addMapLayer(layer)
    tools = {t.tool_name: t for t in qgis_tools(iface, project)}
    tools["zoom_to_layer"](layer_name="Target")
    assert iface.activeLayer() is layer


def test_zoom_to_layer_transforms_extent_when_layer_crs_differs(monkeypatch) -> None:
    """A layer in a different CRS than the canvas must have its extent
    transformed before ``setExtent`` is called.

    Without the transform, a GeoJSON loaded as EPSG:4326 hands lat/lon
    coordinates (-115.x, 36.x) to a Web-Mercator canvas (EPSG:3857),
    which interprets them as metres near (0, 0) and renders blank.
    Faking a tiny ``qgis.core`` so the transform path runs end-to-end
    in tests without needing real QGIS.
    """
    import sys
    import types

    project = MockQGISProject()
    iface = MockQGISIface(project=project)
    canvas = iface.mapCanvas()

    layer_extent = (-115.3, 36.1, -115.0, 36.3)  # lat/lon
    layer = MockQGISLayer("Buildings", "/tmp/b.shp", extent=layer_extent)
    project.addMapLayer(layer)

    class _Crs:
        """Provide a test double for Crs."""

        def __init__(self, name):
            self.name = name

        def __eq__(self, other):
            return isinstance(other, _Crs) and self.name == other.name

        def __hash__(self):
            return hash(self.name)

    layer_crs = _Crs("EPSG:4326")
    canvas_crs = _Crs("EPSG:3857")
    layer.crs = lambda: layer_crs  # type: ignore[method-assign]

    class _MapSettings:
        """Provide a test double for MapSettings."""

        def destinationCrs(self):
            """Return the destination crs."""
            return canvas_crs

    canvas.mapSettings = lambda: _MapSettings()  # type: ignore[method-assign]

    transformed_extent = (-12_835_000.0, 4_330_000.0, -12_801_000.0, 4_355_000.0)

    class _CoordTransform:
        """Provide a test double for CoordTransform."""

        def __init__(self, src, dst, project_arg):
            assert src is layer_crs
            assert dst is canvas_crs

        def transformBoundingBox(self, extent):
            """Transform bounding box."""
            assert extent == layer_extent
            return transformed_extent

    class _QgsProject:
        """Provide a test double for QgsProject."""

        @staticmethod
        def instance():
            """Return the singleton instance."""
            return object()

    fake_qgs_core = types.SimpleNamespace(
        QgsCoordinateTransform=_CoordTransform,
        QgsProject=_QgsProject,
    )
    fake_qgis = types.SimpleNamespace(core=fake_qgs_core)
    monkeypatch.setitem(sys.modules, "qgis", fake_qgis)
    monkeypatch.setitem(sys.modules, "qgis.core", fake_qgs_core)

    tools = {t.tool_name: t for t in qgis_tools(iface, project)}
    tools["zoom_to_layer"](layer_name="Buildings")

    assert canvas.extent() == transformed_extent, (
        "zoom_to_layer must hand the canvas an extent in the canvas CRS, "
        "not in the layer's native CRS"
    )


def test_zoom_to_extent_transforms_bbox_to_canvas_crs(monkeypatch) -> None:
    """``zoom_to_extent`` must reproject the bbox into the canvas CRS.

    The default ``crs`` argument is ``"EPSG:4326"`` because LLMs
    naturally produce place-name extents in lat/lon. A Web-Mercator
    canvas (EPSG:3857) cannot consume those numbers directly without a
    transform — passing them straight to ``setExtent`` zooms to a
    sliver near (0, 0) in metres and renders blank. Faking a tiny
    ``qgis.core`` so the transform path runs end-to-end in tests
    without needing real QGIS.
    """
    import sys
    import types

    iface = MockQGISIface()
    canvas = iface.mapCanvas()

    class _Crs:
        """Provide a test double for Crs."""

        def __init__(self, authid):
            self.authid = authid

        def __eq__(self, other):
            return isinstance(other, _Crs) and self.authid == other.authid

        def __hash__(self):
            return hash(self.authid)

    src_crs = _Crs("EPSG:4326")
    dst_crs = _Crs("EPSG:3857")

    class _MapSettings:
        """Provide a test double for MapSettings."""

        def destinationCrs(self):
            """Return the destination crs."""
            return dst_crs

    canvas.mapSettings = lambda: _MapSettings()  # type: ignore[method-assign]

    seattle_latlon = (-122.5, 47.5, -122.2, 47.7)
    seattle_mercator = (-13_637_000.0, 6_038_000.0, -13_604_000.0, 6_069_000.0)

    # Use a simple 4-tuple as the QgsRectangle stand-in so MockQGISCanvas's
    # ``setExtent`` (which calls ``tuple(extent)``) can record it cleanly.
    def _Rect(w, s, e, n):
        """Return a tuple that stands in for QgsRectangle."""
        return (w, s, e, n)

    class _CoordTransform:
        """Provide a test double for CoordTransform."""

        def __init__(self, src, dst, project_arg):
            assert src == src_crs
            assert dst == dst_crs

        def transformBoundingBox(self, rect):
            """Transform bounding box."""
            assert tuple(rect) == seattle_latlon
            return _Rect(*seattle_mercator)

    class _QgsProject:
        """Provide a test double for QgsProject."""

        @staticmethod
        def instance():
            """Return the singleton instance."""
            return object()

    fake_qgs_core = types.SimpleNamespace(
        QgsCoordinateReferenceSystem=lambda authid: _Crs(authid),
        QgsCoordinateTransform=_CoordTransform,
        QgsProject=_QgsProject,
        QgsRectangle=_Rect,
    )
    fake_qgis = types.SimpleNamespace(core=fake_qgs_core)
    monkeypatch.setitem(sys.modules, "qgis", fake_qgis)
    monkeypatch.setitem(sys.modules, "qgis.core", fake_qgs_core)

    project = MockQGISProject()
    tools = {t.tool_name: t for t in qgis_tools(iface, project)}
    tools["zoom_to_extent"](
        west=seattle_latlon[0],
        south=seattle_latlon[1],
        east=seattle_latlon[2],
        north=seattle_latlon[3],
    )

    assert canvas.extent() == seattle_mercator, (
        "zoom_to_extent must hand the canvas a bbox in the canvas CRS, "
        "not in the LLM-supplied lat/lon CRS"
    )


def test_zoom_to_layer_uses_setExtent_when_extent_available() -> None:
    """Layers with ``extent()`` must drive ``setExtent`` + ``refresh``.

    ``iface.zoomToActiveLayer()`` updates the canvas extent but does
    not always trigger XYZ tile providers (Google Satellite, OSM, etc.)
    to refetch tiles at the new zoom-pyramid level — basemaps stay
    pixelated on upscaled lower-resolution tiles. Routing through
    ``canvas.setExtent()`` + ``canvas.refresh()`` mirrors the path
    QGIS uses for user-driven zoom and resolves the tile pyramid
    correctly.
    """
    project = MockQGISProject()
    iface = MockQGISIface(project=project)
    canvas = iface.mapCanvas()
    layer = MockQGISLayer(
        "Buildings", "/tmp/b.shp", extent=(-115.3, 36.1, -115.0, 36.3)
    )
    project.addMapLayer(layer)
    tools = {t.tool_name: t for t in qgis_tools(iface, project)}
    before_refresh = canvas.refresh_count
    tools["zoom_to_layer"](layer_name="Buildings")
    assert canvas.extent() == (-115.3, 36.1, -115.0, 36.3)
    assert canvas.refresh_count == before_refresh + 1


def test_zoom_to_layer_raises_when_missing() -> None:
    """Verify that zoom to layer raises when missing."""
    project = MockQGISProject()
    iface = MockQGISIface(project=project)
    tools = {t.tool_name: t for t in qgis_tools(iface, project)}
    with pytest.raises(LookupError):
        tools["zoom_to_layer"](layer_name="NotThere")


def test_refresh_canvas_increments_counter() -> None:
    """Verify that refresh canvas increments counter."""
    iface = MockQGISIface()
    project = MockQGISProject()
    tools = {t.tool_name: t for t in qgis_tools(iface, project)}
    before = iface.mapCanvas().refresh_count
    tools["refresh_canvas"]()
    assert iface.mapCanvas().refresh_count == before + 1


def test_set_center_and_scale_update_canvas() -> None:
    """Verify that set center and scale update canvas."""
    iface = MockQGISIface()
    project = MockQGISProject()
    tools = {t.tool_name: t for t in qgis_tools(iface, project)}
    tools["set_center"](lat=10, lon=20, scale=500)
    assert iface.mapCanvas().scale_value == 500
    assert iface.mapCanvas().refresh_count == 1
    tools["set_scale"](scale=250)
    assert iface.mapCanvas().scale_value == 250


def test_add_xyz_tile_layer_uses_raster_fallback() -> None:
    """Verify that add xyz tile layer uses raster fallback."""
    project = MockQGISProject()
    iface = MockQGISIface(project=project)
    tools = {t.tool_name: t for t in qgis_tools(iface, project)}
    out = tools["add_xyz_tile_layer"](
        url="https://example.com/{z}/{x}/{y}.png",
        name="Tiles",
    )
    assert "Added XYZ" in out
    assert "Tiles" in project.mapLayers()


def test_buffer_active_layer_runs_processing_and_loads_output(
    monkeypatch, tmp_path
) -> None:
    """Verify the active-layer buffer convenience tool wraps Processing."""
    import types

    project = MockQGISProject()
    iface = MockQGISIface(project=project)
    layer = MockQGISLayer("Roads", "/tmp/roads.gpkg", "vector")
    project.addMapLayer(layer)
    iface.setActiveLayer(layer)

    output_path = tmp_path / "roads_buffer.gpkg"
    captured = {}

    def _run(algorithm_id, parameters):
        captured["algorithm_id"] = algorithm_id
        captured["parameters"] = dict(parameters)
        output_path.write_text("buffer", encoding="utf-8")
        return {"OUTPUT": str(output_path)}

    monkeypatch.setitem(sys.modules, "processing", types.SimpleNamespace(run=_run))

    tools = {t.tool_name: t for t in qgis_tools(iface, project)}
    result = tools["buffer_active_layer"].__wrapped__(
        distance_meters=1000,
        output_layer_name="Road buffer",
        output_path=str(output_path),
    )

    assert captured["algorithm_id"] == "native:buffer"
    assert captured["parameters"]["INPUT"] is layer
    assert captured["parameters"]["DISTANCE"] == 1000.0
    assert result["success"] is True
    assert result["output"] == str(output_path)
    assert "Road buffer" in project.mapLayers()


def test_layer_opacity_selection_and_summary() -> None:
    """Verify that layer opacity selection and summary."""
    project = MockQGISProject()
    layer = MockQGISLayer(
        "Parcels",
        "parcels.gpkg",
        fields=[{"name": "owner", "type": "string"}],
        extent=(-84, 35, -83, 36),
    )
    project.addMapLayer(layer)
    iface = MockQGISIface(project=project)
    iface.setActiveLayer(layer)
    tools = {t.tool_name: t for t in qgis_tools(iface, project)}

    tools["set_layer_opacity"](layer_name="Parcels", opacity=0.4)
    assert layer.opacity() == 0.4

    tools["select_features_by_expression"](
        layer_name="Parcels",
        expression='"owner" IS NOT NULL',
    )
    assert layer.selectedFeatureCount() == 1

    tools["zoom_to_selected"]()
    assert iface.mapCanvas().extent() == (-84, 35, -83, 36)

    summary = tools["get_layer_summary"](layer_name="Parcels")
    assert summary["fields"] == [{"name": "owner", "type": "string"}]

    tools["clear_selection"](layer_name="Parcels")
    assert layer.selectedFeatureCount() == 0


def test_save_project_records_path(tmp_path) -> None:
    """Verify that save project records path."""
    project = MockQGISProject()
    iface = MockQGISIface(project=project)
    tools = {t.tool_name: t for t in qgis_tools(iface, project)}
    out = tools["save_project"](path=str(tmp_path / "project.qgz"))
    assert out.endswith("project.qgz")
    assert project.saved_path == out


def test_set_layer_symbology_updates_mock_layer() -> None:
    """Verify simple layer styling is exposed as a QGIS tool."""
    project = MockQGISProject()
    layer = MockQGISLayer("Stream network", "/tmp/streams.shp", "vector")
    project.addMapLayer(layer)
    iface = MockQGISIface(project=project)
    tools = {t.tool_name: t for t in qgis_tools(iface, project)}

    result = tools["set_layer_symbology"](
        layer_name="Stream network",
        color="blue",
        line_width=2,
        opacity=0.75,
    )

    assert result["message"] == "Updated symbology for layer 'Stream network'."
    assert result["color"] == "blue"
    assert result["line_width"] == 2.0
    assert result["opacity"] == 0.75
    assert layer.symbology == {"color": "blue", "line_width": 2.0}
    assert layer.opacity() == 0.75
    assert layer.repaint_count == 1
    assert iface.mapCanvas().refresh_count == 1


def test_set_layer_symbology_does_not_reassign_renderer_owned_symbol() -> None:
    """Verify styling mutates the existing renderer symbol without re-owning it."""

    class MockSymbolLayer:
        """Minimal symbol layer with width controls."""

        def __init__(self) -> None:
            self.width = 0.0

        def setWidth(self, width: float) -> None:
            self.width = width

    class MockSymbol:
        """Minimal renderer-owned symbol."""

        def __init__(self) -> None:
            self.color = None
            self.layer = MockSymbolLayer()

        def setColor(self, color) -> None:
            self.color = color

        def symbolLayer(self, index: int):
            return self.layer if index == 0 else None

    class MockRenderer:
        """Renderer that must keep ownership of its existing symbol."""

        def __init__(self) -> None:
            self.existing_symbol = MockSymbol()

        def symbol(self):
            return self.existing_symbol

        def setSymbol(self, symbol) -> None:
            raise AssertionError("setSymbol must not be called with an owned symbol")

    project = MockQGISProject()
    layer = MockQGISLayer("Stream network", "/tmp/streams.shp", "vector")
    renderer = MockRenderer()
    layer.renderer = lambda: renderer
    project.addMapLayer(layer)
    iface = MockQGISIface(project=project)
    tools = {t.tool_name: t for t in qgis_tools(iface, project)}

    result = tools["set_layer_symbology"](
        layer_name="Stream network",
        color="blue",
        line_width=2,
    )

    assert result["message"] == "Updated symbology for layer 'Stream network'."
    assert renderer.existing_symbol.color == "blue"
    assert renderer.existing_symbol.layer.width == 2.0
    assert layer.repaint_count == 1


def test_run_pyqgis_script_executes_on_qgis_context() -> None:
    """Verify fallback PyQGIS snippets can mutate active QGIS objects."""
    project = MockQGISProject()
    layer = MockQGISLayer("NAIP", "/tmp/naip.tif", "raster")
    project.addMapLayer(layer)
    iface = MockQGISIface(project=project)
    iface.setActiveLayer(layer)
    tools = {t.tool_name: t for t in qgis_tools(iface, project)}

    result = tools["run_pyqgis_script"].__wrapped__(
        "active_layer.setOpacity(0.5)\n"
        "canvas.refresh()\n"
        "print(active_layer.name())",
        description="Set active layer opacity.",
    )

    assert result["success"] is True
    assert result["message"] == "Set active layer opacity."
    assert result["stdout"] == "NAIP"
    assert layer.opacity() == 0.5
    assert iface.mapCanvas().refresh_count == 1


def test_run_pyqgis_script_rejects_non_qgis_imports() -> None:
    """Verify generated snippets cannot import non-QGIS modules."""
    tools = {t.tool_name: t for t in qgis_tools(MockQGISIface(), MockQGISProject())}

    try:
        tools["run_pyqgis_script"].__wrapped__("import os\nos.getcwd()")
    except ValueError as exc:
        assert "Only qgis/PyQt and math imports" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected unsafe import to be rejected")


def test_run_pyqgis_script_requires_confirmation() -> None:
    """Verify arbitrary PyQGIS execution is confirmation-gated."""
    tools = {t.tool_name: t for t in qgis_tools(MockQGISIface(), MockQGISProject())}
    meta = getattr(tools["run_pyqgis_script"], "_geoagent_meta")

    assert meta.requires_confirmation is True
    assert meta.destructive is True


def test_qgis_tools_route_calls_through_gui_marshal(monkeypatch) -> None:
    """Verify that qgis tools route calls through gui marshal."""
    iface = MockQGISIface()
    project = MockQGISProject()
    tools = {t.tool_name: t for t in qgis_tools(iface, project)}
    calls: list[str] = []

    def _marshal(func):
        """Run the function through the marshal hook."""
        calls.append("called")
        return func()

    monkeypatch.setitem(qgis_tools.__globals__, "run_on_qt_gui_thread", _marshal)
    tools["zoom_in"]()
    assert calls == ["called"]
