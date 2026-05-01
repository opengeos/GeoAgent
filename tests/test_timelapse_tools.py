"""Tests for the Timelapse GeoAgent tool factory."""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

import pytest

from geoagent import for_timelapse
from geoagent.testing import MockQGISIface, MockQGISProject
from geoagent.tools import timelapse_tools
import geoagent.tools.timelapse as timelapse


class _MockModel:
    """Tiny model stand-in for GeoAgent factory tests."""

    stateful = False


class _FakeTimelapseCore:
    """Small Timelapse core stand-in for dispatch tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.initialized = True

    def is_ee_initialized(self) -> bool:
        """Return mock Earth Engine state."""
        return self.initialized

    def initialize_ee(self, project=None, force=False) -> bool:
        """Record and report successful Earth Engine initialization."""
        self.calls.append(("initialize_ee", {"project": project, "force": force}))
        self.initialized = True
        return True

    def get_ee_project(self):
        """Return a fake Earth Engine project id."""
        return "test-project"

    def bbox_to_ee_geometry(self, west, south, east, north):
        """Return a JSON-friendly ROI marker."""
        return ("roi", west, south, east, north)

    def create_landsat_timelapse(self, **kwargs):
        """Record Landsat timelapse creation."""
        self.calls.append(("create_landsat_timelapse", kwargs))
        return kwargs["out_gif"]


def test_timelapse_module_imports_without_qgis_or_plugin() -> None:
    """Verify Timelapse tools are import-safe outside QGIS."""
    assert "geoagent.tools.timelapse" in sys.modules
    if "qgis" in sys.modules:
        pytest.skip("qgis is already imported in this environment.")
    assert "qgis" not in sys.modules
    assert "timelapse" not in sys.modules


def test_timelapse_tools_returns_empty_for_none_iface() -> None:
    """Verify the Timelapse factory returns no tools without iface."""
    assert timelapse_tools(None) == []


def test_timelapse_tools_expose_expected_surface() -> None:
    """Verify Timelapse tool names and confirmation metadata."""
    tools = {tool.tool_name: tool for tool in timelapse_tools(object())}

    assert set(tools) == {
        "list_timelapse_imagery_types",
        "get_current_timelapse_extent",
        "initialize_timelapse_earth_engine",
        "create_timelapse",
        "open_timelapse_panel",
        "open_timelapse_settings",
    }
    assert tools[
        "initialize_timelapse_earth_engine"
    ]._geoagent_meta.requires_confirmation
    assert tools["create_timelapse"]._geoagent_meta.requires_confirmation
    assert tools["create_timelapse"]._geoagent_meta.long_running
    assert tools["open_timelapse_panel"]._geoagent_meta.requires_confirmation
    assert tools["open_timelapse_settings"]._geoagent_meta.requires_confirmation


def test_get_current_timelapse_extent_uses_canvas_extent() -> None:
    """Verify current extent returns a WGS84 bbox from the QGIS canvas."""
    iface = MockQGISIface()
    iface.mapCanvas().setExtent((-85.0, 34.0, -83.0, 36.0))
    tools = {tool.tool_name: tool for tool in timelapse_tools(iface)}

    result = tools["get_current_timelapse_extent"].__wrapped__()

    assert result["success"] is True
    assert result["bbox"] == [-85.0, 34.0, -83.0, 36.0]


def test_initialize_timelapse_earth_engine_uses_core(monkeypatch) -> None:
    """Verify Earth Engine initialization delegates to Timelapse core."""
    core = _FakeTimelapseCore()
    core.initialized = False
    monkeypatch.setattr(timelapse, "_load_timelapse_core", lambda plugin=None: core)
    tools = {tool.tool_name: tool for tool in timelapse_tools(object())}

    result = tools["initialize_timelapse_earth_engine"].__wrapped__("project-1")

    assert result["success"] is True
    assert result["initialized"] is True
    assert core.calls == [("initialize_ee", {"project": "project-1", "force": False})]


def test_load_timelapse_core_continues_when_plugin_gate_is_false(monkeypatch) -> None:
    """Verify runtime dependency checks, not plugin gate state, decide readiness."""
    fake_core = types.ModuleType("timelapse.core.timelapse_core")
    fake_core.reload_dependencies = lambda: {"earthengine-api": True, "Pillow": True}

    fake_venv = types.ModuleType("timelapse.core.venv_manager")
    fake_venv.ensure_venv_packages_available = lambda: True

    fake_core_package = types.ModuleType("timelapse.core")
    fake_core_package.timelapse_core = fake_core
    fake_core_package.venv_manager = fake_venv

    fake_package = types.ModuleType("timelapse")
    fake_package.core = fake_core_package

    monkeypatch.setitem(sys.modules, "timelapse", fake_package)
    monkeypatch.setitem(sys.modules, "timelapse.core", fake_core_package)
    monkeypatch.setitem(sys.modules, "timelapse.core.timelapse_core", fake_core)
    monkeypatch.setitem(sys.modules, "timelapse.core.venv_manager", fake_venv)

    class _Plugin:
        plugin_dir = "/tmp/qgis-timelapse-plugin/timelapse"

        def __init__(self) -> None:
            self.checked = False

        def _ensure_deps(self) -> bool:
            self.checked = True
            return False

    plugin = _Plugin()

    assert timelapse._load_timelapse_core(plugin) is fake_core
    assert plugin.checked is True


def test_default_output_path_uses_dedicated_unique_temp_dir() -> None:
    """Verify default Timelapse outputs do not overwrite earlier GIFs."""
    first = timelapse._default_output_path("Landsat")
    second = timelapse._default_output_path("Landsat")

    assert first != second
    assert Path(first).parent == Path(tempfile.gettempdir()) / "open_geoagent_timelapse"
    assert Path(second).parent == Path(first).parent
    assert Path(first).name.startswith("landsat_timelapse_")
    assert Path(first).suffix == ".gif"


def test_create_timelapse_dispatches_to_landsat_core(monkeypatch, tmp_path) -> None:
    """Verify create_timelapse builds ROI and calls the selected core function."""
    core = _FakeTimelapseCore()
    monkeypatch.setattr(timelapse, "_load_timelapse_core", lambda plugin=None: core)
    output = tmp_path / "landsat.gif"
    project = MockQGISProject()
    iface = MockQGISIface(project)
    tools = {tool.tool_name: tool for tool in timelapse_tools(iface, project)}

    result = tools["create_timelapse"].__wrapped__(
        imagery_type="Landsat",
        bbox="-85,34,-83,36",
        output_path=str(output),
        start_year=2001,
        end_year=2003,
        bands="NIR, Red, Green",
        create_mp4=True,
        bbox_layer_name="Test Timelapse BBOX",
    )

    assert result["success"] is True
    assert result["output_path"] == str(output)
    assert result["mp4_path"] == str(Path(tmp_path / "landsat.mp4"))
    assert result["images"] == [
        {
            "path": str(output),
            "format": "gif",
            "mime_type": "image/gif",
            "alt": "Landsat timelapse",
        }
    ]
    assert result["bbox"] == [-85.0, 34.0, -83.0, 36.0]
    assert result["bbox_layer"] == {
        "success": True,
        "layer_name": "Test Timelapse BBOX",
        "bbox": [-85.0, 34.0, -83.0, 36.0],
    }
    assert "Test Timelapse BBOX" in project.mapLayers()
    assert project.mapLayers()["Test Timelapse BBOX"].extent() == (
        -85.0,
        34.0,
        -83.0,
        36.0,
    )
    assert iface.mapCanvas().extent() == (-85.0, 34.0, -83.0, 36.0)
    name, kwargs = core.calls[-1]
    assert name == "create_landsat_timelapse"
    assert kwargs["roi"] == ("roi", -85.0, 34.0, -83.0, 36.0)
    assert kwargs["start_year"] == 2001
    assert kwargs["end_year"] == 2003
    assert kwargs["bands"] == ["NIR", "Red", "Green"]


def test_open_timelapse_panels_use_plugin_instance() -> None:
    """Verify panel opening delegates to the supplied Timelapse plugin."""

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
            self._timelapse_dock = None
            self._settings_dock = None
            self.timelapse_toggled = False
            self.settings_toggled = False

        def toggle_timelapse_dock(self) -> None:
            self.timelapse_toggled = True
            self._timelapse_dock = _Dock()

        def toggle_settings_dock(self) -> None:
            self.settings_toggled = True
            self._settings_dock = _Dock()

    plugin = _Plugin()
    tools = {tool.tool_name: tool for tool in timelapse_tools(object(), plugin=plugin)}

    panel_result = tools["open_timelapse_panel"].__wrapped__()
    settings_result = tools["open_timelapse_settings"].__wrapped__()

    assert panel_result == {"success": True, "opened": True}
    assert settings_result == {"success": True, "opened": True}
    assert plugin.timelapse_toggled is True
    assert plugin.settings_toggled is True
    assert plugin._timelapse_dock.shown is True
    assert plugin._timelapse_dock.raised is True
    assert plugin._settings_dock.shown is True
    assert plugin._settings_dock.raised is True


def test_for_timelapse_registers_timelapse_and_qgis_tools() -> None:
    """Verify the Timelapse factory combines Timelapse and QGIS tools."""
    agent = for_timelapse(
        MockQGISIface(),
        MockQGISProject(),
        model=_MockModel(),
    )
    names = set(agent.strands_agent.tool_names)

    assert "list_timelapse_imagery_types" in names
    assert "get_current_timelapse_extent" in names
    assert "create_timelapse" in names
    assert "list_project_layers" in names
    assert agent.context.metadata["integration"] == "timelapse"
