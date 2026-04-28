"""Tests for the NASA Earthdata GeoAgent tool factory."""

from __future__ import annotations

import sys
import types

from geoagent import for_nasa_earthdata
from geoagent.tools.nasa_earthdata import earthdata_tools
from geoagent.testing import MockQGISIface, MockQGISProject


class _MockModel:
    """Provide a test double for MockModel."""

    stateful = False


def test_nasa_earthdata_module_imports_without_qgis() -> None:
    """Verify NASA Earthdata tools are import-safe outside QGIS."""
    assert "geoagent.tools.nasa_earthdata" in sys.modules
    assert "qgis" not in sys.modules


def test_earthdata_tools_returns_empty_for_none_iface() -> None:
    """Verify the Earthdata factory returns no tools without a QGIS iface."""
    assert earthdata_tools(None) == []


def test_earthdata_tools_expose_expected_surface() -> None:
    """Verify NASA Earthdata tool names are available without QGIS imports."""
    tools = {tool.tool_name: tool for tool in earthdata_tools(object())}

    assert "search_earthdata_catalog" in tools
    assert "get_earthdata_dataset_info" in tools
    assert "search_earthdata_data" in tools
    assert "display_earthdata_footprints" in tools
    assert "load_earthdata_raster" in tools


def test_for_nasa_earthdata_registers_earthdata_and_qgis_tools() -> None:
    """Verify the factory combines NASA Earthdata and QGIS tool surfaces."""
    agent = for_nasa_earthdata(
        MockQGISIface(),
        MockQGISProject(),
        model=_MockModel(),
    )
    names = set(agent.strands_agent.tool_names)

    assert "search_earthdata_catalog" in names
    assert "get_earthdata_dataset_info" in names
    assert "list_project_layers" in names
    assert agent.context.metadata["integration"] == "nasa_earthdata"


def test_for_nasa_earthdata_chat_on_gui_thread_fails_closed(monkeypatch) -> None:
    """Verify synchronous Earthdata chat is blocked on the QGIS GUI thread."""

    class _FakeThread:
        """Provide a test double for FakeThread."""

        def __eq__(self, other: object) -> bool:
            return isinstance(other, _FakeThread)

    class _QThread:
        """Provide a test double for QThread."""

        @staticmethod
        def currentThread():
            """Return current thread."""
            return _FakeThread()

    class _App:
        """Provide a test double for App."""

        def thread(self):
            """Return GUI thread."""
            return _FakeThread()

    class _QApplication:
        """Provide a test double for QApplication."""

        @staticmethod
        def instance():
            """Return app instance."""
            return _App()

    fake_qt_core = types.SimpleNamespace(QThread=_QThread)
    fake_qt_widgets = types.SimpleNamespace(QApplication=_QApplication)
    fake_pyqt = types.SimpleNamespace(QtCore=fake_qt_core, QtWidgets=fake_qt_widgets)
    fake_qgis = types.SimpleNamespace(PyQt=fake_pyqt)
    monkeypatch.setitem(sys.modules, "qgis", fake_qgis)
    monkeypatch.setitem(sys.modules, "qgis.PyQt", fake_pyqt)
    monkeypatch.setitem(sys.modules, "qgis.PyQt.QtCore", fake_qt_core)
    monkeypatch.setitem(sys.modules, "qgis.PyQt.QtWidgets", fake_qt_widgets)

    agent = for_nasa_earthdata(
        MockQGISIface(),
        MockQGISProject(),
        model=_MockModel(),
    )
    resp = agent.chat("search Earthdata")

    assert resp.success is False
    assert "NASA Earthdata chat should be launched from a worker thread" in str(
        resp.error_message
    )
