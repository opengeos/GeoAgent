"""Tests for the NASA Earthdata GeoAgent tool factory."""

from __future__ import annotations

import io
import sys
import types

import pytest

from geoagent import for_nasa_earthdata
from geoagent.tools import nasa_earthdata as nasa_earthdata_module
from geoagent.tools.nasa_earthdata import earthdata_tools
from geoagent.testing import MockQGISIface, MockQGISProject


class _MockModel:
    """Provide a test double for MockModel."""

    stateful = False


def test_nasa_earthdata_module_imports_without_qgis() -> None:
    """Verify NASA Earthdata tools are import-safe outside QGIS."""
    assert "geoagent.tools.nasa_earthdata" in sys.modules
    if "qgis" in sys.modules:
        pytest.skip("qgis is already imported in this environment.")
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


_FAKE_TSV = (
    "ShortName\tEntryTitle\tSummary\tPlatform\tInstrument\n"
    "MOD11A1\tMODIS Land Surface Temperature\tDaily LST product\tTerra\tMODIS\n"
    "HLSL30\tHarmonized Landsat Sentinel-2 Landsat\tSurface reflectance\tLandsat\tOLI\n"
    "GPM_3IMERGDF\tGPM IMERG Daily\tPrecipitation\tGPM\tIMERG\n"
)


def _patched_urlopen(monkeypatch, payload: str = _FAKE_TSV) -> None:
    """Replace urlopen in the NASA Earthdata module with a fake TSV response."""

    class _FakeResponse(io.BytesIO):
        def __enter__(self):  # type: ignore[override]
            return self

        def __exit__(self, *exc) -> None:  # type: ignore[override]
            self.close()

    def _fake_urlopen(url, *args, **kwargs):
        assert str(url).lower().startswith("https://")
        return _FakeResponse(payload.encode("utf-8"))

    monkeypatch.setattr(nasa_earthdata_module, "urlopen", _fake_urlopen)


def test_search_earthdata_catalog_filters_and_max_results(monkeypatch) -> None:
    """Verify catalog search filters rows and honors max_results."""
    _patched_urlopen(monkeypatch)
    tools = {tool.tool_name: tool for tool in earthdata_tools(object())}
    search = tools["search_earthdata_catalog"]

    result = search(
        query="modis",
        max_results=10,
        catalog_url="https://example.com/catalog.tsv",
    )
    assert result["count"] == 1
    assert result["datasets"][0]["ShortName"] == "MOD11A1"

    capped = search(
        query="",
        max_results=2,
        catalog_url="https://example.com/catalog.tsv",
    )
    assert capped["count"] == 3
    assert capped["shown"] == 2
    assert len(capped["datasets"]) == 2


def test_load_catalog_rows_rejects_non_https(monkeypatch) -> None:
    """Verify the catalog loader refuses non-HTTPS URLs."""

    def _should_not_be_called(*_args, **_kwargs):
        raise AssertionError("urlopen must not be invoked for non-HTTPS URLs")

    monkeypatch.setattr(nasa_earthdata_module, "urlopen", _should_not_be_called)

    with pytest.raises(ValueError, match="HTTPS"):
        nasa_earthdata_module._load_catalog_rows("http://example.com/catalog.tsv")
