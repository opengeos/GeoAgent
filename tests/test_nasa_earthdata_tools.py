"""Tests for the NASA Earthdata GeoAgent tool factory."""

from __future__ import annotations

import io
import os
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
    assert "create_earthdata_rgb_composite" in tools


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
    "ShortName\tEntryTitle\tSummary\tPlatform\tInstrument\tconcept-id\t"
    "provider-id\tVersion\n"
    "MOD11A1\tMODIS Land Surface Temperature\tDaily LST product\tTerra\t"
    "MODIS\tC123-LPDAAC\tLPDAAC\t6.1\n"
    "HLSL30\tHarmonized Landsat Sentinel-2 Landsat\tSurface reflectance\t"
    "Landsat\tOLI\tC2021957657-LPCLOUD\tLPCLOUD\t2.0\n"
    "GPM_3IMERGDF\tGPM IMERG Daily\tPrecipitation\tGPM\tIMERG\t"
    "C456-GES_DISC\tGES_DISC\t7\n"
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
    assert result["datasets"][0]["concept-id"] == "C123-LPDAAC"

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


def _install_fake_earthaccess(monkeypatch):
    """Install a fake earthaccess module and return captured search calls."""
    calls = []

    def _login(strategy):
        return types.SimpleNamespace(authenticated=True)

    def _search_data(**kwargs):
        calls.append(kwargs)
        return [
            {
                "meta": {
                    "native-id": "HLS.S30.T10SEG.2025168T184941.v2.0",
                    "concept-id": "G3577700089-LPCLOUD",
                    "collection-concept-id": "C2021957295-LPCLOUD",
                },
                "umm": {
                    "TemporalExtent": {
                        "RangeDateTime": {
                            "BeginningDateTime": "2025-06-17T19:04:28.304Z",
                            "EndingDateTime": "2025-06-17T19:04:28.304Z",
                        }
                    }
                },
            }
        ]

    def _download(*_args, **_kwargs):
        raise AssertionError("RGB composite streaming should not download COGs")

    fake_earthaccess = types.SimpleNamespace(
        login=_login,
        search_data=_search_data,
        download=_download,
    )
    monkeypatch.setitem(sys.modules, "earthaccess", fake_earthaccess)
    return calls


def test_search_earthdata_data_ignores_zero_orbit_number(monkeypatch) -> None:
    """Verify LLM/UI default zero does not become a CMR orbit filter."""
    calls = _install_fake_earthaccess(monkeypatch)
    tools = {tool.tool_name: tool for tool in earthdata_tools(object())}
    search = tools["search_earthdata_data"]

    result = search(
        short_name="HLSS30",
        bbox=[-122.55, 37.68, -122.32, 37.84],
        start_date="2025-06-01",
        end_date="2025-08-31",
        max_results=20,
        provider="LPCLOUD",
        version="2.0",
        cloud_cover_min=0,
        cloud_cover_max=5,
        orbit_number=0,
    )

    assert result["count"] == 1
    assert calls[0]["short_name"] == "HLSS30"
    assert calls[0]["bounding_box"] == (-122.55, 37.68, -122.32, 37.84)
    assert calls[0]["cloud_cover"] == (0, 5)
    assert "orbit_number" not in calls[0]


def test_search_earthdata_data_prefers_concept_id(monkeypatch) -> None:
    """Verify catalog concept IDs can disambiguate CMR granule searches."""
    calls = _install_fake_earthaccess(monkeypatch)
    tools = {tool.tool_name: tool for tool in earthdata_tools(object())}
    search = tools["search_earthdata_data"]

    result = search(
        short_name="HLSS30",
        concept_id="C2021957295-LPCLOUD",
        bbox="-122.55,37.68,-122.32,37.84",
        max_results=5,
        orbit_number=123,
    )

    assert result["concept_id"] == "C2021957295-LPCLOUD"
    assert calls[0]["concept_id"] == "C2021957295-LPCLOUD"
    assert "short_name" not in calls[0]
    assert calls[0]["orbit_number"] == 123


def test_create_earthdata_rgb_composite_builds_vrt(monkeypatch) -> None:
    """Verify RGB composites stream COGs through GDAL VRTs."""
    _install_fake_earthaccess(monkeypatch)
    red = "https://example.test/HLS.L30.B05.tif"
    green = "https://example.test/HLS.L30.B04.tif"
    blue = "https://example.test/HLS.L30.B03.tif"

    captured = {}

    class _FakeBand:
        def __init__(self):
            self.color_interp = None

        def SetColorInterpretation(self, value):
            self.color_interp = value

    class _FakeVrt:
        def __init__(self, path):
            self.path = path
            self.bands = [_FakeBand(), _FakeBand(), _FakeBand()]

        def GetRasterBand(self, index):
            return self.bands[index - 1]

        def FlushCache(self):
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("<VRTDataset><SRS>EPSG:32610</SRS></VRTDataset>")

    class _FakeGdal:
        GCI_RedBand = 3
        GCI_GreenBand = 4
        GCI_BlueBand = 5

        @staticmethod
        def SetConfigOption(name, value):
            captured.setdefault("config", {})[name] = value

        @staticmethod
        def BuildVRTOptions(**kwargs):
            captured["options"] = kwargs
            return kwargs

        @staticmethod
        def BuildVRT(path, sources, options=None):
            captured["path"] = path
            captured["sources"] = sources
            captured["build_options"] = options
            return _FakeVrt(path)

    fake_osgeo = types.ModuleType("osgeo")
    fake_osgeo.gdal = _FakeGdal
    monkeypatch.setitem(sys.modules, "osgeo", fake_osgeo)
    monkeypatch.setitem(sys.modules, "osgeo.gdal", _FakeGdal)

    def _fake_add_layer(_iface, _project_getter, path_or_uri, layer_name):
        captured["loaded_path"] = path_or_uri
        captured["layer_name"] = layer_name
        return {"success": True, "layer_name": layer_name}

    monkeypatch.setattr(
        nasa_earthdata_module,
        "_add_qgis_rgb_raster_layer",
        _fake_add_layer,
    )

    tools = {tool.tool_name: tool for tool in earthdata_tools(object())}
    create_composite = tools["create_earthdata_rgb_composite"]

    result = create_composite(
        red_url=red,
        green_url=green,
        blue_url=blue,
        layer_name="HLS false color",
    )

    assert result["success"] is True
    assert result["layer_name"] == "HLS false color"
    assert result["path"].endswith(".vrt")
    assert os.path.exists(result["path"])
    assert captured["sources"] == [
        f"/vsicurl/{red}",
        f"/vsicurl/{green}",
        f"/vsicurl/{blue}",
    ]
    assert captured["options"] == {"separate": True}
    assert captured["loaded_path"] == result["path"]
    assert result["source_type"] == "stream"
    assert result["red_source"] == f"/vsicurl/{red}"
    assert captured["config"]["GDAL_DISABLE_READDIR_ON_OPEN"] == "EMPTY_DIR"
