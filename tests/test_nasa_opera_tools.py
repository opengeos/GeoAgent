"""Tests for the NASA OPERA GeoAgent tool factory."""

from __future__ import annotations

import threading
import sys
import types

import pytest

from geoagent import for_nasa_opera
from geoagent.tools.nasa_opera import nasa_opera_tools, submit_nasa_opera_chat_task
from geoagent.testing import MockQGISIface, MockQGISLayer, MockQGISProject


class _MockModel:
    """Provide a test double for MockModel."""

    stateful = False


def test_nasa_opera_module_imports_without_qgis() -> None:
    """Verify NASA OPERA tools are import-safe outside QGIS."""
    assert "geoagent.tools.nasa_opera" in sys.modules
    if "qgis" in sys.modules:
        pytest.skip("qgis is already imported in this environment.")
    assert "qgis" not in sys.modules
    assert "nasa_opera.ai.tools" not in sys.modules


def test_nasa_opera_tools_returns_empty_for_none_iface() -> None:
    """Verify the OPERA factory returns no tools without a QGIS iface."""
    assert nasa_opera_tools(None) == []


def test_nasa_opera_dataset_tools_use_geoagent_surface() -> None:
    """Verify OPERA dataset metadata tools are available without plugin AI tools."""
    tools = {t.tool_name: t for t in nasa_opera_tools(object())}

    datasets = tools["get_available_datasets"]()
    names = {item["short_name"] for item in datasets}

    assert "OPERA_L3_DSWX-HLS_V1" in names
    assert "OPERA_L2_RTC-S1_V1" in names
    assert "nasa_opera.ai.tools" not in sys.modules


def test_for_nasa_opera_registers_opera_and_qgis_tools() -> None:
    """Verify the NASA OPERA factory combines OPERA and QGIS tool surfaces."""
    agent = for_nasa_opera(
        MockQGISIface(),
        MockQGISProject(),
        model=_MockModel(),
    )
    names = set(agent.strands_agent.tool_names)

    assert "get_available_datasets" in names
    assert "get_dataset_info" in names
    assert "display_footprints" in names
    assert "list_project_layers" in names
    assert agent.context.metadata["integration"] == "nasa_opera"


def test_count_water_pixels_uses_loaded_mosaic_source(monkeypatch) -> None:
    """Verify OPERA water counting resolves a loaded QGIS raster layer."""
    project = MockQGISProject()
    project.addMapLayer(
        MockQGISLayer("OPERA Mosaic", "/tmp/opera_mosaic.vrt", "raster")
    )
    iface = MockQGISIface(project)
    tools = {t.tool_name: t for t in nasa_opera_tools(iface, project)}
    captured = {}

    def _fake_count(source, *, band, class_values):
        captured["source"] = source
        captured["band"] = band
        captured["class_values"] = class_values
        return {
            "success": True,
            "band": band,
            "width": 3,
            "height": 2,
            "total_pixels": 6,
            "valid_pixels": 6,
            "class_counts": {1: 2, 2: 1},
            "matched_pixel_count": 3,
            "nodata": 255,
        }

    monkeypatch.setattr(
        "geoagent.tools.nasa_opera._count_raster_class_values",
        _fake_count,
    )

    result = tools["count_water_pixels"](water_values="1,2")

    assert captured == {
        "source": "/tmp/opera_mosaic.vrt",
        "band": 1,
        "class_values": [1, 2],
    }
    assert result["success"] is True
    assert result["layer_name"] == "OPERA Mosaic"
    assert result["water_pixel_count"] == 3
    assert result["class_counts"] == {1: 2, 2: 1}


def test_analyze_categorical_raster_adds_labels_and_percentages(monkeypatch) -> None:
    """Verify categorical raster summaries include labels and percentages."""
    project = MockQGISProject()
    project.addMapLayer(MockQGISLayer("Classification", "/tmp/classes.tif", "raster"))
    iface = MockQGISIface(project)
    tools = {t.tool_name: t for t in nasa_opera_tools(iface, project)}
    captured = {}

    def _fake_count(source, *, band, class_values=None, max_categories=50):
        captured["source"] = source
        captured["band"] = band
        captured["class_values"] = class_values
        captured["max_categories"] = max_categories
        return {
            "success": True,
            "band": band,
            "width": 3,
            "height": 2,
            "total_pixels": 6,
            "valid_pixels": 5,
            "class_counts": {4: 3, 1: 2},
            "matched_pixel_count": 5,
            "category_count": 2,
            "returned_category_count": 2,
            "truncated": False,
            "nodata": 255,
        }

    monkeypatch.setattr(
        "geoagent.tools.nasa_opera._count_raster_class_values",
        _fake_count,
    )

    result = tools["analyze_categorical_raster"](
        layer_name="Class",
        category_values="1,4",
        category_labels={"1": "water", "4": "vegetation"},
        max_categories=10,
    )

    assert captured == {
        "source": "/tmp/classes.tif",
        "band": 1,
        "class_values": [1, 4],
        "max_categories": 10,
    }
    assert result["success"] is True
    assert result["layer_name"] == "Classification"
    assert result["categories"] == [
        {
            "value": 4,
            "label": "vegetation",
            "pixel_count": 3,
            "percent_of_valid_pixels": 60.0,
        },
        {
            "value": 1,
            "label": "water",
            "pixel_count": 2,
            "percent_of_valid_pixels": 40.0,
        },
    ]


def test_for_nasa_opera_chat_in_background_returns_thread(monkeypatch) -> None:
    """Verify QGIS OPERA users can start chat without blocking the caller."""
    agent = for_nasa_opera(
        MockQGISIface(),
        MockQGISProject(),
        model=_MockModel(),
    )
    called = threading.Event()

    def _chat(query: str, **kwargs):
        called.set()
        return query

    monkeypatch.setattr(agent, "chat", _chat)

    thread = agent.chat_in_background("show OPERA datasets")
    thread.join(timeout=2)

    assert called.is_set()
    assert not thread.is_alive()


def test_for_nasa_opera_chat_on_gui_thread_fails_closed(monkeypatch) -> None:
    """Verify synchronous OPERA chat is blocked on the QGIS GUI thread."""

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

    agent = for_nasa_opera(
        MockQGISIface(),
        MockQGISProject(),
        model=_MockModel(),
    )
    resp = agent.chat("search OPERA")

    assert resp.success is False
    assert "submit_nasa_opera_search_task" in str(resp.error_message)


def test_submit_nasa_opera_chat_task_is_importable() -> None:
    """Verify the QGIS chat task helper is exposed."""
    assert callable(submit_nasa_opera_chat_task)
