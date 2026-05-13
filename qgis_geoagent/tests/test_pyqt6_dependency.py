"""Skip sentinel for optional PyQt6-backed plugin tests."""

from importlib import import_module

import pytest

try:
    PyQtCore = import_module("PyQt6.QtCore")
    PyQtGui = import_module("PyQt6.QtGui")
    PyQtNetwork = import_module("PyQt6.QtNetwork")
    PyQtWidgets = import_module("PyQt6.QtWidgets")
except ImportError:
    PyQtCore = None
    PyQtGui = None
    PyQtNetwork = None
    PyQtWidgets = None


@pytest.mark.skipif(
    any(module is None for module in (PyQtCore, PyQtGui, PyQtNetwork, PyQtWidgets)),
    reason="PyQt6 is required for qgis_geoagent tests.",
)
def test_pyqt6_available_for_qgis_plugin_tests() -> None:
    """Confirm PyQt6 is available before running plugin import tests."""
    assert PyQtCore is not None
    assert PyQtGui is not None
    assert PyQtNetwork is not None
    assert PyQtWidgets is not None
    assert hasattr(PyQtCore, "QObject")
    assert hasattr(PyQtGui, "QGuiApplication")
    assert hasattr(PyQtNetwork, "QNetworkAccessManager")
    assert hasattr(PyQtWidgets, "QWidget")
