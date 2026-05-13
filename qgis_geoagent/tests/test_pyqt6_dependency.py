"""Skip sentinel for optional PyQt6-backed plugin tests."""

from importlib import import_module

import pytest

try:
    PyQtCore = import_module("PyQt6.QtCore")
except ImportError:
    PyQtCore = None


@pytest.mark.skipif(
    PyQtCore is None, reason="PyQt6 is required for qgis_geoagent tests."
)
def test_pyqt6_available_for_qgis_plugin_tests() -> None:
    """Confirm PyQt6 is available before running plugin import tests."""
    assert PyQtCore is not None
    assert hasattr(PyQtCore, "QObject")
