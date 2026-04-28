"""Shared pytest fixtures.

Stubs the ``qgis`` package so the plugin's modules can be imported without a
running QGIS instance. The stub reproduces the real ``qgis.PyQt`` shim
behavior on Qt6: it re-exports ``QAction``, ``QActionGroup`` and ``QShortcut``
from ``PyQt6.QtGui`` under ``qgis.PyQt.QtWidgets`` (they moved out of
``QtWidgets`` in Qt6).

PyQt6 is required for these tests. If it is not installed (for example in a
contributor's bare environment), ``pytest.importorskip`` skips the suite
instead of failing collection. CI installs PyQt6 explicitly.
"""

import sys
import types

import pytest

PyQtCore = pytest.importorskip("PyQt6.QtCore")
PyQtGui = pytest.importorskip("PyQt6.QtGui")
PyQtNetwork = pytest.importorskip("PyQt6.QtNetwork")
PyQtWidgets = pytest.importorskip("PyQt6.QtWidgets")


def _make_strict_module(name: str, attrs: dict) -> types.ModuleType:
    """Create a fail-fast stub module.

    Pre-populates ``attrs`` and installs a module-level ``__getattr__`` that
    raises ``AttributeError`` for any other name. This catches missed/renamed
    QGIS symbols at import time instead of letting a permissive ``MagicMock``
    fabricate them silently.
    """
    module = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)

    def __getattr__(item):
        raise AttributeError(
            f"{name!r} stub has no attribute {item!r}. If the plugin now uses "
            f"this symbol, add it to tests/conftest.py."
        )

    module.__getattr__ = __getattr__
    return module


class _MessageLevel:
    """Stand-in for ``Qgis.MessageLevel`` enum used in plugin logging.

    Only the qualified-form members the plugin uses are defined; bare
    ``Qgis.Info`` / ``Qgis.Warning`` / ``Qgis.Critical`` access (which Qt6
    rejects) will raise ``AttributeError`` because they are not on this class.
    """

    Info = 0
    Warning = 1
    Critical = 2
    Success = 3


class _Qgis:
    """Stand-in for ``qgis.core.Qgis``."""

    MessageLevel = _MessageLevel


class _QgsBlockingNetworkRequest:
    """Stand-in for ``qgis.core.QgsBlockingNetworkRequest``."""

    NoError = 0


def _install_qgis_stub() -> None:
    qgis = types.ModuleType("qgis")
    qgis.__path__ = []
    sys.modules["qgis"] = qgis

    qgis_pyqt = types.ModuleType("qgis.PyQt")
    qgis_pyqt.__path__ = []
    sys.modules["qgis.PyQt"] = qgis_pyqt
    qgis.PyQt = qgis_pyqt

    pyqt_submodules = {
        "QtCore": PyQtCore,
        "QtGui": PyQtGui,
        "QtNetwork": PyQtNetwork,
        "QtWidgets": PyQtWidgets,
    }
    for name, real in pyqt_submodules.items():
        alias = types.ModuleType(f"qgis.PyQt.{name}")
        for attr in dir(real):
            if not attr.startswith("_"):
                setattr(alias, attr, getattr(real, attr))
        sys.modules[f"qgis.PyQt.{name}"] = alias
        setattr(qgis_pyqt, name, alias)

    # Qt6: QAction, QActionGroup, and QShortcut live in QtGui. The real
    # qgis.PyQt.QtWidgets shim re-exports them, so mirror that here.
    qtwidgets_alias = sys.modules["qgis.PyQt.QtWidgets"]
    for attr in ("QAction", "QActionGroup", "QShortcut"):
        setattr(qtwidgets_alias, attr, getattr(PyQtGui, attr))

    # Strict stub for qgis.core: only the symbols the plugin actually imports
    # are exposed. Anything else raises AttributeError at import time.
    qgis_core = _make_strict_module(
        "qgis.core",
        {
            "Qgis": _Qgis,
            "QgsMessageLog": types.SimpleNamespace(logMessage=lambda *a, **kw: None),
            "QgsBlockingNetworkRequest": _QgsBlockingNetworkRequest,
            "QgsProject": types.SimpleNamespace(instance=lambda: None),
            "QgsMapLayerProxyModel": types.SimpleNamespace(),
        },
    )
    sys.modules["qgis.core"] = qgis_core
    qgis.core = qgis_core


_install_qgis_stub()
