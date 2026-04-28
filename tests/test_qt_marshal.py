"""Regression tests for Qt GUI-thread marshalling."""

from __future__ import annotations

import sys
import types

from geoagent.tools._qt_marshal import run_on_qt_gui_thread


def test_run_on_qt_gui_thread_runs_inline_when_current_equals_gui(monkeypatch) -> None:
    """Avoid deadlock: equal QThreads must bypass invokeMethod.

    In PyQt, wrappers for the same C++ QThread may not be identical objects.
    If marshal logic uses ``is`` instead of ``==``, it can incorrectly think
    it's off-thread and call BlockingQueuedConnection against the GUI thread,
    which freezes QGIS.
    """

    class _FakeThread:
        """Provide a test double for FakeThread."""

        def __eq__(self, other: object) -> bool:
            return isinstance(other, _FakeThread)

    class _QThread:
        """Provide a test double for QThread."""

        @staticmethod
        def currentThread():
            """Return the current thread."""
            return _FakeThread()

    class _App:
        """Provide a test double for App."""

        def thread(self):
            """Return the associated thread."""
            return _FakeThread()

    class _QApplication:
        """Provide a test double for QApplication."""

        @staticmethod
        def instance():
            """Return the singleton instance."""
            return _App()

    class _QObject:
        """Provide a test double for QObject."""

        def moveToThread(self, _thread):
            """Move to thread."""
            return None

    class _QMetaObject:
        """Provide a test double for QMetaObject."""

        @staticmethod
        def invokeMethod(*_args, **_kwargs):
            """Invoke method."""
            raise AssertionError("invokeMethod must not run on GUI thread")

    class _Qt:
        """Provide a test double for Qt."""

        BlockingQueuedConnection = 0

    fake_qt_core = types.SimpleNamespace(
        QMetaObject=_QMetaObject,
        QObject=_QObject,
        QThread=_QThread,
        Qt=_Qt,
        pyqtSlot=lambda *args, **kwargs: (lambda fn: fn),
    )
    fake_qt_widgets = types.SimpleNamespace(QApplication=_QApplication)
    fake_pyqt = types.SimpleNamespace(QtCore=fake_qt_core, QtWidgets=fake_qt_widgets)
    fake_qgis = types.SimpleNamespace(PyQt=fake_pyqt)
    monkeypatch.setitem(sys.modules, "qgis", fake_qgis)
    monkeypatch.setitem(sys.modules, "qgis.PyQt", fake_pyqt)
    monkeypatch.setitem(sys.modules, "qgis.PyQt.QtCore", fake_qt_core)
    monkeypatch.setitem(sys.modules, "qgis.PyQt.QtWidgets", fake_qt_widgets)

    assert run_on_qt_gui_thread(lambda: "ok") == "ok"
