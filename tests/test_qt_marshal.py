"""Tests for the QGIS / Qt main-thread marshaling helper.

These run without QGIS / PyQt installed: the helper's contract for the
no-Qt case is that it short-circuits to a direct call. We monkeypatch
:func:`geoagent.tools._qt_marshal.run_on_qgis_main_thread`'s internal
Qt import to also exercise the Qt-loaded short-circuit (caller already
on the main thread) without needing a real Qt event loop.
"""

from __future__ import annotations

import sys
import types

import pytest

from geoagent.tools._qt_marshal import run_on_qgis_main_thread


def test_short_circuits_when_qt_not_available() -> None:
    """No Qt → call ``fn`` directly on the calling thread.

    The helper must be safe to import in CI without QGIS, and it must
    behave as a passthrough so ``MockQGISIface``-based tests stay fast
    and deterministic.
    """
    calls: list[tuple[tuple, dict]] = []

    def f(*args, **kwargs):
        calls.append((args, kwargs))
        return "ok"

    result = run_on_qgis_main_thread(f, 1, 2, k="v")

    assert result == "ok"
    assert calls == [((1, 2), {"k": "v"})]


def test_propagates_exception_when_qt_not_available() -> None:
    """An exception in ``fn`` must surface on the calling thread."""

    class _Boom(RuntimeError):
        pass

    def f():
        raise _Boom("fail")

    with pytest.raises(_Boom, match="fail"):
        run_on_qgis_main_thread(f)


def test_short_circuits_when_already_on_main_thread(monkeypatch) -> None:
    """Qt loaded but caller is on the main thread → direct call.

    The marshaling path uses ``BlockingQueuedConnection``, which would
    deadlock if we dispatched to the main thread *from* the main thread.
    The short-circuit prevents that. We fake a ``qgis.PyQt.QtCore``
    where ``QThread.currentThread() == app.thread()`` so the helper
    must take the direct-call path without touching ``invokeMethod``.
    """
    invoke_method_calls: list[object] = []

    class _ThreadObj:
        pass

    main_thread = _ThreadObj()

    class _App:
        def thread(self):
            return main_thread

    class _QCoreApplication:
        @staticmethod
        def instance():
            return _App()

    class _QThread:
        @staticmethod
        def currentThread():
            return main_thread

    class _QMetaObject:
        @staticmethod
        def invokeMethod(*args, **kwargs):
            invoke_method_calls.append((args, kwargs))

    class _Qt:
        BlockingQueuedConnection = object()

    fake_qtcore = types.SimpleNamespace(
        QCoreApplication=_QCoreApplication,
        QMetaObject=_QMetaObject,
        Qt=_Qt,
        QThread=_QThread,
    )
    fake_pyqt = types.SimpleNamespace(QtCore=fake_qtcore)
    fake_qgis = types.SimpleNamespace(PyQt=fake_pyqt)

    monkeypatch.setitem(sys.modules, "qgis", fake_qgis)
    monkeypatch.setitem(sys.modules, "qgis.PyQt", fake_pyqt)
    monkeypatch.setitem(sys.modules, "qgis.PyQt.QtCore", fake_qtcore)

    def f(x):
        return x * 2

    result = run_on_qgis_main_thread(f, 21)

    assert result == 42
    assert invoke_method_calls == [], "must not dispatch when already on main thread"


def test_marshals_via_invokeMethod_when_off_thread(monkeypatch) -> None:
    """Off-thread call dispatches via ``invokeMethod`` and returns the result.

    Simulates the LangGraph ``ToolNode`` worker-thread case: the helper
    sees that the current thread is not the main thread, so it must
    package the callable and dispatch through
    ``QMetaObject.invokeMethod(..., BlockingQueuedConnection)``. The
    fake invokeMethod runs the runner inline (as a real
    ``BlockingQueuedConnection`` would, but synchronously here) so we
    can verify the result + exception channels.
    """

    class _ThreadObj:
        pass

    main_thread = _ThreadObj()
    worker_thread = _ThreadObj()

    class _App:
        def thread(self):
            return main_thread

    class _QCoreApplication:
        @staticmethod
        def instance():
            return _App()

    class _QThread:
        @staticmethod
        def currentThread():
            return worker_thread

    class _QMetaObject:
        calls: list[object] = []

        @staticmethod
        def invokeMethod(receiver, runner, connection):
            _QMetaObject.calls.append((receiver, runner, connection))
            runner()

    class _Qt:
        BlockingQueuedConnection = object()

    fake_qtcore = types.SimpleNamespace(
        QCoreApplication=_QCoreApplication,
        QMetaObject=_QMetaObject,
        Qt=_Qt,
        QThread=_QThread,
    )
    fake_pyqt = types.SimpleNamespace(QtCore=fake_qtcore)
    fake_qgis = types.SimpleNamespace(PyQt=fake_pyqt)

    monkeypatch.setitem(sys.modules, "qgis", fake_qgis)
    monkeypatch.setitem(sys.modules, "qgis.PyQt", fake_pyqt)
    monkeypatch.setitem(sys.modules, "qgis.PyQt.QtCore", fake_qtcore)

    def f(x, y):
        return x + y

    result = run_on_qgis_main_thread(f, 2, 3)

    assert result == 5
    assert len(_QMetaObject.calls) == 1
    assert _QMetaObject.calls[0][2] is _Qt.BlockingQueuedConnection


def test_marshals_propagates_exception_off_thread(monkeypatch) -> None:
    """Exceptions raised in the runner must surface to the worker."""

    class _ThreadObj:
        pass

    main_thread = _ThreadObj()
    worker_thread = _ThreadObj()

    class _App:
        def thread(self):
            return main_thread

    class _QCoreApplication:
        @staticmethod
        def instance():
            return _App()

    class _QThread:
        @staticmethod
        def currentThread():
            return worker_thread

    class _QMetaObject:
        @staticmethod
        def invokeMethod(receiver, runner, connection):
            runner()

    class _Qt:
        BlockingQueuedConnection = object()

    fake_qtcore = types.SimpleNamespace(
        QCoreApplication=_QCoreApplication,
        QMetaObject=_QMetaObject,
        Qt=_Qt,
        QThread=_QThread,
    )
    fake_pyqt = types.SimpleNamespace(QtCore=fake_qtcore)
    fake_qgis = types.SimpleNamespace(PyQt=fake_pyqt)

    monkeypatch.setitem(sys.modules, "qgis", fake_qgis)
    monkeypatch.setitem(sys.modules, "qgis.PyQt", fake_pyqt)
    monkeypatch.setitem(sys.modules, "qgis.PyQt.QtCore", fake_qtcore)

    class _Boom(RuntimeError):
        pass

    def f():
        raise _Boom("off-thread failure")

    with pytest.raises(_Boom, match="off-thread failure"):
        run_on_qgis_main_thread(f)
