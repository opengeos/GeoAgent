"""Tests for the QGIS / Qt main-thread marshaling helper.

These tests run without QGIS / PyQt installed: the helper's contract
for the no-Qt case is that it short-circuits to a direct call. We
monkeypatch :mod:`qgis.PyQt.QtCore` to also exercise the Qt-loaded
short-circuit (caller already on the main thread) and the off-thread
dispatch path (signal emit lands on the dispatcher's slot via the
fake's blocking-connection emulator). We deliberately avoid spinning
a real Qt event loop — those tests would require a display server in
CI and are covered by the manual QGIS smoke test in the PR.
"""

from __future__ import annotations

import sys
import types

import pytest

import geoagent.tools._qt_marshal as marshal_mod
from geoagent.tools._qt_marshal import run_on_qgis_main_thread


@pytest.fixture(autouse=True)
def _reset_dispatcher_singleton():
    """Each test starts with a fresh dispatcher singleton.

    The marshaler caches the dispatcher in a module global; tests that
    monkeypatch ``qgis.PyQt.QtCore`` need to reset it so the next test
    re-runs ``_make_dispatcher`` against the new fake module.
    """
    marshal_mod._dispatcher = None
    yield
    marshal_mod._dispatcher = None


def test_short_circuits_when_qt_not_available() -> None:
    """No Qt → call ``fn`` directly.

    The helper must be safe to import in CI without QGIS, and must
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


def _install_fake_qtcore(
    monkeypatch: pytest.MonkeyPatch,
    *,
    on_main_thread: bool,
) -> None:
    """Wire a minimal fake ``qgis.PyQt.QtCore`` into ``sys.modules``.

    Args:
        monkeypatch: pytest fixture for scoped sys.modules changes.
        on_main_thread: When True, ``QThread.currentThread()`` returns
            the same object as ``QCoreApplication.instance().thread()``,
            so the helper takes the main-thread short-circuit. When
            False, they differ and the helper enters the dispatcher
            path.
    """

    class _Thread:
        pass

    main_thread = _Thread()
    current_thread = main_thread if on_main_thread else _Thread()

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
            return current_thread

    fake_qtcore = types.SimpleNamespace(
        QCoreApplication=_QCoreApplication,
        QThread=_QThread,
    )
    fake_pyqt = types.SimpleNamespace(QtCore=fake_qtcore)
    fake_qgis = types.SimpleNamespace(PyQt=fake_pyqt)

    monkeypatch.setitem(sys.modules, "qgis", fake_qgis)
    monkeypatch.setitem(sys.modules, "qgis.PyQt", fake_pyqt)
    monkeypatch.setitem(sys.modules, "qgis.PyQt.QtCore", fake_qtcore)


def test_short_circuits_when_already_on_main_thread(monkeypatch) -> None:
    """Qt loaded but caller is on the main thread → direct call.

    The dispatcher path uses ``BlockingQueuedConnection``, which would
    deadlock if we dispatched to the main thread *from* the main
    thread. The short-circuit prevents that.
    """
    _install_fake_qtcore(monkeypatch, on_main_thread=True)

    # If the helper accidentally tries to build the dispatcher, this
    # would fail loudly because the fake QtCore lacks ``QObject`` and
    # the rest of the dispatcher's import surface — exactly the
    # behaviour we want to assert against.
    def f(x):
        return x * 2

    assert run_on_qgis_main_thread(f, 21) == 42


def test_off_thread_dispatches_via_signal(monkeypatch) -> None:
    """Off-thread call goes through the dispatcher singleton.

    Replaces ``_make_dispatcher`` with a stand-in whose ``request``
    object collects emitted runners and runs them inline (a real
    ``BlockingQueuedConnection`` emit would do the same synchronously
    on the receiver's thread). Asserts the runner was dispatched and
    the result flowed back.
    """
    _install_fake_qtcore(monkeypatch, on_main_thread=False)

    class _StubSignal:
        def __init__(self):
            self.calls: list[Any] = []

        def emit(self, runner):
            self.calls.append(runner)
            runner()

    class _StubDispatcher:
        def __init__(self):
            self.request = _StubSignal()

    stub = _StubDispatcher()
    monkeypatch.setattr(marshal_mod, "_make_dispatcher", lambda: stub)

    def f(x, y):
        return x + y

    result = run_on_qgis_main_thread(f, 2, 3)

    assert result == 5
    assert len(stub.request.calls) == 1


def test_off_thread_propagates_exception(monkeypatch) -> None:
    """Exceptions raised inside the marshaled call are re-raised."""
    _install_fake_qtcore(monkeypatch, on_main_thread=False)

    class _StubSignal:
        def emit(self, runner):
            runner()

    class _StubDispatcher:
        request = _StubSignal()

    monkeypatch.setattr(marshal_mod, "_make_dispatcher", lambda: _StubDispatcher())

    class _Boom(RuntimeError):
        pass

    def f():
        raise _Boom("off-thread failure")

    with pytest.raises(_Boom, match="off-thread failure"):
        run_on_qgis_main_thread(f)


def test_dispatcher_singleton_is_reused(monkeypatch) -> None:
    """``_make_dispatcher`` is called only once across multiple marshals.

    LangGraph fans tool calls out across worker threads; if every call
    rebuilt the dispatcher we'd leak ``QObject``s and lose the
    main-thread affinity guarantee. The lock-guarded singleton in
    ``_ensure_dispatcher`` prevents that.
    """
    _install_fake_qtcore(monkeypatch, on_main_thread=False)

    construction_count = [0]

    class _StubSignal:
        def emit(self, runner):
            runner()

    class _StubDispatcher:
        request = _StubSignal()

    def _factory():
        construction_count[0] += 1
        return _StubDispatcher()

    monkeypatch.setattr(marshal_mod, "_make_dispatcher", _factory)

    for i in range(5):
        run_on_qgis_main_thread(lambda x=i: x)

    assert construction_count[0] == 1
