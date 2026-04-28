"""GeoAgent.chat with mocked Strands invocation."""

from __future__ import annotations

import sys
import time
import types
from types import SimpleNamespace
from unittest.mock import MagicMock
import threading

from geoagent import for_leafmap, for_qgis
from geoagent.testing import MockLeafmap, MockQGISIface, MockQGISProject


class _MockModel:
    stateful = False


def test_chat_success_from_mocked_strands() -> None:
    m = MockLeafmap()
    agent = for_leafmap(m, model=_MockModel())

    metrics = SimpleNamespace(tool_metrics={"list_layers": object()})
    msg = {"role": "assistant", "content": [{"text": "done"}]}
    fake_result = SimpleNamespace(
        stop_reason="end_turn",
        metrics=metrics,
        message=msg,
    )

    mock_agent = MagicMock(return_value=fake_result)
    agent._strands = mock_agent  # noqa: SLF001 — test seam

    resp = agent.chat("list layers")
    assert resp.success
    assert resp.answer_text == "done"
    assert "list_layers" in resp.executed_tools
    mock_agent.assert_called_once()


def test_chat_in_background_returns_and_invokes_callback() -> None:
    m = MockLeafmap()
    agent = for_leafmap(m, model=_MockModel())

    metrics = SimpleNamespace(tool_metrics={"list_layers": object()})
    msg = {"role": "assistant", "content": [{"text": "done"}]}
    fake_result = SimpleNamespace(
        stop_reason="end_turn",
        metrics=metrics,
        message=msg,
    )
    agent._strands = MagicMock(return_value=fake_result)  # noqa: SLF001

    done = threading.Event()
    box = {}

    def _on_result(resp):
        box["resp"] = resp
        done.set()

    th = agent.chat_in_background("list layers", on_result=_on_result)
    assert isinstance(th, threading.Thread)
    assert done.wait(2.0), "background chat callback did not fire"
    assert box["resp"].success is True
    assert box["resp"].answer_text == "done"


def test_qgis_chat_on_gui_thread_runs_worker_and_pumps_events(monkeypatch) -> None:
    main_thread = threading.current_thread()
    processed = threading.Event()
    calls = {"process_events": 0}

    class _FakeThread:
        def __eq__(self, other: object) -> bool:
            return isinstance(other, _FakeThread)

    class _QThread:
        @staticmethod
        def currentThread():
            return _FakeThread()

    class _App:
        def thread(self):
            return _FakeThread()

        def processEvents(self):
            calls["process_events"] += 1
            processed.set()

    class _QApplication:
        @staticmethod
        def instance():
            return _App()

    class _QObject:
        def moveToThread(self, _thread):
            return None

    class _QMetaObject:
        @staticmethod
        def invokeMethod(*_args, **_kwargs):
            return True

    class _Qt:
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

    agent = for_qgis(MockQGISIface(), MockQGISProject(), model=_MockModel())
    worker_thread: list[threading.Thread] = []

    def _strands(_query: str):
        worker_thread.append(threading.current_thread())
        deadline = time.time() + 2.0
        while not processed.is_set() and time.time() < deadline:
            time.sleep(0.01)
        metrics = SimpleNamespace(tool_metrics={})
        msg = {"role": "assistant", "content": [{"text": "done"}]}
        return SimpleNamespace(
            stop_reason="end_turn",
            metrics=metrics,
            message=msg,
        )

    agent._strands = _strands  # noqa: SLF001

    resp = agent.chat("list layers")

    assert resp.success is True
    assert resp.answer_text == "done"
    assert worker_thread[0] is not main_thread
    assert calls["process_events"] > 0
