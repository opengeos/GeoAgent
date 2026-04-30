"""GeoAgent.chat with mocked Strands invocation."""

from __future__ import annotations

import asyncio
import sys
import time
import types
from types import SimpleNamespace
from unittest.mock import MagicMock
import threading

from geoagent import for_leafmap, for_qgis
from geoagent.testing import MockLeafmap, MockQGISIface, MockQGISProject


class _MockModel:
    """Provide a test double for MockModel."""

    stateful = False


class _MockStreamingAgent:
    """Provide a test double for a Strands agent with stream_async."""

    def __init__(self, events):
        self.events = events
        self.calls = []

    async def stream_async(self, query):
        """Yield mocked streaming events."""
        self.calls.append(query)
        for event in self.events:
            yield event


def _install_fake_qgis_qt(monkeypatch, calls: dict[str, int] | None = None) -> None:
    """Install fake qgis.PyQt modules that report the current thread as GUI."""

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

        def processEvents(self):
            """Process events."""
            if calls is not None:
                calls["process_events"] += 1

    app = _App()

    class _QApplication:
        """Provide a test double for QApplication."""

        @staticmethod
        def instance():
            """Return the singleton instance."""
            return app

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
            return True

    class _Qt:
        """Provide a test double for Qt."""

        BlockingQueuedConnection = 0

    fake_qt_core = types.SimpleNamespace(
        QMetaObject=_QMetaObject,
        QObject=_QObject,
        QThread=_QThread,
        Qt=_Qt,
        pyqtSlot=lambda *args, **kwargs: lambda fn: fn,
    )
    fake_qt_widgets = types.SimpleNamespace(QApplication=_QApplication)
    fake_pyqt = types.SimpleNamespace(QtCore=fake_qt_core, QtWidgets=fake_qt_widgets)
    fake_qgis = types.SimpleNamespace(PyQt=fake_pyqt)
    monkeypatch.setitem(sys.modules, "qgis", fake_qgis)
    monkeypatch.setitem(sys.modules, "qgis.PyQt", fake_pyqt)
    monkeypatch.setitem(sys.modules, "qgis.PyQt.QtCore", fake_qt_core)
    monkeypatch.setitem(sys.modules, "qgis.PyQt.QtWidgets", fake_qt_widgets)


def test_chat_success_from_mocked_strands() -> None:
    """Verify that chat success from mocked strands."""
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


def test_chat_passes_multimodal_content_blocks_to_strands() -> None:
    """Verify chat accepts Strands multimodal content blocks."""
    m = MockLeafmap()
    agent = for_leafmap(m, model=_MockModel())

    metrics = SimpleNamespace(tool_metrics={})
    msg = {"role": "assistant", "content": [{"text": "image described"}]}
    fake_result = SimpleNamespace(
        stop_reason="end_turn",
        metrics=metrics,
        message=msg,
    )
    content = [
        {"text": "Describe this image."},
        {"image": {"format": "png", "source": {"bytes": b"fake-png"}}},
    ]

    mock_agent = MagicMock(return_value=fake_result)
    agent._strands = mock_agent  # noqa: SLF001

    resp = agent.chat(content)

    assert resp.success
    assert resp.answer_text == "image described"
    mock_agent.assert_called_once_with(content)


def test_chat_json_parse_error_returns_actionable_guidance() -> None:
    """Verify malformed tool-call JSON gets user-facing correction guidance."""
    m = MockLeafmap()
    agent = for_leafmap(m, model=_MockModel())
    agent._strands = MagicMock(  # noqa: SLF001
        side_effect=ValueError(
            "failed to parse JSON: unexpected end of JSON input (status code: -1)"
        )
    )

    resp = agent.chat("create color shaded relief")

    assert resp.success is False
    assert "malformed or incomplete JSON" in resp.error_message
    assert "Original error:" in resp.error_message
    assert "Break a long workflow into smaller steps" in resp.error_message


def test_stream_chat_yields_mocked_strands_events() -> None:
    """Verify stream_chat yields Strands streaming events."""
    m = MockLeafmap()
    agent = for_leafmap(m, model=_MockModel())
    events = [
        {"data": "he"},
        {"data": "llo"},
        {"current_tool_use": {"name": "list_layers"}},
        {"result": SimpleNamespace(stop_reason="end_turn")},
    ]
    mock_agent = _MockStreamingAgent(events)
    agent._strands = mock_agent  # noqa: SLF001

    async def _collect():
        return [event async for event in agent.stream_chat("list layers")]

    seen = asyncio.run(_collect())

    assert seen == events
    assert mock_agent.calls == ["list layers"]


def test_stream_chat_clears_previous_run_state() -> None:
    """Verify stream_chat starts from fresh cancellation/tool-call state."""
    m = MockLeafmap()
    agent = for_leafmap(m, model=_MockModel())
    agent._cancelled.append("old_tool")  # noqa: SLF001
    agent._tool_calls.append({"name": "old_tool"})  # noqa: SLF001
    agent._strands = _MockStreamingAgent([{"data": "done"}])  # noqa: SLF001

    async def _drain():
        return [event async for event in agent.stream_chat("hello")]

    asyncio.run(_drain())

    assert agent._cancelled == []  # noqa: SLF001
    assert agent._tool_calls == []  # noqa: SLF001


def test_chat_in_background_returns_and_invokes_callback() -> None:
    """Verify that chat in background returns and invokes callback."""
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
        """Handle result."""
        box["resp"] = resp
        done.set()

    th = agent.chat_in_background("list layers", on_result=_on_result)
    assert isinstance(th, threading.Thread)
    assert done.wait(2.0), "background chat callback did not fire"
    assert box["resp"].success is True
    assert box["resp"].answer_text == "done"


def test_qgis_chat_on_gui_thread_runs_worker_and_pumps_events(monkeypatch) -> None:
    """Verify that qgis chat on gui thread runs worker and pumps events."""
    main_thread = threading.current_thread()
    processed = threading.Event()
    calls = {"process_events": 0}
    _install_fake_qgis_qt(monkeypatch, calls)

    def _mark_processed():
        processed.set()
        calls["process_events"] += 1

    # Replace the fake processEvents method for this test so the mocked Strands
    # call can wait until the GUI event loop has been pumped at least once.
    sys.modules["qgis.PyQt.QtWidgets"].QApplication.instance().processEvents = (
        _mark_processed
    )

    agent = for_qgis(MockQGISIface(), MockQGISProject(), model=_MockModel())
    worker_thread: list[threading.Thread] = []

    def _strands(_query: str):
        """Return the mocked Strands response."""
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


def test_qgis_stream_chat_on_gui_thread_runs_worker_and_pumps_events(
    monkeypatch,
) -> None:
    """Verify qgis stream_chat on gui thread streams via worker."""
    main_thread = threading.current_thread()
    processed = threading.Event()
    calls = {"process_events": 0}
    _install_fake_qgis_qt(monkeypatch, calls)

    agent = for_qgis(MockQGISIface(), MockQGISProject(), model=_MockModel())
    worker_thread: list[threading.Thread] = []

    class _DelayedStreamingAgent:
        """Provide a streaming agent that waits for Qt event pumping."""

        async def stream_async(self, _query: str):
            """Yield after the GUI side has pumped Qt events."""
            worker_thread.append(threading.current_thread())
            deadline = time.time() + 2.0
            while not processed.is_set() and time.time() < deadline:
                await asyncio.sleep(0.01)
            yield {"data": "do"}
            yield {"data": "ne"}

    def _process_events():
        calls["process_events"] += 1
        processed.set()

    sys.modules["qgis.PyQt.QtWidgets"].QApplication.instance().processEvents = (
        _process_events
    )
    agent._strands = _DelayedStreamingAgent()  # noqa: SLF001

    async def _collect():
        return [event async for event in agent.stream_chat("list layers")]

    seen = asyncio.run(_collect())

    assert seen == [{"data": "do"}, {"data": "ne"}]
    assert worker_thread[0] is not main_thread
    assert calls["process_events"] > 0
