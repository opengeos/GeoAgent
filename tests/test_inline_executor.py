"""Tests for the inline-tool-execution context manager.

The previous attempt at QGIS thread safety used Qt main-thread
marshaling (``pyqtSignal`` + ``Qt.BlockingQueuedConnection``), which
deadlocked QGIS: the main thread sat inside ``agent.invoke()`` for the
whole call and could not pump Qt events to drain the queued
connection. The fix sidesteps the threading entirely by patching
LangGraph's ``ToolNode`` executor for the duration of the call.

These tests pin the patch's behaviour without launching a real
LangGraph agent: they reach into ``langgraph.prebuilt.tool_node``
directly, exercise the inline executor's ``map`` / ``submit`` /
context-manager surface, and verify the ``inline_tool_execution``
context manager swaps + restores the executor factory.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future

import langgraph.prebuilt.tool_node as _tool_node_mod
import pytest

from geoagent.tools._inline_executor import _InlineExecutor, inline_tool_execution


def test_inline_executor_map_runs_inline() -> None:
    calling_thread = threading.current_thread()
    seen_threads: list[threading.Thread] = []

    def f(x: int) -> int:
        seen_threads.append(threading.current_thread())
        return x * 2

    with _InlineExecutor() as ex:
        results = list(ex.map(f, [1, 2, 3]))

    assert results == [2, 4, 6]
    # All callables ran on the calling thread, not a worker thread.
    assert all(t is calling_thread for t in seen_threads)


def test_inline_executor_map_propagates_exceptions() -> None:
    """Eager evaluation of map() must surface the first exception."""

    class _Boom(RuntimeError):
        pass

    def f(x: int) -> int:
        if x == 2:
            raise _Boom("kaboom")
        return x

    with pytest.raises(_Boom, match="kaboom"):
        list(_InlineExecutor().map(f, [1, 2, 3]))


def test_inline_executor_submit_returns_completed_future() -> None:
    fut = _InlineExecutor().submit(lambda x: x + 1, 4)
    assert isinstance(fut, Future)
    assert fut.done() is True
    assert fut.result() == 5


def test_inline_executor_submit_captures_exception_in_future() -> None:
    class _Boom(RuntimeError):
        pass

    fut = _InlineExecutor().submit(lambda: (_ for _ in ()).throw(_Boom("x")))
    assert fut.done() is True
    with pytest.raises(_Boom, match="x"):
        fut.result()


def test_inline_tool_execution_patches_and_restores() -> None:
    """The context manager must swap and then restore the executor factory.

    ``ToolNode._func`` calls ``get_executor_for_config(config)`` once
    per node invocation. Inside the context, that should return our
    inline executor. After the context exits, the original factory
    must be back so other code paths (Streamlit UI, leafmap) keep
    LangGraph's default thread pool.
    """
    original = _tool_node_mod.get_executor_for_config

    with inline_tool_execution():
        ex = _tool_node_mod.get_executor_for_config({})
        assert isinstance(ex, _InlineExecutor)

    assert _tool_node_mod.get_executor_for_config is original


def test_inline_tool_execution_restores_on_exception() -> None:
    """Even if the body raises, the original executor factory is restored."""
    original = _tool_node_mod.get_executor_for_config

    with pytest.raises(RuntimeError, match="boom"):
        with inline_tool_execution():
            raise RuntimeError("boom")

    assert _tool_node_mod.get_executor_for_config is original
