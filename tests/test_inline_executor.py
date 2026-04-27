"""Tests for the inline-tool-execution context manager.

The previous attempt at QGIS thread safety used Qt main-thread
marshaling (``pyqtSignal`` + ``Qt.BlockingQueuedConnection``), which
deadlocked QGIS: the main thread sat inside ``agent.invoke()`` for the
whole call and could not pump Qt events to drain the queued
connection. The fix sidesteps the threading entirely: the
``inline_tool_execution`` context manager flips a
:class:`contextvars.ContextVar` that a stable wrapper around
LangGraph's ``get_executor_for_config`` consults on every call. When
the flag is set the wrapper returns an in-process inline executor;
otherwise it delegates to the captured original factory.

These tests pin that behaviour without launching a real LangGraph
agent: they reach into ``langgraph.prebuilt.tool_node`` directly,
exercise the inline executor's ``map`` / ``submit`` /
context-manager surface, and verify the ``ContextVar`` gating across
nested entries, threads, and async tasks.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future

import langgraph.prebuilt.tool_node as _tool_node_mod
import pytest

from geoagent.tools import _inline_executor as _inline_mod
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


def test_inline_tool_execution_returns_inline_executor_inside_context() -> None:
    """Inside the context, ``get_executor_for_config`` must return inline."""
    with inline_tool_execution():
        ex = _tool_node_mod.get_executor_for_config({})
        assert isinstance(ex, _InlineExecutor)


def test_inline_tool_execution_delegates_outside_context() -> None:
    """Outside the context, the wrapper must delegate to the original.

    Other code paths (Streamlit UI, leafmap) must continue using
    LangGraph's default ``ContextThreadPoolExecutor``.
    """
    # Trigger the lazy patch install if not already done.
    with inline_tool_execution():
        pass
    assert _inline_mod._patched is True

    ex = _tool_node_mod.get_executor_for_config({})
    assert not isinstance(ex, _InlineExecutor), (
        "outside the context, the wrapper must return whatever the original "
        "factory returns (a ContextThreadPoolExecutor in real LangGraph)"
    )


def test_inline_tool_execution_resets_flag_on_exception() -> None:
    """Even if the body raises, the gating flag must reset."""
    with pytest.raises(RuntimeError, match="boom"):
        with inline_tool_execution():
            raise RuntimeError("boom")
    # The wrapper is permanent; the gating flag is what should reset.
    assert _inline_mod._inline_active.get() is False
    ex = _tool_node_mod.get_executor_for_config({})
    assert not isinstance(ex, _InlineExecutor)


def test_inline_tool_execution_is_thread_local_via_contextvar() -> None:
    """Concurrent threads must each carry their own gating flag.

    A naive global swap would let one thread's "inline" state leak into
    another thread that never entered the context. The
    :class:`contextvars.ContextVar` approach prevents that: each thread
    sees the value set in its own context.
    """
    started = threading.Event()
    can_check = threading.Event()
    other_saw_inline: list[bool] = []

    def worker() -> None:
        started.wait()
        # The other thread is inside ``inline_tool_execution`` right now.
        # This worker has not entered, so the wrapper must delegate.
        ex = _tool_node_mod.get_executor_for_config({})
        other_saw_inline.append(isinstance(ex, _InlineExecutor))
        can_check.set()

    t = threading.Thread(target=worker)
    t.start()

    with inline_tool_execution():
        started.set()
        can_check.wait()
        # In *this* thread the gating flag is still True.
        assert isinstance(_tool_node_mod.get_executor_for_config({}), _InlineExecutor)

    t.join()
    assert other_saw_inline == [
        False
    ], "another thread must NOT see the inline gating from this thread"


def test_inline_tool_execution_propagates_through_await() -> None:
    """``ContextVar`` semantics propagate the flag to awaited code in
    the same task. Code in *new* tasks (e.g. ``asyncio.create_task``)
    inherits the value at the time the task was created, which is the
    standard contextvar contract — verified here so the doc string's
    promise is enforced.
    """

    async def inside_same_task() -> bool:
        await asyncio.sleep(0)  # actually yield
        ex = _tool_node_mod.get_executor_for_config({})
        return isinstance(ex, _InlineExecutor)

    async def driver() -> bool:
        with inline_tool_execution():
            return await inside_same_task()

    assert asyncio.run(driver()) is True


def test_inline_tool_execution_isolates_concurrent_async_tasks() -> None:
    """Two tasks racing in the same event loop see independent gating.

    Task A enters ``inline_tool_execution``; task B never does. Task B
    must still see the original factory even while task A has the
    flag set.
    """

    async def task_a(start: asyncio.Event, finish: asyncio.Event) -> bool:
        with inline_tool_execution():
            start.set()
            await finish.wait()
            return isinstance(
                _tool_node_mod.get_executor_for_config({}), _InlineExecutor
            )

    async def task_b(start: asyncio.Event, finish: asyncio.Event) -> bool:
        await start.wait()
        try:
            return isinstance(
                _tool_node_mod.get_executor_for_config({}), _InlineExecutor
            )
        finally:
            finish.set()

    async def driver() -> tuple[bool, bool]:
        start = asyncio.Event()
        finish = asyncio.Event()
        a_fut = asyncio.create_task(task_a(start, finish))
        b_fut = asyncio.create_task(task_b(start, finish))
        a_inline, b_inline = await asyncio.gather(a_fut, b_fut)
        return a_inline, b_inline

    a_inline, b_inline = asyncio.run(driver())
    assert a_inline is True
    assert b_inline is False, "task B must not inherit task A's inline gating"
