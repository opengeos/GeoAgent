"""Force LangGraph's ToolNode to run tools inline on the calling thread.

LangGraph's :class:`langgraph.prebuilt.tool_node.ToolNode` dispatches
every tool call through ``executor.map(...)`` over a
``ContextThreadPoolExecutor`` (see
``langchain_core.runnables.config.get_executor_for_config``). When
``agent.invoke()`` is called from QGIS's Python console (the Qt main
thread), this offloads tool bodies onto worker threads. Our QGIS tools
must run on the Qt main thread or QGIS corrupts ``iface``, layers stop
rendering, and the process eventually segfaults.

The earlier attempt — marshaling each tool body to the main thread via
``QMetaObject.invokeMethod`` / ``pyqtSignal`` with
``Qt.BlockingQueuedConnection`` — *deadlocked* QGIS: the main thread
sits inside ``agent.invoke()`` for the whole call and cannot pump Qt
events to drain the queued connection, so the worker thread blocks
forever.

The fix here sidesteps the threading entirely. The
:func:`inline_tool_execution` context manager replaces
``langgraph.prebuilt.tool_node.get_executor_for_config`` with a factory
that returns an in-process :class:`_InlineExecutor`. Inside the
context, ``ToolNode``'s ``executor.map()`` / ``executor.submit()`` runs
each tool synchronously on the calling thread. Because the user's
``agent.invoke()`` runs on the Qt main thread, every tool body runs on
the main thread too, and the QGIS API is touched only from the thread
that owns the ``QApplication``.

The patch is scoped: it is restored on exit, so other code paths (the
Streamlit UI, leafmap-based runs in a Jupyter kernel) keep LangGraph's
default thread pool.
"""

from __future__ import annotations

import contextlib
from concurrent.futures import Future
from typing import Any, Callable, Iterable, Iterator


class _InlineExecutor:
    """Minimal executor that runs callables inline on the calling thread.

    Implements only the surface LangGraph's ``ToolNode`` and the
    ``langchain_core`` runnable infrastructure use: ``map``, ``submit``,
    ``shutdown``, plus the context-manager protocol so
    ``with get_executor_for_config(config) as executor:`` works.
    """

    def __enter__(self) -> "_InlineExecutor":
        return self

    def __exit__(self, *_exc: Any) -> None:
        return None

    def map(
        self,
        fn: Callable[..., Any],
        *iterables: Iterable[Any],
        timeout: Any = None,
        chunksize: int = 1,
    ) -> Iterator[Any]:
        # Eagerly evaluate so an exception propagates the same way the
        # ThreadPoolExecutor would when the caller does
        # ``list(executor.map(...))`` (the exact pattern in
        # ``ToolNode._func``).
        return iter([fn(*args) for args in zip(*iterables)])

    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Future:
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        return None


@contextlib.contextmanager
def inline_tool_execution() -> Iterator[None]:
    """Context manager: force LangGraph ``ToolNode`` to run tools inline.

    Replaces ``langgraph.prebuilt.tool_node.get_executor_for_config``
    with a factory that returns an :class:`_InlineExecutor` for the
    duration of the block; restores the original on exit.

    The patch is targeted at the symbol bound *inside*
    ``langgraph.prebuilt.tool_node`` rather than at the source module
    (``langchain_core.runnables.config``) because that module imports
    ``get_executor_for_config`` by name with ``from ... import``, which
    creates an independent local binding. Patching the source module
    would not affect the binding ``ToolNode`` already captured.
    """
    import langgraph.prebuilt.tool_node as _tool_node_mod

    original = _tool_node_mod.get_executor_for_config
    _tool_node_mod.get_executor_for_config = lambda _config: _InlineExecutor()
    try:
        yield
    finally:
        _tool_node_mod.get_executor_for_config = original


__all__ = ["inline_tool_execution"]
