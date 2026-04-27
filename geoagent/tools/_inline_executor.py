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
:func:`inline_tool_execution` context manager flips a
:class:`contextvars.ContextVar` that a stable wrapper around
``langgraph.prebuilt.tool_node.get_executor_for_config`` consults on
every call. When the flag is set the wrapper returns an in-process
:class:`_InlineExecutor`; otherwise it delegates to the original
factory unchanged. Because the wrapper is installed *once* and the
gating state lives in a ``ContextVar``:

- Concurrent ``invoke()`` calls in different threads do not stomp on
  each other's gating (each thread has its own ContextVar value).
- ``await`` boundaries do not leak the inline executor across
  unrelated async tasks (Python copies the context on task creation).
- Other code paths in the same process (Streamlit UI, leafmap-based
  runs in a Jupyter kernel) keep LangGraph's default thread pool
  whenever they have not entered :func:`inline_tool_execution`.
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
from concurrent.futures import Future
from typing import Any, Callable, Iterable, Iterator, Optional

# Per-context flag that the stable wrapper around
# ``get_executor_for_config`` checks. Default ``False`` means delegate
# to the original factory; ``inline_tool_execution`` sets it to ``True``
# for the duration of a single ``invoke`` / ``stream`` / ``ainvoke`` /
# ``astream`` / ``astream_events`` call.
_inline_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "geoagent_inline_tool_execution", default=False
)


_patch_lock = threading.Lock()
_original_get_executor_for_config: Optional[Callable[..., Any]] = None
_patched: bool = False


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


def _ensure_patched() -> None:
    """Install the ContextVar-gated wrapper around the executor factory.

    The wrapper is installed *once* and stays in place for the life of
    the process. Each call consults :data:`_inline_active`; when set,
    the wrapper returns an :class:`_InlineExecutor`, otherwise it
    delegates to the captured original factory. This avoids the
    correctness hazards of repeatedly swapping a module global from
    multiple threads or async tasks.
    """
    global _patched, _original_get_executor_for_config
    if _patched:
        return
    with _patch_lock:
        if _patched:
            return
        import langgraph.prebuilt.tool_node as _tool_node_mod

        _original_get_executor_for_config = _tool_node_mod.get_executor_for_config

        def _scoped_get_executor_for_config(config: Any) -> Any:
            if _inline_active.get():
                return _InlineExecutor()
            assert _original_get_executor_for_config is not None
            return _original_get_executor_for_config(config)

        _tool_node_mod.get_executor_for_config = _scoped_get_executor_for_config
        _patched = True


@contextlib.contextmanager
def inline_tool_execution() -> Iterator[None]:
    """Context manager: force LangGraph ``ToolNode`` to run tools inline.

    On entry, ensure the stable wrapper around
    ``langgraph.prebuilt.tool_node.get_executor_for_config`` is in
    place, then set a :class:`contextvars.ContextVar` flag the wrapper
    consults. On exit the flag is reset to its prior value via the
    ``ContextVar`` token, so nested entries / parallel uses in
    different contexts (threads, async tasks) do not interfere.

    Concurrency contract:

    - Two threads can enter this context manager simultaneously without
      racing on a global swap. Each thread carries its own
      ``ContextVar`` value.
    - ``await`` boundaries inside the context propagate the ``True``
      value to awaited code that runs inside the same task; new tasks
      spawned via ``asyncio.create_task`` inherit the value at task
      creation time (Python's standard contextvar semantics).
    - Code paths that never enter the context manager keep LangGraph's
      default ``ContextThreadPoolExecutor``.
    """
    _ensure_patched()
    token = _inline_active.set(True)
    try:
        yield
    finally:
        _inline_active.reset(token)


__all__ = ["inline_tool_execution"]
