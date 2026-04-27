"""Qt main-thread marshaling for QGIS tool bodies.

LangGraph's :class:`langgraph.prebuilt.tool_node.ToolNode` dispatches every
tool call through a ``ContextThreadPoolExecutor`` (see
``langchain_core/runnables/config.py``). That means every body in
:mod:`geoagent.tools.qgis` runs on a *worker* thread even when the user
calls ``agent.invoke()`` synchronously from QGIS's main thread. Touching
``iface`` / ``mapCanvas`` / ``QgsProject`` / ``layerTreeRoot`` from a
non-main thread violates Qt's thread affinity, corrupts canvas + layer
tree state, emits ``QObject::startTimer`` warnings, and eventually
segfaults the QGIS process.

This module provides one helper, :func:`run_on_qgis_main_thread`, that
forwards a callable to the Qt main thread (the thread that owns the
``QApplication``) and blocks the caller until the call returns. The
helper is import-safe outside QGIS — the Qt import is lazy and the
helper degrades to a direct call when no Qt application is running, so
``import geoagent.tools.qgis`` continues to work in CI and unit tests
without QGIS installed (pinned by
:mod:`tests.test_qgis_import_safe`).
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


def run_on_qgis_main_thread(
    fn: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Run ``fn(*args, **kwargs)`` on the Qt main thread.

    Behaviour:

    - If Qt is not importable (no QGIS / no PyQt), call ``fn`` directly.
      Tests and non-QGIS environments take this path.
    - If no ``QCoreApplication`` exists yet (Qt loaded but no app
      running), call ``fn`` directly. This covers some CI shapes where
      ``qgis.PyQt`` imports succeed but the GUI app is absent.
    - If the current thread *is* the main thread, call ``fn`` directly.
      This is a fast path so re-entrant calls and main-thread direct
      use don't pay marshaling cost (and don't deadlock on the blocking
      queued connection).
    - Otherwise dispatch ``fn`` via
      ``QMetaObject.invokeMethod(app, runner, Qt.BlockingQueuedConnection)``,
      block the worker thread until the main thread runs ``runner``,
      then return the result. Any exception raised inside ``fn`` is
      captured and re-raised on the worker thread, so LangChain's
      tool-error machinery still sees it.

    Args:
        fn: The callable to execute.
        *args: Positional arguments forwarded to ``fn``.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        Whatever ``fn(*args, **kwargs)`` returns.

    Raises:
        Anything raised by ``fn`` (re-raised on the calling thread when
        the call was marshaled).
    """
    try:
        from qgis.PyQt.QtCore import (  # type: ignore[import-not-found]
            QCoreApplication,
            QMetaObject,
            Qt,
            QThread,
        )
    except ImportError:
        return fn(*args, **kwargs)

    app = QCoreApplication.instance()
    if app is None or QThread.currentThread() == app.thread():
        return fn(*args, **kwargs)

    holder: dict[str, Any] = {}

    def runner() -> None:
        try:
            holder["value"] = fn(*args, **kwargs)
        except BaseException as exc:  # propagate everything, incl. KeyboardInterrupt
            holder["exception"] = exc

    # PyQt5 >= 5.10 (QGIS LTR ships 5.15) accepts a Python callable as the
    # second argument to invokeMethod. ``BlockingQueuedConnection`` blocks
    # the worker thread until the main thread has executed ``runner``.
    QMetaObject.invokeMethod(app, runner, Qt.BlockingQueuedConnection)

    if "exception" in holder:
        raise holder["exception"]
    return holder.get("value")  # type: ignore[return-value]


__all__ = ["run_on_qgis_main_thread"]
