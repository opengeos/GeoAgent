"""Qt main-thread marshaling for QGIS tool bodies.

LangGraph's :class:`langgraph.prebuilt.tool_node.ToolNode` dispatches every
tool call through a ``ContextThreadPoolExecutor``, so each body in
:mod:`geoagent.tools.qgis` runs on a *worker* thread even when
``agent.invoke()`` is called synchronously from QGIS's main thread.
Touching ``iface`` / ``mapCanvas`` / ``QgsProject`` / ``layerTreeRoot``
from a non-main thread violates Qt's thread affinity, corrupts canvas +
layer-tree state, emits ``QObject::startTimer: Timers cannot be started
from another thread`` warnings, and eventually segfaults the QGIS
process.

This module exposes one helper, :func:`run_on_qgis_main_thread`, that
forwards a callable to the Qt main thread (the thread that owns the
``QApplication``) and blocks the caller until the call returns. It is
import-safe outside QGIS — the Qt import is lazy and the helper
degrades to a direct call when no Qt application is running, so
``import geoagent.tools.qgis`` continues to succeed in CI without QGIS
installed (pinned by :mod:`tests.test_qgis_import_safe`).

Implementation note: ``QMetaObject.invokeMethod`` accepts a Python
callable as the second argument only in newer PyQt5/PyQt6 builds — the
PyQt5 shipped with QGIS LTR does not. The portable approach is a small
``QObject`` subclass with a ``pyqtSignal(object)`` connected to its
slot via ``Qt.BlockingQueuedConnection``: emitting the signal from a
worker thread blocks the emitter until the slot runs on the receiver's
(main) thread.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, TypeVar

T = TypeVar("T")


_dispatcher: Any = None
_dispatcher_lock = threading.Lock()


def _make_dispatcher() -> Any:
    """Construct the main-thread dispatcher ``QObject``.

    The dispatcher exposes a ``request`` signal (carrying a Python
    callable) connected to its slot via ``Qt.BlockingQueuedConnection``.
    Emitting from a worker thread blocks the emitter until the slot
    runs on the dispatcher's owning thread (set to the main thread via
    ``moveToThread``).

    Constructing this object on a worker thread is safe: the worker
    initially owns it, and the immediate ``moveToThread(app.thread())``
    transfers affinity so the slot will run on the main thread.
    """
    from qgis.PyQt.QtCore import (  # type: ignore[import-not-found]
        QCoreApplication,
        QObject,
        Qt,
        pyqtSignal,
        pyqtSlot,
    )

    class _MainThreadDispatcher(QObject):  # type: ignore[misc]
        request = pyqtSignal(object)

        def __init__(self) -> None:
            super().__init__()
            app = QCoreApplication.instance()
            if app is not None:
                # Re-affiliate this QObject with the main thread so the
                # slot connected via BlockingQueuedConnection executes
                # there, not on the worker thread that constructed us.
                self.moveToThread(app.thread())
            self.request.connect(self._on_request, Qt.BlockingQueuedConnection)

        @pyqtSlot(object)
        def _on_request(self, runner: Callable[[], None]) -> None:
            runner()

    return _MainThreadDispatcher()


def _ensure_dispatcher() -> Any:
    """Return the singleton dispatcher, creating it on first use.

    Lock-guarded because multiple LangGraph worker threads may call the
    marshaler concurrently and we must not build two dispatchers.
    """
    global _dispatcher
    with _dispatcher_lock:
        if _dispatcher is None:
            _dispatcher = _make_dispatcher()
        return _dispatcher


def run_on_qgis_main_thread(
    fn: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Run ``fn(*args, **kwargs)`` on the Qt main thread.

    Behaviour:

    - **No Qt available** → call ``fn`` directly. Tests / non-QGIS
      environments take this path so ``import geoagent.tools.qgis``
      stays import-safe outside QGIS.
    - **No ``QCoreApplication``** (Qt loaded but no app running) → call
      ``fn`` directly.
    - **Already on the main thread** → call ``fn`` directly. This is
      a fast path so re-entrant calls don't pay marshaling cost (and
      don't deadlock on the blocking queued connection).
    - **Off-thread** → emit a signal carrying a runner closure to a
      lazily-constructed dispatcher whose slot is connected with
      ``Qt.BlockingQueuedConnection``. The emit blocks until the
      slot has run on the main thread; the closure stashes the
      result / exception in a local dict, which is then returned /
      re-raised on the worker thread.

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

    _ensure_dispatcher().request.emit(runner)

    if "exception" in holder:
        raise holder["exception"]
    return holder.get("value")  # type: ignore[return-value]


__all__ = ["run_on_qgis_main_thread"]
