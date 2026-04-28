"""Qt marshalling helpers for safely calling QGIS GUI APIs.

This module is import-safe outside QGIS/PyQt environments. When Qt cannot be
imported (e.g. CI, plain Python), helpers fall back to direct invocation.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


def is_qt_gui_thread() -> bool:
    """Return ``True`` when the current thread is Qt's GUI thread."""
    try:
        from qgis.PyQt.QtCore import QThread  # type: ignore[import-not-found]
        from qgis.PyQt.QtWidgets import QApplication  # type: ignore[import-not-found]
    except Exception:
        return False

    app = QApplication.instance()
    if app is None:
        return False
    return QThread.currentThread() == app.thread()


def process_qt_events() -> None:
    """Process pending Qt events when a QGIS/PyQt application is active."""
    try:
        from qgis.PyQt.QtWidgets import QApplication  # type: ignore[import-not-found]
    except Exception:
        return

    app = QApplication.instance()
    if app is None:
        return
    try:
        app.processEvents()
    except Exception:
        return


def run_on_qt_gui_thread(func: Callable[[], T]) -> T:
    """Run ``func`` on Qt's GUI thread when available.

    QGIS canvas, layer-tree, and iface methods must execute on the GUI thread.
    When a caller is already on the GUI thread, this runs ``func`` inline.
    Otherwise it posts to the GUI thread via ``QMetaObject.invokeMethod`` and
    blocks until completion.

    Outside Qt/QGIS (tests, CI), this degrades to a direct call.
    """
    try:
        from qgis.PyQt.QtCore import QMetaObject, QObject, Qt, QThread, pyqtSlot  # type: ignore[import-not-found]
        from qgis.PyQt.QtWidgets import QApplication  # type: ignore[import-not-found]
    except Exception:
        return func()

    app = QApplication.instance()
    if app is None:
        return func()

    # PyQt can wrap the same C++ QThread with different Python objects.
    # Use equality, not identity, to avoid false "different thread" detection
    # that can deadlock with BlockingQueuedConnection.
    if is_qt_gui_thread():
        return func()

    source_thread = QThread.currentThread()
    gui = app.thread()

    class _Invoker(QObject):
        """Qt object used to invoke a callable on the GUI thread."""

        def __init__(self) -> None:
            super().__init__()
            self.value: Any = None
            self.error: BaseException | None = None

        @pyqtSlot()
        def run(self) -> None:
            """Run the worker body."""
            try:
                self.value = func()
            except BaseException as exc:  # pragma: no cover - passthrough path
                self.error = exc
            finally:
                # The object was created in the caller thread and moved to the
                # GUI thread for invocation. Move it back before Python drops
                # the wrapper, otherwise Qt may destroy a GUI-affine QObject
                # from the worker thread and crash QGIS.
                try:
                    self.moveToThread(source_thread)
                except Exception:
                    pass

    invoker = _Invoker()
    invoker.moveToThread(gui)
    blocking = getattr(Qt, "BlockingQueuedConnection", None)
    if blocking is None:
        blocking = Qt.ConnectionType.BlockingQueuedConnection
    ok = QMetaObject.invokeMethod(
        invoker,
        "run",
        blocking,
    )
    if ok is False:
        raise RuntimeError("Failed to marshal QGIS API call to the Qt GUI thread.")
    if invoker.error is not None:
        raise invoker.error
    return invoker.value
