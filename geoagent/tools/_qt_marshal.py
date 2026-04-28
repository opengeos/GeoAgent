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
        from qgis.PyQt.QtCore import QMetaObject, QObject, Qt, pyqtSlot  # type: ignore[import-not-found]
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

    gui = app.thread()

    class _Invoker(QObject):
        def __init__(self) -> None:
            super().__init__()
            self.value: Any = None
            self.error: BaseException | None = None

        @pyqtSlot()
        def run(self) -> None:
            try:
                self.value = func()
            except BaseException as exc:  # pragma: no cover - passthrough path
                self.error = exc

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
    if not ok:
        return func()
    if invoker.error is not None:
        raise invoker.error
    return invoker.value
