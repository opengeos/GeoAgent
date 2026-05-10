"""WebSocket command broker for browser-hosted maps."""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from typing import Any


class BrowserMapTimeoutError(TimeoutError):
    """Raised when the browser does not answer a map command in time."""


class BrowserMapSession:
    """Bridge synchronous GeoAgent tools to one async browser WebSocket.

    Strands tools are synchronous functions. The browser connection is async.
    This session schedules outgoing command messages on the server event loop
    and blocks the calling tool thread until the matching browser result arrives.
    """

    def __init__(
        self,
        *,
        websocket: Any,
        loop: asyncio.AbstractEventLoop,
        session_id: str,
        map_id: str = "default",
        timeout_seconds: float = 30.0,
    ) -> None:
        self.websocket = websocket
        self.loop = loop
        self.session_id = session_id
        self.map_id = map_id or "default"
        self.timeout_seconds = float(timeout_seconds)
        self._pending: dict[str, tuple[threading.Event, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def call(
        self,
        command: str,
        args: dict[str, Any] | None = None,
        *,
        map_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> Any:
        """Send a map command to the browser and wait for its result."""
        command_id = str(uuid.uuid4())
        event = threading.Event()
        holder: dict[str, Any] = {}
        with self._lock:
            self._pending[command_id] = (event, holder)

        payload = {
            "type": "map_command",
            "id": command_id,
            "sessionId": self.session_id,
            "mapId": map_id or self.map_id,
            "command": command,
            "args": args or {},
        }
        wait_timeout = (
            self.timeout_seconds if timeout_seconds is None else float(timeout_seconds)
        )
        # Treat ``wait_timeout`` as a total deadline so the send and response
        # phases share one budget instead of each consuming a full ``wait_timeout``.
        deadline = time.monotonic() + wait_timeout
        future = asyncio.run_coroutine_threadsafe(
            self.websocket.send_json(payload),
            self.loop,
        )
        try:
            future.result(timeout=wait_timeout)
        except Exception:
            self._drop_pending(command_id)
            raise

        remaining = max(0.0, deadline - time.monotonic())
        if not event.wait(remaining):
            self._drop_pending(command_id)
            raise BrowserMapTimeoutError(
                f"Browser did not return a result for {command!r} within "
                f"{wait_timeout:g} seconds."
            )

        result = holder.get("message", {})
        if not result.get("ok", False):
            error = result.get("error") or f"Browser command {command!r} failed."
            raise RuntimeError(str(error))
        return result.get("result")

    def resolve_result(self, message: dict[str, Any]) -> bool:
        """Resolve a pending command from a browser ``map_command_result``."""
        command_id = str(message.get("id") or "")
        if not command_id:
            return False
        with self._lock:
            pending = self._pending.pop(command_id, None)
        if pending is None:
            return False
        event, holder = pending
        holder["message"] = message
        event.set()
        return True

    def fail_all(self, error: str) -> None:
        """Fail all pending commands, usually after WebSocket disconnect."""
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for event, holder in pending:
            holder["message"] = {"ok": False, "error": error}
            event.set()

    def _drop_pending(self, command_id: str) -> None:
        with self._lock:
            self._pending.pop(command_id, None)
