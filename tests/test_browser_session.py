"""Concurrency tests for ``BrowserMapSession``."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from geoagent.browser.session import BrowserMapSession, BrowserMapTimeoutError


class _FakeWebSocket:
    """Minimal async websocket stand-in that records sent payloads."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_json(self, payload: dict[str, Any]) -> None:
        """Record ``payload`` so tests can assert on outbound traffic."""
        self.sent.append(payload)


class _LoopThread:
    """Run an asyncio event loop on a background thread for tests."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def stop(self) -> None:
        """Stop the loop and join the worker thread."""
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=2.0)
        self.loop.close()


@pytest.fixture
def loop_thread() -> _LoopThread:
    """Provide a background event loop and tear it down after the test."""
    runner = _LoopThread()
    try:
        yield runner
    finally:
        runner.stop()


def _make_session(loop_thread: _LoopThread, **kwargs: Any) -> BrowserMapSession:
    """Build a session bound to the background loop with a fake websocket."""
    return BrowserMapSession(
        websocket=_FakeWebSocket(),
        loop=loop_thread.loop,
        session_id="test-session",
        timeout_seconds=kwargs.pop("timeout_seconds", 1.0),
        **kwargs,
    )


def test_call_times_out_when_browser_does_not_reply(
    loop_thread: _LoopThread,
) -> None:
    """``call`` raises ``BrowserMapTimeoutError`` when no result arrives."""
    session = _make_session(loop_thread, timeout_seconds=0.1)
    with pytest.raises(BrowserMapTimeoutError):
        session.call("noop")


def test_call_returns_when_resolve_result_unblocks(
    loop_thread: _LoopThread,
) -> None:
    """``resolve_result`` releases the waiting ``call`` thread."""
    session = _make_session(loop_thread, timeout_seconds=2.0)

    def _resolve_after_delay() -> None:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if session.websocket.sent:
                command_id = session.websocket.sent[-1]["id"]
                session.resolve_result(
                    {"id": command_id, "ok": True, "result": {"value": 42}}
                )
                return
            time.sleep(0.01)

    threading.Thread(target=_resolve_after_delay, daemon=True).start()
    assert session.call("get_value") == {"value": 42}


def test_call_propagates_browser_error(loop_thread: _LoopThread) -> None:
    """A non-ok result surfaces as ``RuntimeError`` to the caller."""
    session = _make_session(loop_thread, timeout_seconds=2.0)

    def _resolve_after_delay() -> None:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if session.websocket.sent:
                command_id = session.websocket.sent[-1]["id"]
                session.resolve_result({"id": command_id, "ok": False, "error": "boom"})
                return
            time.sleep(0.01)

    threading.Thread(target=_resolve_after_delay, daemon=True).start()
    with pytest.raises(RuntimeError, match="boom"):
        session.call("explode")


def test_fail_all_unblocks_every_pending_call(
    loop_thread: _LoopThread,
) -> None:
    """``fail_all`` releases all waiting threads with the supplied error."""
    session = _make_session(loop_thread, timeout_seconds=2.0)
    errors: list[BaseException] = []
    barrier = threading.Barrier(3)

    def _waiter() -> None:
        barrier.wait()
        try:
            session.call("noop")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_waiter, daemon=True) for _ in range(2)]
    for thread in threads:
        thread.start()

    barrier.wait()

    deadline = time.monotonic() + 1.0
    while len(session.websocket.sent) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)

    session.fail_all("Browser WebSocket disconnected.")
    for thread in threads:
        thread.join(timeout=2.0)

    assert len(errors) == 2
    for err in errors:
        assert isinstance(err, RuntimeError)
        assert "Browser WebSocket disconnected." in str(err)


def test_call_honors_explicit_zero_timeout(loop_thread: _LoopThread) -> None:
    """Passing ``timeout_seconds=0`` does not silently fall back to the default."""
    session = _make_session(loop_thread, timeout_seconds=30.0)
    started = time.monotonic()
    with pytest.raises(Exception):  # noqa: B017,PT011 - exact type is fast-path detail
        session.call("noop", timeout_seconds=0.0)
    elapsed = time.monotonic() - started
    assert elapsed < 5.0, "Explicit zero timeout fell back to the session default."


def test_call_timeout_is_total_deadline(loop_thread: _LoopThread) -> None:
    """``wait_timeout`` covers send + response combined, not each phase separately."""
    session = _make_session(loop_thread, timeout_seconds=0.3)
    started = time.monotonic()
    with pytest.raises(BrowserMapTimeoutError):
        session.call("noop")
    elapsed = time.monotonic() - started
    # Without a shared deadline the send + response phases would each consume
    # ``timeout_seconds``, doubling the total wait. Allow a small scheduling
    # margin but reject anything that approaches the doubled budget.
    assert (
        elapsed < 0.5
    ), f"call() exceeded its single-deadline timeout budget (took {elapsed:.2f}s)."
