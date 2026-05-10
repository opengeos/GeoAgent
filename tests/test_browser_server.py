"""Tests for the browser WebSocket server."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from geoagent.browser import server
from geoagent.browser.server import create_browser_app


def test_browser_websocket_accepts_connection() -> None:
    """Verify the browser WebSocket route accepts and sends session metadata."""
    fastapi_testclient = pytest.importorskip("fastapi.testclient")

    client = fastapi_testclient.TestClient(create_browser_app())
    with client.websocket_connect("/geoagent/ws") as websocket:
        message = websocket.receive_json()

    assert message["type"] == "session"
    assert message["sessionId"]
    assert message["mapId"] == "default"


def test_browser_code_confirmation_is_explicitly_opt_in() -> None:
    """Verify only the browser code flag approves run_maplibre_script."""
    code_request = SimpleNamespace(
        tool_name="run_maplibre_script",
        args={},
        category="browser_map",
    )
    remove_request = SimpleNamespace(
        tool_name="remove_layer",
        args={},
        category="browser_map",
    )

    assert (
        server._browser_confirm_callback(False, False)(code_request) is False
    )  # noqa: SLF001
    assert (
        server._browser_confirm_callback(True, False)(code_request) is True
    )  # noqa: SLF001
    assert (
        server._browser_confirm_callback(True, False)(remove_request) is False
    )  # noqa: SLF001
    assert (
        server._browser_confirm_callback(False, True)(remove_request) is True
    )  # noqa: SLF001


def test_browser_websocket_streams_chat_deltas(monkeypatch) -> None:
    """Verify browser chat uses stream_chat and emits incremental deltas."""
    fastapi_testclient = pytest.importorskip("fastapi.testclient")

    class FakeAgent:
        """Tiny browser agent test double."""

        def __init__(self) -> None:
            self._tool_calls = [{"name": "list_layers", "result": []}]
            self._cancelled = []

        async def stream_chat(self, prompt: str):
            assert prompt == "hello"
            yield {"data": "he"}
            yield {"data": "llo"}
            yield {"current_tool_use": {"name": "list_layers"}}
            yield {
                "result": SimpleNamespace(
                    message={"content": [{"text": "hello"}]},
                    metrics=SimpleNamespace(tool_metrics={"list_layers": {}}),
                    stop_reason="end_turn",
                )
            }

    monkeypatch.setattr(
        server,
        "for_browser_maplibre",
        lambda **_kwargs: FakeAgent(),
    )

    client = fastapi_testclient.TestClient(create_browser_app())
    with client.websocket_connect("/geoagent/ws") as websocket:
        assert websocket.receive_json()["type"] == "session"
        websocket.send_json({"type": "chat", "message": "hello", "mapId": "default"})

        assert websocket.receive_json() == {"type": "chat_status", "status": "running"}
        assert websocket.receive_json() == {"type": "chat_delta", "text": "he"}
        assert websocket.receive_json() == {"type": "chat_delta", "text": "llo"}
        assert websocket.receive_json() == {"type": "chat_tool", "name": "list_layers"}
        final = websocket.receive_json()

    assert final["type"] == "chat_result"
    assert final["ok"] is True
    assert final["answer"] == "hello"
    assert final["executed_tools"] == ["list_layers"]
    assert final["tool_calls"] == [{"name": "list_layers", "result": []}]
    assert final["cancelled_tools"] == []
    assert final["streamed"] is True
