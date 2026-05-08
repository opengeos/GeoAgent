"""FastAPI WebSocket server for embedded browser GeoAgent chat."""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from geoagent.browser.session import BrowserMapSession
from geoagent.core.factory import for_browser_maplibre
from geoagent.core.safety import auto_approve_safe_only


async def _send_json(websocket: Any, payload: dict[str, Any]) -> None:
    """Send JSON while tolerating disconnected clients."""
    try:
        await websocket.send_json(payload)
    except Exception:
        pass


async def _run_chat_turn(
    *,
    websocket: Any,
    session: BrowserMapSession,
    message: str,
    provider: str | None,
    model_id: str | None,
) -> None:
    """Run one GeoAgent chat turn in a worker thread."""
    try:
        await _send_json(websocket, {"type": "chat_status", "status": "running"})
        agent = for_browser_maplibre(
            session=session,
            provider=provider,
            model_id=model_id,
            confirm=auto_approve_safe_only,
        )
        response = await asyncio.to_thread(agent.chat, message)
        payload: dict[str, Any] = {
            "type": "chat_result",
            "ok": bool(response.success),
            "answer": response.answer_text or "",
            "executed_tools": list(response.executed_tools or []),
            "tool_calls": list(response.tool_calls or []),
            "cancelled_tools": list(response.cancelled_tools or []),
        }
        if response.error_message:
            payload["error"] = response.error_message
        await _send_json(websocket, payload)
    except Exception as exc:
        await _send_json(
            websocket,
            {
                "type": "chat_result",
                "ok": False,
                "answer": "",
                "error": str(exc),
            },
        )


def create_browser_app(
    *,
    provider: str | None = None,
    model_id: str | None = None,
    command_timeout_seconds: float = 30.0,
) -> Any:
    """Create the FastAPI app for browser-embedded GeoAgent chat."""
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    except ImportError as exc:  # pragma: no cover - exercised by CLI fallback.
        raise RuntimeError(
            "FastAPI is required for `geoagent browser`. Install with "
            "`pip install GeoAgent[browser]`."
        ) from exc

    app = FastAPI(title="GeoAgent Browser")

    @app.get("/geoagent/health")
    async def health() -> dict[str, Any]:
        return {"ok": True}

    @app.websocket("/geoagent/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        loop = asyncio.get_running_loop()
        session_id = str(uuid4())
        session = BrowserMapSession(
            websocket=websocket,
            loop=loop,
            session_id=session_id,
            timeout_seconds=command_timeout_seconds,
        )
        active_chat: asyncio.Task[None] | None = None
        await websocket.send_json(
            {
                "type": "session",
                "sessionId": session_id,
                "mapId": session.map_id,
            }
        )

        try:
            while True:
                message = await websocket.receive_json()
                msg_type = message.get("type")
                if msg_type == "map_command_result":
                    session.resolve_result(message)
                    continue
                if msg_type != "chat":
                    await websocket.send_json(
                        {
                            "type": "error",
                            "error": f"Unsupported message type: {msg_type!r}",
                        }
                    )
                    continue
                if active_chat is not None and not active_chat.done():
                    await websocket.send_json(
                        {
                            "type": "chat_result",
                            "ok": False,
                            "answer": "",
                            "error": "A chat turn is already running for this session.",
                        }
                    )
                    continue

                session.map_id = str(message.get("mapId") or "default")
                chat_message = str(message.get("message") or "").strip()
                if not chat_message:
                    await websocket.send_json(
                        {
                            "type": "chat_result",
                            "ok": False,
                            "answer": "",
                            "error": "Chat message is empty.",
                        }
                    )
                    continue

                active_chat = asyncio.create_task(
                    _run_chat_turn(
                        websocket=websocket,
                        session=session,
                        message=chat_message,
                        provider=provider,
                        model_id=model_id,
                    )
                )
        except WebSocketDisconnect:
            session.fail_all("Browser WebSocket disconnected.")
            if active_chat is not None:
                active_chat.cancel()
        except Exception as exc:
            session.fail_all(str(exc))
            if active_chat is not None:
                active_chat.cancel()
            raise

    return app


def run_browser_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    provider: str | None = None,
    model_id: str | None = None,
    command_timeout_seconds: float = 30.0,
) -> None:
    """Run the browser GeoAgent server with uvicorn."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exercised by CLI fallback.
        raise RuntimeError(
            "Uvicorn is required for `geoagent browser`. Install with "
            "`pip install GeoAgent[browser]`."
        ) from exc

    app = create_browser_app(
        provider=provider,
        model_id=model_id,
        command_timeout_seconds=command_timeout_seconds,
    )
    uvicorn.run(app, host=host, port=int(port))


__all__ = ["create_browser_app", "run_browser_server"]
