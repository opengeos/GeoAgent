"""FastAPI WebSocket server for embedded browser GeoAgent chat."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any
from uuid import uuid4

from geoagent.browser.session import BrowserMapSession
from geoagent.core.factory import for_browser_maplibre
from geoagent.core.safety import auto_approve_safe_only
from geoagent.ui.app import build_prompt_with_context

_BROWSER_CONFIRM_TOOL_NAMES = frozenset(
    {
        "remove_layer",
        "clear_layers",
        "run_maplibre_script",
    }
)


def _stream_result_to_text(result: Any) -> str:
    """Extract assistant text from a final Strands streaming result."""
    message = getattr(result, "message", None)
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and "text" in block:
            parts.append(str(block["text"]))
    return "\n".join(parts).strip()


def _stream_result_tool_names(result: Any) -> list[str]:
    """Extract executed tool names from a final Strands streaming result."""
    metrics = getattr(result, "metrics", None)
    tool_metrics = getattr(metrics, "tool_metrics", {}) if metrics is not None else {}
    return list(tool_metrics.keys()) if isinstance(tool_metrics, dict) else []


def _chat_result_payload(
    *,
    agent: Any,
    result: Any,
    answer_text: str,
    elapsed: float,
) -> dict[str, Any]:
    """Build the final browser chat payload for streamed and non-streamed turns."""
    final_text = _stream_result_to_text(result) if result is not None else ""
    stop = str(getattr(result, "stop_reason", "end_turn")) if result is not None else ""
    success = stop not in ("cancelled", "guardrail_intervened")
    payload: dict[str, Any] = {
        "type": "chat_result",
        "ok": success,
        "answer": final_text or answer_text,
        "executed_tools": _stream_result_tool_names(result),
        "tool_calls": list(getattr(agent, "_tool_calls", []) or []),
        "cancelled_tools": list(getattr(agent, "_cancelled", []) or []),
        "execution_time": elapsed,
        "streamed": True,
    }
    if not success:
        payload["error"] = f"stop_reason={stop}"
    return payload


def _browser_confirm_callback(
    allow_browser_code: bool,
    auto_approve_browser_tools: bool,
):
    """Return the browser backend confirmation policy."""

    def _confirm(request: Any) -> bool:
        if auto_approve_browser_tools and (
            getattr(request, "category", None) == "browser_map"
            or request.tool_name in _BROWSER_CONFIRM_TOOL_NAMES
        ):
            return True
        if allow_browser_code and request.tool_name == "run_maplibre_script":
            return True
        return auto_approve_safe_only(request)

    return _confirm


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
    allow_browser_code: bool,
    auto_approve_browser_tools: bool,
    history: list[dict[str, Any]] | None = None,
) -> None:
    """Run one GeoAgent chat turn and stream deltas to the browser."""
    try:
        await _send_json(websocket, {"type": "chat_status", "status": "running"})
        agent = for_browser_maplibre(
            session=session,
            provider=provider,
            model_id=model_id,
            confirm=_browser_confirm_callback(
                allow_browser_code,
                auto_approve_browser_tools,
            ),
            allow_browser_code=allow_browser_code,
        )
        prompt = build_prompt_with_context(history, message) if history else message
        loop = asyncio.get_running_loop()
        events: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        async def _consume_stream() -> None:
            """Drain GeoAgent streaming events in the worker thread."""
            try:
                async for event in agent.stream_chat(prompt):
                    loop.call_soon_threadsafe(events.put_nowait, ("event", event))
            except BaseException as exc:  # pragma: no cover - defensive worker path
                loop.call_soon_threadsafe(events.put_nowait, ("error", exc))
            finally:
                loop.call_soon_threadsafe(events.put_nowait, ("done", None))

        def _worker() -> None:
            asyncio.run(_consume_stream())

        started = time.time()
        thread = threading.Thread(
            target=_worker,
            daemon=True,
            name="GeoAgent-browser-stream-chat",
        )
        thread.start()

        answer_parts: list[str] = []
        final_result: Any = None
        while True:
            kind, event = await events.get()
            if kind == "done":
                break
            if kind == "error":
                raise event
            if not isinstance(event, dict):
                continue
            if "data" in event:
                text = str(event.get("data") or "")
                if text:
                    answer_parts.append(text)
                    await _send_json(
                        websocket,
                        {
                            "type": "chat_delta",
                            "text": text,
                        },
                    )
                continue
            if "current_tool_use" in event:
                tool_use = event.get("current_tool_use")
                tool_name = ""
                if isinstance(tool_use, dict):
                    tool_name = str(tool_use.get("name") or "")
                await _send_json(
                    websocket,
                    {
                        "type": "chat_tool",
                        "name": tool_name,
                    },
                )
                continue
            if "result" in event:
                final_result = event.get("result")

        thread.join(timeout=0)
        await _send_json(
            websocket,
            _chat_result_payload(
                agent=agent,
                result=final_result,
                answer_text="".join(answer_parts),
                elapsed=time.time() - started,
            ),
        )
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
    allow_browser_code: bool = False,
    auto_approve_browser_tools: bool = False,
) -> Any:
    """Create the FastAPI app for browser-embedded GeoAgent chat."""
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    except ImportError as exc:  # pragma: no cover - exercised by CLI fallback.
        raise RuntimeError(
            "FastAPI is required for `geoagent browser`. Install with "
            "`pip install GeoAgent[browser]`."
        ) from exc
    # Endpoint annotations are postponed by ``from __future__ import annotations``.
    # FastAPI resolves them against module globals, not this function's locals.
    globals()["WebSocket"] = WebSocket
    globals()["WebSocketDisconnect"] = WebSocketDisconnect

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

                raw_history = message.get("history")
                history: list[dict[str, Any]] | None = None
                if isinstance(raw_history, list):
                    history = [item for item in raw_history if isinstance(item, dict)]

                active_chat = asyncio.create_task(
                    _run_chat_turn(
                        websocket=websocket,
                        session=session,
                        message=chat_message,
                        provider=provider,
                        model_id=model_id,
                        allow_browser_code=allow_browser_code,
                        auto_approve_browser_tools=auto_approve_browser_tools,
                        history=history,
                    )
                )
        except WebSocketDisconnect:
            session.fail_all("Browser WebSocket disconnected.")
        except Exception as exc:
            session.fail_all(str(exc))
            raise
        finally:
            # Streaming chat runs in a worker thread so browser map tools can
            # synchronously wait on this WebSocket loop. The worker thread is not
            # cancellable, so drop the reference and let the task finish quietly.
            active_chat = None

    return app


def run_browser_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    provider: str | None = None,
    model_id: str | None = None,
    command_timeout_seconds: float = 30.0,
    allow_browser_code: bool = False,
    auto_approve_browser_tools: bool = False,
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
        allow_browser_code=allow_browser_code,
        auto_approve_browser_tools=auto_approve_browser_tools,
    )
    uvicorn.run(app, host=host, port=int(port))


__all__ = ["create_browser_app", "run_browser_server"]
