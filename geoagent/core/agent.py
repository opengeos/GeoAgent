"""High-level :class:`GeoAgent` over the Strands :class:`~strands.agent.agent.Agent`."""

from __future__ import annotations

import asyncio
import base64
import binascii
import queue
import re
import threading
import time
from typing import Any, AsyncIterator, Optional

from strands import Agent
from strands.tools.executors.sequential import SequentialToolExecutor
from geoagent.core.config import GeoAgentConfig
from geoagent.core.confirmation_hook import ConfirmationHookProvider
from geoagent.core.context import GeoAgentContext
from geoagent.core.model import resolve_model
from geoagent.core.prompts import DEFAULT_SYSTEM_PROMPT, FAST_SYSTEM_PROMPT
from geoagent.core.registry import GeoToolRegistry
from geoagent.core.result import GeoAgentResponse
from geoagent.core.safety import ConfirmCallback, auto_approve_safe_only
from geoagent.tools._qt_marshal import is_qt_gui_thread, process_qt_events

_IMAGE_MIME_BY_FORMAT = {
    "gif": "image/gif",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}
_IMAGE_FORMAT_BY_MIME = {value: key for key, value in _IMAGE_MIME_BY_FORMAT.items()}
_IMAGE_FORMAT_BY_MIME["image/jpeg"] = "jpg"
_IMAGE_PATH_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif")


def _result_content_blocks(result: Any) -> list[dict[str, Any]]:
    """Return top-level assistant content blocks from a Strands result."""
    msg = getattr(result, "message", None)
    if not isinstance(msg, dict):
        return []
    content = msg.get("content")
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]


def _image_format_from_mime(mime_type: str | None) -> str:
    """Return a compact image format name for a MIME type."""
    return _IMAGE_FORMAT_BY_MIME.get(str(mime_type or "").lower(), "png")


def _image_mime_from_format(format_name: str | None) -> str:
    """Return a MIME type for a compact image format name."""
    cleaned = str(format_name or "").lower().lstrip(".")
    if "/" in cleaned:
        return cleaned
    return _IMAGE_MIME_BY_FORMAT.get(cleaned, "image/png")


def _data_uri_parts(value: str) -> tuple[str, str] | None:
    """Return MIME type and base64 payload for an image data URI."""
    match = re.match(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.+)$", value, re.S)
    if not match:
        return None
    return match.group(1).lower(), match.group(2)


def _decode_image_bytes(value: Any) -> tuple[bytes | None, str | None]:
    """Decode bytes or base64 image payloads into raw image bytes."""
    if isinstance(value, bytes):
        return value, None
    if isinstance(value, bytearray):
        return bytes(value), None
    if not isinstance(value, str):
        return None, None
    text = value.strip()
    if not text:
        return None, None
    data_uri = _data_uri_parts(text)
    if data_uri is not None:
        mime_type, payload = data_uri
    else:
        mime_type, payload = None, re.sub(r"\s+", "", text)
    try:
        return base64.b64decode(payload, validate=True), mime_type
    except (binascii.Error, ValueError):
        return None, mime_type


def _image_artifact_from_block(block: dict[str, Any]) -> dict[str, Any] | None:
    """Extract one image artifact from a common provider content block."""
    image = block.get("image")
    if not isinstance(image, dict) and block.get("type") in {
        "image",
        "output_image",
        "input_image",
    }:
        image = block

    if isinstance(image, dict):
        source = image.get("source") if isinstance(image.get("source"), dict) else {}
        mime_type = (
            image.get("mime_type")
            or image.get("media_type")
            or source.get("mime_type")
            or source.get("media_type")
        )
        format_name = image.get("format") or source.get("format")
        raw = (
            source.get("bytes")
            or source.get("data")
            or source.get("base64")
            or image.get("bytes")
            or image.get("data")
            or image.get("base64")
        )
        image_bytes, data_mime_type = _decode_image_bytes(raw)
        mime_type = data_mime_type or mime_type or _image_mime_from_format(format_name)
        artifact = {
            "format": str(format_name or _image_format_from_mime(mime_type)),
            "mime_type": str(mime_type),
        }
        path = source.get("path") or image.get("path")
        if isinstance(path, str) and path.strip():
            artifact["path"] = path.strip()
        if image_bytes:
            artifact["bytes"] = image_bytes
        url = source.get("url") or image.get("url") or image.get("image_url")
        if isinstance(url, dict):
            url = url.get("url")
        if isinstance(url, str) and url.strip():
            artifact["url"] = url.strip()
        return (
            artifact
            if artifact.get("bytes") or artifact.get("url") or artifact.get("path")
            else None
        )

    raw_url = block.get("image_url")
    if isinstance(raw_url, dict):
        raw_url = raw_url.get("url")
    if isinstance(raw_url, str) and raw_url.strip():
        text = raw_url.strip()
        data_uri = _data_uri_parts(text)
        if data_uri is not None:
            image_bytes, mime_type = _decode_image_bytes(text)
            if image_bytes:
                return {
                    "format": _image_format_from_mime(mime_type),
                    "mime_type": mime_type or "image/png",
                    "bytes": image_bytes,
                }
        return {"format": "url", "mime_type": "", "url": text}

    if block.get("type") == "image_generation_call":
        image_bytes, mime_type = _decode_image_bytes(block.get("result"))
        if image_bytes:
            return {
                "format": _image_format_from_mime(mime_type),
                "mime_type": mime_type or "image/png",
                "bytes": image_bytes,
            }

    return None


def _images_from_mapping(value: Any) -> list[dict[str, Any]]:
    """Extract image artifacts from nested JSON-like tool results."""
    if not isinstance(value, dict):
        return []

    images: list[dict[str, Any]] = []
    nested = value.get("images")
    if isinstance(nested, list):
        for item in nested:
            if isinstance(item, dict):
                artifact = _image_artifact_from_block({"image": item})
                if artifact is not None:
                    images.append(artifact)
    if images:
        return images

    artifact = _image_artifact_from_block({"image": value})
    if artifact is not None:
        images.append(artifact)

    return images


def _dedupe_images(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return images with duplicate path/url/bytes references removed."""
    deduped: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for image in images:
        key = image.get("path") or image.get("url") or image.get("bytes")
        if key is None:
            key = tuple(sorted((k, str(v)) for k, v in image.items()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(image)
    return deduped


def _image_artifacts_from_value(value: Any) -> list[dict[str, Any]]:
    """Recursively extract image artifacts from JSON-like values."""
    if isinstance(value, dict):
        images = _images_from_mapping(value)
        for key, nested in value.items():
            if key in {"bytes", "data", "base64", "b64_json"}:
                continue
            images.extend(_image_artifacts_from_value(nested))
        return _dedupe_images(images)
    if isinstance(value, (list, tuple)):
        images: list[dict[str, Any]] = []
        for item in value:
            images.extend(_image_artifacts_from_value(item))
        return _dedupe_images(images)
    if isinstance(value, str):
        text = value.strip()
        if text.lower().endswith(_IMAGE_PATH_SUFFIXES):
            return [{"format": text.rsplit(".", 1)[-1].lower(), "path": text}]
    return []


def _tool_calls_to_images(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract image artifacts returned by tool calls."""
    images: list[dict[str, Any]] = []
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        images.extend(_image_artifacts_from_value(call.get("result")))
    return _dedupe_images(images)


def _result_to_images(result: Any) -> list[dict[str, Any]]:
    """Extract image artifacts from a Strands result object."""
    images: list[dict[str, Any]] = []
    for block in _result_content_blocks(result):
        artifact = _image_artifact_from_block(block)
        if artifact is not None:
            images.append(artifact)
    return images


def _result_to_text(result: Any) -> str:
    """Extract response text from a Strands result object."""
    if result is None:
        return ""
    blocks = _result_content_blocks(result)
    if blocks:
        parts: list[str] = []
        for block in blocks:
            if "text" in block:
                parts.append(str(block["text"]))
        extracted = "\n".join(parts).strip()
        if extracted:
            return extracted
    s = str(result).strip()
    return s


def _looks_like_json_parse_failure(exc: Exception) -> bool:
    """Return True when an exception is a malformed JSON/tool-call response."""
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "failed to parse json",
        "jsondecodeerror",
        "unexpected end of json input",
        "expecting value",
        "unterminated string",
    )
    return any(marker in text for marker in markers)


def _format_chat_exception(exc: Exception) -> str:
    """Convert low-level provider/tool-call exceptions into user guidance."""
    original = str(exc).strip() or type(exc).__name__
    if not _looks_like_json_parse_failure(exc):
        return original
    return (
        "The model returned malformed or incomplete JSON while trying to call "
        "a tool. GeoAgent could not safely continue that tool workflow.\n\n"
        f"Original error: {original}\n\n"
        "How to correct it:\n"
        "- Retry with a more specific request, including the layer name and "
        "desired output name.\n"
        "- Break a long workflow into smaller steps if it keeps failing.\n"
        "- Increase the model max tokens if the response may be truncated.\n"
        "- Use a model/provider with reliable tool-calling support; local or "
        "OpenAI-compatible endpoints may need tool/function calling enabled."
    )


class GeoAgent:
    """Public facade: Strands agent + GeoAgent context, tools, and safety hooks."""

    def __init__(
        self,
        *,
        context: Optional[GeoAgentContext] = None,
        config: Optional[GeoAgentConfig] = None,
        tools: Optional[list[Any]] = None,
        registry: Optional[GeoToolRegistry] = None,
        model: Any | None = None,
        provider: str | None = None,
        model_id: str | None = None,
        fast: bool = False,
        confirm: ConfirmCallback | None = None,
        qgis_safe_mode: bool = False,
    ) -> None:
        self._context = context or GeoAgentContext()
        cfg = config or GeoAgentConfig()
        if provider is not None:
            cfg = cfg.model_copy(update={"provider": provider})
        if model_id is not None:
            cfg = cfg.model_copy(update={"model": model_id})
        if fast and cfg.max_tokens > 2048:
            cfg = cfg.model_copy(update={"max_tokens": 2048})
        self._config = cfg
        self._fast = fast
        self._qgis_safe_mode = qgis_safe_mode
        self._registry = registry or GeoToolRegistry()
        self._tool_list = list(tools or [])
        self._cancelled: list[str] = []
        self._tool_calls: list[dict[str, Any]] = []
        self._confirm = confirm or auto_approve_safe_only
        self._model = model or resolve_model(self._config)
        self._rebuild_strands_agent()

    def _rebuild_strands_agent(self) -> None:
        """Recreate the underlying Strands agent from current settings."""
        self._cancelled = []
        prompt = FAST_SYSTEM_PROMPT if self._fast else DEFAULT_SYSTEM_PROMPT
        extra_prompt = self._context.metadata.get("system_prompt")
        if extra_prompt:
            prompt = f"{prompt}\n\n{extra_prompt}"
        hook = ConfirmationHookProvider(
            self._registry,
            self._confirm,
            self._cancelled,
            self._tool_calls,
        )

        self._strands = Agent(
            model=self._model,
            tools=self._tool_list,
            system_prompt=prompt,
            hooks=[hook],
            callback_handler=None,
            tool_executor=SequentialToolExecutor() if self._qgis_safe_mode else None,
        )

    @property
    def context(self) -> GeoAgentContext:
        """GeoAgent runtime context."""
        return self._context

    @property
    def strands_agent(self) -> Agent:
        """The underlying Strands :class:`~strands.agent.agent.Agent`."""
        return self._strands

    @property
    def tool(self) -> Any:
        """Direct Strands tool caller (``agent.tool.some_tool(...)``)."""
        return self._strands.tool

    @property
    def tool_names(self) -> list[str]:
        """Expose Strands tool names on GeoAgent for parity."""
        return list(self._strands.tool_names)

    @property
    def tool_registry(self) -> GeoToolRegistry:
        """GeoAgent metadata registry for tool inspection."""
        return self._registry

    @property
    def config(self) -> GeoAgentConfig:
        """GeoAgent model and runtime configuration."""
        return self._config

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to the underlying Strands agent."""
        return getattr(self._strands, name)

    def with_map(self, m: Any) -> "GeoAgent":
        """Return a new :class:`GeoAgent` bound to a map (rebuilds tools)."""
        from geoagent.core.factory import for_leafmap

        return for_leafmap(
            m,
            config=self._config,
            model=self._model,
            fast=self._fast,
            confirm=self._confirm,
        )

    def chat(
        self,
        query: Any,
        target_map: Any = None,
    ) -> GeoAgentResponse:
        """Run a single user turn and return a :class:`GeoAgentResponse`."""
        if target_map is not None and target_map is not self._context.map_obj:
            from geoagent.core.factory import for_leafmap

            other = for_leafmap(
                target_map,
                config=self._config,
                model=self._model,
                fast=self._fast,
                confirm=self._confirm,
            )
            return other.chat(query)

        if (
            self._context.metadata.get("integration")
            in {"nasa_earthdata", "nasa_opera"}
            and self._qgis_safe_mode
            and is_qt_gui_thread()
        ):
            integration = self._context.metadata.get("integration")
            label = (
                "NASA Earthdata" if integration == "nasa_earthdata" else "NASA OPERA"
            )
            helper = (
                "the NASA Earthdata AI Assistant panel"
                if integration == "nasa_earthdata"
                else (
                    "the NASA OPERA AI Assistant panel or "
                    "geoagent.tools.nasa_opera.submit_nasa_opera_search_task(...) "
                    "for direct QGIS-console workflows"
                )
            )
            return GeoAgentResponse(
                success=False,
                error_message=(
                    f"{label} chat should be launched from a worker thread inside "
                    f"QGIS. Use {helper}."
                ),
                map=self._context.map_obj,
            )

        if self._qgis_safe_mode and is_qt_gui_thread():
            return self._chat_on_qgis_gui_thread(query)

        return self._chat_impl(query)

    def _chat_impl(self, query: Any) -> GeoAgentResponse:
        """Run a single user turn on the current thread."""
        self._cancelled.clear()
        self._tool_calls.clear()
        t0 = time.time()
        try:
            result = self._strands(query)
            elapsed = time.time() - t0
            exec_names = list(getattr(result.metrics, "tool_metrics", {}).keys())
            answer = _result_to_text(result)
            content_blocks = _result_content_blocks(result)
            images = _result_to_images(result) + _tool_calls_to_images(self._tool_calls)
            stop = str(getattr(result, "stop_reason", "end_turn"))
            success = stop not in ("cancelled", "guardrail_intervened")
            err = None if success else f"stop_reason={stop}"
            return GeoAgentResponse(
                answer_text=answer or None,
                success=success,
                error_message=err,
                execution_time=elapsed,
                content_blocks=content_blocks,
                images=images,
                executed_tools=exec_names,
                tool_calls=list(self._tool_calls),
                cancelled_tools=list(self._cancelled),
                map=self._context.map_obj,
                raw=result,
            )
        except Exception as exc:
            elapsed = time.time() - t0
            return GeoAgentResponse(
                success=False,
                error_message=_format_chat_exception(exc),
                execution_time=elapsed,
                tool_calls=list(self._tool_calls),
                cancelled_tools=list(self._cancelled),
                map=self._context.map_obj,
            )

    async def stream_chat(
        self,
        query: Any,
        target_map: Any = None,
    ) -> AsyncIterator[Any]:
        """Stream a single user turn as raw Strands events.

        Text deltas are emitted in events containing ``"data"``. The final
        Strands result is emitted in the event containing ``"result"``.
        """
        if target_map is not None and target_map is not self._context.map_obj:
            from geoagent.core.factory import for_leafmap

            other = for_leafmap(
                target_map,
                config=self._config,
                model=self._model,
                fast=self._fast,
                confirm=self._confirm,
            )
            async for event in other.stream_chat(query):
                yield event
            return

        if self._qgis_safe_mode and is_qt_gui_thread():
            integration = self._context.metadata.get("integration")
            if integration in {"nasa_earthdata", "nasa_opera"}:
                label = (
                    "NASA Earthdata"
                    if integration == "nasa_earthdata"
                    else "NASA OPERA"
                )
                helper = (
                    "the NASA Earthdata AI Assistant panel"
                    if integration == "nasa_earthdata"
                    else (
                        "the NASA OPERA AI Assistant panel or "
                        "geoagent.tools.nasa_opera.submit_nasa_opera_search_task(...) "
                        "for direct QGIS-console workflows"
                    )
                )
                raise RuntimeError(
                    f"{label} streaming chat should be launched from a worker thread "
                    f"inside QGIS. Use {helper}."
                )
            async for event in self._stream_chat_on_qgis_gui_thread(query):
                yield event
            return

        async for event in self._stream_chat_impl(query):
            yield event

    async def _stream_chat_impl(self, query: Any) -> AsyncIterator[Any]:
        """Run a streaming user turn on the current thread."""
        self._cancelled.clear()
        self._tool_calls.clear()
        try:
            async for event in self._strands.stream_async(query):
                yield event
        except Exception as exc:
            if _looks_like_json_parse_failure(exc):
                raise RuntimeError(_format_chat_exception(exc)) from exc
            raise

    async def _stream_chat_on_qgis_gui_thread(self, query: Any) -> AsyncIterator[Any]:
        """Stream QGIS chat from a worker thread while pumping Qt events."""
        events: queue.Queue[tuple[str, Any]] = queue.Queue()

        async def _run_stream() -> None:
            """Execute async streaming work off the GUI thread."""
            async for event in self._stream_chat_impl(query):
                events.put(("event", event))

        def _worker() -> None:
            """Run the async stream and forward events to the GUI thread."""
            try:
                asyncio.run(_run_stream())
            except BaseException as exc:  # pragma: no cover - defensive path
                events.put(("error", exc))
            finally:
                events.put(("done", None))

        thread = threading.Thread(
            target=_worker,
            daemon=True,
            name="GeoAgent-QGIS-stream-chat",
        )
        thread.start()

        while True:
            try:
                kind, payload = events.get_nowait()
            except queue.Empty:
                process_qt_events()
                await asyncio.sleep(0.05)
                continue

            if kind == "event":
                yield payload
            elif kind == "error":
                thread.join(timeout=0)
                raise payload
            elif kind == "done":
                thread.join(timeout=0)
                return

    def _chat_on_qgis_gui_thread(self, query: Any) -> GeoAgentResponse:
        """Run sync QGIS chat without starving the Qt event loop.

        QGIS users often call ``resp = agent.chat(...)`` from the Python
        console, which executes on the GUI thread. The model call needs to run
        away from that thread, but QGIS tools still marshal back to it via
        ``BlockingQueuedConnection``. Pumping Qt events while waiting lets those
        marshalled tool calls run and keeps the application responsive.
        """
        done = threading.Event()
        box: dict[str, Any] = {}

        def _worker() -> None:
            """Execute chat work off the GUI thread."""
            try:
                box["response"] = self._chat_impl(query)
            except BaseException as exc:  # pragma: no cover - defensive path
                box["error"] = exc
            finally:
                done.set()

        thread = threading.Thread(
            target=_worker,
            daemon=True,
            name="GeoAgent-QGIS-chat",
        )
        thread.start()

        while not done.is_set():
            process_qt_events()
            done.wait(0.05)

        thread.join(timeout=0)
        if "error" in box:
            raise box["error"]
        return box["response"]

    def chat_in_background(
        self,
        query: Any,
        *,
        target_map: Any = None,
        on_result: Any | None = None,
        on_error: Any | None = None,
    ) -> threading.Thread:
        """Run ``chat`` on a worker thread and return immediately.

        This is primarily for QGIS console usage where a synchronous ``chat()``
        call blocks the GUI event loop during network/model latency.
        """

        def _worker() -> None:
            """Execute chat work and dispatch callbacks."""
            try:
                resp = self.chat(query, target_map=target_map)
                if on_result is not None:
                    on_result(resp)
            except Exception as exc:  # pragma: no cover - defensive path
                if on_error is not None:
                    on_error(exc)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        return thread


__all__ = ["GeoAgent"]
