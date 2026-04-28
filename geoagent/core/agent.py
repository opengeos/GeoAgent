"""High-level :class:`GeoAgent` over the Strands :class:`~strands.agent.agent.Agent`."""

from __future__ import annotations

import time
import threading
from typing import Any, Optional

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


def _result_to_text(result: Any) -> str:
    """Extract response text from a Strands result object."""
    if result is None:
        return ""
    msg = getattr(result, "message", None)
    if msg and isinstance(msg, dict):
        parts: list[str] = []
        for block in msg.get("content", []):
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        extracted = "\n".join(parts).strip()
        if extracted:
            return extracted
    s = str(result).strip()
    return s


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
        hook = ConfirmationHookProvider(self._registry, self._confirm, self._cancelled)

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
        query: str,
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
            self._context.metadata.get("integration") == "nasa_opera"
            and self._qgis_safe_mode
            and is_qt_gui_thread()
        ):
            return GeoAgentResponse(
                success=False,
                error_message=(
                    "NASA OPERA chat should be launched from a worker thread inside "
                    "QGIS. Use the NASA OPERA AI Assistant panel or "
                    "geoagent.tools.nasa_opera.submit_nasa_opera_search_task(...) "
                    "for direct QGIS-console workflows."
                ),
                map=self._context.map_obj,
            )

        if self._qgis_safe_mode and is_qt_gui_thread():
            return self._chat_on_qgis_gui_thread(query)

        return self._chat_impl(query)

    def _chat_impl(self, query: str) -> GeoAgentResponse:
        """Run a single user turn on the current thread."""
        self._cancelled.clear()
        t0 = time.time()
        try:
            result = self._strands(query)
            elapsed = time.time() - t0
            exec_names = list(getattr(result.metrics, "tool_metrics", {}).keys())
            answer = _result_to_text(result)
            stop = str(getattr(result, "stop_reason", "end_turn"))
            success = stop not in ("cancelled", "guardrail_intervened")
            err = None if success else f"stop_reason={stop}"
            return GeoAgentResponse(
                answer_text=answer or None,
                success=success,
                error_message=err,
                execution_time=elapsed,
                executed_tools=exec_names,
                cancelled_tools=list(self._cancelled),
                map=self._context.map_obj,
                raw=result,
            )
        except Exception as exc:
            elapsed = time.time() - t0
            return GeoAgentResponse(
                success=False,
                error_message=str(exc),
                execution_time=elapsed,
                cancelled_tools=list(self._cancelled),
                map=self._context.map_obj,
            )

    def _chat_on_qgis_gui_thread(self, query: str) -> GeoAgentResponse:
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
        query: str,
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
