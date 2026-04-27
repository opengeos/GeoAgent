"""High-level :class:`GeoAgent` facade over the deepagents runtime.

The legacy v0.x ``GeoAgent`` orchestrated a 4-agent LangGraph pipeline.
Phase 2 replaces it with a thin wrapper around
:func:`geoagent.core.factory.create_geo_agent`. The public API is
preserved (``__init__`` accepts ``llm`` / ``provider`` / ``model`` /
``catalogs``; ``chat()`` returns a :class:`GeoAgentResponse` with the
same field names) but the internals are entirely rebuilt on
deepagents subagents.

The facade also bridges deepagents' interrupt / resume mechanism to a
user-supplied :class:`~geoagent.core.safety.ConfirmCallback`. When a
confirmation-required tool fires, deepagents pauses the graph and
returns a result with an ``__interrupt__`` key; :meth:`GeoAgent.chat`
loops over that, calls the user's callback per pending tool, and
resumes with ``Command(resume={"decisions": [...]})`` until the graph
completes.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable, Optional

from langgraph.types import Command

from .context import GeoAgentContext
from .factory import create_geo_agent
from .result import GeoAgentResponse
from .safety import ConfirmCallback, ConfirmRequest, auto_approve_safe_only

logger = logging.getLogger(__name__)


class GeoAgent:
    """High-level facade over a deepagents-compiled GeoAgent runtime.

    ``GeoAgent`` is the public, stable entry point for building a
    geospatial agent. It owns a single deepagents graph plus a unique
    LangGraph thread, wires confirmation-required tool calls through a
    user-supplied callback, and reconstructs the v0.x-style
    :class:`GeoAgentResponse` shape from the final graph state.

    Args:
        llm: A pre-built LangChain ``BaseChatModel``. Mutually exclusive
            with ``provider`` / ``model``.
        provider: A provider name (``"openai"``, ``"anthropic"``,
            ``"google"``, ``"ollama"``).
        model: A model name string. May be a deepagents-style
            ``"provider:model"`` shorthand when ``provider`` is omitted.
        catalogs: Optional list of STAC catalog URLs to expose to the
            data subagent. Currently informational; the catalog registry
            in :mod:`geoagent.catalogs.registry` continues to drive
            STAC discovery.
        context: Pre-built :class:`GeoAgentContext`. When omitted, an
            empty context is created.
        tools: Additional :class:`langchain_core.tools.BaseTool` objects
            to expose alongside the default subagent tools.
        confirm: Callback invoked before any confirmation-required tool
            executes. Defaults to
            :func:`geoagent.core.safety.auto_approve_safe_only` which
            rejects everything (so destructive tools never run silently).
        checkpointer: A LangGraph ``Checkpointer``. ``None`` (the
            default) creates an in-memory ``MemorySaver``. Pass
            ``False`` to opt out of checkpointing (HITL will be
            disabled).
        thread_id: Override the auto-generated LangGraph thread id.

    Example:
        >>> from geoagent import GeoAgent
        >>> agent = GeoAgent(provider="openai")
        >>> resp = agent.chat("Search Sentinel-2 for Knoxville in July 2024")
        >>> resp.success
        True
    """

    def __init__(
        self,
        llm: Any = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        catalogs: Optional[list] = None,
        *,
        context: Optional[GeoAgentContext] = None,
        tools: Optional[list] = None,
        confirm: Optional[ConfirmCallback] = None,
        checkpointer: Any = None,
        thread_id: Optional[str] = None,
        _tools: Optional[list] = None,
    ) -> None:
        self._context = context or GeoAgentContext()
        self._confirm: ConfirmCallback = confirm or auto_approve_safe_only
        self._catalogs = catalogs or []
        self._thread_id = thread_id or f"geoagent-{uuid.uuid4().hex[:12]}"
        self._tools_by_name: dict[str, Any] = {}
        for t in list(_tools or []) + list(tools or []):
            name = getattr(t, "name", None)
            if name:
                self._tools_by_name[name] = t
        self._graph = create_geo_agent(
            llm=llm,
            provider=provider,
            model=model,
            tools=_tools,
            extra_tools=tools or [],
            context=self._context,
            checkpointer=checkpointer,
        )

    # ------------------------------------------------------------------
    # Public chat interface
    # ------------------------------------------------------------------
    def chat(
        self,
        query: str,
        target_map: Any = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> GeoAgentResponse:
        """Send a natural-language query to the agent.

        Args:
            query: The user's request.
            target_map: Optional live map widget to fold into the
                runtime context for this call. Mutates the agent's
                context so subsequent calls also see the map.
            status_callback: Optional callable invoked with short status
                strings as the agent progresses. Currently used only for
                interrupt / resume notifications; tool-level streaming
                will be added in Phase 3.

        Returns:
            A :class:`GeoAgentResponse` populated from the deepagents
            final state. Tool names that ran are listed in
            :attr:`GeoAgentResponse.executed_tools`; tools the user
            cancelled via the confirm callback are in
            :attr:`GeoAgentResponse.cancelled_tools`.
        """
        if target_map is not None:
            self._context = self._context.with_overrides(map_obj=target_map)

        config = {"configurable": {"thread_id": self._thread_id}}
        executed: list[str] = []
        cancelled: list[str] = []
        start = time.time()

        try:
            result: Any = self._graph.invoke(
                {"messages": [{"role": "user", "content": query}]},
                config=config,
            )
            # Resume loop: while deepagents is paused on an interrupt,
            # walk the user through the pending tool calls and resume
            # with approve / reject decisions.
            while True:
                interrupts = self._extract_interrupts(result)
                if not interrupts:
                    break
                decisions = []
                for req in interrupts:
                    tool_name = req.get("name") or req.get("action_name") or ""
                    args = req.get("args") or {}
                    confirm_req = ConfirmRequest(
                        tool_name=tool_name,
                        args=dict(args),
                        description=self._tool_description(tool_name, req),
                        category=None,
                        metadata={},
                    )
                    if status_callback is not None:
                        try:
                            status_callback(f"awaiting confirmation: {tool_name}")
                        except Exception:  # pragma: no cover - user code
                            logger.debug("status_callback raised", exc_info=True)
                    approved = bool(self._confirm(confirm_req))
                    if approved:
                        decisions.append({"type": "approve"})
                        executed.append(tool_name)
                    else:
                        decisions.append(
                            {"type": "reject", "message": "User cancelled."}
                        )
                        cancelled.append(tool_name)
                result = self._graph.invoke(
                    Command(resume={"decisions": decisions}),
                    config=config,
                )
            elapsed = time.time() - start
            return _adapt_result(result, executed, cancelled, elapsed)
        except Exception as exc:
            elapsed = time.time() - start
            logger.exception("GeoAgent.chat failed")
            return GeoAgentResponse(
                success=False,
                error_message=str(exc),
                execution_time=elapsed,
                executed_tools=executed,
                cancelled_tools=cancelled,
            )

    # ------------------------------------------------------------------
    # Convenience entry points (back-compat with v0.x)
    # ------------------------------------------------------------------
    def search(self, query: str) -> GeoAgentResponse:
        """Search-only run: instructs the coordinator not to analyse or visualise."""
        return self.chat(
            f"Only search and report data. Do not analyse or visualise. Request: {query}"
        )

    def analyze(self, query: str) -> GeoAgentResponse:
        """Analysis run: instructs the coordinator to skip map rendering."""
        return self.chat(
            f"Search and analyse the data. Do not render any maps. Request: {query}"
        )

    def visualize(self, query: str, target_map: Any = None) -> GeoAgentResponse:
        """Alias for :meth:`chat` — visualisation is the default behaviour."""
        return self.chat(query, target_map=target_map)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_interrupts(result: Any) -> list[dict[str, Any]]:
        """Pull pending action_requests from a deepagents invoke result.

        Args:
            result: The dict returned by ``graph.invoke(...)`` (or by a
                resume call). The ``__interrupt__`` key is present when
                the graph paused on an interrupt; absent otherwise.

        Returns:
            A list of ``{"name": ..., "args": ...}`` dicts, one per
            pending tool call. Empty when the graph completed.
        """
        if not isinstance(result, dict):
            return []
        interrupts = result.get("__interrupt__")
        if not interrupts:
            return []
        first = interrupts[0]
        value = getattr(first, "value", first)
        if not isinstance(value, dict):
            return []
        return list(value.get("action_requests") or [])

    def _tool_description(self, tool_name: str, action_request: dict[str, Any]) -> str:
        """Return a short human-readable description for a confirm prompt.

        Prefers the cached tool's own ``description`` (set by
        ``@geo_tool``); falls back to the deepagents-supplied
        description in the action request payload.
        """
        tool = self._tools_by_name.get(tool_name)
        if tool is not None:
            desc = getattr(tool, "description", None)
            if desc:
                return desc
        return action_request.get("description", "")


def _adapt_result(
    result: Any,
    executed: list[str],
    cancelled: list[str],
    elapsed: float,
) -> GeoAgentResponse:
    """Reconstruct a :class:`GeoAgentResponse` from deepagents' final state.

    Args:
        result: The dict returned by the final ``graph.invoke(...)``.
        executed: Names of confirmation-required tools the user
            approved.
        cancelled: Names of confirmation-required tools the user
            rejected.
        elapsed: Wall-clock seconds spent in the agent run.

    Returns:
        A populated :class:`GeoAgentResponse`. Best-effort: pulls
        ``plan`` / ``data`` / ``analysis`` / ``map`` / ``code`` from
        the state if subagents wrote them; otherwise leaves them
        ``None``. The ``answer_text`` field is the content of the last
        ``AIMessage`` in the conversation.
    """
    if not isinstance(result, dict):
        return GeoAgentResponse(
            success=False,
            error_message=f"Unexpected result type: {type(result).__name__}",
            execution_time=elapsed,
            executed_tools=executed,
            cancelled_tools=cancelled,
        )

    messages = result.get("messages") or []
    answer_text: Optional[str] = None
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        if isinstance(content, str) and content.strip():
            answer_text = content
            break

    return GeoAgentResponse(
        plan=result.get("plan"),
        data=result.get("data"),
        analysis=result.get("analysis"),
        map=result.get("map"),
        code=result.get("code", "") or "",
        answer_text=answer_text,
        success=True,
        error_message=None,
        execution_time=elapsed,
        executed_tools=executed,
        cancelled_tools=cancelled,
        messages=messages,
    )


__all__ = ["GeoAgent"]
