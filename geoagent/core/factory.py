"""Agent factory functions.

The :func:`create_geo_agent` function is the canonical entry point for
building a deepagents-based GeoAgent runtime. The :func:`for_leafmap`,
:func:`for_anymap`, and :func:`for_qgis` helpers are convenience wrappers
that bind the factory to a specific runtime resource (a map widget or a
QGIS interface) and select an appropriate default tool set.

deepagents is imported lazily inside the function bodies so that
``import geoagent`` keeps working on a system where deepagents is not yet
installed; callers get a clear ``ImportError`` only if they actually
construct an agent.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from langchain_core.tools import BaseTool

from .context import GeoAgentContext
from .prompts import COORDINATOR_PROMPT
from .safety import ConfirmCallback, build_interrupt_on


def _require_deepagents() -> Any:
    """Import :mod:`deepagents` lazily and raise a friendly error if absent."""
    try:
        from deepagents import create_deep_agent

        return create_deep_agent
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise ImportError(
            "deepagents is not installed. Install it with "
            "`pip install GeoAgent` (deepagents is a base dependency in v1+) "
            "or `pip install deepagents>=0.5.3`."
        ) from exc


def _as_list(value: Optional[Iterable[BaseTool]]) -> list[BaseTool]:
    return list(value) if value is not None else []


# Sentinel used by the convenience factories to mean "no checkpointer
# unless the caller asks for one". ``create_geo_agent`` interprets
# ``checkpointer=False`` as opt-out. We don't want to overwrite a value
# the caller explicitly passes, so we only inject the default when
# ``checkpointer`` is absent from kwargs entirely.
_NO_CHECKPOINTER_DEFAULT = False


def create_geo_agent(
    *,
    tools: Optional[Iterable[BaseTool]] = None,
    extra_tools: Optional[Iterable[BaseTool]] = None,
    context: Optional[GeoAgentContext] = None,
    system_prompt: Optional[str] = None,
    model: Any = None,
    provider: Optional[str] = None,
    llm: Any = None,
    subagents: Optional[list[Any]] = None,
    checkpointer: Any = None,
    confirm: Optional[ConfirmCallback] = None,
    debug: bool = False,
) -> Any:
    """Build a deepagents-compiled GeoAgent runnable.

    Args:
        tools: Explicit base tool list. If ``None``, no base tools are used;
            callers typically obtain a list from the per-package factories
            (:func:`for_leafmap` etc.) or from
            :func:`geoagent.core.registry.get_tools`.
        extra_tools: Additional tools appended to ``tools`` after both inputs
            are normalised to lists.
        context: Runtime context. Defaults to an empty
            :class:`GeoAgentContext`. The dataclass type is passed to
            deepagents as ``context_schema``.
        system_prompt: Coordinator system prompt. Defaults to
            :data:`COORDINATOR_PROMPT`.
        model: A pre-built LangChain ``BaseChatModel``, or a deepagents-style
            string like ``"openai:gpt-5.4"``. Mutually exclusive with
            ``provider`` / ``llm``.
        provider: A provider name (``"openai"``, ``"anthropic"``, ``"google"``,
            ``"ollama"``). Resolved via :func:`geoagent.core.llm.resolve_model`.
        llm: A pre-built LangChain ``BaseChatModel`` (alias of ``model`` for
            users coming from the v0.x API).
        subagents: List of deepagents subagent specs. When ``None`` the
            factory calls :func:`geoagent.agents.coordinator.default_subagents`
            with the active context to assemble the standard set
            (planner, data, analysis, context — plus mapping / qgis /
            geoai / earthdata when their runtime state or optional
            packages are available). Pass an explicit list (including
            ``[]``) to override.
        checkpointer: A LangGraph ``Checkpointer`` for thread
            persistence. Required for the human-in-the-loop interrupt /
            resume flow used by :class:`GeoAgent.chat`. When ``None``
            (the default), an in-memory
            :class:`langgraph.checkpoint.memory.MemorySaver` is created
            so HITL works out of the box. Pass ``False`` to opt out.
        confirm: Reserved for use by the :class:`GeoAgent` facade.
            Currently unused at the factory layer; included so callers
            can construct full configurations now without re-plumbing
            later.
        debug: Forwarded to deepagents.

    Returns:
        A compiled deepagents agent (a LangGraph ``CompiledStateGraph``).
        Invoke via ``.invoke({"messages": [...]})`` or ``.stream(...)``.
    """
    create_deep_agent = _require_deepagents()
    del confirm  # placeholder for the GeoAgent facade

    from .llm import resolve_model

    resolved_model = resolve_model(llm=llm, provider=provider, model=model)

    tool_list = _as_list(tools) + _as_list(extra_tools)

    context = context or GeoAgentContext()

    if subagents is None:
        from geoagent.agents.coordinator import default_subagents

        subagents = default_subagents(context)

    # Build interrupt_on from the union of base tools and every subagent's
    # tools. Confirmation-required tools that live only inside a subagent
    # (e.g. mapping's remove_layer / save_map, qgis's run_processing_algorithm)
    # would otherwise execute without firing the HITL interrupt.
    interrupt_tool_pool: list[BaseTool] = list(tool_list)
    for sub in subagents:
        for t in sub.get("tools") or []:
            interrupt_tool_pool.append(t)
    interrupt_on = build_interrupt_on(interrupt_tool_pool) or None

    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
    elif checkpointer is False:
        # Explicit opt-out: pass None through to deepagents (HITL won't work).
        checkpointer = None

    return create_deep_agent(
        model=resolved_model,
        tools=tool_list,
        system_prompt=system_prompt or COORDINATOR_PROMPT,
        subagents=subagents,
        interrupt_on=interrupt_on,
        context_schema=type(context),
        checkpointer=checkpointer,
        debug=debug,
    )


def for_leafmap(
    m: Any,
    *,
    include_stac: bool = True,
    include_data: bool = False,
    extra_tools: Optional[Iterable[BaseTool]] = None,
    **kwargs: Any,
) -> Any:
    """Build an agent bound to a live ``leafmap.Map`` instance.

    Args:
        m: A ``leafmap.Map`` (or compatible mock) instance.
        include_stac: If ``True``, append ``stac_tools()``.
        include_data: If ``True``, append the raster, vector, and DuckDB
            data tool sets.
        extra_tools: Additional tools appended to the assembled list.
        **kwargs: Forwarded to :func:`create_geo_agent`.

    Returns:
        A compiled deepagents agent. By default no LangGraph checkpointer
        is attached, so the returned graph can be driven via
        ``graph.invoke({"messages": [...]})`` without supplying a
        ``thread_id``. Pass ``checkpointer=<MemorySaver()>`` (or any
        other ``Checkpointer``) explicitly to opt into HITL / thread
        persistence; in that case every ``.invoke()`` call must include
        ``config={"configurable": {"thread_id": ...}}``.
    """
    from geoagent.tools.leafmap import leafmap_tools

    tools: list[BaseTool] = list(leafmap_tools(m))
    if include_stac:
        from geoagent.tools.stac import stac_tools

        tools += stac_tools()
    if include_data:
        from geoagent.tools.data.raster import raster_tools
        from geoagent.tools.data.vector import vector_tools
        from geoagent.tools.data.duckdb import duckdb_tools

        tools += raster_tools() + vector_tools() + duckdb_tools()
    if extra_tools is not None:
        tools += list(extra_tools)
    context = kwargs.pop("context", None) or GeoAgentContext(map_obj=m)
    kwargs.setdefault("checkpointer", _NO_CHECKPOINTER_DEFAULT)
    return create_geo_agent(tools=tools, context=context, **kwargs)


def for_anymap(
    m: Any,
    *,
    include_stac: bool = True,
    include_data: bool = False,
    extra_tools: Optional[Iterable[BaseTool]] = None,
    **kwargs: Any,
) -> Any:
    """Build an agent bound to a live ``anymap.Map`` instance.

    Args:
        m: An ``anymap.Map`` (or compatible mock) instance.
        include_stac: If ``True``, append ``stac_tools()``.
        include_data: If ``True``, append the raster, vector, and DuckDB
            data tool sets.
        extra_tools: Additional tools appended to the assembled list.
        **kwargs: Forwarded to :func:`create_geo_agent`.

    Returns:
        A compiled deepagents agent. By default no LangGraph checkpointer
        is attached, so the returned graph can be driven via
        ``graph.invoke({"messages": [...]})`` without supplying a
        ``thread_id``. Pass ``checkpointer=<MemorySaver()>`` explicitly
        to opt into HITL / thread persistence.
    """
    from geoagent.tools.anymap import anymap_tools

    tools: list[BaseTool] = list(anymap_tools(m))
    if include_stac:
        from geoagent.tools.stac import stac_tools

        tools += stac_tools()
    if include_data:
        from geoagent.tools.data.raster import raster_tools
        from geoagent.tools.data.vector import vector_tools
        from geoagent.tools.data.duckdb import duckdb_tools

        tools += raster_tools() + vector_tools() + duckdb_tools()
    if extra_tools is not None:
        tools += list(extra_tools)
    context = kwargs.pop("context", None) or GeoAgentContext(map_obj=m)
    kwargs.setdefault("checkpointer", _NO_CHECKPOINTER_DEFAULT)
    return create_geo_agent(tools=tools, context=context, **kwargs)


class _SyncToolGraph:
    """Wrap a compiled deepagents graph to keep tool execution synchronous.

    LangGraph's ``ToolNode`` always dispatches tool calls through a
    ``ContextThreadPoolExecutor``, which moves tool bodies onto worker
    threads. Inside QGIS this corrupts ``iface`` and the layer tree
    (Qt requires main-thread affinity for the canvas / project / layer
    tree). The :func:`geoagent.tools._inline_executor.inline_tool_execution`
    context manager monkey-patches LangGraph's executor factory for the
    duration of a call so each tool runs inline on the calling thread.

    Because the user's ``agent.invoke()`` is itself called on the QGIS
    main thread, every tool body then runs on the main thread. No Qt
    thread marshaling is needed and no event-loop pumping is required.

    Other public surface (``ainvoke``, ``astream``, ``stream_events``,
    ``get_state``, etc.) is forwarded unchanged via ``__getattr__`` so
    the wrapper is a transparent stand-in for the inner graph.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        from geoagent.tools._inline_executor import inline_tool_execution

        with inline_tool_execution():
            return self._inner.invoke(*args, **kwargs)

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        from geoagent.tools._inline_executor import inline_tool_execution

        with inline_tool_execution():
            yield from self._inner.stream(*args, **kwargs)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        from geoagent.tools._inline_executor import inline_tool_execution

        with inline_tool_execution():
            return await self._inner.ainvoke(*args, **kwargs)

    async def astream(self, *args: Any, **kwargs: Any) -> Any:
        from geoagent.tools._inline_executor import inline_tool_execution

        with inline_tool_execution():
            async for chunk in self._inner.astream(*args, **kwargs):
                yield chunk

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def for_qgis(
    iface: Any,
    project: Any = None,
    *,
    include_stac: bool = True,
    include_data: bool = False,
    extra_tools: Optional[Iterable[BaseTool]] = None,
    **kwargs: Any,
) -> Any:
    """Build an agent bound to a live QGIS ``QgisInterface``.

    Args:
        iface: The QGIS ``QgisInterface``. Pass ``None`` to build an agent
            with no QGIS-specific tools (useful for testing the path).
        project: Optional ``QgsProject`` instance.
        include_stac: If ``True``, append ``stac_tools()``.
        include_data: If ``True``, append the raster, vector, and DuckDB
            data tool sets.
        extra_tools: Additional tools.
        **kwargs: Forwarded to :func:`create_geo_agent`.

    Returns:
        A compiled deepagents agent wrapped so ``.invoke()`` /
        ``.stream()`` / ``.ainvoke()`` / ``.astream()`` runs every tool
        body inline on the calling thread (the QGIS main thread when
        called from QGIS's Python console). This avoids the Qt
        thread-affinity violations that would otherwise corrupt iface
        and crash QGIS. By default no LangGraph checkpointer is
        attached, so the graph can be driven via
        ``graph.invoke({"messages": [...]})`` without supplying a
        ``thread_id``. Pass ``checkpointer=<MemorySaver()>`` explicitly
        to opt into HITL / thread persistence.
    """
    from geoagent.tools.qgis import qgis_tools

    tools: list[BaseTool] = list(qgis_tools(iface, project))
    if include_stac:
        from geoagent.tools.stac import stac_tools

        tools += stac_tools()
    if include_data:
        from geoagent.tools.data.raster import raster_tools
        from geoagent.tools.data.vector import vector_tools

        tools += raster_tools() + vector_tools()
    if extra_tools is not None:
        tools += list(extra_tools)
    context = kwargs.pop("context", None) or GeoAgentContext(
        qgis_iface=iface, qgis_project=project
    )
    kwargs.setdefault("checkpointer", _NO_CHECKPOINTER_DEFAULT)
    inner = create_geo_agent(tools=tools, context=context, **kwargs)
    return _SyncToolGraph(inner)


__all__ = [
    "create_geo_agent",
    "for_leafmap",
    "for_anymap",
    "for_qgis",
]
