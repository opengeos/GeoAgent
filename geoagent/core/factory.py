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
        tools: Explicit tool list. If ``None``, the registry is used.
        extra_tools: Additional tools appended to ``tools``.
        context: Runtime context. Defaults to an empty
            :class:`GeoAgentContext`. The dataclass type is passed to
            deepagents as ``context_schema``.
        system_prompt: Coordinator system prompt. Defaults to
            :data:`COORDINATOR_PROMPT`.
        model: A pre-built LangChain ``BaseChatModel``, or a deepagents-style
            string like ``"openai:gpt-4o"``. Mutually exclusive with
            ``provider`` / ``llm``.
        provider: A provider name (``"openai"``, ``"anthropic"``, ``"google"``,
            ``"ollama"``). Resolved via :func:`geoagent.core.llm.resolve_model`.
        llm: A pre-built LangChain ``BaseChatModel`` (alias of ``model`` for
            users coming from the v0.x API).
        subagents: List of deepagents subagent specs. Phase 1 ships with no
            built-in subagents; pass an explicit list when ready.
        checkpointer: A LangGraph ``Checkpointer`` for thread persistence.
        confirm: Reserved for use by the :class:`GeoAgent` facade in Phase 2.
            Currently unused at the factory layer; included so callers can
            construct full configurations now without re-plumbing later.
        debug: Forwarded to deepagents.

    Returns:
        A compiled deepagents agent (a LangGraph ``CompiledStateGraph``).
        Invoke via ``.invoke({"messages": [...]})`` or ``.stream(...)``.
    """
    create_deep_agent = _require_deepagents()
    del confirm  # placeholder for the Phase 2 facade

    from .llm import resolve_model

    resolved_model = resolve_model(llm=llm or model, provider=provider)

    tool_list = _as_list(tools) + _as_list(extra_tools)
    interrupt_on = build_interrupt_on(tool_list) or None

    context = context or GeoAgentContext()

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
        A compiled deepagents agent.
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
        A compiled deepagents agent.
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
    return create_geo_agent(tools=tools, context=context, **kwargs)


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
        A compiled deepagents agent.
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
    return create_geo_agent(tools=tools, context=context, **kwargs)


__all__ = [
    "create_geo_agent",
    "for_leafmap",
    "for_anymap",
    "for_qgis",
]
