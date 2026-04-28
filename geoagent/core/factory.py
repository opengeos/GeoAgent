"""Build :class:`strands.agent.agent.Agent` and :class:`geoagent.GeoAgent` instances."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from geoagent.core.config import GeoAgentConfig
from geoagent.core.context import GeoAgentContext
from geoagent.core.registry import (
    GeoToolRegistry,
    collect_tools_for_context,
    packages_available,
)
from geoagent.core.safety import ConfirmCallback
from geoagent.core.agent import GeoAgent
from geoagent.tools.anymap import anymap_tools
from geoagent.tools.leafmap import leafmap_tools
from geoagent.tools.qgis import qgis_tools


def _filter_by_imports(tools: list[Any]) -> list[Any]:
    """Drop tools whose declared optional packages are unavailable."""
    out: list[Any] = []
    for t in tools:
        meta = getattr(t, "_geoagent_meta", None)
        if (
            meta
            and meta.requires_packages
            and not packages_available(meta.requires_packages)
        ):
            continue
        out.append(t)
    return out


def register_all_tools(registry: GeoToolRegistry, tools: Iterable[Any]) -> None:
    """Populate registry from decorated tools."""
    for t in tools:
        meta = getattr(t, "_geoagent_meta", None)
        if meta is not None:
            registry.register_tool(t, meta)


def assemble_tools(
    *,
    context: GeoAgentContext,
    extra_tools: Optional[list[Any]] = None,
    include_leafmap: bool = False,
    include_anymap: bool = False,
    include_qgis: bool = False,
    fast: bool = False,
) -> tuple[list[Any], GeoToolRegistry]:
    """Collect tools for a context and build a metadata registry."""
    registry = GeoToolRegistry()
    collected: list[Any] = []
    if include_leafmap and context.map_obj is not None:
        lt = _filter_by_imports(leafmap_tools(context.map_obj))
        register_all_tools(registry, lt)
        collected.extend(lt)
    if include_anymap and context.map_obj is not None:
        at = _filter_by_imports(anymap_tools(context.map_obj))
        register_all_tools(registry, at)
        collected.extend(at)
    if include_qgis:
        qt = _filter_by_imports(qgis_tools(context.qgis_iface, context.qgis_project))
        register_all_tools(registry, qt)
        collected.extend(qt)
    if extra_tools:
        register_all_tools(registry, extra_tools)
        collected.extend(extra_tools)
    tools = collect_tools_for_context(collected, fast=fast, registry=registry)
    return tools, registry


def create_agent(
    *,
    context: GeoAgentContext | None = None,
    tools: list[Any] | None = None,
    config: GeoAgentConfig | None = None,
    model: Any | None = None,
    provider: str | None = None,
    model_id: str | None = None,
    fast: bool = False,
    confirm: ConfirmCallback | None = None,
) -> GeoAgent:
    """Create a :class:`GeoAgent` with explicit tools and optional model."""
    ctx = context or GeoAgentContext()
    cfg = config or GeoAgentConfig()
    if provider is not None:
        cfg = cfg.model_copy(update={"provider": provider})
    if model_id is not None:
        cfg = cfg.model_copy(update={"model": model_id})
    registry = GeoToolRegistry()
    tool_list = _filter_by_imports(list(tools or []))
    register_all_tools(registry, tool_list)
    tool_list = collect_tools_for_context(tool_list, fast=fast, registry=registry)
    return GeoAgent(
        context=ctx,
        config=cfg,
        tools=tool_list,
        registry=registry,
        model=model,
        provider=provider,
        model_id=model_id,
        fast=fast,
        confirm=confirm,
    )


def for_leafmap(
    m: Any,
    *,
    config: GeoAgentConfig | None = None,
    model: Any | None = None,
    provider: str | None = None,
    model_id: str | None = None,
    fast: bool = False,
    confirm: ConfirmCallback | None = None,
    extra_tools: Optional[list[Any]] = None,
) -> GeoAgent:
    """Bind an agent to a leafmap-compatible map instance."""
    ctx = GeoAgentContext(map_obj=m)
    tools, registry = assemble_tools(
        context=ctx,
        include_leafmap=True,
        extra_tools=extra_tools,
        fast=fast,
    )
    cfg = config or GeoAgentConfig()
    if provider is not None:
        cfg = cfg.model_copy(update={"provider": provider})
    if model_id is not None:
        cfg = cfg.model_copy(update={"model": model_id})
    return GeoAgent(
        context=ctx,
        config=cfg,
        tools=tools,
        registry=registry,
        model=model,
        provider=provider,
        model_id=model_id,
        fast=fast,
        confirm=confirm,
    )


def for_anymap(
    m: Any,
    *,
    config: GeoAgentConfig | None = None,
    model: Any | None = None,
    provider: str | None = None,
    model_id: str | None = None,
    fast: bool = False,
    confirm: ConfirmCallback | None = None,
    extra_tools: Optional[list[Any]] = None,
) -> GeoAgent:
    """Bind an agent to an anymap map instance."""
    ctx = GeoAgentContext(map_obj=m)
    tools, registry = assemble_tools(
        context=ctx,
        include_anymap=True,
        extra_tools=extra_tools,
        fast=fast,
    )
    cfg = config or GeoAgentConfig()
    if provider is not None:
        cfg = cfg.model_copy(update={"provider": provider})
    if model_id is not None:
        cfg = cfg.model_copy(update={"model": model_id})
    return GeoAgent(
        context=ctx,
        config=cfg,
        tools=tools,
        registry=registry,
        model=model,
        provider=provider,
        model_id=model_id,
        fast=fast,
        confirm=confirm,
    )


def for_qgis(
    iface: Any,
    project: Any = None,
    *,
    config: GeoAgentConfig | None = None,
    model: Any | None = None,
    provider: str | None = None,
    model_id: str | None = None,
    fast: bool = False,
    confirm: ConfirmCallback | None = None,
    extra_tools: Optional[list[Any]] = None,
) -> GeoAgent:
    """Bind an agent to QGIS ``iface`` (and optional ``project``)."""
    ctx = GeoAgentContext(qgis_iface=iface, qgis_project=project)
    tools, registry = assemble_tools(
        context=ctx,
        include_qgis=True,
        extra_tools=extra_tools,
        fast=fast,
    )
    cfg = config or GeoAgentConfig()
    if provider is not None:
        cfg = cfg.model_copy(update={"provider": provider})
    if model_id is not None:
        cfg = cfg.model_copy(update={"model": model_id})
    return GeoAgent(
        context=ctx,
        config=cfg,
        tools=tools,
        registry=registry,
        model=model,
        provider=provider,
        model_id=model_id,
        fast=fast,
        confirm=confirm,
        qgis_safe_mode=True,
    )


__all__ = [
    "assemble_tools",
    "create_agent",
    "for_anymap",
    "for_leafmap",
    "for_qgis",
    "register_all_tools",
]
