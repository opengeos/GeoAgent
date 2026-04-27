"""Capability-tagged tool registry.

The registry is a process-wide singleton dict that lets downstream packages
register :func:`geoagent.core.decorators.geo_tool`-decorated tools and lets
the agent factory pick a subset of them at construction time. Tools whose
``requires_packages`` are not importable are silently filtered out so that
``import geoagent.tools.qgis`` works on a non-QGIS Python without raising.

Typical usage in a downstream package's tool module::

    from geoagent.core.registry import register
    from geoagent.core.decorators import geo_tool

    @register
    @geo_tool(category="data", requires_packages=("pystac_client",))
    def my_tool(...): ...

The factory then calls :func:`get_tools` with category filters to assemble
the active tool list for an agent.
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import Iterable, Optional

from langchain_core.tools import BaseTool

from .decorators import get_geo_meta

_TOOLS: dict[str, BaseTool] = {}
_PKG_CACHE: dict[str, bool] = {}


def _have(pkg: str) -> bool:
    """Cached check for whether a Python package is importable.

    Args:
        pkg: A top-level distribution / module name (e.g. ``"qgis"``,
            ``"leafmap"``).

    Returns:
        ``True`` if the package can be located by
        :func:`importlib.util.find_spec`.
    """
    if pkg not in _PKG_CACHE:
        try:
            _PKG_CACHE[pkg] = find_spec(pkg) is not None
        except (ImportError, ValueError):
            _PKG_CACHE[pkg] = False
    return _PKG_CACHE[pkg]


def register(tool: BaseTool) -> BaseTool:
    """Register a single tool in the global registry.

    Idempotent: re-registering a tool with the same name overwrites the
    previous entry. Returns the tool to support decorator-style usage.

    Args:
        tool: A LangChain ``BaseTool`` (typically produced by
            :func:`geo_tool`).

    Returns:
        The same tool, unchanged.
    """
    _TOOLS[tool.name] = tool
    return tool


def register_many(tools: Iterable[BaseTool]) -> None:
    """Register an iterable of tools.

    Args:
        tools: Tools to register.
    """
    for t in tools:
        register(t)


def unregister(name: str) -> None:
    """Remove a tool from the registry by name. No-op if absent."""
    _TOOLS.pop(name, None)


def clear() -> None:
    """Remove all registered tools. Primarily useful in tests."""
    _TOOLS.clear()


def all_tools() -> list[BaseTool]:
    """Return every registered tool, regardless of availability."""
    return list(_TOOLS.values())


def get_tools(
    categories: Optional[Iterable[str]] = None,
    available_only: bool = True,
) -> list[BaseTool]:
    """Return registered tools filtered by category and availability.

    Args:
        categories: If provided, only return tools whose ``category`` is in
            this iterable. ``None`` returns all categories.
        available_only: If ``True`` (the default), drop tools whose
            ``requires_packages`` are not importable.

    Returns:
        A list of matching ``BaseTool`` instances, in insertion order.
    """
    cats = set(categories) if categories else None
    out: list[BaseTool] = []
    for tool in _TOOLS.values():
        meta = get_geo_meta(tool)
        if cats is not None and meta.get("category") not in cats:
            continue
        if available_only:
            required = meta.get("requires_packages", [])
            if required and not all(_have(pkg) for pkg in required):
                continue
        out.append(tool)
    return out


def is_available(tool: BaseTool) -> bool:
    """Return whether ``tool``'s required Python packages are importable."""
    required = get_geo_meta(tool).get("requires_packages", [])
    return all(_have(pkg) for pkg in required)
