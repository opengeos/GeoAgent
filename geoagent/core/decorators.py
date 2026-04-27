"""The ``@geo_tool`` decorator used by all GeoAgent tool adapters.

``geo_tool`` wraps :func:`langchain_core.tools.tool` so the resulting object is
still a :class:`~langchain_core.tools.BaseTool` and can be passed unchanged to
:func:`deepagents.create_deep_agent` as part of its ``tools=...`` argument. It
additionally stamps GeoAgent-specific metadata onto ``tool.metadata["geo"]``
so the tool registry, the safety module, and downstream UIs can introspect:

* ``category``: a coarse capability label (``"map"``, ``"qgis"``, ``"data"``,
  ``"ai"``, ``"io"``).
* ``requires_confirmation``: whether the tool should be wired into deepagents'
  ``interrupt_on`` so the user is prompted before execution.
* ``requires_packages``: optional Python packages whose absence should cause
  the registry to silently skip the tool (e.g. ``("qgis",)`` for QGIS tools).
* ``context_keys``: which :class:`GeoAgentContext` fields the tool consults at
  runtime (informational only — closures handle live objects directly).
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

from langchain_core.tools import BaseTool, tool as _lc_tool

GEO_META_KEY = "geo"
"""The key inside ``BaseTool.metadata`` that holds the GeoAgent metadata dict."""


def geo_tool(
    *,
    category: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    requires_confirmation: bool = False,
    requires_packages: Iterable[str] = (),
    context_keys: Iterable[str] = (),
) -> Callable[[Callable[..., Any]], BaseTool]:
    """Decorate a Python callable as a GeoAgent tool.

    The returned object is a LangChain :class:`BaseTool`, so it can be passed
    directly to :func:`deepagents.create_deep_agent`. GeoAgent metadata is
    stored on ``tool.metadata["geo"]``.

    Args:
        category: Coarse capability label, one of ``"map"``, ``"qgis"``,
            ``"data"``, ``"ai"``, ``"io"``. Used by
            :func:`geoagent.core.registry.get_tools` to filter.
        name: Optional explicit tool name. Defaults to the function name.
        description: Optional explicit description. Defaults to the function's
            docstring.
        requires_confirmation: If true, the tool is added to deepagents'
            ``interrupt_on`` so the user must approve before execution.
        requires_packages: Python packages that must be importable for this
            tool to be available. The registry skips tools whose required
            packages are missing.
        context_keys: :class:`GeoAgentContext` fields the tool reads at
            runtime. Informational only.

    Returns:
        A decorator that turns the function into a :class:`BaseTool`.

    Example:
        >>> @geo_tool(category="map", requires_confirmation=True)
        ... def remove_layer(name: str) -> str:
        ...     '''Remove the named layer from the map.'''
        ...     ...
    """

    def deco(fn: Callable[..., Any]) -> BaseTool:
        if name is not None:
            built = _lc_tool(name)(fn)
        else:
            built = _lc_tool(fn)
        if description is not None:
            built.description = description
        meta = dict(built.metadata or {})
        meta[GEO_META_KEY] = {
            "category": category,
            "requires_confirmation": bool(requires_confirmation),
            "requires_packages": list(requires_packages),
            "context_keys": list(context_keys),
        }
        built.metadata = meta
        return built

    return deco


def get_geo_meta(tool_obj: BaseTool) -> dict[str, Any]:
    """Return the GeoAgent metadata dict stamped on a tool.

    Args:
        tool_obj: A LangChain ``BaseTool``.

    Returns:
        The metadata dict, or an empty dict if the tool was not produced by
        :func:`geo_tool`.
    """
    return (tool_obj.metadata or {}).get(GEO_META_KEY, {})


def needs_confirmation(tool_obj: BaseTool) -> bool:
    """Return whether ``tool_obj`` requires user confirmation before running.

    Args:
        tool_obj: A LangChain ``BaseTool``.

    Returns:
        ``True`` if the tool's GeoAgent metadata sets
        ``requires_confirmation``; ``False`` otherwise.
    """
    return bool(get_geo_meta(tool_obj).get("requires_confirmation"))


def get_category(tool_obj: BaseTool) -> Optional[str]:
    """Return the tool's GeoAgent category, or ``None`` if not stamped."""
    return get_geo_meta(tool_obj).get("category")


def get_required_packages(tool_obj: BaseTool) -> list[str]:
    """Return the list of Python packages required by ``tool_obj``."""
    return list(get_geo_meta(tool_obj).get("requires_packages", []))
