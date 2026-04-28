"""``@geo_tool`` — Strands :func:`strands.tool` plus GeoAgent metadata."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from strands import tool as strands_tool

from geoagent.core.registry import GeoToolMeta

F = TypeVar("F", bound=Callable[..., Any])


def geo_tool(
    *,
    category: str = "general",
    name: str | None = None,
    description: str | None = None,
    requires_confirmation: bool = False,
    destructive: bool = False,
    long_running: bool = False,
    available_in: Iterable[str] = ("full",),
    requires_packages: Iterable[str] = (),
) -> Callable[[F], Any]:
    """Decorate a function as a Strands tool with GeoAgent metadata.

    The returned object is a Strands ``DecoratedFunctionTool``. Metadata is
    stored on ``tool._geoagent_meta`` as :class:`GeoToolMeta`.

    Args:
        category: Logical group (``map``, ``qgis``, ``data``, ...).
        name: Optional override for tool name (forwarded via ``@tool(name=...)``).
        description: Optional override; otherwise the docstring is used.
        requires_confirmation: If True, host must approve via callback.
        destructive: Implies confirmation when True.
        long_running: Marks expensive jobs (typically confirmation-required).
        available_in: Include ``\"fast\"`` for tools exposed in fast mode.
        requires_packages: Skip registering tools when imports fail upstream.
    """

    def deco(fn: F) -> Any:
        kwargs: dict[str, Any] = {}
        if name is not None:
            kwargs["name"] = name
        if description is not None:
            kwargs["description"] = description

        base = strands_tool(**kwargs)(fn)

        meta = GeoToolMeta(
            name=str(base.tool_name),
            description=description or (fn.__doc__ or "").strip(),
            category=category,
            requires_confirmation=requires_confirmation or destructive,
            destructive=destructive,
            long_running=long_running,
            available_in=tuple(available_in),
            requires_packages=tuple(requires_packages),
        )
        setattr(base, "_geoagent_meta", meta)
        return base

    return deco


def get_geo_meta(tool_obj: Any) -> dict[str, Any]:
    """Return GeoAgent metadata as a plain dict for callbacks / UIs."""
    meta: GeoToolMeta | None = getattr(tool_obj, "_geoagent_meta", None)
    if meta is None:
        return {}
    return {
        "category": meta.category,
        "requires_confirmation": meta.requires_confirmation,
        "destructive": meta.destructive,
        "long_running": meta.long_running,
        "available_in": meta.available_in,
        "requires_packages": meta.requires_packages,
        **meta.extra,
    }


def stamp_geo_meta(tool_obj: Any, **fields: Any) -> Any:
    """Attach or merge :class:`GeoToolMeta` fields on a Strands tool."""
    meta: GeoToolMeta | None = getattr(tool_obj, "_geoagent_meta", None)
    if meta is None:
        name = getattr(tool_obj, "tool_name", "unknown")
        meta = GeoToolMeta(name=str(name))
    for k, v in fields.items():
        if hasattr(meta, k):
            setattr(meta, k, v)
    setattr(tool_obj, "_geoagent_meta", meta)
    return tool_obj


def needs_confirmation(tool_obj: Any) -> bool:
    """Return True if tool should use the confirmation callback."""
    meta: GeoToolMeta | None = getattr(tool_obj, "_geoagent_meta", None)
    if meta is None:
        return False
    return bool(meta.requires_confirmation or meta.destructive or meta.long_running)
