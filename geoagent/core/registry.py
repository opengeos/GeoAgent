"""GeoAgent tool metadata registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


@dataclass
class GeoToolMeta:
    """Metadata for a registered Strands tool."""

    name: str
    description: str = ""
    category: str = "general"
    requires_confirmation: bool = False
    destructive: bool = False
    long_running: bool = False
    available_in: tuple[str, ...] = ("full",)
    requires_packages: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


class GeoToolRegistry:
    """Maps tool names to :class:`GeoToolMeta`."""

    def __init__(self) -> None:
        self._by_name: dict[str, GeoToolMeta] = {}

    def register_tool(self, tool_obj: Any, meta: GeoToolMeta) -> None:
        """Register metadata for a Strands decorated tool."""
        name = getattr(tool_obj, "tool_name", None) or meta.name
        self._by_name[str(name)] = meta

    def register(self, meta: GeoToolMeta) -> None:
        """Register metadata by explicit name."""
        self._by_name[meta.name] = meta

    def get(self, tool_name: str) -> GeoToolMeta | None:
        return self._by_name.get(tool_name)

    def list_names(self) -> list[str]:
        """Return registered tool names."""
        return sorted(self._by_name.keys())

    def get_all_tools_config(self) -> list[dict[str, Any]]:
        """Return tool metadata as serializable config records.

        Mirrors the type of inspection users expect from Strands-facing
        registries while preserving GeoAgent-specific metadata.
        """
        out: list[dict[str, Any]] = []
        for name in self.list_names():
            meta = self._by_name[name]
            out.append(
                {
                    "name": meta.name,
                    "description": meta.description,
                    "category": meta.category,
                    "requires_confirmation": meta.requires_confirmation,
                    "destructive": meta.destructive,
                    "long_running": meta.long_running,
                    "available_in": list(meta.available_in),
                    "requires_packages": list(meta.requires_packages),
                    "extra": dict(meta.extra),
                }
            )
        return out

    def needs_user_confirmation(self, meta: GeoToolMeta) -> bool:
        """Return True if the tool should go through the confirm callback."""
        return bool(meta.requires_confirmation or meta.destructive or meta.long_running)


# Names allowed in fast mode when metadata lacks explicit ``available_in``.
FAST_TOOL_FALLBACK: frozenset[str] = frozenset(
    {
        # leafmap / anymap (same function names)
        "list_layers",
        "get_map_state",
        "zoom_in",
        "zoom_out",
        "set_zoom",
        "set_center",
        "change_basemap",
        # qgis (safe navigation / inspection)
        "list_project_layers",
        "get_active_layer",
        "zoom_to_layer",
        "inspect_layer_fields",
        "refresh_canvas",
        "zoom_to_extent",
    }
)


def collect_tools_for_context(
    tool_objects: Sequence[Any],
    *,
    fast: bool,
    registry: GeoToolRegistry,
) -> list[Any]:
    """Filter tools for fast mode using metadata or :data:`FAST_TOOL_FALLBACK`."""

    if not fast:
        return list(tool_objects)

    out: list[Any] = []
    for t in tool_objects:
        name = getattr(t, "tool_name", None)
        if name is None:
            continue
        meta = registry.get(str(name))
        if meta is not None and "fast" in meta.available_in:
            out.append(t)
            continue
        if str(name) in FAST_TOOL_FALLBACK:
            out.append(t)
    return out


def packages_available(requires: Iterable[str]) -> bool:
    """Return True if every named package is importable."""
    for pkg in requires:
        try:
            __import__(pkg)
        except ImportError:
            return False
    return True
