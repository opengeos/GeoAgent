"""GeoAgent tool metadata registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Sequence
import importlib


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

    @property
    def needs_confirmation(self) -> bool:
        """
        Determine whether user confirmation is required

        Returns:
            True if confirmation should be required before execution
        """
        return self.requires_confirmation or self.destructive or self.long_running

    def to_dict(self) -> dict[Any, Any]:
        """
        Convert metadata into serial dictionary

        Returns:
            Dictionary representation of the metadata
        """
        data = asdict(self)

        data["available_in"] = list(self.available_in)
        data["requires_confirmation"] = list(self.requires_packages)

        return data


class GeoToolRegistry:
    """Maps tool names to :class:`GeoToolMeta`."""

    def __init__(self) -> None:
        self._tools: dict[str, GeoToolMeta] = {}

    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def register(self, meta: GeoToolMeta) -> None:
        """
        Register metadata by explicit name.

        Args:
            meta:
                Metadata describing the tool.

        Raises:
            ValueError:
                If a tool with the same name has already been registered.
        """
        if meta.name in self._tools:
            raise ValueError(f"'{meta.name}' already registered.")
        self._tools[meta.name] = meta

    def register_tool(self, tool_obj: Any, meta: GeoToolMeta) -> None:
        """
        Register metadata for a Strands decorated tool.

        Args:
            tool_obj:
                Tool implementation instance

            meta:
                Metadata associated with the tool
        """
        tool_name = getattr(tool_obj, "tool_name", None)

        if tool_name is None:
            tool_name = meta.name

        self.register(
            GeoToolMeta(
                name=str(tool_name),
                description=meta.description,
                category=meta.category,
                requires_confirmation=meta.requires_confirmation,
                destructive=meta.destructive,
                long_running=meta.long_running,
                available_in=meta.available_in,
                requires_packages=meta.requires_packages,
                extra=dict(meta.extra),
            )
        )

    def unregister(self, tool_name: str) -> bool:
        """
        Remove a tool from the registry

        Args:
            tool_name:
                Name of the tool to remove

        Returns:
            True if the tool existed and was removed.
            otherwise False.
        """
        return self._tools.pop(tool_name, None) is not None

    def get(self, tool_name: str) -> GeoToolMeta | None:
        """
        Return metadata for a registered tool name.

        Args:
            tool_name:
                Tool identifier.

        Returns:
            Tool metadata if found, otherwise None.
        """
        return self._tools.get(tool_name)

    def require(self, tool_name: str) -> GeoToolMeta:
        """
        Return tool metadata

        Args:
            tool_name:
                Tool identifier.

        Returns:
            KeyError:
                If the tool is not registered.
        """
        try:
            return self._tools[tool_name]
        except Exception as exc:
            raise KeyError(f"Unknown tool '{tool_name}'") from exc

    def exists(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def list_names(self) -> list[str]:
        """Return registered tool names."""
        return sorted(self._tools)

    def list_tools(self) -> list[GeoToolMeta]:
        return list(self._tools.values())

    def get_all_tools_config(self) -> list[dict[str, Any]]:
        """Return tool metadata as serializable config records.

        Mirrors the type of inspection users expect from Strands-facing
        registries while preserving GeoAgent-specific metadata.
        """
        return [
            meta.to_dict()
            for meta in sorted(self._tools.values(), key=lambda m: m.name)
        ]

    def get_tools_for_context(
        self, tool_objects: Sequence[Any], *, context: str
    ) -> list[Any]:
        """
        Filter tools that are available in a specific execution context.

        Args:
            tool_objects:
                Collection of tool implementation.

            context:
                Execution context name such as ``full`` or ``fast``

        Returns:
            Filtered list of tool objects that may bbe used in the requested
        """
        if context != "fast":
            return list(tool_objects)

        result: list[Any] = []

        for tool in tool_objects:
            name = getattr(tool, "tool_name", None)

            if not name:
                continue

            meta = self.get(name)

            if meta and context in meta.available_in:
                result.append(tool)
                continue

            if name in FAST_TOOL_FALLBACK:
                result.append(tool)
        return result

    def tool_requiring_confirmation(self) -> list[GeoToolMeta]:
        """
        Return tool requiring user confirmation
        """
        return [meta for meta in self._tools.values() if meta.needs_confirmation]

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
        "fly_to",
        "change_basemap",
        "set_layer_visibility",
        "set_layer_opacity",
        "set_layer_symbology",
        "zoom_to_layer",
        # qgis (safe navigation / inspection)
        "list_project_layers",
        "get_active_layer",
        "get_project_state",
        "zoom_to_layer",
        "inspect_layer_fields",
        "get_layer_summary",
        "refresh_canvas",
        "zoom_to_extent",
        "set_scale",
        "add_xyz_tile_layer",
    }
)


@lru_cache(maxsize=128)
def package_available(package: str) -> bool:
    """
    Return True if every named package is importable.

    Args:
        package:
            Package name.

    Returns:
        True if the package is importable, otherwise False.
    """

    try:
        importlib.import_module(package)
        return True
    except ImportError:
        return False


def packages_available(packages: Iterable[str]) -> bool:
    """
    Verify that all required packages are available.

    Args:
        packages:
            Collection of package names.

    Returns:
        True only if every package can be imported.
    """
    return all(package_available(package) for package in packages)


def collect_tools_for_context(
    tool_objects: Sequence[Any], *, fast: bool, registry: GeoToolRegistry
) -> list[Any]:
    """
    Convenience wrapper for context-based tool filtering.

    This function preserves backward compatibility with older code that
    uses a boolean ``fast`` flag instead of explicitly specifying a
    context name.

    Args:
        tool_objects:
            Collection of tool implementations.

        fast:
            Whether fast execution mode is enabled.

        registry:
            Registry used to evaluate tool availability.

    Returns:
        Filtered list of tool objects.
    """
    context = "fast" if fast else "full"

    return registry.get_tools_for_context(tool_objects, context=context)
