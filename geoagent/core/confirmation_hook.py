"""Strands hook that gates destructive tools behind :class:`ConfirmCallback`."""

from __future__ import annotations

from typing import Any

from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import BeforeToolCallEvent

from geoagent.core.registry import GeoToolMeta, GeoToolRegistry
from geoagent.core.safety import ConfirmCallback, ConfirmRequest


class ConfirmationHookProvider(HookProvider):
    """Sets ``cancel_tool`` when the user rejects a confirmation-required tool."""

    def __init__(
        self,
        registry: GeoToolRegistry,
        confirm: ConfirmCallback,
        cancelled: list[str],
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        self._registry = registry
        self._confirm = confirm
        self._cancelled = cancelled
        self._tool_calls = tool_calls

    def register_hooks(
        self, registry: HookRegistry, **kwargs: Any
    ) -> None:  # noqa: ARG002
        """Register confirmation callbacks with the Strands hook registry."""
        registry.add_callback(BeforeToolCallEvent, self._before_tool)

    def _resolve_meta(self, tool_name: str, selected: Any | None) -> GeoToolMeta | None:
        """Resolve metadata from the selected tool or registry."""
        if selected is not None:
            attached: GeoToolMeta | None = getattr(selected, "_geoagent_meta", None)
            if attached is not None:
                return attached
        return self._registry.get(tool_name)

    def _before_tool(self, event: BeforeToolCallEvent) -> None:
        """Handle the pre-tool execution hook."""
        use = event.tool_use
        name = str(use.get("name", ""))
        inp = use.get("input")
        args = inp if isinstance(inp, dict) else {}
        if self._tool_calls is not None:
            self._tool_calls.append({"name": name, "args": dict(args)})

        selected = event.selected_tool
        meta = self._resolve_meta(name, selected)
        if meta is None or not self._registry.needs_user_confirmation(meta):
            return

        description = ""
        if selected is not None:
            ts = selected.tool_spec
            if isinstance(ts, dict):
                description = str(ts.get("description") or "")
        elif meta.description:
            description = meta.description

        req = ConfirmRequest(
            tool_name=name,
            args=args,
            description=description,
            category=meta.category,
            metadata={
                "category": meta.category,
                "requires_confirmation": meta.requires_confirmation,
                "destructive": meta.destructive,
                "long_running": meta.long_running,
            },
        )
        if not self._confirm(req):
            event.cancel_tool = "User denied confirmation for this tool."
            self._cancelled.append(name)
