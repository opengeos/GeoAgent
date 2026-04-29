"""Strands hook that gates destructive tools behind :class:`ConfirmCallback`."""

from __future__ import annotations

from typing import Any

from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import AfterToolCallEvent, BeforeToolCallEvent

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
        registry.add_callback(AfterToolCallEvent, self._after_tool)

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
            record: dict[str, Any] = {"name": name, "args": dict(args)}
            tool_use_id = use.get("toolUseId") or use.get("tool_use_id")
            if tool_use_id:
                record["tool_use_id"] = tool_use_id
            self._tool_calls.append(record)

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

    def _after_tool(self, event: AfterToolCallEvent) -> None:
        """Attach compact tool results to the recorded tool call."""
        if self._tool_calls is None:
            return

        use = event.tool_use
        name = str(use.get("name", ""))
        tool_use_id = use.get("toolUseId") or use.get("tool_use_id")
        payload = {
            "result": _compact_tool_result(event.result),
        }
        if event.exception is not None:
            payload["exception"] = str(event.exception)

        for call in reversed(self._tool_calls):
            same_name = call.get("name") == name
            same_id = tool_use_id and call.get("tool_use_id") == tool_use_id
            if same_id or (same_name and "result" not in call):
                call.update(payload)
                return

        record: dict[str, Any] = {"name": name, **payload}
        if tool_use_id:
            record["tool_use_id"] = tool_use_id
        self._tool_calls.append(record)


def _compact_tool_result(value: Any) -> Any:
    """Return a JSON-friendly, size-limited representation of a tool result."""
    if isinstance(value, dict):
        return {str(k): _compact_tool_result(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_compact_tool_result(item) for item in value[:20]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > 20000:
            return value[:20000] + "\n... [truncated]"
        return value
    get = getattr(value, "get", None)
    if callable(get):
        try:
            keys = ("status", "content", "toolUseId")
            return {
                key: _compact_tool_result(get(key))
                for key in keys
                if get(key) is not None
            }
        except Exception:
            pass
    return str(value)
