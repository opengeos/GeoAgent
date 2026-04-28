"""Approve/deny path coverage for :class:`ConfirmationHookProvider`."""

from __future__ import annotations

from types import SimpleNamespace

from geoagent.core.confirmation_hook import ConfirmationHookProvider
from geoagent.core.registry import GeoToolMeta, GeoToolRegistry
from geoagent.core.safety import ConfirmRequest


def _build(meta: GeoToolMeta, callback) -> tuple[ConfirmationHookProvider, list[str]]:
    registry = GeoToolRegistry()
    registry.register(meta)
    cancelled: list[str] = []
    return ConfirmationHookProvider(registry, callback, cancelled), cancelled


def _event(name: str, args: dict, selected=None) -> SimpleNamespace:
    return SimpleNamespace(
        tool_use={"name": name, "input": args},
        selected_tool=selected,
        cancel_tool=False,
    )


def test_denied_confirmation_cancels_and_records() -> None:
    meta = GeoToolMeta(
        name="remove_layer",
        description="Remove a layer from the map.",
        requires_confirmation=True,
    )
    requests: list[ConfirmRequest] = []

    def deny(req: ConfirmRequest) -> bool:
        requests.append(req)
        return False

    hook, cancelled = _build(meta, deny)
    event = _event("remove_layer", {"layer": "roads"})

    hook._before_tool(event)  # noqa: SLF001 — exercising hook directly

    assert event.cancel_tool == "User denied confirmation for this tool."
    assert cancelled == ["remove_layer"]
    assert len(requests) == 1
    assert requests[0].tool_name == "remove_layer"
    assert requests[0].args == {"layer": "roads"}
    assert requests[0].description == "Remove a layer from the map."


def test_approved_confirmation_allows_tool() -> None:
    meta = GeoToolMeta(name="save_map", destructive=True)

    def approve(req: ConfirmRequest) -> bool:
        return True

    hook, cancelled = _build(meta, approve)
    event = _event("save_map", {"path": "out.html"})

    hook._before_tool(event)  # noqa: SLF001

    assert event.cancel_tool is False
    assert cancelled == []


def test_unconfirmed_tool_skips_callback() -> None:
    meta = GeoToolMeta(name="list_layers")
    calls: list[ConfirmRequest] = []

    def callback(req: ConfirmRequest) -> bool:
        calls.append(req)
        return False

    hook, cancelled = _build(meta, callback)
    event = _event("list_layers", {})

    hook._before_tool(event)  # noqa: SLF001

    assert event.cancel_tool is False
    assert cancelled == []
    assert calls == []
