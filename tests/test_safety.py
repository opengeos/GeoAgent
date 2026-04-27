"""Tests for the safety / interrupt_on plumbing."""

from __future__ import annotations

from geoagent.core.decorators import geo_tool
from geoagent.core.safety import (
    auto_approve_all,
    auto_approve_safe_only,
    build_interrupt_on,
    make_confirm_request,
    ConfirmRequest,
)


@geo_tool(category="data")
def safe() -> str:
    """A safe tool."""
    return "ok"


@geo_tool(category="io", requires_confirmation=True)
def dangerous() -> str:
    """A dangerous tool."""
    return "danger"


@geo_tool(category="map", requires_confirmation=True)
def remove_layer(name: str) -> str:
    """Remove a layer."""
    return f"removed {name}"


def test_build_interrupt_on_only_flags_confirm_required() -> None:
    cfg = build_interrupt_on([safe, dangerous, remove_layer])
    assert set(cfg.keys()) == {"dangerous", "remove_layer"}
    assert all(value is True for value in cfg.values())


def test_build_interrupt_on_returns_empty_when_all_safe() -> None:
    cfg = build_interrupt_on([safe])
    assert cfg == {}


def test_make_confirm_request_populates_fields() -> None:
    request = make_confirm_request(remove_layer, {"name": "NDVI"})
    assert isinstance(request, ConfirmRequest)
    assert request.tool_name == "remove_layer"
    assert request.args == {"name": "NDVI"}
    assert request.category == "map"
    assert request.metadata["requires_confirmation"] is True


def test_auto_approve_safe_only_rejects_everything() -> None:
    request = make_confirm_request(remove_layer, {"name": "NDVI"})
    assert auto_approve_safe_only(request) is False


def test_auto_approve_all_approves_everything() -> None:
    request = make_confirm_request(remove_layer, {"name": "NDVI"})
    assert auto_approve_all(request) is True
