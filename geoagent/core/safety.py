"""Confirmation callbacks and safety helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from geoagent.core.decorators import get_geo_meta


@dataclass
class ConfirmRequest:
    """Pending tool invocation presented to a host UI or CLI."""

    tool_name: str
    args: dict[str, Any]
    description: str = ""
    category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


ConfirmCallback = Callable[[ConfirmRequest], bool]


def make_confirm_request_from_tool_meta(
    tool_name: str,
    args: dict[str, Any],
    tool_obj: Any | None,
) -> ConfirmRequest:
    """Build :class:`ConfirmRequest` using optional Strands tool object."""
    meta = get_geo_meta(tool_obj) if tool_obj is not None else {}
    return ConfirmRequest(
        tool_name=tool_name,
        args=dict(args),
        description=(
            getattr(tool_obj, "tool_spec", {}).get("description", "")
            if isinstance(getattr(tool_obj, "tool_spec", None), dict)
            else str(getattr(tool_obj, "tool_spec", "") or "")
        ),
        category=meta.get("category"),
        metadata=meta,
    )


def auto_approve_safe_only(request: ConfirmRequest) -> bool:
    """Reject every confirmation-required tool (safe default)."""
    return False


def auto_approve_all(request: ConfirmRequest) -> bool:
    """Approve every confirmation request (testing / trusted scripts only)."""
    return True


def build_interrupt_on(*args: Any, **kwargs: Any) -> dict[str, bool]:
    """Removed in 1.0; raise to surface accidental use of the deepagents shim.

    Strands gates confirmation-required tools through
    :class:`geoagent.core.confirmation_hook.ConfirmationHookProvider`, not via
    a LangGraph-style ``interrupt_on`` mapping. Returning ``{}`` silently
    would let downstream code believe nothing needs confirmation.
    """
    raise NotImplementedError(
        "build_interrupt_on was removed in GeoAgent 1.0. Use "
        "geoagent.core.confirmation_hook.ConfirmationHookProvider with a "
        "ConfirmCallback to gate confirmation-required tools."
    )
