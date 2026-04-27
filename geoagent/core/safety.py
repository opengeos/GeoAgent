"""Safety classification and confirmation-callback bridge.

GeoAgent classifies tools into "safe" (read-only inspection, navigation,
preview) and "confirmation-required" (destructive, expensive, or
externally-visible) actions. Confirmation-required tools are wired into
deepagents' ``interrupt_on`` mechanism so the agent pauses before executing
them.

The :class:`ConfirmRequest` / :data:`ConfirmCallback` pair is the integration
point between the GeoAgent runtime and the host environment (Jupyter, Solara,
CLI, custom UI). The host supplies a :data:`ConfirmCallback` to
:class:`geoagent.GeoAgent`, and the facade routes each interrupt through that
callback to decide whether to resume or cancel.

Built-in default callbacks are provided here:

* :func:`auto_approve_safe_only`: reject everything (safe fallback when no
  callback is supplied — confirmation-required tools never execute).
* :func:`auto_approve_all`: approve everything (only suitable for tests or
  trusted scripts).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from langchain_core.tools import BaseTool

from .decorators import get_geo_meta, needs_confirmation


@dataclass
class ConfirmRequest:
    """A pending tool invocation awaiting user approval.

    Attributes:
        tool_name: The name of the tool deepagents is about to call.
        args: The arguments the LLM proposed to pass to the tool.
        description: A short human-readable description of the tool, lifted
            from ``tool.description``.
        category: The tool's GeoAgent category (``"map"``, ``"qgis"``,
            ``"data"``, ``"ai"``, ``"io"``) when known.
        metadata: The full GeoAgent metadata dict for the tool (handy for
            UIs that want to render extra context).
    """

    tool_name: str
    args: dict[str, Any]
    description: str = ""
    category: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


ConfirmCallback = Callable[[ConfirmRequest], bool]
"""A callable that receives a :class:`ConfirmRequest` and returns ``True`` to
approve the call or ``False`` to reject it."""


def build_interrupt_on(tools: Iterable[BaseTool]) -> dict[str, bool]:
    """Build the deepagents ``interrupt_on`` dict from tool metadata.

    Args:
        tools: An iterable of tools (typically the full set passed to
            :func:`create_geo_agent`).

    Returns:
        A mapping of ``{tool_name: True}`` for every tool whose
        :func:`needs_confirmation` returns ``True``. Empty dict if no tool
        requires confirmation.
    """
    return {t.name: True for t in tools if needs_confirmation(t)}


def make_confirm_request(tool: BaseTool, args: dict[str, Any]) -> ConfirmRequest:
    """Build a :class:`ConfirmRequest` for an interrupt firing on ``tool``."""
    meta = get_geo_meta(tool)
    return ConfirmRequest(
        tool_name=tool.name,
        args=dict(args),
        description=tool.description or "",
        category=meta.get("category"),
        metadata=meta,
    )


def auto_approve_safe_only(request: ConfirmRequest) -> bool:
    """Default conservative callback: rejects every confirmation request.

    Used when the user hasn't supplied a ``confirm`` callback to
    :class:`geoagent.GeoAgent`. Because only confirmation-required tools
    reach this callback, the effect is "let safe tools run, block everything
    that needs approval."

    Args:
        request: The pending tool invocation.

    Returns:
        Always ``False``.
    """
    del request
    return False


def auto_approve_all(request: ConfirmRequest) -> bool:
    """Permissive callback that approves every request.

    Suitable for tests or tightly controlled automation. Do not use as a
    default in interactive applications.
    """
    del request
    return True
