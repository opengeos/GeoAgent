"""Solara components for the GeoAgent web workspace."""

from __future__ import annotations

from typing import Any

import solara

from geoagent import __version__
from geoagent.ui.app import (
    PROVIDER_NAMES,
    compact_tool_call,
    create_ui_map_binding,
    default_model_for_provider,
    default_provider,
    dispatch_prompt,
)


def _fenced_block(text: str) -> str:
    """Wrap arbitrary text in a Markdown code fence that survives backticks.

    Uses the smallest run of backticks longer than any backtick run in the
    text, so user prompts containing ````` characters still render
    verbatim instead of being interpreted as Markdown.
    """
    longest = 0
    current = 0
    for ch in text:
        if ch == "`":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    fence = "`" * max(3, longest + 1)
    return f"{fence}\n{text}\n{fence}"


def _message_markdown(message: dict[str, Any]) -> str:
    """Render one chat-history message as Markdown.

    User text is rendered inside a fenced code block so prompts containing
    Markdown syntax, links, or raw HTML do not change meaning in the
    transcript. Assistant messages are rendered as Markdown so tool results
    keep their formatting.
    """
    role = str(message.get("role") or "assistant").title()
    status = str(message.get("status") or "")
    text = str(message.get("text") or "")
    if message.get("role") == "user":
        return f"**{role}**\n\n{_fenced_block(text)}"
    if status == "error":
        return f"**{role} error**\n\n{text}"
    return f"**{role}**\n\n{text}"


def _safe_create_binding() -> tuple[Any | None, str]:
    """Create a map binding without letting render-time import errors escape."""
    try:
        return create_ui_map_binding(), ""
    except Exception as exc:
        return None, str(exc)


@solara.component
def WorkspacePage() -> None:
    """Render the map chat workspace."""
    provider = solara.use_reactive(default_provider())
    model_id = solara.use_reactive(default_model_for_provider(provider.value))
    prompt = solara.use_reactive("")
    fast = solara.use_reactive(False)
    auto_approve = solara.use_reactive(False)
    busy = solara.use_reactive(False)
    history = solara.use_reactive([])
    last_tool_calls = solara.use_reactive([])
    binding_state = solara.use_memo(_safe_create_binding, dependencies=[])

    def _provider_changed(value: str) -> None:
        provider.set(value)
        model_id.set(default_model_for_provider(value))

    def _send() -> None:
        text = prompt.value.strip()
        if not text or busy.value:
            return
        prompt.set("")
        busy.set(True)
        try:
            binding_obj, binding_error = (
                binding_state if isinstance(binding_state, tuple) else (None, "")
            )
            new_history, tool_calls = dispatch_prompt(
                text,
                history=history.value,
                binding=binding_obj,
                binding_error=binding_error,
                provider=provider.value,
                model_id=model_id.value,
                fast=fast.value,
                auto_approve=auto_approve.value,
            )
            history.set(new_history)
            last_tool_calls.set(tool_calls)
        finally:
            busy.set(False)

    binding, binding_error = binding_state
    try:
        map_obj = binding.map_obj if binding is not None else None
        map_label = binding.map_library if binding is not None else "unavailable"
    except Exception as exc:  # pragma: no cover - defensive Solara path
        map_obj = None
        map_label = "unavailable"
        binding_error = str(exc)

    with solara.Column(gap="12px"):
        solara.Markdown(f"## GeoAgent {__version__}")
        with solara.Row(gap="12px"):
            with solara.Column(
                gap="8px", style={"minWidth": "280px", "maxWidth": "360px"}
            ):
                solara.Select(
                    label="Provider",
                    values=list(PROVIDER_NAMES),
                    value=provider.value,
                    on_value=_provider_changed,
                )
                solara.InputText(
                    label="Model",
                    value=model_id.value,
                    on_value=model_id.set,
                )
                solara.Checkbox(
                    label="Fast mode",
                    value=fast.value,
                    on_value=fast.set,
                )
                solara.Checkbox(
                    label="Auto-approve confirmation tools",
                    value=auto_approve.value,
                    on_value=auto_approve.set,
                )
                solara.InputTextArea(
                    label="Prompt",
                    value=prompt.value,
                    on_value=prompt.set,
                    rows=5,
                )
                solara.Button(
                    "Send",
                    on_click=_send,
                    disabled=busy.value or not prompt.value.strip(),
                    color="primary",
                )
                if busy.value:
                    solara.ProgressLinear()
                solara.Markdown(
                    f"Map backend: `{map_label}`. "
                    "Confirmation-required tools are denied unless auto-approve is enabled."
                )

            with solara.Column(gap="8px", style={"minWidth": "360px", "flex": "1"}):
                if binding_error:
                    solara.Markdown(f"**Map unavailable**\n\n{binding_error}")
                elif map_obj is not None:
                    solara.display(map_obj)

                solara.Markdown("### Chat")
                if not history.value:
                    solara.Markdown(
                        "Ask GeoAgent to inspect the map, add layers, change the "
                        "view, or summarize geospatial data."
                    )
                for item in history.value:
                    with solara.Card():
                        solara.Markdown(_message_markdown(item))

                if last_tool_calls.value:
                    solara.Markdown("### Tool Calls")
                    for call in last_tool_calls.value:
                        solara.Markdown(f"`{compact_tool_call(call)}`")


__all__ = ["WorkspacePage"]
