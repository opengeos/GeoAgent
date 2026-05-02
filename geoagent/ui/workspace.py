"""Solara components for the GeoAgent web workspace."""

from __future__ import annotations

from typing import Any

import solara

from geoagent import __version__
from geoagent.ui.app import (
    PROVIDER_NAMES,
    compact_tool_call,
    create_bound_agent,
    create_ui_map_binding,
    default_model_for_provider,
)


def _message_markdown(message: dict[str, Any]) -> str:
    """Render one chat-history message as Markdown."""
    role = str(message.get("role") or "assistant").title()
    status = str(message.get("status") or "")
    text = str(message.get("text") or "")
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
    provider = solara.use_reactive("openai-codex")
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
        last_tool_calls.set([])
        next_history = [*history.value, {"role": "user", "text": text}]
        history.set(next_history)
        try:
            if not isinstance(binding_state, tuple):
                raise RuntimeError("Map binding did not initialize correctly.")
            binding, binding_error = binding_state
            if binding_error:
                raise RuntimeError(binding_error)
            agent = create_bound_agent(
                binding,
                provider=provider.value,
                model_id=model_id.value,
                fast=fast.value,
                auto_approve=auto_approve.value,
            )
            response = agent.chat(text)
            last_tool_calls.set(list(response.tool_calls))
            details: list[str] = []
            if response.executed_tools:
                details.append("Executed tools: " + ", ".join(response.executed_tools))
            if response.cancelled_tools:
                details.append(
                    "Cancelled tools: " + ", ".join(response.cancelled_tools)
                )
            if response.error_message:
                details.append(response.error_message)
            answer = response.answer_text or "\n".join(details) or "No text response."
            if details and response.answer_text:
                answer = answer + "\n\n" + "\n".join(details)
            history.set(
                [
                    *next_history,
                    {
                        "role": "assistant",
                        "text": answer,
                        "status": "error" if not response.success else "ok",
                    },
                ]
            )
        except Exception as exc:
            history.set(
                [
                    *next_history,
                    {
                        "role": "assistant",
                        "text": str(exc),
                        "status": "error",
                    },
                ]
            )
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
