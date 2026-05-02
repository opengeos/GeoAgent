"""Runtime helpers for the Solara GeoAgent web UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from geoagent import (
    GeoAgent,
    GeoAgentConfig,
    auto_approve_all,
    auto_approve_safe_only,
)
from geoagent.core.config import _default_provider_from_env
from geoagent.core.safety import ConfirmCallback, ConfirmRequest

PROVIDER_NAMES: tuple[str, ...] = (
    "openai-codex",
    "openai",
    "anthropic",
    "gemini",
    "bedrock",
    "litellm",
    "ollama",
)

DEFAULT_MODEL_BY_PROVIDER: dict[str, str] = {
    "openai-codex": "gpt-5.5",
    "openai": "gpt-5.5",
    "anthropic": "claude-sonnet-4-6",
    "gemini": "gemini-3.1-pro-preview",
    "bedrock": "us.anthropic.claude-sonnet-4-6",
    "litellm": "openai/gpt-5.5",
    "ollama": "qwen3.5:4b",
}

MAX_CONTEXT_MESSAGES = 12
MAX_CONTEXT_CHARS = 12000


def default_provider() -> str:
    """Return the provider id derived from environment for UI initialization.

    Mirrors :func:`geoagent.core.config._default_provider_from_env` so the UI
    starts on a provider whose credentials are already configured instead of
    forcing the user onto the package-wide default.
    """
    candidate = _default_provider_from_env()
    return candidate if candidate in PROVIDER_NAMES else PROVIDER_NAMES[0]


@dataclass
class UiMapBinding:
    """A map object and the GeoAgent factory needed to bind it."""

    map_obj: Any
    map_library: str
    factory: Callable[..., GeoAgent]


def default_model_for_provider(provider: str | None) -> str:
    """Return the UI's default model id for a provider."""
    return DEFAULT_MODEL_BY_PROVIDER.get(str(provider or ""), "")


def confirmation_callback(auto_approve: bool) -> ConfirmCallback:
    """Return the confirmation policy selected in the UI."""
    return auto_approve_all if auto_approve else auto_approve_safe_only


def confirmation_preview(auto_approve: bool, tool_name: str = "preview") -> bool:
    """Return whether the current confirmation setting would approve a tool."""
    callback = confirmation_callback(auto_approve)
    return callback(ConfirmRequest(tool_name=tool_name, args={}))


def _map_from_anymap() -> UiMapBinding:
    """Create an anymap-backed UI binding."""
    import anymap
    from geoagent import for_anymap

    return UiMapBinding(
        map_obj=anymap.Map(),
        map_library="anymap",
        factory=for_anymap,
    )


def _map_from_leafmap() -> UiMapBinding:
    """Create a leafmap-backed UI binding."""
    import leafmap
    from geoagent import for_leafmap

    return UiMapBinding(
        map_obj=leafmap.Map(),
        map_library="leafmap",
        factory=for_leafmap,
    )


def create_ui_map_binding() -> UiMapBinding:
    """Create the first available web map binding, preferring anymap."""
    errors: list[str] = []
    for create in (_map_from_anymap, _map_from_leafmap):
        try:
            return create()
        except Exception as exc:
            errors.append(f"{create.__name__}: {exc}")
    details = "; ".join(errors) if errors else "no map libraries were tried"
    raise RuntimeError(
        "GeoAgent UI needs a web map package. Install `GeoAgent[anymap,ui]` "
        "or `GeoAgent[leafmap,ui]` and restart the UI. "
        f"Details: {details}"
    )


def create_bound_agent(
    binding: UiMapBinding,
    *,
    provider: str,
    model_id: str | None = None,
    fast: bool = False,
    auto_approve: bool = False,
) -> GeoAgent:
    """Create a map-bound GeoAgent from current UI controls."""
    model = (model_id or "").strip() or None
    config = GeoAgentConfig(
        provider=provider,  # type: ignore[arg-type]
        model=model,
    )
    return binding.factory(
        binding.map_obj,
        config=config,
        fast=bool(fast),
        confirm=confirmation_callback(bool(auto_approve)),
    )


def _compact_value(
    value: Any,
    *,
    per_string: int = 200,
    list_limit: int = 10,
    dict_limit: int = 20,
) -> Any:
    """Shrink a value structurally so its repr stays small.

    Truncates long strings, caps list and dict sizes, and recurses into nested
    containers so the final stringification cannot accidentally serialize a
    multi-megabyte tool-call payload.
    """
    if isinstance(value, str):
        if len(value) <= per_string:
            return value
        return value[:per_string] + "...[truncated]"
    if isinstance(value, dict):
        items = list(value.items())[:dict_limit]
        compact = {
            k: _compact_value(
                v,
                per_string=per_string,
                list_limit=list_limit,
                dict_limit=dict_limit,
            )
            for k, v in items
        }
        if len(value) > dict_limit:
            compact["..."] = f"+{len(value) - dict_limit} more keys"
        return compact
    if isinstance(value, (list, tuple)):
        head = [
            _compact_value(
                v,
                per_string=per_string,
                list_limit=list_limit,
                dict_limit=dict_limit,
            )
            for v in list(value)[:list_limit]
        ]
        if len(value) > list_limit:
            head.append(f"...[+{len(value) - list_limit} more]")
        return head
    return value


def compact_tool_call(call: dict[str, Any], *, max_chars: int = 1200) -> str:
    """Return a compact human-readable tool-call summary.

    Args and results are first compacted structurally so a single oversized
    payload cannot allocate a multi-megabyte intermediate string before the
    final character cap kicks in.
    """
    name = str(call.get("name") or "tool")
    args = call.get("args")
    result = call.get("result")
    exception = call.get("exception")
    parts = [name]
    if args:
        parts.append(f"args={_compact_value(args)}")
    if exception:
        parts.append(f"error={_compact_value(exception)}")
    elif result is not None:
        parts.append(f"result={_compact_value(result)}")
    text = " | ".join(parts)
    if len(text) > max_chars:
        return text[: max_chars - 16].rstrip() + " ... [truncated]"
    return text


def build_prompt_with_context(
    history: Sequence[dict[str, Any]],
    prompt: str,
    *,
    max_messages: int = MAX_CONTEXT_MESSAGES,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """Prepend recent transcript so multi-turn requests retain context.

    Mirrors the QGIS chat dock so follow-up turns like "remove the layer you
    just added" can resolve against earlier messages even though each send
    creates a fresh agent in the web UI.
    """
    history_lines: list[str] = []
    for msg in list(history)[-max_messages:]:
        body = str(msg.get("text") or "").strip()
        if not body:
            continue
        role = "User" if msg.get("role") == "user" else "Assistant"
        history_lines.append(f"{role}: {body}")
    if not history_lines:
        return prompt
    transcript = "\n\n".join(history_lines)
    if len(transcript) > max_chars:
        transcript = "[Earlier history truncated]\n" + transcript[-max_chars:]
    return (
        "Use the recent conversation history for context. The current user "
        "request is the authoritative request to answer now.\n\n"
        f"Recent conversation:\n{transcript}\n\n"
        f"Current user request:\n{prompt}"
    )


def format_response_message(response: Any) -> dict[str, Any]:
    """Build the assistant chat-history entry from a GeoAgentResponse."""
    details: list[str] = []
    executed = list(getattr(response, "executed_tools", None) or [])
    if executed:
        details.append("Executed tools: " + ", ".join(executed))
    cancelled = list(getattr(response, "cancelled_tools", None) or [])
    if cancelled:
        details.append("Cancelled tools: " + ", ".join(cancelled))
    error_message = getattr(response, "error_message", None)
    if error_message:
        details.append(str(error_message))
    answer_text = getattr(response, "answer_text", "") or ""
    if answer_text and details:
        text = answer_text + "\n\n" + "\n".join(details)
    else:
        text = answer_text or "\n".join(details) or "No text response."
    success = bool(getattr(response, "success", True))
    return {
        "role": "assistant",
        "text": text,
        "status": "ok" if success else "error",
    }


def dispatch_prompt(
    text: str,
    *,
    history: Sequence[dict[str, Any]],
    binding: "UiMapBinding | None",
    binding_error: str = "",
    provider: str,
    model_id: str | None,
    fast: bool,
    auto_approve: bool,
    create_agent: Callable[..., GeoAgent] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one UI send and return ``(next_history, tool_calls)``.

    Pure helper extracted from ``WorkspacePage`` so the send pipeline is
    unit-testable without a Solara render.
    """
    factory = create_agent or create_bound_agent
    next_history: list[dict[str, Any]] = [
        *history,
        {"role": "user", "text": text},
    ]
    try:
        if binding is None:
            raise RuntimeError(binding_error or "Map binding unavailable.")
        prompt = build_prompt_with_context(history, text)
        agent = factory(
            binding,
            provider=provider,
            model_id=model_id,
            fast=fast,
            auto_approve=auto_approve,
        )
        response = agent.chat(prompt)
        message = format_response_message(response)
        return [*next_history, message], list(
            getattr(response, "tool_calls", None) or []
        )
    except Exception as exc:
        return (
            [*next_history, {"role": "assistant", "text": str(exc), "status": "error"}],
            [],
        )


__all__ = [
    "DEFAULT_MODEL_BY_PROVIDER",
    "MAX_CONTEXT_CHARS",
    "MAX_CONTEXT_MESSAGES",
    "PROVIDER_NAMES",
    "UiMapBinding",
    "build_prompt_with_context",
    "compact_tool_call",
    "confirmation_callback",
    "confirmation_preview",
    "create_bound_agent",
    "create_ui_map_binding",
    "default_model_for_provider",
    "default_provider",
    "dispatch_prompt",
    "format_response_message",
]
