"""Runtime helpers for the Solara GeoAgent web UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from geoagent import (
    GeoAgent,
    GeoAgentConfig,
    auto_approve_all,
    auto_approve_safe_only,
)
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


def compact_tool_call(call: dict[str, Any], *, max_chars: int = 1200) -> str:
    """Return a compact human-readable tool-call summary."""
    name = str(call.get("name") or "tool")
    args = call.get("args")
    result = call.get("result")
    exception = call.get("exception")
    parts = [name]
    if args:
        parts.append(f"args={args}")
    if exception:
        parts.append(f"error={exception}")
    elif result is not None:
        parts.append(f"result={result}")
    text = " | ".join(parts)
    if len(text) > max_chars:
        return text[: max_chars - 16].rstrip() + " ... [truncated]"
    return text


__all__ = [
    "DEFAULT_MODEL_BY_PROVIDER",
    "PROVIDER_NAMES",
    "UiMapBinding",
    "compact_tool_call",
    "confirmation_callback",
    "confirmation_preview",
    "create_bound_agent",
    "create_ui_map_binding",
    "default_model_for_provider",
]
