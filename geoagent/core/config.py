"""Model and runtime configuration for :class:`geoagent.GeoAgent`."""

from __future__ import annotations

import os
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

ProviderName = Literal["bedrock", "openai", "anthropic", "ollama"]


def _default_provider_from_env() -> ProviderName:
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OLLAMA_HOST") or os.environ.get("USE_OLLAMA") == "1":
        return "ollama"
    return "bedrock"


class GeoAgentConfig(BaseModel):
    """LLM provider and generation settings.

    API keys and region defaults are read from the environment when not
    passed explicitly (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``,
    ``AWS_REGION`` / default AWS credential chain for Bedrock, etc.).

    Attributes:
        provider: Model provider id.
        model: Model id for the provider. If omitted, a provider default is used.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the model response.
        ollama_host: Ollama server base URL (e.g. ``http://127.0.0.1:11434``).
    """

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    provider: ProviderName = Field(default_factory=_default_provider_from_env)
    model: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4096
    ollama_host: Optional[str] = Field(
        default=None, description="Ollama base URL, e.g. http://127.0.0.1:11434"
    )
    # Optional OpenAI / Anthropic client_args (e.g. base_url for Azure)
    client_args: dict[str, Any] = Field(default_factory=dict)

    def with_overrides(self, **kwargs: Any) -> "GeoAgentConfig":
        """Return a copy with field overrides."""
        return self.model_copy(update=kwargs)
