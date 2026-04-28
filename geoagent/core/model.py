"""Resolve Strands :class:`~strands.models.model.Model` instances from config."""

from __future__ import annotations

import os
from typing import Any

from geoagent.core.config import GeoAgentConfig, ProviderName


def resolve_model(config: GeoAgentConfig | None = None, **overrides: Any) -> Any:
    """Build a Strands model from :class:`GeoAgentConfig` or kwargs overrides.

    Raises:
        ImportError: When optional provider client libraries are missing.
        ValueError: When configuration is inconsistent.
    """
    cfg = (
        config.model_copy(update=overrides)
        if config is not None
        else GeoAgentConfig(**overrides)
    )
    provider: ProviderName = cfg.provider
    if provider == "bedrock":
        from strands.models.bedrock import BedrockModel

        model_id = cfg.model or os.environ.get(
            "BEDROCK_MODEL", "us.anthropic.claude-sonnet-4-6"
        )
        params = {"temperature": cfg.temperature, "max_tokens": cfg.max_tokens}
        return BedrockModel(model_id=model_id, params=params)

    if provider == "openai":
        from strands.models.openai import OpenAIModel

        model_id = cfg.model or os.environ.get("OPENAI_MODEL", "gpt-5.5")
        client_args = dict(cfg.client_args)
        return OpenAIModel(
            client_args=client_args or None,
            model_id=model_id,
            params={"temperature": cfg.temperature, "max_tokens": cfg.max_tokens},
        )

    if provider == "anthropic":
        from strands.models.anthropic import AnthropicModel

        model_id = cfg.model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        client_args = dict(cfg.client_args)
        return AnthropicModel(
            client_args=client_args or None,
            model_id=model_id,
            max_tokens=cfg.max_tokens,
            params={"temperature": cfg.temperature},
        )

    if provider == "gemini":
        from strands.models.gemini import GeminiModel

        model_id = cfg.model or os.environ.get(
            "GEMINI_MODEL",
            os.environ.get("GOOGLE_MODEL", "gemini-3.1-pro-preview"),
        )
        client_args = dict(cfg.client_args)
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key and "api_key" not in client_args:
            client_args["api_key"] = api_key
        return GeminiModel(
            client_args=client_args or None,
            model_id=model_id,
            params={
                "temperature": cfg.temperature,
                "max_output_tokens": cfg.max_tokens,
            },
        )

    if provider == "ollama":
        from strands.models.ollama import OllamaModel

        host = cfg.ollama_host or os.environ.get(
            "OLLAMA_HOST", "http://127.0.0.1:11434"
        )
        model_id = cfg.model or os.environ.get("OLLAMA_MODEL", "llama3.1")
        return OllamaModel(
            host,
            model_id=model_id,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    raise ValueError(f"Unknown provider: {provider}")


def get_default_model() -> Any:
    """Default model using environment-derived provider selection."""
    return resolve_model(GeoAgentConfig())


def get_llm(**kwargs: Any) -> Any:
    """Build a model (alias for :func:`resolve_model` with a fresh config)."""
    if not kwargs:
        return get_default_model()
    return resolve_model(GeoAgentConfig.model_validate(kwargs))
