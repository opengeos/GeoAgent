"""LLM provider abstraction for GeoAgent.

Provides a unified interface for multiple LLM providers including
OpenAI, Anthropic, Google Gemini, and Ollama (local).
"""

from typing import Any, Optional, Dict, List
import logging
import os

logger = logging.getLogger(__name__)


# Provider configurations with default models
PROVIDERS: Dict[str, Dict[str, str]] = {
    "openai": {
        "default_model": "gpt-5.5",
        "env_var": "OPENAI_API_KEY",
        "package": "langchain-openai",
    },
    "anthropic": {
        "default_model": "claude-sonnet-4-6",
        "env_var": "ANTHROPIC_API_KEY",
        "package": "langchain-anthropic",
    },
    "google": {
        "default_model": "gemini-3.1-flash-lite-preview",
        "env_var": "GOOGLE_API_KEY",
        "package": "langchain-google-genai",
    },
    "ollama": {
        "default_model": "llama3.1",
        "env_var": None,
        "package": "langchain-ollama",
    },
}


def get_llm(
    provider: str = "openai",
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    **kwargs,
) -> Any:
    """Create an LLM instance for the specified provider.

    Args:
        provider: LLM provider name ("openai", "anthropic", "google", "ollama").
        model: Model name. Uses provider default if None.
        temperature: Sampling temperature (0.0 to 1.0).
        max_tokens: Maximum tokens in the response.
        **kwargs: Additional provider-specific keyword arguments.

    Returns:
        A LangChain BaseChatModel instance.

    Raises:
        ValueError: If the provider is not supported.
        ImportError: If the required package is not installed.
        RuntimeError: If the API key is missing.
    """
    provider = provider.lower()
    if provider not in PROVIDERS:
        supported = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unsupported provider '{provider}'. Supported: {supported}")

    config = PROVIDERS[provider]
    resolved_model = model or config["default_model"]

    # Check API key (not needed for Ollama)
    if config["env_var"] and not os.getenv(config["env_var"]):
        raise RuntimeError(
            f"API key not found. Set the {config['env_var']} environment variable."
        )

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai is not installed. Run: pip install langchain-openai"
            )
        return ChatOpenAI(
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic is not installed. Run: pip install langchain-anthropic"
            )
        return ChatAnthropic(
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    elif provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "langchain-google-genai is not installed. Run: pip install langchain-google-genai"
            )
        return ChatGoogleGenerativeAI(
            model=resolved_model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            **kwargs,
        )

    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "langchain-ollama is not installed. Run: pip install langchain-ollama"
            )
        return ChatOllama(
            model=resolved_model,
            temperature=temperature,
            **kwargs,
        )


def get_default_llm(temperature: float = 0.1, **kwargs) -> Any:
    """Get a default LLM by checking available API keys.

    Tries cloud providers first in the order OpenAI, Anthropic, Google, and
    falls back to a local Ollama model when no cloud provider is usable
    (for example, because no cloud API key is set or because the matching
    LangChain integration package is not installed). The Ollama fallback
    emits a ``logging.WARNING`` so the caller can see why the agent might
    be misbehaving: the default Ollama model (``llama3.1`` 8B Q4) is too
    small to drive the deepagents multi-subagent coordinator reliably, so
    silent fallback would leave the user with an agent that "runs" but
    never fires any tools.

    Args:
        temperature: Sampling temperature.
        **kwargs: Additional keyword arguments passed to the LLM constructor.

    Returns:
        A LangChain ``BaseChatModel`` instance.

    Raises:
        RuntimeError: If no provider is usable (no cloud key in the
            environment AND ``langchain-ollama`` is not importable).
    """
    cloud_providers = [
        (name, config)
        for name, config in PROVIDERS.items()
        if config["env_var"] is not None
    ]
    skipped_no_key: List[str] = []
    skipped_missing_pkg: List[str] = []
    for provider_name, config in cloud_providers:
        if not os.getenv(config["env_var"]):
            skipped_no_key.append(provider_name)
            continue
        try:
            return get_llm(provider=provider_name, temperature=temperature, **kwargs)
        except ImportError:
            skipped_missing_pkg.append(f"{provider_name} ({config['package']})")
            logger.warning(
                f"{config['package']} not installed, skipping {provider_name}"
            )
            continue

    # No cloud provider was usable. Fall back to local Ollama with a loud
    # warning. The deepagents multi-subagent coordinator GeoAgent uses
    # is demanding enough that small local models (e.g. 8B-class Ollama
    # models) routinely emit empty or text-only responses with no
    # tool_calls populated, leaving the agent looking like it works while
    # never actually firing a tool. Surface that risk instead of hiding it.
    if skipped_missing_pkg:
        reason = (
            "no cloud LLM provider was usable: API keys were set but "
            f"required packages are missing for: {', '.join(skipped_missing_pkg)}"
        )
    else:
        reason = (
            "no cloud LLM API key found in environment "
            "(OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY)"
        )

    ollama_config = PROVIDERS.get("ollama")
    if ollama_config is not None:
        try:
            llm = get_llm(provider="ollama", temperature=temperature, **kwargs)
        except ImportError:
            pass
        else:
            logger.warning(
                "%s. Falling back to Ollama '%s'. GeoAgent needs strong "
                "tool-calling; small local models often return empty "
                "content with no tool_calls, so tools may never fire. Set "
                "a cloud API key (and install the matching langchain "
                "package) for reliable behavior, or pass an explicit "
                "`provider`/`model` to GeoAgent that you have verified "
                "can drive the coordinator.",
                reason,
                ollama_config["default_model"],
            )
            return llm

    raise RuntimeError(
        f"No LLM provider is available ({reason}). Install langchain-ollama "
        "and run a local Ollama server, or set OPENAI_API_KEY, "
        "ANTHROPIC_API_KEY, or GOOGLE_API_KEY with the matching package."
    )


def get_available_providers() -> List[str]:
    """Get list of available LLM providers based on installed packages and API keys.

    Returns:
        List of available provider names.
    """
    available = []
    for name, config in PROVIDERS.items():
        env_var = config["env_var"]
        has_key = env_var is None or bool(os.getenv(env_var))

        if not has_key:
            continue

        try:
            if name == "openai":
                import langchain_openai  # noqa: F401
            elif name == "anthropic":
                import langchain_anthropic  # noqa: F401
            elif name == "google":
                import langchain_google_genai  # noqa: F401
            elif name == "ollama":
                import langchain_ollama  # noqa: F401
            available.append(name)
        except ImportError:
            pass

    return available


def check_api_keys() -> Dict[str, bool]:
    """Check which LLM API keys are available in the environment.

    Returns:
        Dictionary mapping provider names to whether their API key is set.
    """
    return {
        name: config["env_var"] is None or bool(os.getenv(config["env_var"]))
        for name, config in PROVIDERS.items()
    }


def resolve_model(
    llm: Optional[Any] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Resolve the tri-form ``llm`` / ``provider`` / ``model`` arguments.

    GeoAgent's public API accepts any of these forms for picking the LLM:

    * an explicit ``llm`` object (a LangChain ``BaseChatModel`` instance),
    * a ``provider`` name (e.g. ``"openai"``) plus an optional ``model``,
    * a ``model`` string in deepagents' ``"provider:model"`` shorthand
      (e.g. ``"google_genai:gemini-3.1-flash-lite-preview"``,
      ``"anthropic:claude-sonnet-4-5"``), which is routed verbatim to
      :func:`langchain.chat_models.init_chat_model` for full parity with
      :func:`deepagents.create_deep_agent`,
    * neither, in which case :func:`get_default_llm` picks the first
      available cloud provider from the environment, or falls back to
      Ollama with a warning.

    This function normalises all forms into a single ``BaseChatModel``
    ready to pass to :func:`deepagents.create_deep_agent` as its
    ``model=`` argument.

    Args:
        llm: A pre-built LangChain chat model. Returned unchanged if given.
        provider: A provider name from :data:`PROVIDERS`.
        model: A model name override for ``provider``, or a
            ``"provider:model"`` shorthand when ``provider`` is omitted.
        **kwargs: Forwarded to the underlying constructor (:func:`get_llm`
            or :func:`init_chat_model`).

    Returns:
        A LangChain ``BaseChatModel``.

    Raises:
        RuntimeError: If no provider is usable and no ``llm`` was supplied.
            See :func:`get_default_llm`.
    """
    if llm is not None:
        return llm
    if provider is not None:
        return get_llm(provider=provider, model=model, **kwargs)
    if isinstance(model, str) and ":" in model:
        # deepagents-style "provider:model" string. Hand it to
        # langchain.chat_models.init_chat_model verbatim so the full set
        # of langchain provider namespaces is supported (google_genai,
        # google_vertexai, groq, together, azure_openai, ...). Matches
        # the behavior of deepagents.create_deep_agent for string models.
        from langchain.chat_models import init_chat_model

        return init_chat_model(model, **kwargs)
    return get_default_llm(**kwargs)
