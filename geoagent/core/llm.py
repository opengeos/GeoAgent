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
        "default_model": "gpt-4.1",
        "env_var": "OPENAI_API_KEY",
        "package": "langchain-openai",
    },
    "anthropic": {
        "default_model": "claude-sonnet-4-5-20250929",
        "env_var": "ANTHROPIC_API_KEY",
        "package": "langchain-anthropic",
    },
    "google": {
        "default_model": "gemini-2.5-flash",
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

    Checks environment variables in order: OpenAI, Anthropic, Google, Ollama.
    Returns the first available provider.

    Args:
        temperature: Sampling temperature.
        **kwargs: Additional keyword arguments passed to the LLM constructor.

    Returns:
        A LangChain BaseChatModel instance.

    Raises:
        RuntimeError: If no LLM provider is available.
    """
    # Try providers in priority order
    for provider_name, config in PROVIDERS.items():
        env_var = config["env_var"]

        # Ollama has no API key requirement
        if env_var is None:
            try:
                return get_llm(
                    provider=provider_name, temperature=temperature, **kwargs
                )
            except ImportError:
                continue

        # Check if API key is set
        if os.getenv(env_var):
            try:
                return get_llm(
                    provider=provider_name, temperature=temperature, **kwargs
                )
            except ImportError:
                logger.warning(
                    f"{config['package']} not installed, skipping {provider_name}"
                )
                continue

    # Return MockLLM when no providers are available
    logger.warning("No LLM provider available, using MockLLM")
    return MockLLM()


class MockLLM:
    """Mock LLM for testing and development when no real LLM is available."""

    def __init__(self, name: str = "MockLLM"):
        self.name = name

    def invoke(self, prompt: str) -> str:
        """Mock LLM invocation.

        Args:
            prompt: Input prompt

        Returns:
            Mock response
        """
        return f"Mock response to: {prompt[:100]}..."

    def __str__(self) -> str:
        return self.name


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
