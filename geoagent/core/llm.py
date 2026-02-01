"""LLM provider abstraction with unified interface for multiple providers."""

import os
from enum import Enum
from typing import Optional, Dict, Any

from langchain_core.language_models.chat_models import BaseChatModel


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic" 
    GOOGLE = "google"
    OLLAMA = "ollama"


# Default models per provider
DEFAULT_MODELS = {
    LLMProvider.OPENAI: "gpt-4o",
    LLMProvider.ANTHROPIC: "claude-sonnet-4-20250514",
    LLMProvider.GOOGLE: "gemini-2.0-flash", 
    LLMProvider.OLLAMA: "llama3.1"
}

# Environment variables for API key detection
ENV_VARS = {
    LLMProvider.OPENAI: "OPENAI_API_KEY",
    LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    LLMProvider.GOOGLE: "GOOGLE_API_KEY",
    LLMProvider.OLLAMA: None  # Ollama typically runs locally without API key
}


def get_llm(
    provider: str, 
    model: Optional[str] = None,
    **kwargs: Any
) -> BaseChatModel:
    """
    Factory function to create LLM instances from different providers.
    
    Args:
        provider: The LLM provider ("openai", "anthropic", "google", "ollama")
        model: Model name. Uses default if not specified.
        **kwargs: Additional parameters like temperature, max_tokens
        
    Returns:
        LangChain BaseChatModel instance
        
    Raises:
        ValueError: If provider is not supported
        ImportError: If required package is not installed
        RuntimeError: If API key is missing for providers that require it
    """
    try:
        provider_enum = LLMProvider(provider.lower())
    except ValueError:
        supported = ", ".join([p.value for p in LLMProvider])
        raise ValueError(f"Unsupported provider '{provider}'. Supported: {supported}")
    
    if model is None:
        model = DEFAULT_MODELS[provider_enum]
    
    # Check for required API key
    env_var = ENV_VARS[provider_enum]
    if env_var and not os.getenv(env_var):
        raise RuntimeError(f"Missing required environment variable: {env_var}")
    
    if provider_enum == LLMProvider.OPENAI:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "OpenAI provider requires 'langchain-openai'. "
                "Install with: pip install langchain-openai"
            )
        
        return ChatOpenAI(
            model=model,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens"),
            **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
        )
    
    elif provider_enum == LLMProvider.ANTHROPIC:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "Anthropic provider requires 'langchain-anthropic'. "
                "Install with: pip install langchain-anthropic"
            )
        
        return ChatAnthropic(
            model=model,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens"),
            **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
        )
    
    elif provider_enum == LLMProvider.GOOGLE:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "Google provider requires 'langchain-google-genai'. "
                "Install with: pip install langchain-google-genai"
            )
        
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=kwargs.get("temperature", 0.0),
            max_output_tokens=kwargs.get("max_tokens"),
            **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
        )
    
    elif provider_enum == LLMProvider.OLLAMA:
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "Ollama provider requires 'langchain-ollama'. "
                "Install with: pip install langchain-ollama"
            )
        
        return ChatOllama(
            model=model,
            temperature=kwargs.get("temperature", 0.0),
            num_predict=kwargs.get("max_tokens"),
            **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
        )


def get_default_llm(**kwargs: Any) -> BaseChatModel:
    """
    Get the first available LLM provider based on environment variables.
    
    Checks for API keys in order: OpenAI, Anthropic, Google, then falls back to Ollama.
    
    Args:
        **kwargs: Additional parameters passed to get_llm()
        
    Returns:
        LangChain BaseChatModel instance
        
    Raises:
        RuntimeError: If no providers are available
    """
    # Check providers in order of preference
    for provider in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC, LLMProvider.GOOGLE]:
        env_var = ENV_VARS[provider]
        if env_var and os.getenv(env_var):
            try:
                return get_llm(provider.value, **kwargs)
            except ImportError:
                continue  # Try next provider if package not installed
    
    # Fall back to Ollama (no API key required)
    try:
        return get_llm(LLMProvider.OLLAMA.value, **kwargs)
    except ImportError:
        pass
    
    raise RuntimeError(
        "No LLM providers available. Please install required packages and set API keys:\n"
        "- OpenAI: pip install langchain-openai, set OPENAI_API_KEY\n"
        "- Anthropic: pip install langchain-anthropic, set ANTHROPIC_API_KEY\n" 
        "- Google: pip install langchain-google-genai, set GOOGLE_API_KEY\n"
        "- Ollama: pip install langchain-ollama (no API key required)"
    )