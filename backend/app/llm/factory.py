"""Construct an LLM provider from a name and settings."""

from __future__ import annotations

from ..config import Settings
from .base import LLMProvider
from .providers import (
    AnthropicProvider,
    OllamaProvider,
    OpenAIProvider,
    RuleBasedProvider,
)


def create_llm(
    provider: str,
    openai_api_key: str | None = None,
    anthropic_api_key: str | None = None,
    ollama_base_url: str = "http://localhost:11434",
    ollama_model: str = "llama3",
) -> LLMProvider:
    if provider == "openai" and openai_api_key:
        return OpenAIProvider(openai_api_key)
    if provider == "anthropic" and anthropic_api_key:
        return AnthropicProvider(anthropic_api_key)
    if provider == "ollama":
        return OllamaProvider(ollama_base_url, ollama_model)
    return RuleBasedProvider()


def create_llm_from_settings(settings: Settings) -> LLMProvider:
    return create_llm(
        settings.llm_provider,
        openai_api_key=settings.openai_api_key,
        anthropic_api_key=settings.anthropic_api_key,
        ollama_base_url=settings.ollama_base_url,
        ollama_model=settings.ollama_model,
    )
