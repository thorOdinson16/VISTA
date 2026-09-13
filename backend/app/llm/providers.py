"""Concrete LLM providers (REST via httpx, no vendor SDK required)."""

from __future__ import annotations

import httpx

from .base import LLMProvider


class RuleBasedProvider(LLMProvider):
    name = "rule-based"

    def complete(
        self, prompt: str, system: str | None = None, temperature: float = 0.0
    ) -> str:
        return ""


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.embedding_model = embedding_model
        self.base_url = base_url.rstrip("/")

    def complete(
        self, prompt: str, system: str | None = None, temperature: float = 0.0
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        if not texts:
            return []
        try:
            response = httpx.post(
                f"{self.base_url}/embeddings",
                json={"model": self.embedding_model, "input": texts},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()["data"]
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            return None
        return [item["embedding"] for item in data]


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(
        self, api_key: str, model: str = "claude-3-5-haiku-latest"
    ) -> None:
        self.api_key = api_key
        self.model = model

    def complete(
        self, prompt: str, system: str | None = None, temperature: float = 0.0
    ) -> str:
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(
        self, base_url: str = "http://localhost:11434", model: str = "llama3"
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def complete(
        self, prompt: str, system: str | None = None, temperature: float = 0.0
    ) -> str:
        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "system": system or "",
                "stream": False,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()["response"]

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        if not texts:
            return []
        try:
            response = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json().get("embeddings")
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            return None
        return data
