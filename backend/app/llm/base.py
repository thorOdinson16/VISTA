"""Pluggable LLM provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def complete(
        self, prompt: str, system: str | None = None, temperature: float = 0.0
    ) -> str:
        """Return the model's completion text ("" if unavailable)."""

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        """Return embeddings for ``texts``, or None when unsupported.

        Providers that have no embedding endpoint must return None rather
        than fabricating vectors, so callers can fall back honestly.
        """
        return None
