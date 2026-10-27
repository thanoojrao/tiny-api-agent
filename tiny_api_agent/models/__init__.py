"""Model adapters. Every adapter exposes `complete(system, messages) -> str`."""

from __future__ import annotations

from .base import Model, ScriptedModel


def load_model(name: str, **kwargs) -> Model:
    """Build a model from a short name like `anthropic:claude-haiku-5-5` or `ollama:qwen2.5:3b`."""
    backend, _, model_id = name.partition(":")
    if backend == "anthropic":
        from .anthropic_model import AnthropicModel

        return AnthropicModel(model_id or None, **kwargs)
    if backend == "ollama":
        from .ollama_model import OllamaModel

        return OllamaModel(model_id or "qwen2.5:3b", **kwargs)
    raise ValueError(f"unknown model backend {backend!r}; use anthropic:<id> or ollama:<name>")


__all__ = ["Model", "ScriptedModel", "load_model"]
