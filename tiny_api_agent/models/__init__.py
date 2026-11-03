"""Model adapters. Every adapter exposes `complete(system, messages) -> str`."""

from __future__ import annotations

from .base import Model, ScriptedModel


def load_model(name: str, **kwargs) -> Model:
    """Build a model from a short name like `anthropic:claude-haiku-5-5`, `ollama:qwen2.5:3b` or `vllm:Qwen/Qwen2.5-3B-Instruct`."""
    backend, _, model_id = name.partition(":")
    if backend == "anthropic":
        from .anthropic_model import AnthropicModel

        return AnthropicModel(model_id or None, **kwargs)
    if backend == "ollama":
        from .ollama_model import OllamaModel

        return OllamaModel(model_id or "qwen2.5:3b", **kwargs)
    if backend == "vllm":
        from .vllm_model import VLLMModel

        if not model_id:
            raise ValueError("vllm needs a model id, e.g. vllm:Qwen/Qwen2.5-3B-Instruct")
        return VLLMModel(model_id, **kwargs)
    raise ValueError(f"unknown model backend {backend!r}; use anthropic:<id>, ollama:<name> or vllm:<hf-id>")


__all__ = ["Model", "ScriptedModel", "load_model"]
