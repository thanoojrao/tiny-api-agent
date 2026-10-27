"""A local model through Ollama's chat endpoint. CPU-only is fine for one
episode at a time; this is for checking the loop, not for training."""

from __future__ import annotations

import os

import httpx


class OllamaModel:
    def __init__(self, model: str = "qwen2.5:3b", host: str | None = None, temperature: float = 0.0):
        self.name = model
        self._host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self._temperature = temperature

    def complete(self, system: str, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.name,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
            "options": {"temperature": self._temperature},
        }
        r = httpx.post(f"{self._host}/api/chat", json=payload, timeout=600)
        r.raise_for_status()
        return r.json()["message"]["content"]
