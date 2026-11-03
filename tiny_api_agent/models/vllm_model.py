"""A model served by vLLM through its OpenAI-compatible chat endpoint.

This is the baseline and training-time path: the same bf16 checkpoint the
trainer updates is the one being evaluated. Point it at any server that
speaks `/v1/chat/completions`.
"""

from __future__ import annotations

import os

import httpx


class VLLMModel:
    def __init__(
        self,
        model: str,
        host: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        client: httpx.Client | None = None,
    ):
        self.name = model
        self._host = (host or os.environ.get("VLLM_HOST") or "http://localhost:8000").rstrip("/")
        self._temperature = temperature
        # A small model with no cap can loop until the server's context limit. One executor
        # turn is a few request blocks or a final answer, so 1024 tokens is generous.
        self._max_tokens = max_tokens
        # One client for the whole run keeps the connection open across turns and episodes.
        # Tests inject a client with a mock transport instead of a real server.
        self._client = client or httpx.Client(base_url=self._host, timeout=600)

    def complete(self, system: str, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.name,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        r = self._client.post("/v1/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""
