"""Claude as the executor. Used as a frontier baseline and as a stand-in while the
small model is not trained yet.

Credentials come from the environment (ANTHROPIC_API_KEY or an `ant auth login`
profile); nothing is hardcoded here.
"""

from __future__ import annotations

import anthropic

DEFAULT_MODEL = "claude-opus-5-5"


class AnthropicModel:
    def __init__(
        self,
        model: str | None = None,
        effort: str = "low",
        max_tokens: int = 8000,
    ):
        self.name = model or DEFAULT_MODEL
        self.effort = effort
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    def complete(self, system: str, messages: list[dict[str, str]]) -> str:
        response = self._client.beta.messages.create(
            model=self.name,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            return ""
        return "".join(block.text for block in response.content if block.type == "text")
