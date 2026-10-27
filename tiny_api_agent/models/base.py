from __future__ import annotations

from typing import Protocol


class Model(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict[str, str]]) -> str:
        """Return the assistant's next message as plain text."""
        ...


class ScriptedModel:
    """Replays a fixed list of replies. For tests and dry runs."""

    name = "scripted"

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, system: str, messages: list[dict[str, str]]) -> str:
        self.calls.append(list(messages))
        if not self._replies:
            return ""
        return self._replies.pop(0)
