"""Run one episode: prompt the model, parse its action, hit the mock server,
feed the response back, repeat until a final answer or the request budget.

The transcript keeps model-generated turns separate from server-generated
turns so a trainer can mask the server's text out of the loss later.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .models.base import Model
from .render import render_markdown
from .schema import ApiSpec
from .server import MockServer, Request, Response

SYSTEM_PROMPT = """You are an API executor. You receive a task and the documentation for one API. Make HTTP requests to complete the task, then return a result.

Rules:
- Reply with exactly one block per turn and nothing else.
- To make a request, reply with a <request> block containing JSON:
  <request>{{"method": "POST", "path": "/things/1/do", "query": {{}}, "headers": {{}}, "body": {{}}}}</request>
  Omit keys you do not need. Paths are relative to the base URL.
- After each request you receive a <response status="..."> block with the body.
- When the task is complete, reply with a <final> block containing JSON:
  <final>{{"ok": true, "data": <the body of the response that completed the task>}}</final>
  or, if the task cannot be done with this API:
  <final>{{"ok": false, "reason": "<short reason>"}}</final>
- Return ok:false only when the API genuinely cannot do the task. Do not give up because a request failed; read the error and fix the request.
- You may make at most {budget} requests."""

_REQUEST_RE = re.compile(r"<request>(.*?)</request>", re.DOTALL)
_FINAL_RE = re.compile(r"<final>(.*?)</final>", re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class Turn:
    role: str  # "model" | "server"
    text: str


@dataclass
class Episode:
    task: dict[str, Any]
    turns: list[Turn] = field(default_factory=list)
    calls: list[tuple[Request, Response, str | None]] = field(default_factory=list)
    final: dict[str, Any] | None = None
    parse_failures: int = 0
    budget: int = 5

    @property
    def n_requests(self) -> int:
        return len(self.calls)

    def to_json(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "turns": [{"role": t.role, "text": t.text} for t in self.turns],
            "calls": [
                {
                    "request": {"method": q.method, "path": q.path, "query": q.query, "body": q.body},
                    "status": r.status,
                    "endpoint": ep,
                }
                for q, r, ep in self.calls
            ],
            "final": self.final,
            "parse_failures": self.parse_failures,
        }


def _parse_json_block(raw: str) -> dict[str, Any] | None:
    cleaned = _FENCE_RE.sub("", raw.strip()).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def parse_action(text: str) -> tuple[str, dict[str, Any] | None]:
    """Return ("request", json) | ("final", json) | ("none", None)."""
    m = _FINAL_RE.search(text)
    if m:
        return "final", _parse_json_block(m.group(1))
    m = _REQUEST_RE.search(text)
    if m:
        return "request", _parse_json_block(m.group(1))
    return "none", None


def build_user_prompt(spec_for_model: ApiSpec, task: dict[str, Any], auth_token: str | None) -> str:
    parts = ["# Task", "", task["instruction"], ""]
    if auth_token:
        parts += ["# Credentials", "", f"API token: `{auth_token}`", ""]
    parts += ["# API documentation", "", render_markdown(spec_for_model)]
    return "\n".join(parts)


def _format_response(resp: Response) -> str:
    body = "" if resp.body is None else json.dumps(resp.body)
    return f'<response status="{resp.status}">{body}</response>'


def run_episode(
    model: Model,
    true_spec: ApiSpec,
    spec_for_model: ApiSpec,
    task: dict[str, Any],
    server: MockServer,
    budget: int = 5,
    max_parse_failures: int = 2,
) -> Episode:
    """Run one task to completion. The caller owns `server` (reset it per episode)."""
    system = SYSTEM_PROMPT.format(budget=budget)
    token = true_spec.auth.token if true_spec.auth else None
    user = build_user_prompt(spec_for_model, task, token)
    messages: list[dict[str, str]] = [{"role": "user", "content": user}]
    ep = Episode(task=task, budget=budget)

    while True:
        text = model.complete(system, messages)
        ep.turns.append(Turn("model", text))
        messages.append({"role": "assistant", "content": text or "(empty)"})

        kind, payload = parse_action(text)

        if kind == "final" and payload is not None:
            ep.final = payload
            return ep

        if kind == "request" and payload is not None:
            if ep.n_requests >= budget:
                feedback = f"Request budget of {budget} exhausted. Reply with a <final> block."
                ep.turns.append(Turn("server", feedback))
                messages.append({"role": "user", "content": feedback})
                if ep.n_requests >= budget + 1:
                    return ep
                continue
            req = Request.from_model_json(payload)
            resp = server.handle(req)
            ep.calls.append(server.log[-1])
            feedback = _format_response(resp)
            ep.turns.append(Turn("server", feedback))
            messages.append({"role": "user", "content": feedback})
            continue

        ep.parse_failures += 1
        if ep.parse_failures > max_parse_failures:
            return ep
        feedback = "Could not parse your reply. Reply with exactly one <request> or <final> block containing valid JSON."
        ep.turns.append(Turn("server", feedback))
        messages.append({"role": "user", "content": feedback})
