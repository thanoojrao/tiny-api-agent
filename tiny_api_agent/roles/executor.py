"""Run one executor episode: prompt the model, parse its turn, hit the mock
server, feed the responses back, repeat until a final answer or the turn budget.

The executor contract:
- Input: a task instruction, optional known facts, credentials, one API's docs.
- Each model turn: one to MAX_REQUESTS_PER_TURN <request> blocks (independent
  calls, run in order, all answered in one server turn), or exactly one <final>
  block. Mixing the two, a truncated block, or too many blocks is a parse failure.
- Budget counts turns that contain requests, not requests. Batching costs no
  turns; each request beyond the golden path still costs score in the reward.

The transcript keeps model-generated turns separate from server-generated
turns so a trainer can mask the server's text out of the loss later.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..models.base import Model
from ..env.render import render_markdown
from ..env.schema import ApiSpec
from ..env.server import MockServer, Request, Response

SYSTEM_PROMPT = """You are an API executor. You receive a task and the documentation for one API. Make HTTP requests to complete the task, then return a result.

Rules:
- Each turn, reply with one or more <request> blocks, or exactly one <final> block, and nothing else.
- A <request> block contains JSON:
  <request>{{"method": "POST", "path": "/things/1/do", "query": {{}}, "headers": {{}}, "body": {{}}}}</request>
  Omit keys you do not need. Paths are relative to the base URL.
- Put several <request> blocks in one turn only when they do not depend on each other, at most {max_requests} per turn. They run in order and you receive every response in the next message, each in a <response status="..." n="..."> block. Never write a <response> block yourself, and never put a <final> block in the same turn as a <request> block: send the request, stop, and wait for the server.
- When the task is complete, reply with a <final> block containing JSON:
  <final>{{"ok": true, "data": <the body of the response that completed the task>}}</final>
  or, if the task cannot be done with this API:
  <final>{{"ok": false, "reason": "<short reason>"}}</final>
- Return ok:false only when the API genuinely cannot do the task. Do not give up because a request failed; read the error and fix the request.
- You may take at most {budget} turns of requests."""

# A block runs from the LAST opener before its closer, so a stray "<request>" in prose
# before a real block does not swallow it.
_REQUEST_RE = re.compile(r"<request>((?:(?!<request>).)*?)</request>", re.DOTALL)
_FINAL_RE = re.compile(r"<final>((?:(?!<final>).)*?)</final>", re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

MAX_REQUESTS_PER_TURN = 8
DEFAULT_BUDGET = 5  # turns of requests; the one default every caller shares


def system_prompt(budget: int) -> str:
    """The one place the system prompt is filled in, so every caller states the same limits."""
    return SYSTEM_PROMPT.format(budget=budget, max_requests=MAX_REQUESTS_PER_TURN)


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
    budget: int = DEFAULT_BUDGET  # turns that contain requests
    n_turns: int = 0  # model turns that contained at least one request
    stop_reason: str | None = None  # "final" | "budget" | "parse"
    system: str = ""  # the exact prompts the model saw, so a run file can rebuild the training input
    prompt: str = ""

    @property
    def n_requests(self) -> int:
        return len(self.calls)

    @property
    def had_failure(self) -> bool:
        """At least one request was rejected. The repair-only metric is computed over these episodes."""
        return any(not resp.ok for _, resp, _ in self.calls)

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
            "n_turns": self.n_turns,
            "budget": self.budget,
            "stop_reason": self.stop_reason,
            "system": self.system,
            "prompt": self.prompt,
        }


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def _parse_json_block(raw: str) -> dict[str, Any] | None:
    """Model text is untrusted: anything json.loads can raise on (huge ints, deep nesting,
    NaN/Infinity) is 'not valid JSON', never an exception out of the loop."""
    cleaned = _FENCE_RE.sub("", raw.strip()).strip()
    try:
        value = json.loads(cleaned, parse_constant=_reject_constant)
        json.dumps(value)  # what we accept must also be writable to the run file (same recursion ceiling)
    except (ValueError, RecursionError):  # JSONDecodeError is a ValueError
        return None
    return value if isinstance(value, dict) else None


def parse_turn(text: str) -> tuple[str, Any]:
    """Return ("final", dict) | ("requests", [dict, ...]) | ("none", reason).

    The whole turn is unparseable (and the model is told why) when: a <final> and
    a <request> appear together, any block is not valid JSON, a block is opened
    but never closed (a truncated reply), or there are more than
    MAX_REQUESTS_PER_TURN request blocks. A truncated reply must not execute its
    earlier blocks, or the model would act on a belief the transcript contradicts.
    """
    text = text or ""
    finals = _FINAL_RE.findall(text)
    raws = _REQUEST_RE.findall(text)
    # Truncation shows as an opener after the last closed block. Openers inside a closed block
    # (a title echoed from a response) or in prose before the blocks are not truncation.
    last_end = max((m.end() for m in _FINAL_RE.finditer(text)), default=0)
    last_end = max(last_end, max((m.end() for m in _REQUEST_RE.finditer(text)), default=0))
    tail = text[last_end:]
    if "<final>" in tail or "<request>" in tail:
        return "none", "a block was opened but never closed; the reply looks truncated"
    if "<response" in text:
        return "none", "you wrote a <response> block yourself; only the server writes those. Send your <request> blocks and wait for the real responses"
    if finals and raws:
        return "none", "a <final> block and a <request> block in the same turn"
    if len(finals) > 1:
        return "none", "more than one <final> block"
    if finals:
        payload = _parse_json_block(finals[0])
        return ("final", payload) if payload is not None else ("none", "the <final> block is not valid JSON")
    if not raws:
        return "none", "no <request> or <final> block"
    if len(raws) > MAX_REQUESTS_PER_TURN:
        return "none", f"{len(raws)} <request> blocks; the limit is {MAX_REQUESTS_PER_TURN} per turn"
    payloads = [_parse_json_block(r) for r in raws]
    if any(p is None for p in payloads):
        return "none", "a <request> block is not valid JSON"
    return "requests", payloads


def build_user_prompt(
    spec_for_model: ApiSpec,
    task: dict[str, Any],
    auth_token: str | None,
    facts: dict[str, Any] | None = None,
) -> str:
    """Task, then known facts (from an orchestrator), then credentials, then docs."""
    parts = ["# Task", "", task["instruction"], ""]
    facts = facts if facts is not None else task.get("facts")
    if facts is not None and not isinstance(facts, dict):
        raise ValueError(f"task {task.get('id')!r}: facts must be a JSON object, got {type(facts).__name__}")
    if facts:
        # Every value is JSON-encoded, strings included: a fact comes from an orchestrator (another model)
        # and must not be able to open a new section of this prompt.
        parts += ["# Known", ""] + [f"- {json.dumps(str(k))}: {json.dumps(v)}" for k, v in facts.items()] + [""]
    if auth_token:
        parts += ["# Credentials", "", f"API token: `{auth_token}`", ""]
    parts += ["# API documentation", "", render_markdown(spec_for_model)]
    return "\n".join(parts)


def _format_responses(resps: list[Response]) -> str:
    out = []
    for i, resp in enumerate(resps, start=1):
        body = "" if resp.body is None else json.dumps(resp.body)
        out.append(f'<response status="{resp.status}" n="{i}">{body}</response>')
    return "\n".join(out)


def run_episode(
    model: Model,
    true_spec: ApiSpec,
    spec_for_model: ApiSpec,
    task: dict[str, Any],
    server: MockServer,
    budget: int = DEFAULT_BUDGET,
    max_parse_failures: int = 2,
    facts: dict[str, Any] | None = None,
) -> Episode:
    """Run one task to completion. The caller owns `server` (reset it per episode).

    `facts` is what an orchestrator relays into the Known block; it overrides
    `task["facts"]` when given (an explicit `{}` suppresses the task's facts).
    """
    if budget < 1:
        raise ValueError(f"budget must be at least 1 turn, got {budget}")
    system = system_prompt(budget)
    token = true_spec.auth.token if true_spec.auth else None
    user = build_user_prompt(spec_for_model, task, token, facts=facts)
    messages: list[dict[str, str]] = [{"role": "user", "content": user}]
    ep = Episode(task=task, budget=budget, system=system, prompt=user)
    budget_warned = False

    def say(feedback: str) -> None:
        ep.turns.append(Turn("server", feedback))
        messages.append({"role": "user", "content": feedback})

    while True:
        text = model.complete(system, messages) or ""
        ep.turns.append(Turn("model", text))
        # The API rejects whitespace-only assistant content, so keep the raw text only in the transcript.
        messages.append({"role": "assistant", "content": text if text.strip() else "(empty)"})

        kind, payload = parse_turn(text)

        if kind == "final":
            ep.final = payload
            ep.stop_reason = "final"
            return ep

        if kind == "requests":
            if ep.n_turns >= ep.budget:
                if budget_warned:
                    ep.stop_reason = "budget"
                    return ep
                budget_warned = True
                say(f"Turn budget of {ep.budget} exhausted. Reply with a <final> block.")
                continue
            ep.n_turns += 1
            resps: list[Response] = []
            for p in payload:
                req = Request.from_model_json(p)
                resps.append(server.handle(req))
                ep.calls.append(server.log[-1])
            say(_format_responses(resps))
            continue

        ep.parse_failures += 1
        if ep.parse_failures > max_parse_failures:
            ep.stop_reason = "parse"
            return ep
        reason = payload or "could not parse your reply"
        if budget_warned or ep.n_turns >= ep.budget:
            say(f"Could not parse your reply ({reason}). The turn budget is exhausted: reply with one <final> block containing valid JSON.")
        else:
            say(f"Could not parse your reply ({reason}). Reply with one to {MAX_REQUESTS_PER_TURN} <request> blocks, "
                "or one <final> block, each containing valid JSON.")
