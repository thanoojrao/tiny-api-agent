"""Execution-grounded reward.

The expected outcome is not a reference trajectory. It is computed by replaying
the task's golden request(s) against a fresh server: the resulting state and the
last response body are what the model's episode is compared against.

A task with `golden: null` is a no-fit task: the API cannot do it, and the
correct behavior is `ok: false` with no successful mutating call.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from ..roles.executor import Episode
from ..env.schema import ApiSpec
from ..env.server import MockServer, Request

PER_EXTRA_REQUEST_PENALTY = 0.05
ENDPOINT_SHAPING = 0.2
FORMAT_SHAPING = 0.1
SCORE_FLOOR = -1.0  # a runaway batch must not dominate a GRPO group's normalisation
_MISSING = object()


def _canon(v: Any) -> Any:
    """Dict keys become strings so a table keyed by both 'b1' and 4 still compares."""
    if isinstance(v, dict):
        return {str(k): _canon(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_canon(x) for x in v]
    return v


def _same(a: Any, b: Any) -> bool:
    """JSON-strict equality: true != 1, missing != null, key order ignored."""
    if a is _MISSING or b is _MISSING:
        return False
    try:
        return json.dumps(_canon(a), sort_keys=True) == json.dumps(_canon(b), sort_keys=True)
    except (TypeError, ValueError, RecursionError):
        return False


@dataclass
class Expected:
    state: dict[str, dict[Any, dict[str, Any]]]
    data: Any
    endpoint_ids: list[str]
    n_requests: int
    no_fit: bool


def expected_outcome(spec: ApiSpec, task: dict[str, Any]) -> Expected:
    server = MockServer(spec, verbosity=3)
    golden = task.get("golden")
    if not golden:
        return Expected(server.snapshot(), None, [], 0, no_fit=True)
    headers = {}
    if spec.auth:
        headers[spec.auth.header] = f"{spec.auth.scheme} {spec.auth.token}".strip()
    last = None
    endpoint_ids: list[str] = []
    for g in golden:
        req = Request.from_model_json({**g, "headers": {**headers, **(g.get("headers") or {})}})
        last = server.handle(req)
        _, _, ep_id = server.log[-1]
        if not last.ok:
            raise ValueError(f"golden request for task {task.get('id')} failed: {last.status} {last.body}")
        if ep_id:
            endpoint_ids.append(ep_id)
    return Expected(server.snapshot(), last.body if last else None, endpoint_ids, len(golden), no_fit=False)


@dataclass
class Reward:
    success: bool
    format_ok: bool
    endpoint_hit: bool
    state_ok: bool
    data_ok: bool
    n_requests: int
    n_turns: int
    extra_requests: int
    had_failure: bool  # at least one request was rejected
    no_fit: bool  # the task has no golden path; a rejected probe here is the right move, not a repair
    score: float

    @property
    def repair_candidate(self) -> bool:
        """Episodes the repair-only metric is measured over: a request was rejected on a doable task."""
        return self.had_failure and not self.no_fit

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_reward(spec: ApiSpec, task: dict[str, Any], episode: Episode, final_state: dict) -> Reward:
    exp = expected_outcome(spec, task)
    final = episode.final
    format_ok = isinstance(final, dict) and "ok" in final

    hit_ids = {ep_id for _, _, ep_id in episode.calls if ep_id}
    endpoint_hit = bool(set(exp.endpoint_ids) & hit_ids) if exp.endpoint_ids else False
    state_ok = _same(final_state, exp.state)

    if exp.no_fit:
        mutated = any(resp.ok and req.method != "GET" for req, resp, _ in episode.calls)
        data_ok = format_ok and final.get("ok") is False
        success = data_ok and not mutated and state_ok
        extra = max(0, episode.n_requests - 2)
    else:
        data_ok = format_ok and final.get("ok") is True and _same(final.get("data", _MISSING), exp.data)
        success = state_ok and data_ok
        extra = max(0, episode.n_requests - exp.n_requests)

    if success:
        score = 1.0
    else:
        score = (ENDPOINT_SHAPING if endpoint_hit else 0.0) + (FORMAT_SHAPING if format_ok else 0.0)
    score = max(SCORE_FLOOR, score - PER_EXTRA_REQUEST_PENALTY * extra)

    return Reward(
        success=success,
        format_ok=format_ok,
        endpoint_hit=endpoint_hit,
        state_ok=state_ok,
        data_ok=data_ok,
        n_requests=episode.n_requests,
        n_turns=episode.n_turns,
        extra_requests=extra,
        had_failure=episode.had_failure,
        no_fit=exp.no_fit,
        score=round(score, 4),
    )
