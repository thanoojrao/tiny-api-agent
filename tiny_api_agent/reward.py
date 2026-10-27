"""Execution-grounded reward.

The expected outcome is not a reference trajectory. It is computed by replaying
the task's golden request(s) against a fresh server: the resulting state and the
last response body are what the model's episode is compared against.

A task with `golden: null` is a no-fit task: the API cannot do it, and the
correct behavior is `ok: false` with no successful mutating call.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .rollout import Episode
from .schema import ApiSpec
from .server import MockServer, Request

PER_EXTRA_REQUEST_PENALTY = 0.05
ENDPOINT_SHAPING = 0.2
FORMAT_SHAPING = 0.1


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
    extra_requests: int
    score: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_reward(spec: ApiSpec, task: dict[str, Any], episode: Episode, final_state: dict) -> Reward:
    exp = expected_outcome(spec, task)
    final = episode.final
    format_ok = isinstance(final, dict) and "ok" in final

    hit_ids = {ep_id for _, _, ep_id in episode.calls if ep_id}
    endpoint_hit = bool(set(exp.endpoint_ids) & hit_ids) if exp.endpoint_ids else False
    state_ok = final_state == exp.state

    if exp.no_fit:
        mutated = any(resp.ok and req.method != "GET" for req, resp, _ in episode.calls)
        data_ok = format_ok and final.get("ok") is False
        success = data_ok and not mutated and state_ok
        extra = max(0, episode.n_requests - 2)
    else:
        data_ok = format_ok and final.get("ok") is True and final.get("data") == exp.data
        success = state_ok and data_ok
        extra = max(0, episode.n_requests - exp.n_requests)

    if success:
        score = 1.0
    else:
        score = (ENDPOINT_SHAPING if endpoint_hit else 0.0) + (FORMAT_SHAPING if format_ok else 0.0)
    score -= PER_EXTRA_REQUEST_PENALTY * extra

    return Reward(
        success=success,
        format_ok=format_ok,
        endpoint_hit=endpoint_hit,
        state_ok=state_ok,
        data_ok=data_ok,
        n_requests=episode.n_requests,
        extra_requests=extra,
        score=round(score, 4),
    )
