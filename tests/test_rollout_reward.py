import json
from pathlib import Path

from tiny_api_agent.models import ScriptedModel
from tiny_api_agent.reward import compute_reward, expected_outcome
from tiny_api_agent.rollout import parse_action, run_episode
from tiny_api_agent.schema import ApiSpec
from tiny_api_agent.server import MockServer

SPEC = ApiSpec.load(Path(__file__).parent.parent / "examples" / "library.json")
TASKS = {t["id"]: t for t in json.loads((Path(__file__).parent.parent / "examples" / "library_tasks.json").read_text())["tasks"]}
H = {"Authorization": "Bearer lib_test_7f3a"}


def req(method, path, **kw):
    return "<request>" + json.dumps({"method": method, "path": path, "headers": H, **kw}) + "</request>"


def final(**kw):
    return "<final>" + json.dumps(kw) + "</final>"


def run(task_id, replies, verbosity=3):
    server = MockServer(SPEC, verbosity=verbosity)
    ep = run_episode(ScriptedModel(replies), SPEC, SPEC, TASKS[task_id], server)
    return ep, compute_reward(SPEC, TASKS[task_id], ep, server.snapshot())


def test_all_golden_requests_succeed():
    for task in TASKS.values():
        expected_outcome(SPEC, task)


def test_parse_action_tolerates_fences():
    kind, payload = parse_action('<request>```json\n{"method": "GET", "path": "/x"}\n```</request>')
    assert kind == "request" and payload["path"] == "/x"
    kind, payload = parse_action("hello")
    assert kind == "none"


def test_perfect_get():
    book = {"id": 1, "title": "Dune", "author": "Frank Herbert", "status": "available", "member_id": None}
    ep, r = run("get_book", [req("GET", "/books/1"), final(ok=True, data=book)])
    assert r.success and r.score == 1.0 and r.n_requests == 1


def test_wrong_data_fails_read_task():
    ep, r = run("get_book", [req("GET", "/books/1"), final(ok=True, data={"id": 1})])
    assert not r.success and r.endpoint_hit and r.state_ok and not r.data_ok


def test_repair_after_400_succeeds_with_penalty():
    done = {"id": 3, "title": "Snow Crash", "author": "Neal Stephenson", "status": "checked_out", "member_id": 2}
    ep, r = run("checkout", [
        req("POST", "/books/3/checkout", body={"memberId": 2}),
        req("POST", "/books/3/checkout", body={"member_id": 2}),
        final(ok=True, data=done),
    ])
    assert r.success and r.n_requests == 2 and r.extra_requests == 1
    assert r.score == 0.95
    assert ep.calls[0][1].status == 400 and ep.calls[1][1].status == 200


def test_no_fit_correct_when_ok_false_and_nothing_mutated():
    ep, r = run("no_fit_delete", [req("DELETE", "/books/2"), final(ok=False, reason="no delete endpoint")])
    assert r.success and r.score == 1.0


def test_no_fit_wrong_when_something_else_was_mutated():
    ep, r = run("no_fit_delete", [req("POST", "/books/2/return"), final(ok=False, reason="cannot")])
    assert not r.success


def test_giving_up_on_doable_task_scores_low():
    ep, r = run("checkout", [final(ok=False, reason="nope")])
    assert not r.success and r.format_ok and r.score == 0.1


def test_budget_ends_episode():
    replies = [req("GET", "/books/1")] * 10
    ep, r = run("get_book", replies)
    assert ep.n_requests == 5 and ep.final is None and not r.success


def test_multi_call_task_data_is_last_response():
    done = {"id": 3, "title": "Snow Crash", "author": "Neal Stephenson", "status": "checked_out", "member_id": 2}
    ep, r = run("checkout_by_name", [
        req("GET", "/books", query={"title": "Snow Crash"}),
        req("GET", "/members", query={"name": "Bob Okafor"}),
        req("POST", "/books/3/checkout", body={"member_id": 2}),
        final(ok=True, data=done),
    ])
    assert r.success and r.extra_requests == 0


def test_server_turns_are_separable_for_loss_masking():
    ep, _ = run("get_book", [req("GET", "/books/1"), final(ok=True, data={})])
    roles = [t.role for t in ep.turns]
    assert roles == ["model", "server", "model"]
