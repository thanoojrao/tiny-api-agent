import json
from pathlib import Path

from tiny_api_agent.models import ScriptedModel
from tiny_api_agent.reward.executor import compute_reward, expected_outcome
from tiny_api_agent.roles.executor import build_user_prompt, parse_turn, run_episode
from tiny_api_agent.env.schema import ApiSpec
from tiny_api_agent.env.server import MockServer

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


def test_parse_turn_tolerates_fences():
    kind, payload = parse_turn('<request>```json\n{"method": "GET", "path": "/x"}\n```</request>')
    assert kind == "requests" and payload[0]["path"] == "/x"
    kind, payload = parse_turn("hello")
    assert kind == "none"


def test_parse_turn_several_requests_and_bad_block():
    kind, payload = parse_turn(req("GET", "/a") + "\n" + req("GET", "/b"))
    assert kind == "requests" and [p["path"] for p in payload] == ["/a", "/b"]
    kind, payload = parse_turn(req("GET", "/a") + "<request>not json</request>")
    assert kind == "none"
    kind, payload = parse_turn(req("GET", "/a") + final(ok=True, data={}))
    assert kind == "none"  # a final and a request in one turn is a parse failure


def test_batched_turn_runs_in_order_and_answers_in_one_server_turn():
    done = {"id": 3, "title": "Snow Crash", "author": "Neal Stephenson", "status": "checked_out", "member_id": 2}
    ep, r = run("checkout_by_name", [
        req("GET", "/books", query={"title": "Snow Crash"}) + "\n" + req("GET", "/members", query={"name": "Bob Okafor"}),
        req("POST", "/books/3/checkout", body={"member_id": 2}),
        final(ok=True, data=done),
    ])
    assert r.success and r.n_requests == 3 and r.n_turns == 2 and r.extra_requests == 0
    roles = [t.role for t in ep.turns]
    assert roles == ["model", "server", "model", "server", "model"]
    assert ep.turns[1].text.count("<response") == 2 and 'n="2"' in ep.turns[1].text
    assert 'status="200" n="1"' in ep.turns[3].text  # single responses carry n too, as the prompt promises


def test_known_facts_render_in_prompt():
    task = {**TASKS["checkout"], "facts": {"member_email": "bob@example.com", "due_days": 14}}
    prompt = build_user_prompt(SPEC, task, "tok")
    assert "# Known" in prompt and '- "member_email": "bob@example.com"' in prompt and '- "due_days": 14' in prompt
    assert prompt.index("# Task") < prompt.index("# Known") < prompt.index("# Credentials") < prompt.index("# API documentation")
    assert "# Known" not in build_user_prompt(SPEC, TASKS["checkout"], "tok")


def test_perfect_get():
    book = {"id": 1, "title": "Dune", "author": "Frank Herbert", "status": "available", "member_id": None}
    ep, r = run("get_book", [req("GET", "/books/1"), final(ok=True, data=book)])
    assert r.success and r.score == 1.0 and r.n_requests == 1 and r.n_turns == 1 and not r.had_failure


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
    assert r.had_failure and r.n_turns == 2
    assert ep.calls[0][1].status == 400 and ep.calls[1][1].status == 200


def test_no_fit_correct_when_ok_false_and_nothing_mutated():
    ep, r = run("no_fit_delete", [req("DELETE", "/books/2"), final(ok=False, reason="no delete endpoint")])
    assert r.success and r.score == 1.0
    # the probe was rejected, but a no-fit task is never a repair candidate
    assert r.had_failure and r.no_fit and not r.repair_candidate


def test_repair_candidate_is_rejected_request_on_doable_task():
    done = {"id": 3, "title": "Snow Crash", "author": "Neal Stephenson", "status": "checked_out", "member_id": 2}
    ep, r = run("checkout", [req("POST", "/books/3/checkout", body={"memberId": 2}),
                             req("POST", "/books/3/checkout", body={"member_id": 2}), final(ok=True, data=done)])
    assert r.repair_candidate and r.success and ep.stop_reason == "final"
    ep, r = run("checkout", [req("POST", "/books/3/checkout", body={"member_id": 2}), final(ok=True, data=done)])
    assert not r.repair_candidate


def test_none_and_whitespace_replies_do_not_crash():
    ep, _ = run("get_book", [None, "   \n", final(ok=False, reason="x")])
    assert ep.parse_failures == 2 and ep.final is not None


def test_non_dict_facts_rejected():
    import pytest
    with pytest.raises(ValueError):
        build_user_prompt(SPEC, {**TASKS["checkout"], "facts": ["a"]}, "tok")


def test_no_fit_wrong_when_something_else_was_mutated():
    ep, r = run("no_fit_delete", [req("POST", "/books/2/return"), final(ok=False, reason="cannot")])
    assert not r.success


def test_giving_up_on_doable_task_scores_low():
    ep, r = run("checkout", [final(ok=False, reason="nope")])
    assert not r.success and r.format_ok and r.score == 0.1


def test_budget_counts_turns_not_requests():
    # loop states themselves are pinned in tests/test_executor_loop.py; this pins the reward side
    replies = [req("GET", "/books/1") + req("GET", "/books/2")] * 10
    ep, r = run("get_book", replies)
    assert ep.n_turns == 5 and ep.n_requests == 10 and not r.had_failure and not r.success


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
