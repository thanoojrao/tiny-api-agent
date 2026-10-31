from pathlib import Path

import pytest

from tiny_api_agent.env.schema import ApiSpec
from tiny_api_agent.env.server import MockServer, Request

SPEC = ApiSpec.load(Path(__file__).parent.parent / "examples" / "library.json")
AUTH = {"Authorization": "Bearer lib_test_7f3a"}


def srv(verbosity=3):
    return MockServer(SPEC, verbosity=verbosity)


def test_unknown_route_404():
    r = srv().handle(Request("GET", "/book/1", headers=AUTH))
    assert r.status == 404


def test_wrong_method_405_lists_allowed():
    r = srv().handle(Request("DELETE", "/books/1", headers=AUTH))
    assert r.status == 405
    assert "GET" in r.body["details"][0]["hint"]


def test_missing_auth_401():
    r = srv().handle(Request("GET", "/books/1"))
    assert r.status == 401


def test_base_path_is_stripped():
    r = srv().handle(Request("GET", "/v1/books/1", headers=AUTH))
    assert r.status == 200 and r.body["title"] == "Dune"


def test_full_url_in_path_is_accepted():
    req = Request.from_model_json({"method": "GET", "path": "https://library.example.com/v1/books?status=available", "headers": AUTH})
    r = srv().handle(req)
    assert r.status == 200 and r.body["total"] == 2


def test_unknown_body_field_400_with_hint_at_verbosity_3():
    r = srv(3).handle(Request("POST", "/books/3/checkout", headers=AUTH, body={"memberId": 2}))
    assert r.status == 400
    messages = [d["message"] for d in r.body["details"]]
    assert any("unknown body field 'memberId'" in m for m in messages)
    assert any("missing required body field 'member_id'" in m for m in messages)
    hints = [d.get("hint", "") for d in r.body["details"]]
    assert any("member_id" in h for h in hints)


@pytest.mark.parametrize("verbosity,expect", [(0, None), (1, {"error": "Bad Request"})])
def test_low_verbosity_hides_details(verbosity, expect):
    r = srv(verbosity).handle(Request("POST", "/books/3/checkout", headers=AUTH, body={"memberId": 2}))
    assert r.status == 400 and r.body == expect


def test_verbosity_2_has_messages_without_hints():
    r = srv(2).handle(Request("POST", "/books/3/checkout", headers=AUTH, body={}))
    assert r.body["details"] == ["missing required body field 'member_id'"]


def test_create_assigns_next_id_and_defaults():
    s = srv()
    r = s.handle(Request("POST", "/books", headers=AUTH, body={"title": "Hyperion", "author": "Dan Simmons"}))
    assert r.status == 201
    assert r.body == {"id": 4, "title": "Hyperion", "author": "Dan Simmons", "status": "available", "member_id": None}
    assert s.state["books"][4]["title"] == "Hyperion"


def test_failed_create_does_not_consume_id():
    s = srv()
    s.handle(Request("POST", "/books", headers=AUTH, body={"title": "x"}))
    r = s.handle(Request("POST", "/books", headers=AUTH, body={"title": "x", "author": "y"}))
    assert r.body["id"] == 4


def test_action_precondition_409():
    r = srv().handle(Request("POST", "/books/2/checkout", headers=AUTH, body={"member_id": 2}))
    assert r.status == 409
    assert "status" in r.body["details"][0]["message"]


def test_action_applies_sets():
    s = srv()
    r = s.handle(Request("POST", "/books/3/checkout", headers=AUTH, body={"member_id": 2}))
    assert r.status == 200
    assert s.state["books"][3]["status"] == "checked_out"
    assert s.state["books"][3]["member_id"] == 2
    r = s.handle(Request("POST", "/books/3/return", headers=AUTH))
    assert r.status == 200 and s.state["books"][3]["member_id"] is None


def test_query_type_coercion_and_enum():
    s = srv()
    r = s.handle(Request("GET", "/books", headers=AUTH, query={"limit": "1"}))
    assert r.status == 200 and len(r.body["items"]) == 1 and r.body["total"] == 3
    r = s.handle(Request("GET", "/books", headers=AUTH, query={"status": "lost"}))
    assert r.status == 400


def test_negative_paging_is_rejected():
    r = srv().handle(Request("GET", "/books", headers=AUTH, query={"offset": "-1"}))
    assert r.status == 400 and "offset" in r.body["details"][0]["message"]
    r = srv().handle(Request("GET", "/books", headers=AUTH, query={"limit": "-1"}))
    assert r.status == 400


def test_filter_equality_is_not_fooled_by_none_or_booleans():
    from tiny_api_agent.env.server import _eq
    assert _eq(2, "2") and _eq(None, None)
    assert not _eq(None, "None") and not _eq(1, True) and not _eq("1", True)


def test_path_param_type_checked():
    r = srv().handle(Request("GET", "/books/abc", headers=AUTH))
    assert r.status == 400


def test_reset_restores_seed():
    s = srv()
    s.handle(Request("POST", "/books", headers=AUTH, body={"title": "a", "author": "b"}))
    s.reset()
    assert 4 not in s.state["books"] and s.log == []
