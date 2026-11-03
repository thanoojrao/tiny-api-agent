"""Drift transforms: each lie changes the docs, leaves the server on truth, and earns the expected rejection."""

import json
import random
from pathlib import Path

import pytest

from tiny_api_agent.env.drift import (
    TRANSFORMS, DriftRecord, apply_drift, auth_header, golden_endpoints, hide_param, move_endpoint, no_drift,
    parse_kinds, rename_enum, rename_param, swap_method, unrequire_param,
)
from tiny_api_agent.env.render import render_markdown
from tiny_api_agent.env.schema import ApiSpec
from tiny_api_agent.env.server import MockServer, Request

ROOT = Path(__file__).parent.parent
SPEC = ApiSpec.load(ROOT / "examples" / "library.json")
TASKS = {t["id"]: t for t in json.loads((ROOT / "examples" / "library_tasks.json").read_text())["tasks"]}
H = {"Authorization": "Bearer lib_test_7f3a"}


def rng(seed=1):
    return random.Random(seed)


def handle(method, path, **kw):
    server = MockServer(SPEC, verbosity=3)
    return server.handle(Request.from_model_json({"method": method, "path": path, "headers": H, **kw}))


# ---- every kind: docs change, truth does not ----

@pytest.mark.parametrize("kind", list(TRANSFORMS))
def test_each_kind_changes_the_docs_and_not_the_true_spec(kind):
    before = render_markdown(SPEC)
    result = TRANSFORMS[kind](SPEC, rng(), [])
    assert result is not None
    drifted, entry = result
    assert entry["kind"] == kind
    assert render_markdown(drifted) != before
    assert SPEC == ApiSpec.load(ROOT / "examples" / "library.json")  # never mutated
    assert MockServer(drifted, verbosity=3)  # the lie is still a valid spec, so it can be rendered


def test_no_drift_is_a_deep_copy_with_an_empty_record():
    copy, record = no_drift(SPEC)
    assert copy == SPEC and copy is not SPEC and not record and record.to_json() == {"applied": []}


# ---- each kind returns None when there is no place for the lie ----

def test_kinds_return_none_without_a_target():
    assert rename_param(SPEC, rng(), ["get_book"]) is None  # no query or body
    assert hide_param(SPEC, rng(), ["get_book", "list_books"]) is None  # nothing required
    assert unrequire_param(SPEC, rng(), ["list_books"]) is None
    assert rename_enum(SPEC, rng(), ["create_book"]) is None  # no enum field
    assert move_endpoint(SPEC, rng(), ["no_such_endpoint"]) is None
    assert swap_method(SPEC, rng(), ["no_such_endpoint"]) is None
    assert auth_header(SPEC.model_copy(update={"auth": None}), rng(), []) is None


def test_swap_method_never_lands_on_a_real_sibling():
    # /books/{book_id} has GET and PATCH; a PATCH->PUT or PATCH->POST lie is fine, but never PATCH->GET
    for seed in range(20):
        drifted, entry = swap_method(SPEC, rng(seed), ["update_book"])
        siblings = {e.method for e in SPEC.endpoints if e.path == SPEC.endpoint("update_book").path}
        assert entry["to"] not in siblings and drifted.endpoint("update_book").method == entry["to"]


# ---- following the lie earns the expected rejection from the real server ----

def test_rename_param_lie_is_an_unknown_field():
    drifted, entry = rename_param(SPEC, rng(), ["checkout_book"])
    assert entry["from"] == "member_id" and entry["to"] != "member_id"
    assert drifted.endpoint("checkout_book").body[0].name == entry["to"]
    resp = handle("POST", "/books/3/checkout", body={entry["to"]: 2})
    assert resp.status == 400 and f"unknown body field '{entry['to']}'" in json.dumps(resp.body)


def test_hide_param_lie_is_a_missing_required_field():
    drifted, entry = hide_param(SPEC, rng(), ["create_book"])
    documented = [f.name for f in drifted.endpoint("create_book").body]
    assert entry["name"] not in documented and len(documented) == 1
    resp = handle("POST", "/books", body={documented[0]: "x"})
    assert resp.status == 400 and f"missing required body field '{entry['name']}'" in json.dumps(resp.body)


def test_unrequire_param_lie_bites_only_when_the_field_is_omitted():
    drifted, entry = unrequire_param(SPEC, rng(), ["checkout_book"])
    assert drifted.endpoint("checkout_book").body[0].required is False and entry["name"] == "member_id"
    assert handle("POST", "/books/3/checkout", body={}).status == 400
    assert handle("POST", "/books/3/checkout", body={"member_id": 2}).status == 200


def test_move_endpoint_lie_is_a_404_and_never_an_existing_path():
    for seed in range(20):
        drifted, entry = move_endpoint(SPEC, rng(seed), ["get_book"])
        assert entry["to"] != entry["from"] and entry["to"] not in {e.path for e in SPEC.endpoints}
        assert "{book_id}" in entry["to"]  # the placeholder survives, so the drifted spec still validates
    resp = handle("GET", entry["to"].replace("{book_id}", "1"))
    assert resp.status == 404


def test_swap_method_lie_is_a_405_naming_the_real_methods():
    drifted, entry = swap_method(SPEC, rng(), ["update_book"])
    resp = handle(entry["to"], "/books/1", body={"title": "x"})
    assert resp.status == 405 and "PATCH" in json.dumps(resp.body)


def test_rename_enum_lie_is_consistent_in_docs_and_rejected_by_the_server():
    drifted, entry = rename_enum(SPEC, rng(), ["list_books"])
    enum_lines = [line for line in render_markdown(drifted).splitlines() if "enum: " in line]
    assert len(enum_lines) == 2  # the endpoint table and the resource table
    assert all(entry["to"] in line and entry["from"] not in line for line in enum_lines)  # both lie, consistently
    resp = handle("GET", "/books", query={"status": entry["to"]})
    assert resp.status == 400 and "allowed values: available, checked_out" in json.dumps(resp.body)


def test_auth_header_lie_is_a_401():
    drifted, entry = auth_header(SPEC, rng(), [])
    assert entry["endpoint"] == "" and drifted.auth.header != "Authorization" or drifted.auth.scheme != "Bearer"
    header = {drifted.auth.header: f"{drifted.auth.scheme} lib_test_7f3a".strip()}
    server = MockServer(SPEC, verbosity=3)
    resp = server.handle(Request.from_model_json({"method": "GET", "path": "/books/1", "headers": header}))
    assert resp.status == 401


# ---- the driver ----

def test_driver_applies_n_distinct_kinds_aimed_at_targets():
    drifted, record = apply_drift(SPEC, 3, rng(7), targets=["checkout_book"])
    assert len(record.applied) == 3 and len(set(record.kinds)) == 3
    assert set(record.endpoints) <= {"checkout_book"}  # auth_header has no endpoint; everything else is aimed
    assert render_markdown(drifted) != render_markdown(SPEC)


def test_driver_zero_is_identity_and_too_many_stops_at_what_fits():
    copy, record = apply_drift(SPEC, 0, rng())
    assert copy == SPEC and not record
    _, record = apply_drift(SPEC, 100, rng(), targets=["get_book"])
    assert set(record.kinds) <= {"move_endpoint", "swap_method", "auth_header"} and len(record.kinds) == len(set(record.kinds))
    with pytest.raises(ValueError):
        apply_drift(SPEC, -1, rng())


def test_driver_respects_the_kinds_list_and_the_seed():
    _, a = apply_drift(SPEC, 2, rng(3), kinds=["rename_param", "move_endpoint"], targets=["checkout_book"])
    _, b = apply_drift(SPEC, 2, rng(3), kinds=["rename_param", "move_endpoint"], targets=["checkout_book"])
    assert a.applied == b.applied and set(a.kinds) == {"rename_param", "move_endpoint"}
    _, none = apply_drift(SPEC, 1, rng(), kinds=["rename_enum"], targets=["get_book"])
    assert not none  # held-out kinds that do not fit the target simply do not land


def test_parse_kinds():
    assert parse_kinds(None) == parse_kinds("") == parse_kinds("all") == list(TRANSFORMS)
    assert parse_kinds(" move_endpoint, rename_param ,move_endpoint") == ["move_endpoint", "rename_param"]
    with pytest.raises(ValueError, match="unknown drift kind"):
        parse_kinds("rename_param,nope")


def test_golden_endpoints_follow_the_task_and_are_empty_for_no_fit():
    assert golden_endpoints(SPEC, TASKS["checkout"]) == ["checkout_book"]
    assert golden_endpoints(SPEC, TASKS["checkout_by_name"]) == ["checkout_book", "list_books", "list_members"]
    assert golden_endpoints(SPEC, TASKS["no_fit_delete"]) == []
    assert golden_endpoints(SPEC, {"golden": [{"method": "GET", "path": "/nowhere"}]}) == []


def test_record_json_round_trips():
    _, record = apply_drift(SPEC, 2, rng(5), targets=["create_book"])
    again = DriftRecord(**json.loads(json.dumps(record.to_json())))
    assert again.applied == record.applied and again.kinds == record.kinds
