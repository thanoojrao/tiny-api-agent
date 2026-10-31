"""Generic mock server driven by an ApiSpec.

No network. The rollout harness calls `MockServer.handle` directly, which keeps
rollouts fast and deterministic. Wrap it in FastAPI later if a real HTTP
surface is ever needed.

Error verbosity (the second difficulty axis):
  0  status code only, empty body
  1  {"error": "<status text>"}
  2  + "details": list of plain messages ("missing required field 'member_id'")
  3  + hints on each detail (allowed fields, enum values, expected types)
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qsl, urlparse

from .schema import ApiSpec, Endpoint, Field

STATUS_TEXT = {
    200: "OK",
    201: "Created",
    204: "No Content",
    400: "Bad Request",
    401: "Unauthorized",
    404: "Not Found",
    405: "Method Not Allowed",
    409: "Conflict",
}


@dataclass
class Request:
    method: str
    path: str
    query: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    body: dict[str, Any] | None = None

    @classmethod
    def from_model_json(cls, d: dict[str, Any]) -> "Request":
        """Build a request from the loose JSON a model emits.

        Accepts a full URL or a path with a query string, and tolerates
        missing optional keys.
        """
        raw_path = str(d.get("path") or d.get("url") or "")
        try:
            parsed = urlparse(raw_path)
            path = parsed.path or "/"
            query = dict(parse_qsl(parsed.query)) if parsed.query else {}
        except ValueError:
            # e.g. "http://[bad": a model's malformed URL must become a routing miss, not an exception
            path, query = raw_path or "/", {}
        extra_q = d.get("query") or d.get("params") or {}
        if isinstance(extra_q, dict):
            query.update(extra_q)
        body = d.get("body")
        if body is not None and not isinstance(body, dict):
            body = {"_raw": body}
        headers = d.get("headers") or {}
        if not isinstance(headers, dict):
            headers = {}
        return cls(
            method=str(d.get("method", "GET")).upper(),
            path=path,
            query=query,
            headers={str(k): str(v) for k, v in headers.items()},
            body=body,
        )


@dataclass
class Response:
    status: int
    body: Any = None

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


@dataclass
class Detail:
    message: str
    hint: str = ""


class _Reject(Exception):
    def __init__(self, status: int, details: list[Detail] | None = None):
        self.status = status
        self.details = details or []


class MockServer:
    def __init__(self, spec: ApiSpec, verbosity: int = 3):
        if not 0 <= verbosity <= 3:
            raise ValueError("verbosity must be 0..3")
        self.spec = spec
        self.verbosity = verbosity
        self._routes = [(e, self._compile(e.path)) for e in spec.endpoints]
        self.reset()

    # ----- state -----------------------------------------------------------

    def reset(self) -> None:
        self.state: dict[str, dict[Any, dict[str, Any]]] = {}
        self._next_id: dict[str, int] = {}
        self.log: list[tuple[Request, Response, str | None]] = []
        for r in self.spec.resources:
            records = deepcopy(self.spec.seed.get(r.name, []))
            self.state[r.name] = {rec[r.id_field]: rec for rec in records}
            ints = [rec[r.id_field] for rec in records if isinstance(rec[r.id_field], int)]
            self._next_id[r.name] = (max(ints) + 1) if ints else 1

    def snapshot(self) -> dict[str, dict[Any, dict[str, Any]]]:
        return deepcopy(self.state)

    def base_path(self) -> str:
        return urlparse(self.spec.base_url).path.rstrip("/")

    # ----- dispatch --------------------------------------------------------

    def handle(self, req: Request) -> Response:
        endpoint_id: str | None = None
        try:
            endpoint, path_params = self._route(req)
            endpoint_id = endpoint.id
            self._check_auth(req)
            params = self._validate(endpoint, req, path_params)
            resp = self._execute(endpoint, params)
        except _Reject as rej:
            resp = Response(rej.status, self._error_body(rej))
        self.log.append((req, resp, endpoint_id))
        return resp

    def _route(self, req: Request) -> tuple[Endpoint, dict[str, str]]:
        path = req.path
        base = self.base_path()
        if base and path.startswith(base):
            path = path[len(base):] or "/"
        if len(path) > 1:
            path = path.rstrip("/")
        allowed: list[str] = []
        for endpoint, rx in self._routes:
            m = rx.match(path)
            if not m:
                continue
            if endpoint.method == req.method:
                return endpoint, m.groupdict()
            allowed.append(endpoint.method)
        if allowed:
            raise _Reject(405, [Detail(
                f"method {req.method} not allowed for {path}",
                f"allowed: {', '.join(sorted(set(allowed)))}",
            )])
        raise _Reject(404, [Detail(f"no route matches {req.method} {path}")])

    def _check_auth(self, req: Request) -> None:
        auth = self.spec.auth
        if auth is None:
            return
        value = next((v for k, v in req.headers.items() if k.lower() == auth.header.lower()), None)
        expected = f"{auth.scheme} {auth.token}".strip()
        if value is None:
            raise _Reject(401, [Detail(
                f"missing {auth.header} header",
                f"expected '{auth.header}: {auth.scheme} <token>'",
            )])
        if value.strip() != expected:
            raise _Reject(401, [Detail("invalid credentials")])

    # ----- validation ------------------------------------------------------

    def _validate(self, ep: Endpoint, req: Request, path_params: dict[str, str]) -> dict[str, Any]:
        details: list[Detail] = []
        params: dict[str, Any] = {}

        params.update(self._validate_group("path", ep.path_params, path_params, details, coerce=True))
        params.update(self._validate_group("query", ep.query, req.query, details, coerce=True))

        body = req.body or {}
        if ep.body or body:
            params.update(self._validate_group("body", ep.body, body, details, coerce=False))

        if details:
            raise _Reject(400, details)
        return params

    def _validate_group(
        self,
        where: str,
        fields: list[Field],
        given: dict[str, Any],
        details: list[Detail],
        coerce: bool,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        by_name = {f.name: f for f in fields}
        allowed = ", ".join(f.name for f in fields) or "(none)"

        for name in given:
            if name not in by_name:
                details.append(Detail(
                    f"unknown {where} field '{name}'",
                    f"allowed {where} fields: {allowed}",
                ))
        for f in fields:
            if f.name not in given:
                if f.required:
                    enum_hint = f"; allowed: {', '.join(f.enum)}" if f.enum else ""
                    details.append(Detail(
                        f"missing required {where} field '{f.name}'",
                        f"'{f.name}' is {f.type}{enum_hint}",
                    ))
                continue
            value = given[f.name]
            ok, converted, why = self._check_type(f, value, coerce)
            if not ok:
                details.append(Detail(f"invalid value for {where} field '{f.name}'", why))
                continue
            out[f.name] = converted
        return out

    @staticmethod
    def _check_type(f: Field, value: Any, coerce: bool) -> tuple[bool, Any, str]:
        t = f.type
        try:
            if t == "integer":
                if isinstance(value, bool):
                    raise ValueError
                if coerce and isinstance(value, str):
                    value = int(value)
                if not isinstance(value, int):
                    raise ValueError
                return True, value, ""
            if t == "number":
                if isinstance(value, bool):
                    raise ValueError
                if coerce and isinstance(value, str):
                    value = float(value)
                if not isinstance(value, (int, float)):
                    raise ValueError
                return True, value, ""
            if t == "boolean":
                if coerce and isinstance(value, str):
                    if value.lower() in ("true", "1"):
                        return True, True, ""
                    if value.lower() in ("false", "0"):
                        return True, False, ""
                    raise ValueError
                if not isinstance(value, bool):
                    raise ValueError
                return True, value, ""
            if t == "enum":
                if not isinstance(value, str) or value not in (f.enum or []):
                    return False, None, f"allowed values: {', '.join(f.enum or [])}"
                return True, value, ""
            # string
            if not isinstance(value, str):
                raise ValueError
            return True, value, ""
        except (ValueError, TypeError):
            return False, None, f"'{f.name}' must be {t}"

    # ----- execution -------------------------------------------------------

    def _execute(self, ep: Endpoint, params: dict[str, Any]) -> Response:
        op = ep.operation
        res = self.spec.resource(op.resource)
        table = self.state[res.name]

        def lookup() -> dict[str, Any]:
            key = params.get(op.id_param or "")
            rec = table.get(key)
            if rec is None:
                raise _Reject(404, [Detail(f"{res.name} {key!r} not found")])
            return rec

        if op.kind == "create":
            new_id = self._next_id[res.name]
            self._next_id[res.name] += 1
            rec = {res.id_field: new_id}
            for f in res.fields:
                if f.name == res.id_field:
                    continue
                rec[f.name] = params.get(f.name, deepcopy(f.default))
            table[new_id] = rec
            return Response(201, deepcopy(rec))

        if op.kind == "get":
            return Response(200, deepcopy(lookup()))

        if op.kind == "list":
            items = list(table.values())
            for name in op.filters:
                if name in params:
                    items = [r for r in items if _eq(r.get(name), params[name])]
            total = len(items)
            offset = int(params.get("offset", 0) or 0)
            limit = params.get("limit")
            paging = [(n, v) for n, v in (("offset", offset), ("limit", limit)) if v is not None and int(v) < 0]
            if paging:
                raise _Reject(400, [Detail(f"query parameter '{n}' must not be negative (got {v})") for n, v in paging])
            items = items[offset:]
            if limit is not None:
                items = items[: int(limit)]
            return Response(200, {"items": deepcopy(items), "total": total})

        if op.kind == "update":
            rec = lookup()
            for name, value in params.items():
                if name == op.id_param or name == res.id_field:  # the key must never drift from the record
                    continue
                if res.field(name) is not None:
                    rec[name] = value
            return Response(200, deepcopy(rec))

        if op.kind == "delete":
            rec = lookup()
            del table[rec[res.id_field]]
            return Response(204, None)

        if op.kind == "action":
            rec = lookup()
            for name, want in op.precondition.items():
                if not _eq(rec.get(name), want):
                    raise _Reject(409, [Detail(
                        f"{res.name} {rec[res.id_field]!r} has {name}={rec.get(name)!r}",
                        f"requires {name}={want!r}",
                    )])
            for name, value in op.sets.items():
                if isinstance(value, str) and value.startswith("$"):
                    rec[name] = params.get(value[1:])
                elif value == "null":
                    rec[name] = None
                else:
                    rec[name] = value
            return Response(200, deepcopy(rec))

        raise _Reject(500, [Detail(f"unsupported operation {op.kind}")])

    # ----- error rendering -------------------------------------------------

    def _error_body(self, rej: _Reject) -> Any:
        v = self.verbosity
        if v == 0:
            return None
        body: dict[str, Any] = {"error": STATUS_TEXT.get(rej.status, "Error")}
        if v == 1:
            return body
        if v == 2:
            body["details"] = [d.message for d in rej.details]
            return body
        body["details"] = [
            {"message": d.message, **({"hint": d.hint} if d.hint else {})} for d in rej.details
        ]
        return body

    @staticmethod
    def _compile(path: str) -> re.Pattern[str]:
        pattern = re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", path.rstrip("/") or "/")
        return re.compile(f"^{pattern}$")


def _eq(a: Any, b: Any) -> bool:
    """Loose equality for filters and preconditions: "2" matches 2, but None matches only None
    and booleans never match integers."""
    if a is None or b is None:
        return a is b
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if a == b:
        return True
    return str(a) == str(b)
