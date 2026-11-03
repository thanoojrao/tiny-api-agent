"""Spec drift: corrupt a copy of the true ApiSpec before rendering.

The mock server always runs on the true spec. A transform makes one deliberate lie
in a copy, and that copy is what render.py turns into the docs the model reads.
The task, the golden path, the reward and the server stay on truth, so the only
way through a drifted episode is to read the error and repair.

Each transform is `(spec, rng, targets) -> (drifted_copy, entry) | None`:
  - it never mutates its input
  - it returns None when the spec has no place for that lie among `targets`
  - `entry` records what changed, for the run file and for held-out patterns

`apply_drift` composes up to N transforms chosen from a mix, aimed at a set of endpoints.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from .schema import ApiSpec, Endpoint, Field, Method

Entry = dict[str, str]
DriftFn = Callable[[ApiSpec, random.Random, list[str]], "tuple[ApiSpec, Entry] | None"]


@dataclass
class DriftRecord:
    applied: list[Entry] = field(default_factory=list)

    @property
    def kinds(self) -> list[str]:
        return [e["kind"] for e in self.applied]

    @property
    def endpoints(self) -> list[str]:
        return sorted({e["endpoint"] for e in self.applied if e.get("endpoint")})

    def __bool__(self) -> bool:
        return bool(self.applied)

    def to_json(self) -> dict[str, Any]:
        return {"applied": [dict(e) for e in self.applied]}


def no_drift(spec: ApiSpec) -> tuple[ApiSpec, DriftRecord]:
    return spec.model_copy(deep=True), DriftRecord()


# ----- helpers --------------------------------------------------------------

def _endpoints(spec: ApiSpec, targets: list[str]) -> list[Endpoint]:
    """Endpoints a transform may touch: the targets, or every endpoint when none are given."""
    wanted = set(targets) if targets else {e.id for e in spec.endpoints}
    return [e for e in spec.endpoints if e.id in wanted]


def _slots(ep: Endpoint) -> list[tuple[str, list[Field]]]:
    return [("query", ep.query), ("body", ep.body)]


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(p.capitalize() for p in rest)


def _entry(kind: str, ep: Endpoint | None, **kw: Any) -> Entry:
    return {"kind": kind, "endpoint": ep.id if ep else "", **{k: str(v) for k, v in kw.items()}}


_FIELD_SYNONYMS = {
    "id": "identifier", "name": "label", "title": "name", "author": "writer", "email": "mail",
    "status": "state", "member": "user", "limit": "max", "offset": "skip",
}


def _alias(name: str, rng: random.Random, taken: set[str]) -> str:
    """A plausible stale-doc alias for a field name, never colliding with a sibling."""
    options: list[str] = []
    if "_" in name:
        options.append(_camel(name))
    parts = name.split("_")
    for key, alt in _FIELD_SYNONYMS.items():
        if key in parts:
            options.append("_".join(alt if p == key else p for p in parts))
    options.append(name[:-3] if name.endswith("_id") else name + "_id")
    options = [o for o in options if o != name and o not in taken]
    return rng.choice(options) if options else name + "_v2"


# ----- transforms -----------------------------------------------------------

def rename_param(spec: ApiSpec, rng: random.Random, targets: list[str]) -> tuple[ApiSpec, Entry] | None:
    """Docs call a query or body field by another name. Server: unknown field."""
    spec = spec.model_copy(deep=True)
    slots = [(ep, where, f) for ep in _endpoints(spec, targets) for where, fs in _slots(ep) for f in fs]
    if not slots:
        return None
    ep, where, f = rng.choice(slots)
    taken = {g.name for _, fs in _slots(ep) for g in fs} | {g.name for g in ep.path_params}
    old, f.name = f.name, _alias(f.name, rng, taken)
    return spec, _entry("rename_param", ep, where=where, **{"from": old, "to": f.name})


def hide_param(spec: ApiSpec, rng: random.Random, targets: list[str]) -> tuple[ApiSpec, Entry] | None:
    """Docs omit a required field entirely. Server: missing required field."""
    spec = spec.model_copy(deep=True)
    slots = [(ep, where, fs, f) for ep in _endpoints(spec, targets) for where, fs in _slots(ep) for f in fs if f.required]
    if not slots:
        return None
    ep, where, fs, f = rng.choice(slots)
    fs.remove(f)
    return spec, _entry("hide_param", ep, where=where, name=f.name)


def unrequire_param(spec: ApiSpec, rng: random.Random, targets: list[str]) -> tuple[ApiSpec, Entry] | None:
    """Docs mark a required field optional. Server: missing required field, if the model skips it."""
    spec = spec.model_copy(deep=True)
    slots = [(ep, where, f) for ep in _endpoints(spec, targets) for where, fs in _slots(ep) for f in fs if f.required]
    if not slots:
        return None
    ep, where, f = rng.choice(slots)
    f.required = False
    return spec, _entry("unrequire_param", ep, where=where, name=f.name)


_PATH_ALIASES = {"checkout": "borrow", "return": "give-back", "cancel": "abort", "archive": "retire", "search": "find"}


def move_endpoint(spec: ApiSpec, rng: random.Random, targets: list[str]) -> tuple[ApiSpec, Entry] | None:
    """Docs show the endpoint at another path. Server: 404 no route."""
    spec = spec.model_copy(deep=True)
    eps = _endpoints(spec, targets)
    if not eps:
        return None
    ep = rng.choice(eps)
    segments = ep.path.strip("/").split("/")
    options: list[list[str]] = [["v2"] + segments, ["api"] + segments]
    for i, seg in enumerate(segments):
        if seg.startswith("{"):
            continue
        alt = _PATH_ALIASES.get(seg) or (seg[:-1] if seg.endswith("s") and len(seg) > 3 else seg + "s")
        options.append(segments[:i] + [alt] + segments[i + 1:])
    existing = {e.path for e in spec.endpoints}
    choices = [c for c in ("/" + "/".join(o) for o in options) if c not in existing]
    old, ep.path = ep.path, rng.choice(choices) if choices else ep.path + "/do"
    return spec, _entry("move_endpoint", ep, **{"from": old, "to": ep.path})


_METHOD_SWAPS: dict[str, list[Method]] = {
    "PATCH": ["PUT", "POST"], "PUT": ["PATCH", "POST"], "POST": ["PUT"], "DELETE": ["POST"], "GET": ["POST"],
}


def swap_method(spec: ApiSpec, rng: random.Random, targets: list[str]) -> tuple[ApiSpec, Entry] | None:
    """Docs list the wrong HTTP method. Server: 405 with the allowed methods."""
    spec = spec.model_copy(deep=True)
    candidates: list[tuple[Endpoint, list[Method]]] = []
    for ep in _endpoints(spec, targets):
        siblings = {e.method for e in spec.endpoints if e.path == ep.path}
        options = [m for m in _METHOD_SWAPS[ep.method] if m not in siblings]  # a swap onto a real sibling is not a lie
        if options:
            candidates.append((ep, options))
    if not candidates:
        return None
    ep, options = rng.choice(candidates)
    old, ep.method = ep.method, rng.choice(options)
    return spec, _entry("swap_method", ep, **{"from": old, "to": ep.method})


_ENUM_ALIASES = {"checked_out": "on_loan", "available": "in_stock", "active": "enabled", "inactive": "disabled",
                 "pending": "queued", "done": "completed", "open": "unresolved", "closed": "resolved"}


def rename_enum(spec: ApiSpec, rng: random.Random, targets: list[str]) -> tuple[ApiSpec, Entry] | None:
    """Docs list an enum value by another name. Server: allowed values listed."""
    spec = spec.model_copy(deep=True)
    slots = [(ep, where, f) for ep in _endpoints(spec, targets) for where, fs in _slots(ep) for f in fs if f.type == "enum" and f.enum]
    if not slots:
        return None
    ep, where, f = rng.choice(slots)
    old = rng.choice(f.enum or [])
    new = _ENUM_ALIASES.get(old) or (_camel(old) if "_" in old else old + "_v2")
    for g in (f, *(rf for r in spec.resources for rf in r.fields)):  # the docs' resource table lists the same enum
        if g.type == "enum" and g.enum and old in g.enum:
            g.enum[g.enum.index(old)] = new
    return spec, _entry("rename_enum", ep, where=where, field=f.name, **{"from": old, "to": new})


_AUTH_ALIASES = [("X-API-Key", ""), ("X-Auth-Token", ""), ("Authorization", "Token"), ("Api-Key", "")]


def auth_header(spec: ApiSpec, rng: random.Random, targets: list[str]) -> tuple[ApiSpec, Entry] | None:
    """Docs name a different auth header or scheme. Server: 401. Global, so `targets` is ignored."""
    if spec.auth is None:
        return None
    spec = spec.model_copy(deep=True)
    assert spec.auth is not None
    old = f"{spec.auth.header}: {spec.auth.scheme}".rstrip(": ")
    options = [(h, s) for h, s in _AUTH_ALIASES if (h, s) != (spec.auth.header, spec.auth.scheme)]
    spec.auth.header, spec.auth.scheme = rng.choice(options)
    new = f"{spec.auth.header}: {spec.auth.scheme}".rstrip(": ")
    return spec, _entry("auth_header", None, **{"from": old, "to": new})


TRANSFORMS: dict[str, DriftFn] = {
    "rename_param": rename_param,
    "hide_param": hide_param,
    "unrequire_param": unrequire_param,
    "move_endpoint": move_endpoint,
    "swap_method": swap_method,
    "rename_enum": rename_enum,
    "auth_header": auth_header,
}


def parse_kinds(text: str | None) -> list[str]:
    """`"rename_param,move_endpoint"` -> kinds. Empty, None or `all` means every kind. Unknown names raise."""
    if not text or text.strip().lower() == "all":
        return list(TRANSFORMS)
    kinds = [k.strip() for k in text.split(",") if k.strip()]
    unknown = [k for k in kinds if k not in TRANSFORMS]
    if unknown:
        raise ValueError(f"unknown drift kind(s) {unknown}; choose from {', '.join(TRANSFORMS)}")
    return list(dict.fromkeys(kinds))


def apply_drift(
    spec: ApiSpec,
    n: int,
    rng: random.Random,
    kinds: list[str] | None = None,
    targets: list[str] | None = None,
) -> tuple[ApiSpec, DriftRecord]:
    """Apply up to `n` lies, each a different kind, aimed at `targets` (endpoint ids; empty = any).

    Fewer than `n` land when the spec has no place for the remaining kinds; the record says how many did.
    The true spec is never touched.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    kinds = list(kinds) if kinds is not None else list(TRANSFORMS)
    targets = list(targets or [])
    out, record = no_drift(spec)
    order = list(kinds)
    rng.shuffle(order)
    for kind in order:
        if len(record.applied) >= n:
            break
        result = TRANSFORMS[kind](out, rng, targets)
        if result is None:
            continue
        out, entry = result
        record.applied.append(entry)
    return out, record


def golden_endpoints(spec: ApiSpec, task: dict[str, Any]) -> list[str]:
    """Endpoint ids a task's golden path touches, so drift can be aimed where it bites. Empty for no-fit tasks."""
    from .server import MockServer, Request, _Reject

    server = MockServer(spec, verbosity=0)
    ids: list[str] = []
    for g in task.get("golden") or []:
        try:
            ep, _ = server._route(Request.from_model_json(g))
        except _Reject:
            continue
        ids.append(ep.id)
    return sorted(set(ids))
