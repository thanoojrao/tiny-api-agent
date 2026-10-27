"""True API schema. This is what the mock server enforces.

The model never sees this directly; it sees a rendered (and possibly drifted)
document produced from it. See render.py and drift.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field as PField, model_validator

FieldType = Literal["string", "integer", "number", "boolean", "enum"]
Method = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
OpKind = Literal["create", "get", "list", "update", "delete", "action"]


class Field(BaseModel):
    name: str
    type: FieldType = "string"
    required: bool = False
    enum: list[str] | None = None
    default: Any = None
    description: str = ""

    @model_validator(mode="after")
    def _check_enum(self) -> "Field":
        if self.type == "enum" and not self.enum:
            raise ValueError(f"field {self.name!r} is enum but has no values")
        return self


class Resource(BaseModel):
    name: str
    fields: list[Field]
    id_field: str = "id"

    def field(self, name: str) -> Field | None:
        return next((f for f in self.fields if f.name == name), None)


class Operation(BaseModel):
    """How the server executes an endpoint against its state."""

    kind: OpKind
    resource: str
    id_param: str | None = None
    # list: which query fields act as equality filters
    filters: list[str] = PField(default_factory=list)
    # action: field -> "$param_name" | "null" | literal
    sets: dict[str, Any] = PField(default_factory=dict)
    # action: required current field values, else 409
    precondition: dict[str, Any] = PField(default_factory=dict)


class Endpoint(BaseModel):
    id: str
    method: Method
    path: str
    description: str = ""
    path_params: list[Field] = PField(default_factory=list)
    query: list[Field] = PField(default_factory=list)
    body: list[Field] = PField(default_factory=list)
    operation: Operation

    @property
    def mutating(self) -> bool:
        return self.method != "GET"

    @model_validator(mode="after")
    def _check_path_params(self) -> "Endpoint":
        import re

        in_path = set(re.findall(r"\{(\w+)\}", self.path))
        declared = {f.name for f in self.path_params}
        if in_path != declared:
            raise ValueError(
                f"endpoint {self.id}: path params {sorted(in_path)} != declared {sorted(declared)}"
            )
        for f in self.path_params:
            f.required = True
        return self


class Auth(BaseModel):
    header: str = "Authorization"
    scheme: str = "Bearer"
    token: str


class ApiSpec(BaseModel):
    name: str
    description: str = ""
    base_url: str = "https://api.example.com"
    auth: Auth | None = None
    resources: list[Resource]
    endpoints: list[Endpoint]
    # initial records per resource
    seed: dict[str, list[dict[str, Any]]] = PField(default_factory=dict)

    def resource(self, name: str) -> Resource:
        for r in self.resources:
            if r.name == name:
                return r
        raise KeyError(name)

    def endpoint(self, id: str) -> Endpoint:
        for e in self.endpoints:
            if e.id == id:
                return e
        raise KeyError(id)

    @model_validator(mode="after")
    def _check_refs(self) -> "ApiSpec":
        names = {r.name for r in self.resources}
        for e in self.endpoints:
            if e.operation.resource not in names:
                raise ValueError(f"endpoint {e.id} references unknown resource {e.operation.resource}")
        for name in self.seed:
            if name not in names:
                raise ValueError(f"seed references unknown resource {name}")
        return self

    @classmethod
    def load(cls, path: str | Path) -> "ApiSpec":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
