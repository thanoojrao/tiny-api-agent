"""Render an ApiSpec into the documentation the model sees.

One renderer for now (markdown). The point of the project is that the spec
the model reads may be wrong; drift.py corrupts a copy of the ApiSpec before
it gets here, so this module stays honest and simple.
"""

from __future__ import annotations

from .schema import ApiSpec, Endpoint, Field


def _type_label(f: Field) -> str:
    if f.type == "enum":
        return "enum: " + ", ".join(f.enum or [])
    return f.type


def _fields_table(title: str, fields: list[Field], show_required: bool = True) -> str:
    if not fields:
        return ""
    if show_required:
        header = "| name | type | required | description |\n|---|---|---|---|"
        rows = "\n".join(
            f"| `{f.name}` | {_type_label(f)} | {'yes' if f.required else 'no'} | {f.description} |"
            for f in fields
        )
    else:
        header = "| name | type | description |\n|---|---|---|"
        rows = "\n".join(f"| `{f.name}` | {_type_label(f)} | {f.description} |" for f in fields)
    return f"**{title}**\n\n{header}\n{rows}\n"


def render_endpoint(e: Endpoint) -> str:
    parts = [f"### {e.method} {e.path}", ""]
    if e.description:
        parts += [e.description, ""]
    for title, fields in (
        ("Path parameters", e.path_params),
        ("Query parameters", e.query),
        ("JSON body", e.body),
    ):
        t = _fields_table(title, fields)
        if t:
            parts.append(t)
    return "\n".join(parts).rstrip() + "\n"


def render_markdown(spec: ApiSpec) -> str:
    parts = [f"# {spec.name}", ""]
    if spec.description:
        parts += [spec.description, ""]
    parts += [f"Base URL: `{spec.base_url}`", ""]
    if spec.auth:
        parts += [
            "## Authentication",
            "",
            f"Send `{spec.auth.header}: {spec.auth.scheme} <token>` on every request.",
            "",
        ]
    parts += ["## Resources", ""]
    for r in spec.resources:
        parts += [f"### {r.name}", "", _fields_table("Fields", r.fields, show_required=False)]
    parts += ["## Endpoints", ""]
    for e in spec.endpoints:
        parts.append(render_endpoint(e))
    return "\n".join(parts).rstrip() + "\n"
