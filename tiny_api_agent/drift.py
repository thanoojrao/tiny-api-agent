"""Spec drift: corrupt a copy of the true ApiSpec before rendering.

Milestone 1 ships only the identity transform so the rest of the loop can be
built against clean specs. Each drift type will be a small function
ApiSpec -> (ApiSpec, DriftRecord) so the rate and mix are controllable and
whole patterns can be held out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import ApiSpec


@dataclass
class DriftRecord:
    applied: list[dict[str, str]] = field(default_factory=list)


def no_drift(spec: ApiSpec) -> tuple[ApiSpec, DriftRecord]:
    return spec.model_copy(deep=True), DriftRecord()
