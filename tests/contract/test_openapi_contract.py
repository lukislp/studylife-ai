"""Contract tests against StudyLife's committed OpenAPI spec (`docs/api/openapi.json` in the
main `studylife` repo) - audit finding D2/D3: this repo hand-mirrors StudyLife's DTOs
(`studylife/models.py`) and hardcodes endpoint knowledge (`studylife/client.py`) with zero
drift detection. These tests fetch that spec and assert:

1. every StudyLife endpoint `studylife/client.py` calls still exists in the spec (method +
   path), and
2. every field each Pydantic DTO mirror here declares still exists on the matching spec
   component schema, with a separate, more severe check for fields the model *requires*.

A server-side rename/removal of an endpoint or field currently fails silently on this repo's
side - `client.py` would 404, or `model_validate()` would either raise (a REQUIRED field
vanishes) or just silently drop the data (`extra="ignore"`, an OPTIONAL field renamed). This
module turns that into a loud CI failure here, at the point of drift, instead of a live-traffic
surprise.

Spec source (`STUDYLIFE_OPENAPI_SPEC`): a local file path or an http(s) URL. Defaults to the
studylife repo's own raw GitHub URL - there is no package/artifact distribution channel for the
spec (see docs/decisions.md "No Swagger/OpenAPI in the StudyLife repo", now resolved upstream by
the spec's addition at docs/api/openapi.json).

Reachability: in CI, an unreachable/unparsable spec FAILS this module outright - skipping would
defeat the entire point of a contract test. Locally, only the fully-default case (no env var
set, AND the default URL happens to be unreachable - e.g. no internet) is treated as "offline
dev" and skipped with a clear message; an explicitly configured `STUDYLIFE_OPENAPI_SPEC` that
turns out to be unreachable is still a real error and fails, since the developer pointed at it
on purpose.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

from studylife_ai.studylife.models import (
    CourseDto,
    CourseGoalDto,
    StudyLifeNote,
    StudySessionDto,
)

DEFAULT_SPEC_URL = "https://raw.githubusercontent.com/lukislp/studylife/main/docs/api/openapi.json"

# Every StudyLife endpoint studylife_ai/studylife/client.py calls, read exhaustively from that
# file (its only `self._client.get(...)`/`self._client.post(...)` call sites - grep confirms no
# other module in this repo talks to the StudyLife API directly). None of these are actually
# path-templated today (no `/api/sessions/{id}`-style calls exist yet), but `_path_regex()`
# below still handles `{param}` segments structurally for whenever one is added.
CALLED_ENDPOINTS: list[tuple[str, str]] = [
    ("GET", "/api/notes"),  # get_notes()
    ("GET", "/api/courses"),  # get_courses()
    ("GET", "/api/sessions/history"),  # get_sessions_history()
    ("GET", "/api/coursegoals"),  # get_course_goals()
    ("POST", "/api/sessions"),  # create_session()
    ("POST", "/api/notes"),  # create_note()
]

# Model -> the server-side C# DTO name it mirrors (StudyLife.Shared/Dtos.cs), i.e. the OpenAPI
# component schema name to check it against. Confirmed against docs/decisions.md, which
# references these DTOs by these exact names (e.g. "no `UpdatedAt` on `StudySessionDto`").
# `StudyLifeNote` is the one mismatch - the Python class is named for local clarity, not after
# the wire DTO, which is `NoteDto`.
MODEL_TO_SCHEMA: dict[type, str] = {
    StudyLifeNote: "NoteDto",
    CourseDto: "CourseDto",
    StudySessionDto: "StudySessionDto",
    CourseGoalDto: "CourseGoalDto",
}


def _is_ci() -> bool:
    return os.environ.get("CI", "").strip().lower() == "true"


def _fetch_spec(source: str) -> dict[str, Any]:
    if source.startswith(("http://", "https://")):
        response = httpx.get(source, timeout=15.0, follow_redirects=True)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data
    parsed: dict[str, Any] = json.loads(Path(source).read_text(encoding="utf-8"))
    return parsed


@pytest.fixture(scope="session")
def spec() -> dict[str, Any]:
    """Session-scoped so the spec is fetched/parsed once for the whole contract run, not once
    per test. A `pytest.skip`/`pytest.fail` raised in here applies to every test that depends on
    this fixture, so the reachability policy lives in exactly one place."""
    configured = os.environ.get("STUDYLIFE_OPENAPI_SPEC")
    explicit = configured is not None
    source = configured if explicit else DEFAULT_SPEC_URL
    try:
        return _fetch_spec(source)
    except Exception as exc:
        if _is_ci():
            pytest.fail(
                f"Could not load the StudyLife OpenAPI spec from {source!r} in CI - contract "
                f"tests MUST fail here, not skip (that's the point of the check): {exc!r}"
            )
        if explicit:
            pytest.fail(
                f"STUDYLIFE_OPENAPI_SPEC={source!r} was set explicitly but could not be "
                f"loaded - this is a real configuration error, not the offline-dev fallback "
                f"below: {exc!r}"
            )
        pytest.skip(
            f"Skipping contract tests: STUDYLIFE_OPENAPI_SPEC is unset and the default spec "
            f"URL ({source!r}) is unreachable - assuming offline local dev. Set "
            f"STUDYLIFE_OPENAPI_SPEC to a local docs/api/openapi.json path to run these tests "
            f"offline. Original error: {exc!r}"
        )


def _path_regex(template: str) -> re.Pattern[str]:
    """Turns an OpenAPI path template like `/api/sessions/{id}` into a regex matching any
    concrete path with that shape, e.g. `/api/sessions/42`. `{param}` segments never contain
    `/`, so `[^/]+` is an exact match for one path segment, not just an approximation."""
    marker = "\x00"
    placeholder_free = re.sub(r"\{[^/}]+\}", marker, template)
    pattern = re.escape(placeholder_free).replace(marker, "[^/]+")
    return re.compile(f"^{pattern}$")


def _find_matching_path_template(spec_paths: dict[str, Any], concrete_path: str) -> str | None:
    """An exact literal match (e.g. `/api/sessions/history`) always wins over a templated one
    (e.g. `/api/sessions/{id}`, whose `[^/]+` would otherwise also match the literal segment
    "history") - the same static-beats-parameterized precedence ASP.NET Core's own router
    uses. Only falls back to templated matching once no literal path is present at all."""
    if concrete_path in spec_paths:
        return concrete_path
    for template in spec_paths:
        if _path_regex(template).match(concrete_path):
            return template
    return None


@pytest.mark.parametrize(
    "method,path",
    CALLED_ENDPOINTS,
    ids=[f"{method}_{path}" for method, path in CALLED_ENDPOINTS],
)
def test_called_endpoint_exists_in_spec(spec: dict[str, Any], method: str, path: str) -> None:
    spec_paths = spec.get("paths", {})
    template = _find_matching_path_template(spec_paths, path)
    assert template is not None, (
        f"{method} {path} is called by studylife_ai/studylife/client.py but no matching path "
        f"exists in the StudyLife OpenAPI spec's `paths` - the endpoint was removed or renamed "
        f"server-side (or client.py has a typo). Spec paths: {sorted(spec_paths)}"
    )
    methods_at_path = {m.lower() for m in spec_paths[template]}
    assert method.lower() in methods_at_path, (
        f"{method} {path} matched spec path template {template!r}, but that path does not "
        f"support {method} in the spec - only {sorted(methods_at_path)} do. The endpoint's "
        f"HTTP method changed server-side."
    )


def _resolve_schema(spec: dict[str, Any], schema_name: str) -> dict[str, Any]:
    schemas = spec.get("components", {}).get("schemas", {})
    assert schema_name in schemas, (
        f"Component schema {schema_name!r} not found in the StudyLife OpenAPI spec's "
        f"components/schemas - the DTO was renamed or removed server-side. Available schemas: "
        f"{sorted(schemas)}"
    )
    schema: dict[str, Any] = schemas[schema_name]
    return schema


def _schema_properties(schema: dict[str, Any]) -> dict[str, Any]:
    """Merges in any `allOf` branches (NSwag/Swashbuckle sometimes emits inheritance this way)
    so property lookups below see the full, flattened set - not just whatever's declared
    directly on this schema object."""
    properties: dict[str, Any] = dict(schema.get("properties", {}))
    for branch in schema.get("allOf", []):
        properties.update(branch.get("properties", {}))
    return properties


def _model_field_cases() -> list[tuple[type, str]]:
    return [(model, field_name) for model in MODEL_TO_SCHEMA for field_name in model.model_fields]


def _required_model_field_cases() -> list[tuple[type, str]]:
    return [
        (model, field_name)
        for model, field_name in _model_field_cases()
        if model.model_fields[field_name].is_required()
    ]


def _case_id(model_and_field: tuple[type, str]) -> str:
    model, field_name = model_and_field
    return f"{model.__name__}.{field_name}"


@pytest.mark.parametrize(
    "model,field_name",
    _required_model_field_cases(),
    ids=[_case_id(c) for c in _required_model_field_cases()],
)
def test_required_model_field_exists_in_spec_schema(
    spec: dict[str, Any], model: type, field_name: str
) -> None:
    """(a) Every field a DTO mirror REQUIRES (no default) must be present in the spec - if the
    server ever omits a required field, `Model.model_validate()` raises `pydantic.ValidationError`
    at runtime (get_notes()/get_courses()/etc. would start hard-failing on every synced item)."""
    schema_name = MODEL_TO_SCHEMA[model]
    schema = _resolve_schema(spec, schema_name)
    properties = _schema_properties(schema)
    field_info = model.model_fields[field_name]
    wire_name = field_info.alias or field_name

    assert wire_name in properties, (
        f"{model.__name__}.{field_name} (wire name {wire_name!r}) is REQUIRED on the Python "
        f"model (no default) but is missing from the spec's {schema_name!r} schema properties "
        f"- if the server ever omits it, model_validate() raises pydantic.ValidationError at "
        f"runtime. Spec {schema_name!r} properties: {sorted(properties)}"
    )


@pytest.mark.parametrize(
    "model,field_name",
    _model_field_cases(),
    ids=[_case_id(c) for c in _model_field_cases()],
)
def test_declared_model_field_exists_in_spec_schema(
    spec: dict[str, Any], model: type, field_name: str
) -> None:
    """(b) Every field a DTO mirror declares at all (required or optional) must be present in
    the spec - catches a server-side rename/removal of an OPTIONAL field too, which would
    otherwise fail silently (pydantic's `extra="ignore"` default just drops the unmatched wire
    data, no error) instead of failing loudly here."""
    schema_name = MODEL_TO_SCHEMA[model]
    schema = _resolve_schema(spec, schema_name)
    properties = _schema_properties(schema)
    field_info = model.model_fields[field_name]
    wire_name = field_info.alias or field_name

    assert wire_name in properties, (
        f"{model.__name__}.{field_name} (wire name {wire_name!r}) has no matching property in "
        f"the spec's {schema_name!r} schema - the server renamed or removed this field. Spec "
        f"{schema_name!r} properties: {sorted(properties)}"
    )
