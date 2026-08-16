"""Pydantic models mirroring StudyLife's REST API DTOs.

Field names follow StudyLife's `NoteDto` (`StudyLife.Shared/Dtos.cs`); the
API serializes camelCase JSON (ASP.NET Core's default for
`AddControllersWithViews`), so a camelCase alias generator maps that onto
Python's snake_case convention.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class _StudyLifeDto(BaseModel):
    """Shared config for all StudyLife API DTOs: the API serializes camelCase
    JSON (ASP.NET Core's default for `AddControllersWithViews`), so a camelCase
    alias generator maps that onto Python's snake_case convention. One shared
    base instead of repeating this on every DTO, so a future tweak (e.g.
    `str_strip_whitespace`) can't accidentally land on only some of them."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class StudyLifeNote(_StudyLifeDto):
    """A note as returned by `GET /api/notes`."""

    id: int
    title: str
    content: str
    created_at: datetime
    updated_at: datetime
    course_id: int | None = None
    session_id: int | None = None
    is_markdown: bool = False


class CourseDto(_StudyLifeDto):
    """A course as returned by `GET /api/courses`. No date fields."""

    id: int
    semester: int = 1
    name: str = ""
    code: str = ""
    topics: list[str] = []
    ects: int = 5
    group: str | None = None


class StudySessionDto(_StudyLifeDto):
    """A study session (calendar entry) as returned by `GET /api/sessions/history`.

    `start_time`/`end_time` are naive local timestamps, not UTC (same quirk
    as `StudyLifeNote.updated_at` — see docs/decisions.md).
    """

    id: int
    course_id: int
    course_name: str = ""
    start_time: datetime
    end_time: datetime
    topic: str | None = None
    notes: str | None = None
    is_completed: bool = False


class CourseGoalDto(_StudyLifeDto):
    """A course goal as returned by `GET /api/coursegoals`.

    No own `id` field — `course_id` is its natural unique key (one goal per
    course, per the API's own design).
    """

    course_id: int
    course_name: str = ""
    target_date: datetime | None = None
    completion_note: str | None = None
    completed_at: datetime | None = None
    grade: float | None = None
    completed_topics: str = ""
    tag: str | None = None
