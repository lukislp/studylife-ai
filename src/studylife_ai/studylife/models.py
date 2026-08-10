"""Pydantic models mirroring StudyLife's REST API DTOs.

Field names follow StudyLife's `NoteDto` (`StudyLife.Shared/Dtos.cs`); the
API serializes camelCase JSON (ASP.NET Core's default for
`AddControllersWithViews`), so a camelCase alias generator maps that onto
Python's snake_case convention.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class StudyLifeNote(BaseModel):
    """A note as returned by `GET /api/notes`."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    id: int
    title: str
    content: str
    created_at: datetime
    updated_at: datetime
    course_id: int | None = None
    session_id: int | None = None
