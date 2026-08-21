"""Request/response models for the internal StudyLife-to-studylife-ai endpoints."""

from pydantic import BaseModel


class RegisterKeyRequest(BaseModel):
    user_id: str
    ai_api_key: str


class RevokeKeyRequest(BaseModel):
    user_id: str


class EnrichCaptureRequest(BaseModel):
    user_id: str
    # The real StudyLife note id - used as the Qdrant entity_id for immediate ingestion
    # (rag/enrichment.py's _ingest_note) and to exclude this note from its own related-notes
    # search (_find_related_notes), plus log correlation.
    note_id: int
    title: str
    content: str
    source_url: str | None = None
    # UserSettingsDto.SelectedCourseIds from StudyLife's side - scopes course-matching to the
    # user's currently-active courses (see rag/enrichment.py's _match_course docstring). Default
    # empty list, not a required field, so an older StudyLife.Server build that hasn't been
    # updated to send this yet degrades to "no active courses to match against" rather than a
    # validation error - matches the general never-raises philosophy of this whole feature.
    active_course_ids: list[int] = []


class EnrichCaptureResponse(BaseModel):
    course_id: int | None
    course_confidence: float | None
    tags: list[str]
    summary: str | None
    related_note_ids: list[int]
