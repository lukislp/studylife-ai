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


class EnrichCaptureResponse(BaseModel):
    course_id: int | None
    course_confidence: float | None
    tags: list[str]
    summary: str | None
    related_note_ids: list[int]
