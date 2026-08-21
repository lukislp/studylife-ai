"""Request/response models for the internal StudyLife-to-studylife-ai endpoints."""

from pydantic import BaseModel


class RegisterKeyRequest(BaseModel):
    user_id: str
    ai_api_key: str


class RevokeKeyRequest(BaseModel):
    user_id: str


class EnrichCaptureRequest(BaseModel):
    user_id: str
    # Purely a log-correlation id (which note this enrichment run was for) - not used to look
    # anything up on this side, StudyLife's own background task already knows which note to
    # write the result back onto.
    note_id: int
    title: str
    content: str
    source_url: str | None = None


class EnrichCaptureResponse(BaseModel):
    course_id: int | None
    course_confidence: float | None
    tags: list[str]
    summary: str | None
