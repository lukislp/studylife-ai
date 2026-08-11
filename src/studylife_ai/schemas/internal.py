"""Request models for the internal StudyLife-to-studylife-ai registration endpoints."""

from pydantic import BaseModel


class RegisterKeyRequest(BaseModel):
    user_id: str
    ai_api_key: str


class RevokeKeyRequest(BaseModel):
    user_id: str
