"""Request/response models for the /chat endpoint."""

from typing import Literal

from pydantic import BaseModel, Field

from studylife_ai.ingestion.qdrant_store import ContentType


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Request body for POST /chat.

    `messages` is the full conversation history, oldest first. `model`
    optionally overrides the server-configured default LiteLLM model
    identifier for this request - kept for forward compat (e.g. a future
    per-user model picker), even though no deployed caller sets it today.
    Audit F15: the server pays for whatever model gets named here, so a
    request naming anything outside `Settings.allowed_chat_models` (plus the
    always-implicitly-allowed `Settings.llm_model` itself) gets a 400 before
    any LLM call is made - see `api/chat.py`'s `_resolve_model`.
    """

    messages: list[ChatMessage] = Field(min_length=1)
    model: str | None = None


class Source(BaseModel):
    """One entry of the SSE `sources` event: an entity retrieved for RAG context."""

    content_type: ContentType
    entity_id: int
    title: str
    course_id: int | None = None
