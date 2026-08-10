"""Request/response models for the /chat endpoint."""

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Request body for POST /chat.

    `messages` is the full conversation history, oldest first. `model`
    optionally overrides the server-configured default LiteLLM model
    identifier for this request.
    """

    messages: list[ChatMessage] = Field(min_length=1)
    model: str | None = None


class NoteSource(BaseModel):
    """One entry of the SSE `sources` event: a note that was retrieved for RAG context."""

    note_id: int
    title: str
    course_id: int | None = None
