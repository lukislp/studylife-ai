"""Request/response models for the /agent and /agent/confirm endpoints."""

from typing import Any, Literal

from pydantic import BaseModel


class AgentRequest(BaseModel):
    """Request body for POST /agent. Single-turn, like /chat - no
    conversation memory across unrelated turns, only within one
    propose-then-confirm pairing (see docs/decisions.md "M4 agent stack").

    No per-request `model` override (unlike `ChatRequest`): the agent graph
    is compiled once at startup with one bound model - swapping it per
    request would mean rebuilding the whole graph, not a cheap parameter.
    """

    message: str


class PendingAction(BaseModel):
    """A write tool the agent wants to run, awaiting confirmation.

    `thread_id` must be echoed back to POST /agent/confirm - it's what lets
    the paused LangGraph state be resumed, potentially in a separate process
    (the SQLite checkpointer survives a restart between propose and confirm).
    """

    tool: str
    args: dict[str, Any]
    description: str
    thread_id: str


class AgentResponse(BaseModel):
    """Response for both POST /agent and POST /agent/confirm.

    Exactly one of `answer`/`pending_actions` is set: a plain answer needs no
    confirmation; a non-empty `pending_actions` means nothing was executed
    yet. Usually one item - a list because the model can propose more than
    one write in a single turn (e.g. "create a session and save a note"),
    and `POST /agent/confirm` applies its single decision to all of them
    together (see docs/decisions.md "M4 agent stack") - there's no way to
    approve one and reject another in the same turn.
    """

    answer: str | None = None
    pending_actions: list[PendingAction] = []


class ConfirmRequest(BaseModel):
    """Request body for POST /agent/confirm.

    `decision` applies to every action in that turn's `pending_actions` -
    see `AgentResponse`.
    """

    thread_id: str
    decision: Literal["approve", "reject"]
    message: str | None = None
