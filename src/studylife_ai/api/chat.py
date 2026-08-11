"""Chat endpoint: RAG-augmented, streams an LLM completion back via SSE.

Every request is retrieval-augmented (docs/decisions.md "Prompt design"):
the latest user message is used to fetch relevant note chunks, which are
injected as a system message ahead of the conversation. Streamed events:
`{"delta": "..."}` per token (or one `{"error": "..."}` if the LLM call
fails mid-stream), then always one `{"sources": [...]}` event listing the
notes that were actually retrieved (deduplicated, independent of whether
the model cited them or whether streaming succeeded), then `data: [DONE]`.

Requires the signed proxy token described in `api/identity.py` (see
docs/decisions.md "M4.5 Multi-user support") - `user_id` scopes retrieval to
the calling user's own Qdrant partition. Unlike `/agent`, /chat never calls
StudyLife's own API (pure RAG over already-ingested Qdrant data), so it
needs no `StudyLifeClient` and no `STUDYLIFE_API_BASE_URL` - identity is
fully verified locally via the token signature.
"""

import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from studylife_ai.api.identity import ResolvedIdentity, resolve_identity
from studylife_ai.api.rate_limit import enforce_rate_limit
from studylife_ai.config import Settings, get_settings
from studylife_ai.ingestion.qdrant_store import QdrantStore, RetrievedChunk
from studylife_ai.llm.client import stream_chat_completion
from studylife_ai.rag.prompt import build_context_system_message, sources_payload
from studylife_ai.rag.retrieval import retrieve_with_rerank
from studylife_ai.schemas.chat import ChatMessage, ChatRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def _latest_user_message(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return messages[-1].content


async def _retrieve_context(
    query: str, settings: Settings, store: QdrantStore, user_id: str
) -> list[RetrievedChunk]:
    return await retrieve_with_rerank(query, store=store, settings=settings, user_id=user_id)


async def _sse_event_stream(
    request: ChatRequest, store: QdrantStore, user_id: str
) -> AsyncIterator[str]:
    settings = get_settings()
    model = request.model or settings.llm_model

    try:
        chunks = await _retrieve_context(
            _latest_user_message(request.messages), settings, store, user_id
        )
    except Exception:
        logger.exception("Retrieval failed, continuing without note context")
        chunks = []

    context_message = ChatMessage(role="system", content=build_context_system_message(chunks))
    # Without this, the model has no way to know "today" and answers relative-date questions
    # ("what's on today/this week") against training data instead - same fix, same reasoning,
    # as api/agent.py's identical injection (found live: /chat asserted a specific "today" that
    # was months off). Local time, not UTC - matches StudyLife's own sessions, which store naive
    # local timestamps (see docs/decisions.md).
    now = datetime.now().strftime("%Y-%m-%d %H:%M, %A")
    date_message = ChatMessage(role="system", content=f"The current date and time is {now}.")
    augmented_messages = [context_message, date_message, *request.messages]
    # Computed once, up front, from the already-retrieved chunks — independent of
    # whether the LLM call below succeeds, so a client always learns which notes
    # were consulted, and a bug here can never be mislabeled as an LLM failure.
    sources_event = (
        f"data: {json.dumps({'sources': [s.model_dump() for s in sources_payload(chunks)]})}\n\n"
    )

    try:
        async for delta in stream_chat_completion(
            augmented_messages,
            model=model,
            api_base=settings.llm_api_base,
            timeout=settings.llm_request_timeout_seconds,
            call_site="chat",
        ):
            yield f"data: {json.dumps({'delta': delta})}\n\n"
    except Exception:
        logger.exception("LLM streaming failed for model=%s", model)
        yield f"data: {json.dumps({'error': 'LLM request failed'})}\n\n"
    yield sources_event
    yield "data: [DONE]\n\n"


@router.post("/chat", dependencies=[Depends(enforce_rate_limit)])
async def chat(
    request: ChatRequest,
    http_request: Request,
    identity: ResolvedIdentity = Depends(resolve_identity),
) -> StreamingResponse:
    store: QdrantStore = http_request.app.state.qdrant_store
    return StreamingResponse(
        _sse_event_stream(request, store, identity.user_id), media_type="text/event-stream"
    )
