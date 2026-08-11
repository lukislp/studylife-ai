"""Chat endpoint: RAG-augmented, streams an LLM completion back via SSE.

Every request is retrieval-augmented (docs/decisions.md "Prompt design"):
the latest user message is used to fetch relevant note chunks, which are
injected as a system message ahead of the conversation. Streamed events:
`{"delta": "..."}` per token (or one `{"error": "..."}` if the LLM call
fails mid-stream), then always one `{"sources": [...]}` event listing the
notes that were actually retrieved (deduplicated, independent of whether
the model cited them or whether streaming succeeded), then `data: [DONE]`.

Requires the identity headers described in `api/identity.py` (see
docs/decisions.md "M4.5 Multi-user support") - `user_id` scopes retrieval
to the calling user's own Qdrant partition. Also requires
`STUDYLIFE_API_BASE_URL` to be set: `ai_api_key` is verified against
StudyLife (`verify_identity()`) before any streaming starts - a new
dependency /chat didn't previously have, accepted as the cost of
defense-in-depth beyond network isolation (see docs/decisions.md "Key
validity check").
"""

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from studylife_ai.api.identity import ResolvedIdentity, resolve_identity, verify_identity
from studylife_ai.config import Settings, get_settings
from studylife_ai.ingestion.qdrant_store import QdrantStore, RetrievedChunk
from studylife_ai.llm.client import stream_chat_completion
from studylife_ai.rag.prompt import build_context_system_message, sources_payload
from studylife_ai.rag.retrieval import retrieve_with_rerank
from studylife_ai.schemas.chat import ChatMessage, ChatRequest
from studylife_ai.studylife.client import StudyLifeClient

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

_NOT_CONFIGURED_DETAIL = "Chat not available - STUDYLIFE_API_BASE_URL must be set."


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
    augmented_messages = [context_message, *request.messages]
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
        ):
            yield f"data: {json.dumps({'delta': delta})}\n\n"
    except Exception:
        logger.exception("LLM streaming failed for model=%s", model)
        yield f"data: {json.dumps({'error': 'LLM request failed'})}\n\n"
    yield sources_event
    yield "data: [DONE]\n\n"


@router.post("/chat")
async def chat(
    request: ChatRequest,
    http_request: Request,
    identity: ResolvedIdentity = Depends(resolve_identity),
) -> StreamingResponse:
    settings = get_settings()
    if not settings.studylife_api_base_url:
        raise HTTPException(status_code=503, detail=_NOT_CONFIGURED_DETAIL)
    # Verified before any streaming starts - once the SSE response begins,
    # the 200 status is already committed and a rejected key could only
    # surface as an in-stream error event, not a real 401.
    async with StudyLifeClient(
        base_url=settings.studylife_api_base_url, api_key=identity.ai_api_key
    ) as studylife_client:
        await verify_identity(studylife_client)

    store: QdrantStore = http_request.app.state.qdrant_store
    return StreamingResponse(
        _sse_event_stream(request, store, identity.user_id), media_type="text/event-stream"
    )
