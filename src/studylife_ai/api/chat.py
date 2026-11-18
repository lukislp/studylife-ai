"""Chat endpoint: context-augmented, streams an LLM completion back via SSE.

Every request is augmented with a context system message built by
`rag/chat_context.py` from the latest user message: retrieved note chunks for
content questions (docs/decisions.md "Prompt design"), or a structured listing
read live from the StudyLife API for enumeration questions ("which notes do I
have in course X" - docs/decisions.md "Structured listings for enumeration
questions"). Streamed events: `{"delta": "..."}` per token (or one
`{"error": "..."}` if the LLM call fails mid-stream), then always one
`{"sources": [...]}` event listing the entities the context was built from
(deduplicated, independent of whether the model cited them or whether
streaming succeeded), then `data: [DONE]`.

Requires the signed proxy token described in `api/identity.py` (see
docs/decisions.md "M4.5 Multi-user support") - `user_id` scopes retrieval to
the calling user's own Qdrant partition. Listings and course names are read
with the user's own registered AiApiKey (the same key and consent gate as
/agent); without `STUDYLIFE_API_BASE_URL` or a registered key /chat still
answers, from retrieval alone.
"""

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from studylife_ai.api.consent import require_ai_consent
from studylife_ai.api.identity import ResolvedIdentity, resolve_identity
from studylife_ai.api.rate_limit import enforce_rate_limit
from studylife_ai.clock import local_now
from studylife_ai.config import Settings, get_settings
from studylife_ai.ingestion.qdrant_store import QdrantStore, RetrievedChunk
from studylife_ai.llm.client import stream_chat_completion
from studylife_ai.rag.chat_context import build_chat_context
from studylife_ai.rag.prompt import current_date_message
from studylife_ai.rag.retrieval import retrieve_with_rerank
from studylife_ai.schemas.chat import ChatMessage, ChatRequest
from studylife_ai.studylife.client import StudyLifeClient
from studylife_ai.studylife.registered_keys import RegisteredKeyStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def _allowed_models(settings: Settings) -> set[str]:
    """The set of LiteLLM model strings a `ChatRequest.model` override may name (audit F15) -
    `settings.llm_model` is always implicitly a member, in addition to whatever
    `settings.allowed_chat_models` (comma-separated) adds. With the default empty
    `allowed_chat_models`, this is exactly `{llm_model}` - the tightest useful default."""
    allowed = {settings.llm_model}
    allowed.update(
        model.strip() for model in settings.allowed_chat_models.split(",") if model.strip()
    )
    return allowed


def _resolve_model(requested: str | None, settings: Settings) -> str:
    """Returns the LiteLLM model string this request should use, or raises a 400 before any LLM
    call is made if `requested` isn't in `_allowed_models` (audit F15: the server, not the
    caller, pays for whatever model gets named here). `requested=None` (no override in the
    request body - true of every deployed caller today) always resolves to `settings.llm_model`
    without consulting the allowlist at all."""
    if requested is None:
        return settings.llm_model
    allowed = _allowed_models(settings)
    if requested not in allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"model '{requested}' is not allowed. Configure ALLOWED_CHAT_MODELS to permit "
                "it, or omit `model` to use the server default."
            ),
        )
    return requested


def _latest_user_message(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return messages[-1].content


async def _retrieve_context(
    query: str, settings: Settings, store: QdrantStore, user_id: str
) -> list[RetrievedChunk]:
    return await retrieve_with_rerank(query, store=store, settings=settings, user_id=user_id)


async def _studylife_source(
    key_store: RegisteredKeyStore, user_id: str, settings: Settings
) -> StudyLifeClient | None:
    """The calling user's own StudyLife API client (their registered AiApiKey, so every read is
    scoped to their account), or `None` when that isn't possible - /chat then answers from
    retrieval alone, as it did before structured listings existed."""
    if not settings.studylife_api_base_url:
        return None
    try:
        api_key = await key_store.get(user_id)
    except Exception:
        logger.exception("Registered-key lookup failed for user_id=%s", user_id)
        return None
    if api_key is None:
        return None
    return StudyLifeClient(
        base_url=settings.studylife_api_base_url,
        api_key=api_key,
        ca_cert_path=settings.studylife_ca_cert_path,
    )


async def _sse_event_stream(
    request: ChatRequest,
    store: QdrantStore,
    key_store: RegisteredKeyStore,
    user_id: str,
    model: str,
) -> AsyncIterator[str]:
    settings = get_settings()
    query = _latest_user_message(request.messages)
    # Without this, the model has no way to know "today" and answers relative-date questions
    # ("what's on today/this week") against training data instead - same fix, same reasoning,
    # as api/agent.py's identical injection (found live: /chat asserted a specific "today" that
    # was months off). Local time in Settings.timezone, not UTC - matches StudyAtrium's own
    # sessions, which store naive local timestamps (see docs/decisions.md).
    now = local_now(settings.timezone)

    source = await _studylife_source(key_store, user_id, settings)
    try:
        context = await build_chat_context(
            query,
            retrieve=lambda: _retrieve_context(query, settings, store, user_id),
            source=source,
            settings=settings,
            now=now,
            user_id=user_id,
        )
    finally:
        if source is not None:
            await source.aclose()

    context_message = ChatMessage(role="system", content=context.system_message)
    # States the exact Mon-Sun week boundaries as ground truth, not just "today" - found live
    # (2026-08-12): even with the correct sessions retrieved and listed, the answering model's
    # own framing sentence mislabeled a correctly-resolved "last week" range as "the week before
    # last week" (see rag/prompt.py's current_date_message()).
    date_message = ChatMessage(
        role="system", content=current_date_message(now, timezone=settings.timezone)
    )
    augmented_messages = [context_message, date_message, *request.messages]
    # Computed once, up front, from the already-built context — independent of
    # whether the LLM call below succeeds, so a client always learns which notes
    # were consulted, and a bug here can never be mislabeled as an LLM failure.
    sources_event = (
        f"data: {json.dumps({'sources': [s.model_dump() for s in context.sources]})}\n\n"
    )

    try:
        async for delta in stream_chat_completion(
            augmented_messages,
            model=model,
            api_base=settings.llm_api_base,
            timeout=settings.llm_request_timeout_seconds,
            call_site="chat",
            user_id=user_id,
            reasoning_effort=settings.llm_reasoning_effort,
        ):
            yield f"data: {json.dumps({'delta': delta})}\n\n"
    except Exception:
        logger.exception("LLM streaming failed for model=%s", model)
        yield f"data: {json.dumps({'error': 'LLM request failed'})}\n\n"
    yield sources_event
    yield "data: [DONE]\n\n"


@router.post("/chat", dependencies=[Depends(enforce_rate_limit), Depends(require_ai_consent)])
async def chat(
    request: ChatRequest,
    http_request: Request,
    identity: ResolvedIdentity = Depends(resolve_identity),
) -> StreamingResponse:
    # Resolved (and, if `request.model` is set, allowlist-checked - audit F15) before the
    # StreamingResponse is ever constructed: once streaming starts, the response is already
    # committed as a 200 (see the LLM-failure path below, which can only ever emit an SSE
    # `error` event, not a real HTTP error status) - a disallowed model must fail as a clean 400
    # instead.
    model = _resolve_model(request.model, get_settings())
    store: QdrantStore = http_request.app.state.qdrant_store
    key_store: RegisteredKeyStore = http_request.app.state.registered_key_store
    return StreamingResponse(
        _sse_event_stream(request, store, key_store, identity.user_id, model),
        media_type="text/event-stream",
    )
