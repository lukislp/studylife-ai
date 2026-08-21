"""Internal endpoints StudyLife's backend calls to keep studylife-ai's
per-user AiApiKey registry in sync (see docs/decisions.md "M4.5 Multi-user
support" - "Registration-on-generate"). Not part of the public /chat, /agent
surface - meant to be reachable only from StudyLife's backend, same trust
boundary as the rest of the service.

Authenticated by a constant-time comparison of a shared secret (the same
`Settings.studylife_shared_secret` used to verify per-request proxy tokens
in api/identity.py) - a plain bearer-secret check, not the signed-token
scheme, since these aren't per-user requests.
"""

import hmac
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from studylife_ai.config import get_settings
from studylife_ai.ingestion.qdrant_store import QdrantStore
from studylife_ai.ingestion.sync import sync_user
from studylife_ai.rag.enrichment import enrich_capture
from studylife_ai.schemas.internal import (
    EnrichCaptureRequest,
    EnrichCaptureResponse,
    RegisterKeyRequest,
    RevokeKeyRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["internal"])

SHARED_SECRET_HEADER = "X-StudyLife-Shared-Secret"


def _require_valid_secret(http_request: Request) -> None:
    settings = get_settings()
    if not settings.studylife_shared_secret:
        raise HTTPException(
            status_code=503, detail="STUDYLIFE_SHARED_SECRET must be set for this endpoint."
        )
    provided = http_request.headers.get(SHARED_SECRET_HEADER)
    if not provided or not hmac.compare_digest(provided, settings.studylife_shared_secret):
        raise HTTPException(status_code=401, detail="Invalid or missing shared secret.")


async def _sync_new_registration(user_id: str, ai_api_key: str, store: QdrantStore) -> None:
    """Runs after the HTTP response for /internal/register-key has already gone out (see
    BackgroundTasks below) - a slow/failed first sync must never make StudyLife's own
    "generate my AiApiKey" button hang or error. Failures are logged, not raised: there is
    nothing left to report them to by the time this runs, and the next scheduled `sync_all()`
    run (or another registration) will retry the same diff-against-Qdrant logic regardless."""
    try:
        await sync_user(
            user_id=user_id, ai_api_key=ai_api_key, settings=get_settings(), store=store
        )
        logger.info("Auto-ingestion after registration succeeded for user_id=%s", user_id)
    except Exception:
        logger.exception("Auto-ingestion after registration failed for user_id=%s", user_id)


@router.post("/internal/register-key")
async def register_key(
    request: RegisterKeyRequest, http_request: Request, background_tasks: BackgroundTasks
) -> dict[str, bool]:
    _require_valid_secret(http_request)
    await http_request.app.state.registered_key_store.set(request.user_id, request.ai_api_key)
    logger.info("Registered AiApiKey for user_id=%s", request.user_id)
    # Fire-and-forget: the caller (StudyLife's GenerateAiApiKey) gets its response immediately
    # rather than waiting out a full sync - see docs/decisions.md "Auto-ingestion on register".
    settings = get_settings()
    if settings.studylife_api_base_url:
        background_tasks.add_task(
            _sync_new_registration,
            request.user_id,
            request.ai_api_key,
            http_request.app.state.qdrant_store,
        )
    return {"ok": True}


@router.post("/internal/revoke-key")
async def revoke_key(request: RevokeKeyRequest, http_request: Request) -> dict[str, bool]:
    _require_valid_secret(http_request)
    await http_request.app.state.registered_key_store.delete(request.user_id)
    logger.info("Revoked AiApiKey for user_id=%s", request.user_id)
    return {"ok": True}


@router.post("/internal/enrich-capture")
async def enrich_capture_endpoint(
    request: EnrichCaptureRequest, http_request: Request
) -> EnrichCaptureResponse:
    """Called by StudyLife's BackgroundTaskService (CaptureEnrichment sub-task) shortly after a
    studylife-capture browser-extension save - course-matching, related-notes lookup, tag/
    summary generation, and immediate Qdrant ingestion for one note. See rag/enrichment.py for
    the actual logic; this endpoint is just the internal-trust-boundary wrapper (same auth as
    register-key/revoke-key above)."""
    _require_valid_secret(http_request)
    settings = get_settings()
    result = await enrich_capture(
        request.note_id,
        request.title,
        request.content,
        user_id=request.user_id,
        settings=settings,
        store=http_request.app.state.qdrant_store,
    )
    logger.info(
        "Capture enrichment for user_id=%s note_id=%d: course_id=%s confidence=%s tags=%d "
        "related=%d",
        request.user_id,
        request.note_id,
        result.course_id,
        result.course_confidence,
        len(result.tags),
        len(result.related_note_ids),
    )
    return EnrichCaptureResponse(
        course_id=result.course_id,
        course_confidence=result.course_confidence,
        tags=result.tags,
        summary=result.summary,
        related_note_ids=result.related_note_ids,
    )
