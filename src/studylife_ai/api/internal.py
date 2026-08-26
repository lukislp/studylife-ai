"""Internal endpoints StudyLife's backend calls to keep studylife-ai's
per-user AiApiKey registry in sync (see docs/decisions.md "M4.5 Multi-user
support" - "Registration-on-generate"). Not part of the public /chat, /agent
surface - meant to be reachable only from StudyLife's backend, same trust
boundary as the rest of the service.

Authenticated by a constant-time comparison of a shared secret against
`Settings.studylife_internal_api_secret` (a plain bearer-secret check, not
the signed-token scheme in api/identity.py, since these aren't per-user
requests) - split out from the token-signing secret in audit A5 (2026-08-26,
see docs/decisions.md "Split the shared secret (audit A5)"): previously the
single `Settings.studylife_shared_secret` both signed per-user proxy tokens
AND authenticated this trust boundary, so anyone holding it could also
administer the registry, not just impersonate a user. `studylife_shared_secret`
is still accepted here as a legacy fallback while configured, and
`studylife_internal_api_secret` may itself be a comma-separated list of
*accepted* values (rotation - see config.py).
"""

import hmac
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from studylife_ai.config import Settings, get_settings
from studylife_ai.ingestion.qdrant_store import QdrantStore
from studylife_ai.ingestion.sync import purge_user, sync_user
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


def _accepted_internal_secrets(settings: Settings) -> list[str]:
    """Every bearer value `/internal/*` accepts (audit A5): each comma-separated entry of
    `studylife_internal_api_secret` (StudyLife's own `AiProxyClient` always SENDS only the
    first value it has configured - this side is what holds multiple *accepted* values during
    a rotation), plus the legacy `studylife_shared_secret` while that fallback is still
    configured."""
    accepted: list[str] = []
    if settings.studylife_internal_api_secret:
        accepted.extend(
            value.strip()
            for value in settings.studylife_internal_api_secret.split(",")
            if value.strip()
        )
    if settings.studylife_shared_secret:
        accepted.append(settings.studylife_shared_secret)
    return accepted


def _require_valid_secret(http_request: Request) -> None:
    settings = get_settings()
    accepted = _accepted_internal_secrets(settings)
    if not accepted:
        raise HTTPException(
            status_code=503,
            detail=(
                "STUDYLIFE_INTERNAL_API_SECRET (or the legacy STUDYLIFE_SHARED_SECRET) must "
                "be set for this endpoint."
            ),
        )
    provided = http_request.headers.get(SHARED_SECRET_HEADER)
    if not provided or not any(hmac.compare_digest(provided, secret) for secret in accepted):
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
    """Full purge, not just the registration row (audit F5/F13): a revoke that only deleted
    `registered_keys` left the user's Qdrant partition and agent-checkpoint threads retrievable
    via /chat and /agent forever. See `ingestion.sync.purge_user` for the deletion order and
    why - shared with the sync loop's own zombie-registration cleanup so there is one purge
    implementation, not two."""
    _require_valid_secret(http_request)
    await purge_user(
        user_id=request.user_id,
        store=http_request.app.state.qdrant_store,
        checkpointer=http_request.app.state.agent_checkpointer,
        key_store=http_request.app.state.registered_key_store,
    )
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
        active_course_ids=request.active_course_ids,
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
