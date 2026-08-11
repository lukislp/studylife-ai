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

from fastapi import APIRouter, HTTPException, Request

from studylife_ai.config import get_settings
from studylife_ai.schemas.internal import RegisterKeyRequest, RevokeKeyRequest

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


@router.post("/internal/register-key")
async def register_key(request: RegisterKeyRequest, http_request: Request) -> dict[str, bool]:
    _require_valid_secret(http_request)
    await http_request.app.state.registered_key_store.set(request.user_id, request.ai_api_key)
    logger.info("Registered AiApiKey for user_id=%s", request.user_id)
    return {"ok": True}


@router.post("/internal/revoke-key")
async def revoke_key(request: RevokeKeyRequest, http_request: Request) -> dict[str, bool]:
    _require_valid_secret(http_request)
    await http_request.app.state.registered_key_store.delete(request.user_id)
    logger.info("Revoked AiApiKey for user_id=%s", request.user_id)
    return {"ok": True}
