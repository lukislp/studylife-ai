"""FastAPI glue for the server-side AI consent gate (see studylife/consent.py, GDPR audit F-20).

`require_ai_consent` guards the user-facing routes that send user content to a model
(/chat, /agent, /agent/confirm); `enforce_ai_consent` is the same check as a plain coroutine for
routes whose user id comes from the request body instead of a proxy token
(/internal/enrich-capture). Both map the domain errors onto HTTP: 403 `ai-consent-required` when
the user has no consent, 503 when consent cannot be established (fail closed).
"""

from fastapi import Depends, HTTPException, Request

from studylife_ai.api.identity import ResolvedIdentity, resolve_identity
from studylife_ai.config import get_settings
from studylife_ai.studylife.consent import ConsentRequired, ConsentUnverifiable, ensure_ai_consent


async def enforce_ai_consent(http_request: Request, user_id: str) -> None:
    try:
        await ensure_ai_consent(
            user_id,
            settings=get_settings(),
            key_store=http_request.app.state.registered_key_store,
        )
    except ConsentRequired:
        raise HTTPException(
            status_code=403,
            detail="ai-consent-required: the user has not agreed to AI processing.",
        ) from None
    except ConsentUnverifiable:
        raise HTTPException(
            status_code=503,
            detail="AI consent could not be verified; nothing was sent to a model.",
        ) from None


async def require_ai_consent(
    http_request: Request, identity: ResolvedIdentity = Depends(resolve_identity)
) -> None:
    await enforce_ai_consent(http_request, identity.user_id)
