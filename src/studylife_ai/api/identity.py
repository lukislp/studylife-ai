"""Resolves the calling StudyLife user's identity from request headers.

Forwarded by StudyLife's own ASP.NET Core backend, which proxies chat/agent
requests on behalf of an already-authenticated (passkey session) user - see
docs/decisions.md "M4.5 Multi-user support".

`resolve_identity()` only parses headers. `verify_identity()` is a separate,
explicit defense-in-depth step (the user's call, per CLAUDE.md's
assist-only security-design process - see docs/decisions.md "Key validity
check") that confirms `ai_api_key` is a real, non-revoked StudyLife
credential via one live API call, on top of network isolation (studylife-ai
should still not be reachable except from StudyLife's backend). It does
NOT confirm the key belongs to the forwarded `user_id` specifically -
that needs a StudyLife-side "whoami" endpoint that doesn't exist yet
(tracked alongside the still-pending backend proxy work).
"""

from dataclasses import dataclass

import httpx
from fastapi import HTTPException, Request

from studylife_ai.studylife.client import StudyLifeClient

USER_ID_HEADER = "X-StudyLife-User-Id"
AI_API_KEY_HEADER = "X-StudyLife-Ai-Key"


@dataclass
class ResolvedIdentity:
    user_id: str
    ai_api_key: str


def resolve_identity(request: Request) -> ResolvedIdentity:
    user_id = request.headers.get(USER_ID_HEADER)
    ai_api_key = request.headers.get(AI_API_KEY_HEADER)
    if not user_id or not ai_api_key:
        raise HTTPException(
            status_code=401,
            detail=f"Missing {USER_ID_HEADER}/{AI_API_KEY_HEADER} headers.",
        )
    return ResolvedIdentity(user_id=user_id, ai_api_key=ai_api_key)


async def verify_identity(studylife_client: StudyLifeClient) -> None:
    """Raises 401 if `studylife_client`'s ai_api_key is invalid/revoked.
    Uses GET /api/courses (no dedicated lightweight validation endpoint
    exists in StudyLife yet) - the response is discarded, only used to
    confirm the key authenticates."""
    try:
        await studylife_client.get_courses()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            raise HTTPException(status_code=401, detail="Invalid AiApiKey.") from None
        raise
