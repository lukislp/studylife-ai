"""Resolves the calling StudyLife user's identity from a signed proxy token.

Forwarded by StudyLife's own ASP.NET Core backend, which proxies chat/agent
requests on behalf of an already-authenticated (passkey session) user - see
docs/decisions.md "M4.5 Multi-user support" ("Auth flow, take two: a
short-lived signed proxy token").

The token is NOT the user's `AiApiKey` (StudyLife only ever stores a hash of
that, never the plaintext, after the moment it's generated - so the backend
cannot forward it). It's a short-lived `{user_id}.{expiry}.{hmac_sig}` token,
minted by StudyLife per request from a real, already-authenticated session,
and verified here purely locally (no network round-trip) against a shared
secret (`Settings.studylife_shared_secret`) both services are configured
with. Verifying the signature IS verifying the claimed `user_id` - unlike
the old `AiApiKey`-based design, there is no separate "does this key really
belong to this user_id" gap left to close.

A verified `ResolvedIdentity` only proves *who is asking* - it is not itself
a credential StudyLife's own `/api/*` gate would accept. `/agent`, which
needs to make real StudyLife API calls, looks up that user's actual
`AiApiKey` from `studylife.registered_keys.RegisteredKeyStore` separately.
"""

import base64
import hashlib
import hmac
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request

from studylife_ai.config import get_settings

PROXY_TOKEN_HEADER = "X-StudyLife-Proxy-Token"


@dataclass
class ResolvedIdentity:
    user_id: str


def _sign(payload: str, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def resolve_identity(request: Request) -> ResolvedIdentity:
    settings = get_settings()
    if not settings.studylife_shared_secret:
        raise HTTPException(
            status_code=503, detail="STUDYLIFE_SHARED_SECRET must be set to verify identity."
        )

    token = request.headers.get(PROXY_TOKEN_HEADER)
    if not token:
        raise HTTPException(status_code=401, detail=f"Missing {PROXY_TOKEN_HEADER} header.")

    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(status_code=401, detail="Malformed proxy token.")
    user_id, expiry_str, signature = parts

    payload = f"{user_id}.{expiry_str}"
    expected_signature = _sign(payload, settings.studylife_shared_secret)
    if not hmac.compare_digest(expected_signature, signature):
        raise HTTPException(status_code=401, detail="Invalid proxy token signature.")

    try:
        expiry = int(expiry_str)
    except ValueError:
        raise HTTPException(status_code=401, detail="Malformed proxy token.") from None
    if expiry < int(time.time()):
        raise HTTPException(status_code=401, detail="Proxy token expired.")

    return ResolvedIdentity(user_id=user_id)
