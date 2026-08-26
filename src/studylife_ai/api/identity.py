"""Resolves the calling StudyLife user's identity from a signed proxy token.

Forwarded by StudyLife's own ASP.NET Core backend, which proxies chat/agent
requests on behalf of an already-authenticated (passkey session) user - see
docs/decisions.md "M4.5 Multi-user support" ("Auth flow, take two: a
short-lived signed proxy token").

The token is NOT the user's `AiApiKey` (StudyLife only ever stores a hash of
that, never the plaintext, after the moment it's generated - so the backend
cannot forward it). It's a short-lived token, minted by StudyLife per request
from a real, already-authenticated session, and verified here purely locally
(no network round-trip). Verifying the signature IS verifying the claimed
`user_id` - unlike the old `AiApiKey`-based design, there is no separate
"does this key really belong to this user_id" gap left to close.

Audit A5 (2026-08-26, see docs/decisions.md "Split the shared secret (audit
A5)"): split the single `Settings.studylife_shared_secret` - which used to
both sign proxy tokens AND authenticate `/internal/*` (api/internal.py),
letting anyone holding it mint a token for ANY `user_id` - into a dedicated
`Settings.studylife_token_signing_secret`, with a key-id (`kid`) so rotation
doesn't require a simultaneous redeploy. Two token formats are accepted:
  - New: `{user_id}.{expiry}.{kid}.{sig}` - verified against
    `studylife_token_signing_secret`'s "kid:secret,..." entries, looked up by
    the token's own `kid`.
  - Legacy: `{user_id}.{expiry}.{sig}` (no kid) - verified against
    `studylife_shared_secret`, accepted ONLY while that legacy setting is
    still configured (rollout compatibility - lets StudyLife's backend and
    studylife-ai deploy the split independently, in either order).

A verified `ResolvedIdentity` only proves *who is asking* - it is not itself
a credential StudyLife's own `/api/*` gate would accept. `/agent`, which
needs to make real StudyLife API calls, looks up that user's actual
`AiApiKey` from `studylife.registered_keys.RegisteredKeyStore` separately.
"""

import base64
import hashlib
import hmac
import logging
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request

from studylife_ai.config import get_settings

PROXY_TOKEN_HEADER = "X-StudyLife-Proxy-Token"

logger = logging.getLogger(__name__)

# Module-level, deliberately (same reasoning as internal.py's constant-time comparison list):
# a startup-lifetime "have we already warned about this" flag, not per-request state. Reset in
# tests via conftest.py's autouse fixture, same pattern as rate_limit._windows.
_warned_legacy_signing_fallback = False


@dataclass
class ResolvedIdentity:
    user_id: str


def _sign(payload: str, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _parse_signing_keys(config: str) -> dict[str, str]:
    """Parses `studylife_token_signing_secret` ("kid1:secret1,kid2:secret2,...") into a
    kid -> secret map. Every entry is a valid verification key here - StudyLife's own backend
    picks which one to SIGN with (always the first, see its AiProxyTokenService.Mint), this
    side just needs to look up whichever kid a given token claims."""
    keys: dict[str, str] = {}
    for raw_entry in config.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        kid, sep, secret = entry.partition(":")
        if not sep or not kid or not secret:
            raise ValueError(
                f"Malformed STUDYLIFE_TOKEN_SIGNING_SECRET entry {entry!r} - expected 'kid:secret'."
            )
        keys[kid] = secret
    if not keys:
        raise ValueError(
            "STUDYLIFE_TOKEN_SIGNING_SECRET must contain at least one 'kid:secret' entry."
        )
    return keys


def _warn_legacy_signing_fallback_once() -> None:
    global _warned_legacy_signing_fallback
    if _warned_legacy_signing_fallback:
        return
    logger.warning(
        "STUDYLIFE_TOKEN_SIGNING_SECRET is not set - falling back to the legacy "
        "STUDYLIFE_SHARED_SECRET for proxy-token verification (audit A5). Set "
        "STUDYLIFE_TOKEN_SIGNING_SECRET before removing STUDYLIFE_SHARED_SECRET."
    )
    _warned_legacy_signing_fallback = True


def resolve_identity(request: Request) -> ResolvedIdentity:
    settings = get_settings()
    if not settings.studylife_token_signing_secret and not settings.studylife_shared_secret:
        raise HTTPException(
            status_code=503,
            detail=(
                "STUDYLIFE_TOKEN_SIGNING_SECRET (or the legacy STUDYLIFE_SHARED_SECRET) must "
                "be set to verify identity."
            ),
        )

    token = request.headers.get(PROXY_TOKEN_HEADER)
    if not token:
        raise HTTPException(status_code=401, detail=f"Missing {PROXY_TOKEN_HEADER} header.")

    parts = token.split(".")

    if len(parts) == 4:
        # New format: {user_id}.{expiry}.{kid}.{sig}.
        if not settings.studylife_token_signing_secret:
            raise HTTPException(status_code=401, detail="Invalid proxy token signature.")
        user_id, expiry_str, kid, signature = parts
        try:
            signing_keys = _parse_signing_keys(settings.studylife_token_signing_secret)
        except ValueError:
            logger.exception("Malformed STUDYLIFE_TOKEN_SIGNING_SECRET configuration.")
            raise HTTPException(
                status_code=503, detail="Proxy-token signing keys misconfigured."
            ) from None
        secret = signing_keys.get(kid)
        if secret is None:
            raise HTTPException(status_code=401, detail="Unknown proxy token key id.")
        payload = f"{user_id}.{expiry_str}"
        expected_signature = _sign(payload, secret)
    elif len(parts) == 3:
        # Legacy format: {user_id}.{expiry}.{sig} - only while STUDYLIFE_SHARED_SECRET is
        # still configured (audit A5 rollout compatibility).
        if not settings.studylife_shared_secret:
            raise HTTPException(status_code=401, detail="Invalid proxy token signature.")
        if not settings.studylife_token_signing_secret:
            _warn_legacy_signing_fallback_once()
        user_id, expiry_str, signature = parts
        payload = f"{user_id}.{expiry_str}"
        expected_signature = _sign(payload, settings.studylife_shared_secret)
    else:
        raise HTTPException(status_code=401, detail="Malformed proxy token.")

    if not hmac.compare_digest(expected_signature, signature):
        raise HTTPException(status_code=401, detail="Invalid proxy token signature.")

    try:
        expiry = int(expiry_str)
    except ValueError:
        raise HTTPException(status_code=401, detail="Malformed proxy token.") from None
    if expiry < int(time.time()):
        raise HTTPException(status_code=401, detail="Proxy token expired.")

    return ResolvedIdentity(user_id=user_id)
