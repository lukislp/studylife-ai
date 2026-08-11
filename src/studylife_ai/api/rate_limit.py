"""In-process rate limiting, per resolved user, for the LLM-cost-incurring endpoints
(`/chat`, `/agent`, `/agent/confirm`) - see docs/decisions.md "Rate limiting".

A fixed-window counter, not a sliding log or token bucket: good enough to catch a runaway
client loop or a leaked token being hammered, not meant as precise API metering. In-memory
only, safe because studylife-ai runs as a single replica (`Deployment` strategy: `Recreate`,
see k8s/04-app.yaml) - there's no second instance for the counter to be wrong about.
"""

import time

from fastapi import Depends, HTTPException

from studylife_ai.api.identity import ResolvedIdentity, resolve_identity
from studylife_ai.config import get_settings

# user_id -> (window_start, request_count_in_window). Module-level, not per-request - the whole
# point is to persist counts across requests for the lifetime of the process.
_windows: dict[str, tuple[float, int]] = {}


def _check_and_increment(user_id: str, *, limit: int, window_seconds: int) -> int | None:
    """Returns `None` if the request is allowed (and records it), or the number of seconds
    until the caller should retry if the limit is already hit this window.

    Not guarded by a lock: FastAPI/Starlette runs sync dependencies like this one to
    completion within a single event-loop iteration - since this function never awaits, no
    other request's coroutine can interleave inside it, even with multiple concurrent
    requests from different users.
    """
    now = time.monotonic()
    window_start, count = _windows.get(user_id, (now, 0))
    if now - window_start >= window_seconds:
        window_start, count = now, 0
    if count >= limit:
        return int(window_seconds - (now - window_start)) + 1
    _windows[user_id] = (window_start, count + 1)
    return None


def enforce_rate_limit(identity: ResolvedIdentity = Depends(resolve_identity)) -> None:
    """FastAPI dependency - raises 429 once `settings.rate_limit_requests` is exceeded within
    `settings.rate_limit_window_seconds` for the calling (already-authenticated) user.
    Depends on `resolve_identity` itself rather than taking a bare user_id, so it can't be
    wired in ahead of auth by mistake - and FastAPI resolves `resolve_identity` at most once
    per request even if both this and the route handler depend on it."""
    settings = get_settings()
    retry_after = _check_and_increment(
        identity.user_id,
        limit=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail="Too many requests - please slow down.",
            headers={"Retry-After": str(retry_after)},
        )
