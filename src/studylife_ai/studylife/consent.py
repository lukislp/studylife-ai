"""Server-side AI consent gate (GDPR audit F-20).

StudyLife's backend already checks the user's AI consent before it proxies a request here, but
this service is the one that actually sends user content to a model, so it must not rely on the
caller having checked. `ensure_ai_consent` asks StudyLife itself: it makes one cheap
authenticated read with the user's registered AiApiKey, and StudyAtrium answers any such request
with 403 `ai-consent-required` while the user has no consent covering the current provider
(the same signal ingestion already treats as a pause, see ingestion/sync.py).

Outcomes:
- consent in place: returns (positive answers are cached for `ai_consent_cache_seconds`)
- no consent: raises `ConsentRequired`
- cannot be established (no registered key, key rejected, StudyLife unreachable or erroring):
  raises `ConsentUnverifiable` - fail closed, nothing is sent to a model
- check disabled (`AI_CONSENT_CHECK=false`) or no `STUDYLIFE_API_BASE_URL`: returns without
  asking (nothing to ask / operator opted out)
"""

import logging
import time

import httpx

from studylife_ai.config import Settings
from studylife_ai.ingestion.sync import is_consent_required
from studylife_ai.studylife.client import StudyLifeClient
from studylife_ai.studylife.registered_keys import RegisteredKeyStore

logger = logging.getLogger(__name__)


class ConsentRequired(Exception):
    """The user has no AI consent covering the current provider."""


class ConsentUnverifiable(Exception):
    """Consent could not be established; callers must treat this as "no"."""


# user_id -> monotonic deadline until which a positive answer is reused. Module-level on
# purpose (process-lifetime cache, like api/rate_limit._windows); reset in tests.
_allowed_until: dict[str, float] = {}


def forget_consent(user_id: str | None = None) -> None:
    """Drops the cached positive answer for one user, or for everybody."""
    if user_id is None:
        _allowed_until.clear()
    else:
        _allowed_until.pop(user_id, None)


async def ensure_ai_consent(
    user_id: str,
    *,
    settings: Settings,
    key_store: RegisteredKeyStore,
    transport: httpx.AsyncBaseTransport | None = None,
) -> None:
    if not settings.ai_consent_check or not settings.studylife_api_base_url:
        return

    deadline = _allowed_until.get(user_id)
    if deadline is not None and deadline > time.monotonic():
        return

    api_key = await key_store.get(user_id)
    if api_key is None:
        logger.warning("AI consent not verifiable for user_id=%s: no registered AiApiKey", user_id)
        raise ConsentUnverifiable("No registered AiApiKey for this user.")

    try:
        async with StudyLifeClient(
            base_url=settings.studylife_api_base_url,
            api_key=api_key,
            ca_cert_path=settings.studylife_ca_cert_path,
            transport=transport,
        ) as studylife:
            await studylife.check_ai_access()
    except httpx.HTTPStatusError as exc:
        _allowed_until.pop(user_id, None)
        if is_consent_required(exc):
            raise ConsentRequired from exc
        logger.warning(
            "AI consent not verifiable for user_id=%s: StudyAtrium answered %d",
            user_id,
            exc.response.status_code,
        )
        raise ConsentUnverifiable("StudyAtrium rejected the consent probe.") from exc
    except httpx.HTTPError as exc:
        _allowed_until.pop(user_id, None)
        logger.warning(
            "AI consent not verifiable for user_id=%s: StudyAtrium unreachable (%s)",
            user_id,
            type(exc).__name__,
        )
        raise ConsentUnverifiable("StudyAtrium could not be reached.") from exc

    _allowed_until[user_id] = time.monotonic() + settings.ai_consent_cache_seconds
