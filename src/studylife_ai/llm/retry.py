"""Shared retry/backoff for transient LiteLLM failures (see docs/decisions.md "LLM call
retry").

One implementation, used by every direct `litellm.a*()` call site (`llm/client.py`,
`llm/embeddings.py`) instead of separate copies - `rag/rerank.py` and `rag/date_parse.py`
already go through `llm/client.py`'s `complete_chat()`, so they inherit this for free without
being touched directly.

`is_transient_llm_error()` is also reused by `api/agent.py`'s `_invoke_and_handle_failure` to
decide whether a failed agent run's paused checkpoint should survive a one-off provider
hiccup, so the client can retry `POST /agent/confirm` with the same `thread_id` instead of
losing the whole propose-confirm flow.

Considered LiteLLM's own `num_retries` kwarg instead of a hand-rolled helper - rejected:
tracing through `litellm/main.py` shows it wraps the call in a bare `tenacity.Retrying`/
`AsyncRetrying` with no exception filter, i.e. it would retry a 400/401 (a wrong request or a
bad API key) exactly as eagerly as a timeout, wasting attempts on failures a retry can never
fix. A ~30-line helper that only retries the classes worth retrying is simpler to reason
about, and - unlike an opaque kwarg passed into a third-party client - directly unit-testable
with a fake that fails twice then succeeds (see `tests/test_llm_retry.py`).
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from litellm.exceptions import Timeout as LiteLLMTimeout

logger = logging.getLogger(__name__)

# 1 initial attempt + 2 retries - enough to ride out a one-off provider hiccup without turning
# a real outage into a long hang. Exponential backoff between attempts (0.5s, then 1.0s).
_MAX_ATTEMPTS = 3
_BASE_DELAY_SECONDS = 0.5


def is_transient_llm_error(exc: BaseException) -> bool:
    """True for the narrow class of errors worth retrying: a timeout, HTTP 429 (rate limit),
    or any 5xx - the provider (or the connection to it) had a bad moment, not the request
    itself. Everything else (400 bad request, 401 auth, a content-policy rejection, a
    context-window overflow, ...) means the *request* is wrong and would fail identically on
    a retry, so those propagate immediately, unretried.

    Works off LiteLLM's own exception hierarchy (`litellm.exceptions`), which normalizes
    every provider's errors (OpenAI, Anthropic, Ollama, ...) onto one OpenAI-style set with a
    `.status_code` attribute - so this one check covers whichever provider `llm_model`/
    `embedding_model`/`rerank_model`/`date_parse_model` happens to be configured to, not just
    a specific one. `litellm.exceptions.Timeout` is checked by type rather than status code,
    since its default `status_code` (408) can be overridden per-provider - a timeout is
    transient regardless of what status code it happens to carry.
    """
    if isinstance(exc, LiteLLMTimeout):
        return True
    status_code = getattr(exc, "status_code", None)
    return isinstance(status_code, int) and (status_code == 429 or 500 <= status_code < 600)


async def with_retry[T](call: Callable[[], Awaitable[T]], *, call_site: str) -> T:
    """Runs `call()`, retrying up to `_MAX_ATTEMPTS` times with exponential backoff whenever
    the failure is transient (see `is_transient_llm_error`). A non-transient exception
    propagates immediately on its first occurrence; a transient one propagates only once
    `_MAX_ATTEMPTS` is exhausted - either way, the exception reaching the caller is always the
    original one, unwrapped. Retry is purely an internal robustness layer, not a new error
    type callers need to handle.

    `call` is a zero-argument thunk (not the coroutine itself) so each retry attempt starts a
    fresh call - an already-awaited coroutine object can't be awaited a second time.
    """
    attempt = 1
    while True:
        try:
            return await call()
        except Exception as exc:
            if not is_transient_llm_error(exc) or attempt >= _MAX_ATTEMPTS:
                raise
            delay = _BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Transient LLM error at call_site=%s (attempt %d/%d), retrying in %.1fs: %s",
                call_site,
                attempt,
                _MAX_ATTEMPTS,
                delay,
                exc,
            )
            await asyncio.sleep(delay)
            attempt += 1
