"""Tests for the shared LLM retry helper (llm/retry.py) - see docs/decisions.md "LLM call
retry"."""

from litellm.exceptions import (
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
)
from litellm.exceptions import Timeout as LiteLLMTimeout
from pytest import MonkeyPatch, raises

from studylife_ai.llm import retry as retry_module
from studylife_ai.llm.retry import is_transient_llm_error, with_retry


def _no_sleep_monkeypatch(monkeypatch: MonkeyPatch) -> None:
    """Retry tests don't need to actually wait out the real backoff delay."""

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(retry_module.asyncio, "sleep", _no_sleep)


def test_is_transient_llm_error_true_for_timeout_429_and_5xx() -> None:
    assert is_transient_llm_error(LiteLLMTimeout(message="timed out", model="m", llm_provider="p"))
    assert is_transient_llm_error(
        RateLimitError(message="rate limited", model="m", llm_provider="p")
    )
    assert is_transient_llm_error(
        InternalServerError(message="internal error", model="m", llm_provider="p")
    )
    assert is_transient_llm_error(
        ServiceUnavailableError(message="unavailable", model="m", llm_provider="p")
    )


def test_is_transient_llm_error_false_for_bad_request_and_auth_errors() -> None:
    """A 400/401 means the request itself is wrong (bad params, invalid key) - retrying would
    just fail identically, so these must NOT be classified as transient."""
    assert not is_transient_llm_error(BadRequestError(message="bad", model="m", llm_provider="p"))
    assert not is_transient_llm_error(
        AuthenticationError(message="unauthorized", model="m", llm_provider="p")
    )


def test_is_transient_llm_error_false_for_unrelated_exceptions() -> None:
    assert not is_transient_llm_error(RuntimeError("something else"))
    assert not is_transient_llm_error(ValueError("bad value"))


async def test_with_retry_succeeds_after_transient_failures(monkeypatch: MonkeyPatch) -> None:
    _no_sleep_monkeypatch(monkeypatch)
    attempts = 0

    async def flaky_call() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise LiteLLMTimeout(message="timed out", model="m", llm_provider="p")
        return "ok"

    result = await with_retry(flaky_call, call_site="test")

    assert result == "ok"
    assert attempts == 3


async def test_with_retry_gives_up_after_max_attempts(monkeypatch: MonkeyPatch) -> None:
    _no_sleep_monkeypatch(monkeypatch)
    attempts = 0

    async def always_times_out() -> str:
        nonlocal attempts
        attempts += 1
        raise LiteLLMTimeout(message="timed out", model="m", llm_provider="p")

    with raises(LiteLLMTimeout):
        await with_retry(always_times_out, call_site="test")

    assert attempts == 3  # 1 initial attempt + 2 retries, then gives up


async def test_with_retry_does_not_retry_non_transient_errors(monkeypatch: MonkeyPatch) -> None:
    _no_sleep_monkeypatch(monkeypatch)
    attempts = 0

    async def bad_request() -> str:
        nonlocal attempts
        attempts += 1
        raise BadRequestError(message="bad", model="m", llm_provider="p")

    with raises(BadRequestError):
        await with_retry(bad_request, call_site="test")

    assert attempts == 1
