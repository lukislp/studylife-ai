"""Tests for llm/client.py's complete_chat/stream_chat_completion - including their shared
retry wiring (llm/retry.py, see docs/decisions.md "LLM call retry")."""

from types import SimpleNamespace

import pytest
from litellm.exceptions import BadRequestError
from litellm.exceptions import Timeout as LiteLLMTimeout
from pytest import MonkeyPatch

from studylife_ai.llm import retry as retry_module
from studylife_ai.llm.client import complete_chat, stream_chat_completion
from studylife_ai.schemas.chat import ChatMessage


def _no_sleep(monkeypatch: MonkeyPatch) -> None:
    """Retry tests don't need to actually wait out the real backoff delay."""

    async def _sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(retry_module.asyncio, "sleep", _sleep)


async def test_complete_chat_retries_a_transient_failure_then_succeeds(
    monkeypatch: MonkeyPatch,
) -> None:
    _no_sleep(monkeypatch)
    attempts = 0

    async def fake_acompletion(*_args: object, **_kwargs: object) -> object:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise LiteLLMTimeout(message="timed out", model="m", llm_provider="p")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    monkeypatch.setattr("studylife_ai.llm.client.litellm.acompletion", fake_acompletion)

    result = await complete_chat(
        [ChatMessage(role="user", content="hi")],
        model="ollama/llama3.2",
        api_base=None,
        timeout=30.0,
    )

    assert result == "ok"
    assert attempts == 2


async def test_complete_chat_does_not_retry_a_non_transient_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    _no_sleep(monkeypatch)
    attempts = 0

    async def fake_acompletion(*_args: object, **_kwargs: object) -> object:
        nonlocal attempts
        attempts += 1
        raise BadRequestError(message="bad", model="m", llm_provider="p")

    monkeypatch.setattr("studylife_ai.llm.client.litellm.acompletion", fake_acompletion)

    with pytest.raises(BadRequestError):
        await complete_chat(
            [ChatMessage(role="user", content="hi")],
            model="ollama/llama3.2",
            api_base=None,
            timeout=30.0,
        )

    assert attempts == 1


async def test_stream_chat_completion_retries_establishing_the_stream(
    monkeypatch: MonkeyPatch,
) -> None:
    """Retry only has to cover getting the stream started - once chunks are flowing, a
    mid-stream failure propagates directly (see stream_chat_completion's docstring)."""
    _no_sleep(monkeypatch)
    attempts = 0

    async def fake_stream() -> object:
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello"))])

    async def fake_acompletion(*_args: object, **_kwargs: object) -> object:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise LiteLLMTimeout(message="timed out", model="m", llm_provider="p")
        return fake_stream()

    monkeypatch.setattr("studylife_ai.llm.client.litellm.acompletion", fake_acompletion)

    deltas = [
        delta
        async for delta in stream_chat_completion(
            [ChatMessage(role="user", content="hi")],
            model="ollama/llama3.2",
            api_base=None,
            timeout=30.0,
        )
    ]

    assert deltas == ["Hello"]
    assert attempts == 2
