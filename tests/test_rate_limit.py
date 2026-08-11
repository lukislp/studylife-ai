import pytest
from fastapi import HTTPException
from httpx import AsyncClient
from pytest import MonkeyPatch

from studylife_ai.api import rate_limit as rate_limit_module
from studylife_ai.api.identity import ResolvedIdentity
from studylife_ai.api.rate_limit import _check_and_increment, enforce_rate_limit
from studylife_ai.config import Settings

from .test_chat import _make_fake_stream, _mock_no_retrieval


def test_check_and_increment_allows_up_to_the_limit_then_blocks() -> None:
    for _ in range(3):
        assert _check_and_increment("alice", limit=3, window_seconds=60) is None

    retry_after = _check_and_increment("alice", limit=3, window_seconds=60)

    assert retry_after is not None
    assert retry_after > 0


def test_check_and_increment_tracks_users_independently() -> None:
    for _ in range(3):
        assert _check_and_increment("alice", limit=3, window_seconds=60) is None

    # bob has never made a request - his own window is fresh regardless of alice's count.
    assert _check_and_increment("bob", limit=3, window_seconds=60) is None


def test_check_and_increment_resets_after_the_window_expires(monkeypatch: MonkeyPatch) -> None:
    clock = {"now": 1000.0}
    monkeypatch.setattr(rate_limit_module.time, "monotonic", lambda: clock["now"])

    for _ in range(3):
        assert _check_and_increment("alice", limit=3, window_seconds=60) is None
    assert _check_and_increment("alice", limit=3, window_seconds=60) is not None

    clock["now"] += 61  # past the 60s window

    assert _check_and_increment("alice", limit=3, window_seconds=60) is None


def test_enforce_rate_limit_raises_429_with_retry_after_once_over_limit(
    monkeypatch: MonkeyPatch,
) -> None:
    def fake_settings() -> Settings:
        return Settings(rate_limit_requests=1, rate_limit_window_seconds=60)  # type: ignore[call-arg]

    monkeypatch.setattr(rate_limit_module, "get_settings", fake_settings)
    identity = ResolvedIdentity(user_id="alice")

    enforce_rate_limit(identity)  # first call: allowed

    with pytest.raises(HTTPException) as exc_info:
        enforce_rate_limit(identity)  # second call, same window: blocked

    assert exc_info.value.status_code == 429
    assert "Retry-After" in exc_info.value.headers  # type: ignore[operator]


async def test_chat_endpoint_returns_429_once_rate_limited(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    def fake_settings() -> Settings:
        return Settings(rate_limit_requests=1, rate_limit_window_seconds=60)  # type: ignore[call-arg]

    monkeypatch.setattr(rate_limit_module, "get_settings", fake_settings)

    async def fake_acompletion(*_args: object, **_kwargs: object) -> object:
        return _make_fake_stream(["hi"])

    monkeypatch.setattr("studylife_ai.llm.client.litellm.acompletion", fake_acompletion)
    _mock_no_retrieval(monkeypatch)

    body = {"messages": [{"role": "user", "content": "Hi"}]}
    first = await client.post("/chat", json=body)
    second = await client.post("/chat", json=body)

    assert first.status_code == 200
    assert second.status_code == 429
    assert "Retry-After" in second.headers
