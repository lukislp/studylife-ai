"""Server-side AI consent gate (GDPR audit F-20): studylife/consent.py and its API wiring."""

import logging
from unittest.mock import AsyncMock

import httpx
import pytest
from httpx import AsyncClient
from pytest import LogCaptureFixture, MonkeyPatch

from studylife_ai.api import internal as internal_module
from studylife_ai.api.internal import SHARED_SECRET_HEADER
from studylife_ai.config import Settings
from studylife_ai.main import app
from studylife_ai.rag.enrichment import CaptureEnrichment
from studylife_ai.studylife import consent as consent_module
from studylife_ai.studylife.client import StudyLifeClient
from studylife_ai.studylife.consent import (
    ConsentRequired,
    ConsentUnverifiable,
    ensure_ai_consent,
    forget_consent,
)
from studylife_ai.studylife.registered_keys import RegisteredKeyStore
from tests.conftest import TEST_AI_KEY_ENCRYPTION_KEY, TEST_SHARED_SECRET

CONSENT_PROBLEM = {"type": "ai-consent-required", "title": "AI consent required"}


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "studylife_api_base_url": "http://studylife.test",
        "unsafe_disable_ai_consent_check": False,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


async def _store_with(user_id: str, key: str | None) -> RegisteredKeyStore:
    store = RegisteredKeyStore(":memory:", TEST_AI_KEY_ENCRYPTION_KEY)
    await store.setup()
    if key is not None:
        await store.set(user_id, key)
    return store


class _Recorder:
    def __init__(self, response: httpx.Response | Exception) -> None:
        self.calls: list[httpx.Request] = []
        self._response = response

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _transport(recorder: _Recorder) -> httpx.MockTransport:
    return httpx.MockTransport(recorder)


async def test_consent_in_place_passes_and_probes_with_the_users_own_key() -> None:
    store = await _store_with("alice", "key-a")
    recorder = _Recorder(httpx.Response(200, json=[]))

    await ensure_ai_consent(
        "alice", settings=_settings(), key_store=store, transport=_transport(recorder)
    )

    assert len(recorder.calls) == 1
    assert recorder.calls[0].headers["X-Api-Key"] == "key-a"
    await store.close()


async def test_missing_consent_raises_consent_required() -> None:
    store = await _store_with("alice", "key-a")
    recorder = _Recorder(httpx.Response(403, json=CONSENT_PROBLEM))

    with pytest.raises(ConsentRequired):
        await ensure_ai_consent(
            "alice", settings=_settings(), key_store=store, transport=_transport(recorder)
        )
    await store.close()


async def test_other_403_is_unverifiable_not_a_consent_answer() -> None:
    store = await _store_with("alice", "key-a")
    recorder = _Recorder(httpx.Response(403, json={"type": "something-else"}))

    with pytest.raises(ConsentUnverifiable):
        await ensure_ai_consent(
            "alice", settings=_settings(), key_store=store, transport=_transport(recorder)
        )
    await store.close()


async def test_rejected_key_and_server_errors_fail_closed() -> None:
    store = await _store_with("alice", "key-a")
    for status in (401, 500):
        recorder = _Recorder(httpx.Response(status))
        with pytest.raises(ConsentUnverifiable):
            await ensure_ai_consent(
                "alice", settings=_settings(), key_store=store, transport=_transport(recorder)
            )
    await store.close()


async def test_unreachable_studylife_fails_closed() -> None:
    store = await _store_with("alice", "key-a")
    recorder = _Recorder(httpx.ConnectError("boom"))

    with pytest.raises(ConsentUnverifiable):
        await ensure_ai_consent(
            "alice", settings=_settings(), key_store=store, transport=_transport(recorder)
        )
    await store.close()


async def test_user_without_registered_key_fails_closed() -> None:
    store = await _store_with("alice", None)
    recorder = _Recorder(httpx.Response(200, json=[]))

    with pytest.raises(ConsentUnverifiable):
        await ensure_ai_consent(
            "alice", settings=_settings(), key_store=store, transport=_transport(recorder)
        )

    assert recorder.calls == []
    await store.close()


async def test_positive_answer_is_cached_and_forgotten_on_demand() -> None:
    store = await _store_with("alice", "key-a")
    recorder = _Recorder(httpx.Response(200, json=[]))
    transport = _transport(recorder)

    for _ in range(3):
        await ensure_ai_consent("alice", settings=_settings(), key_store=store, transport=transport)
    assert len(recorder.calls) == 1

    forget_consent("alice")
    await ensure_ai_consent("alice", settings=_settings(), key_store=store, transport=transport)
    assert len(recorder.calls) == 2
    await store.close()


async def test_refusal_is_never_cached() -> None:
    store = await _store_with("alice", "key-a")
    refusing = _Recorder(httpx.Response(403, json=CONSENT_PROBLEM))
    with pytest.raises(ConsentRequired):
        await ensure_ai_consent(
            "alice", settings=_settings(), key_store=store, transport=_transport(refusing)
        )

    granting = _Recorder(httpx.Response(200, json=[]))
    await ensure_ai_consent(
        "alice", settings=_settings(), key_store=store, transport=_transport(granting)
    )
    assert len(granting.calls) == 1
    await store.close()


async def test_withdrawal_after_cache_expiry_is_picked_up() -> None:
    store = await _store_with("alice", "key-a")
    settings = _settings(ai_consent_cache_seconds=0.0)
    await ensure_ai_consent(
        "alice",
        settings=settings,
        key_store=store,
        transport=_transport(_Recorder(httpx.Response(200, json=[]))),
    )

    with pytest.raises(ConsentRequired):
        await ensure_ai_consent(
            "alice",
            settings=settings,
            key_store=store,
            transport=_transport(_Recorder(httpx.Response(403, json=CONSENT_PROBLEM))),
        )
    await store.close()


async def test_explicit_opt_out_skips_the_check() -> None:
    store = await _store_with("alice", None)  # would be unverifiable if checked
    await ensure_ai_consent(
        "alice", settings=_settings(unsafe_disable_ai_consent_check=True), key_store=store
    )
    await store.close()


@pytest.mark.parametrize("environment", ["local", "staging"])
async def test_missing_studylife_url_fails_closed_outside_production(environment: str) -> None:
    store = await _store_with("alice", "key-a")
    settings = Settings(
        _env_file=None,
        environment=environment,  # type: ignore[arg-type]
        studylife_api_base_url=None,
        unsafe_disable_ai_consent_check=False,
    )
    with pytest.raises(ConsentUnverifiable):
        await ensure_ai_consent("alice", settings=settings, key_store=store)
    await store.close()


async def test_the_environment_name_alone_never_skips_the_check() -> None:
    store = await _store_with("alice", None)
    for environment in ("local", "staging"):
        settings = Settings(
            _env_file=None,
            environment=environment,  # type: ignore[arg-type]
            studylife_api_base_url="http://studylife.test",
            unsafe_disable_ai_consent_check=False,
        )
        with pytest.raises(ConsentUnverifiable):  # no registered key -> cannot verify
            await ensure_ai_consent("alice", settings=settings, key_store=store)
    await store.close()


async def test_unverifiable_log_does_not_contain_the_api_key(caplog: LogCaptureFixture) -> None:
    store = await _store_with("alice", "super-secret-key")
    recorder = _Recorder(httpx.Response(500))
    with caplog.at_level(logging.WARNING), pytest.raises(ConsentUnverifiable):
        await ensure_ai_consent(
            "alice", settings=_settings(), key_store=store, transport=_transport(recorder)
        )
    assert "super-secret-key" not in caplog.text
    await store.close()


# --- API wiring -------------------------------------------------------------------------------


def _refuse(error: Exception):
    async def _raiser(*_args: object, **_kwargs: object) -> None:
        raise error

    return _raiser


async def test_enrich_capture_is_refused_without_consent_and_never_reaches_the_model(
    internal_client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr("studylife_ai.api.consent.ensure_ai_consent", _refuse(ConsentRequired()))
    called = []

    async def fake_enrich_capture(*_args: object, **_kwargs: object) -> CaptureEnrichment:
        called.append(True)
        return CaptureEnrichment(None, None, [], None, [])

    monkeypatch.setattr(internal_module, "enrich_capture", fake_enrich_capture)

    response = await internal_client.post(
        "/internal/enrich-capture",
        json={"user_id": "alice", "note_id": 1, "title": "t", "content": "c"},
        headers={SHARED_SECRET_HEADER: TEST_SHARED_SECRET},
    )

    assert response.status_code == 403
    assert "ai-consent-required" in response.json()["detail"]
    assert called == []


async def test_enrich_capture_fails_closed_when_consent_is_unverifiable(
    internal_client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "studylife_ai.api.consent.ensure_ai_consent", _refuse(ConsentUnverifiable("x"))
    )
    called = []

    async def fake_enrich_capture(*_args: object, **_kwargs: object) -> CaptureEnrichment:
        called.append(True)
        return CaptureEnrichment(None, None, [], None, [])

    monkeypatch.setattr(internal_module, "enrich_capture", fake_enrich_capture)

    response = await internal_client.post(
        "/internal/enrich-capture",
        json={"user_id": "alice", "note_id": 1, "title": "t", "content": "c"},
        headers={SHARED_SECRET_HEADER: TEST_SHARED_SECRET},
    )

    assert response.status_code == 503
    assert called == []


async def test_enrich_capture_asks_studylife_with_the_users_key_end_to_end(
    internal_client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    # Real gate (undo the conftest default), StudyAtrium answers 403 consent-required.
    monkeypatch.setattr(
        "studylife_ai.api.consent.ensure_ai_consent", consent_module.ensure_ai_consent
    )
    recorder = _Recorder(httpx.Response(403, json=CONSENT_PROBLEM))
    real_client = consent_module.StudyLifeClient

    def client_with_mock_transport(**kwargs: object) -> StudyLifeClient:
        kwargs["transport"] = _transport(recorder)
        return real_client(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(consent_module, "StudyLifeClient", client_with_mock_transport)
    monkeypatch.setattr("studylife_ai.api.consent.get_settings", _settings)
    await app.state.registered_key_store.set("alice", "key-a")

    response = await internal_client.post(
        "/internal/enrich-capture",
        json={"user_id": "alice", "note_id": 1, "title": "t", "content": "c"},
        headers={SHARED_SECRET_HEADER: TEST_SHARED_SECRET},
    )

    assert response.status_code == 403
    assert recorder.calls[0].headers["X-Api-Key"] == "key-a"


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/chat", {"messages": [{"role": "user", "content": "hi"}]}),
        ("/agent", {"message": "hi"}),
        ("/agent/confirm", {"thread_id": "test-user:abc", "decision": "approve"}),
    ],
)
async def test_user_facing_ai_routes_are_refused_without_consent(
    client: AsyncClient, monkeypatch: MonkeyPatch, path: str, body: dict[str, object]
) -> None:
    monkeypatch.setattr("studylife_ai.api.consent.ensure_ai_consent", _refuse(ConsentRequired()))

    response = await client.post(path, json=body)

    assert response.status_code == 403
    assert "ai-consent-required" in response.json()["detail"]


async def test_revoke_key_drops_the_cached_consent(
    internal_client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    recorder = _Recorder(httpx.Response(200, json=[]))
    store = await _store_with("alice", "key-a")
    await ensure_ai_consent(
        "alice", settings=_settings(), key_store=store, transport=_transport(recorder)
    )
    assert "alice" in consent_module._allowed_until
    await app.state.registered_key_store.set("alice", "key-a")
    monkeypatch.setattr(app.state, "qdrant_store", AsyncMock())

    await internal_client.post(
        "/internal/revoke-key",
        json={"user_id": "alice"},
        headers={SHARED_SECRET_HEADER: TEST_SHARED_SECRET},
    )

    assert "alice" not in consent_module._allowed_until
    await store.close()
