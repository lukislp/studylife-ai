import logging

from httpx import AsyncClient
from pytest import LogCaptureFixture, MonkeyPatch

from studylife_ai.api.internal import SHARED_SECRET_HEADER, _sync_new_registration
from studylife_ai.config import Settings
from studylife_ai.main import app
from tests.conftest import TEST_SHARED_SECRET


async def test_register_key_stores_the_key(client: AsyncClient) -> None:
    response = await client.post(
        "/internal/register-key",
        json={"user_id": "alice", "ai_api_key": "key-a"},
        headers={SHARED_SECRET_HEADER: TEST_SHARED_SECRET},
    )

    assert response.status_code == 200
    assert await app.state.registered_key_store.get("alice") == "key-a"


async def test_register_key_overwrites_an_existing_entry(client: AsyncClient) -> None:
    await app.state.registered_key_store.set("alice", "old-key")

    response = await client.post(
        "/internal/register-key",
        json={"user_id": "alice", "ai_api_key": "new-key"},
        headers={SHARED_SECRET_HEADER: TEST_SHARED_SECRET},
    )

    assert response.status_code == 200
    assert await app.state.registered_key_store.get("alice") == "new-key"


async def test_register_key_rejects_a_wrong_secret(client: AsyncClient) -> None:
    response = await client.post(
        "/internal/register-key",
        json={"user_id": "alice", "ai_api_key": "key-a"},
        headers={SHARED_SECRET_HEADER: "wrong-secret"},
    )

    assert response.status_code == 401
    assert await app.state.registered_key_store.get("alice") is None


async def test_register_key_rejects_a_missing_secret(client: AsyncClient) -> None:
    response = await client.post(
        "/internal/register-key", json={"user_id": "alice", "ai_api_key": "key-a"}
    )

    assert response.status_code == 401


async def test_register_key_returns_503_when_secret_not_configured(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "studylife_ai.api.internal.get_settings",
        lambda: Settings(studylife_shared_secret=None),  # type: ignore[arg-type]
    )

    response = await client.post(
        "/internal/register-key",
        json={"user_id": "alice", "ai_api_key": "key-a"},
        headers={SHARED_SECRET_HEADER: TEST_SHARED_SECRET},
    )

    assert response.status_code == 503


async def test_register_key_schedules_auto_ingestion_for_the_new_user(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    calls = []

    async def fake_sync_user(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr("studylife_ai.api.internal.sync_user", fake_sync_user)

    response = await client.post(
        "/internal/register-key",
        json={"user_id": "alice", "ai_api_key": "key-a"},
        headers={SHARED_SECRET_HEADER: TEST_SHARED_SECRET},
    )

    assert response.status_code == 200
    assert len(calls) == 1
    assert calls[0]["user_id"] == "alice"
    assert calls[0]["ai_api_key"] == "key-a"


async def test_register_key_skips_auto_ingestion_when_studylife_not_configured(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    calls = []

    async def fake_sync_user(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr("studylife_ai.api.internal.sync_user", fake_sync_user)
    monkeypatch.setattr(
        "studylife_ai.api.internal.get_settings",
        lambda: Settings(  # type: ignore[call-arg]
            studylife_api_base_url=None, studylife_shared_secret=TEST_SHARED_SECRET
        ),
    )

    response = await client.post(
        "/internal/register-key",
        json={"user_id": "alice", "ai_api_key": "key-a"},
        headers={SHARED_SECRET_HEADER: TEST_SHARED_SECRET},
    )

    assert response.status_code == 200
    assert calls == []


async def test_sync_new_registration_logs_and_swallows_a_sync_failure(
    caplog: LogCaptureFixture, monkeypatch: MonkeyPatch
) -> None:
    async def failing_sync_user(**_kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("studylife_ai.api.internal.sync_user", failing_sync_user)

    with caplog.at_level(logging.ERROR):
        await _sync_new_registration("alice", "key-a", store=object())  # type: ignore[arg-type]

    assert "failed" in caplog.text.lower()
    assert "alice" in caplog.text


async def test_revoke_key_removes_the_key(client: AsyncClient) -> None:
    await app.state.registered_key_store.set("alice", "key-a")

    response = await client.post(
        "/internal/revoke-key",
        json={"user_id": "alice"},
        headers={SHARED_SECRET_HEADER: TEST_SHARED_SECRET},
    )

    assert response.status_code == 200
    assert await app.state.registered_key_store.get("alice") is None


async def test_revoke_key_rejects_a_wrong_secret(client: AsyncClient) -> None:
    await app.state.registered_key_store.set("alice", "key-a")

    response = await client.post(
        "/internal/revoke-key",
        json={"user_id": "alice"},
        headers={SHARED_SECRET_HEADER: "wrong-secret"},
    )

    assert response.status_code == 401
    assert await app.state.registered_key_store.get("alice") == "key-a"


async def test_revoke_key_is_a_noop_for_an_unknown_user(client: AsyncClient) -> None:
    response = await client.post(
        "/internal/revoke-key",
        json={"user_id": "does-not-exist"},
        headers={SHARED_SECRET_HEADER: TEST_SHARED_SECRET},
    )

    assert response.status_code == 200
