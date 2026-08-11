from httpx import AsyncClient
from pytest import MonkeyPatch

from studylife_ai.api.internal import SHARED_SECRET_HEADER
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
