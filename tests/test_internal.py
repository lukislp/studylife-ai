import logging

from httpx import AsyncClient
from pytest import LogCaptureFixture, MonkeyPatch

from studylife_ai.api import internal as internal_module
from studylife_ai.api.internal import SHARED_SECRET_HEADER, _sync_new_registration
from studylife_ai.config import Settings
from studylife_ai.main import app
from studylife_ai.rag.enrichment import CaptureEnrichment
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


async def test_enrich_capture_returns_the_enrichment_result(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    calls = []

    async def fake_enrich_capture(
        note_id: int, title: str, content: str, **kwargs: object
    ) -> CaptureEnrichment:
        calls.append({"note_id": note_id, "title": title, "content": content, **kwargs})
        return CaptureEnrichment(
            course_id=7,
            course_confidence=0.9,
            tags=["a", "b"],
            summary="S.",
            related_note_ids=[3, 5],
        )

    monkeypatch.setattr(internal_module, "enrich_capture", fake_enrich_capture)

    response = await client.post(
        "/internal/enrich-capture",
        json={
            "user_id": "alice",
            "note_id": 42,
            "title": "My note",
            "content": "Some captured content",
            "source_url": "https://example.com/article",
            "active_course_ids": [1, 2, 3],
        },
        headers={SHARED_SECRET_HEADER: TEST_SHARED_SECRET},
    )

    assert response.status_code == 200
    assert response.json() == {
        "course_id": 7,
        "course_confidence": 0.9,
        "tags": ["a", "b"],
        "summary": "S.",
        "related_note_ids": [3, 5],
    }
    assert len(calls) == 1
    assert calls[0]["note_id"] == 42
    assert calls[0]["active_course_ids"] == [1, 2, 3]


async def test_enrich_capture_defaults_active_course_ids_to_empty_list(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    calls = []

    async def fake_enrich_capture(
        note_id: int, title: str, content: str, **kwargs: object
    ) -> CaptureEnrichment:
        calls.append(kwargs)
        return CaptureEnrichment(
            course_id=None, course_confidence=None, tags=[], summary=None, related_note_ids=[]
        )

    monkeypatch.setattr(internal_module, "enrich_capture", fake_enrich_capture)

    response = await client.post(
        "/internal/enrich-capture",
        json={"user_id": "alice", "note_id": 42, "title": "T", "content": "C"},
        headers={SHARED_SECRET_HEADER: TEST_SHARED_SECRET},
    )

    assert response.status_code == 200
    assert calls[0]["active_course_ids"] == []


async def test_enrich_capture_rejects_a_wrong_secret(client: AsyncClient) -> None:
    response = await client.post(
        "/internal/enrich-capture",
        json={"user_id": "alice", "note_id": 42, "title": "T", "content": "C"},
        headers={SHARED_SECRET_HEADER: "wrong-secret"},
    )

    assert response.status_code == 401


async def test_enrich_capture_rejects_a_missing_secret(client: AsyncClient) -> None:
    response = await client.post(
        "/internal/enrich-capture",
        json={"user_id": "alice", "note_id": 42, "title": "T", "content": "C"},
    )

    assert response.status_code == 401
