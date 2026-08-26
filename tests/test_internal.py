import logging
from pathlib import Path
from unittest.mock import AsyncMock

from httpx import AsyncClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
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


# --- Audit A5: split STUDYLIFE_INTERNAL_API_SECRET, with a legacy STUDYLIFE_SHARED_SECRET
# fallback (may itself be a comma-separated list of accepted values). ---


async def test_register_key_accepts_the_new_internal_api_secret(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "studylife_ai.api.internal.get_settings",
        lambda: Settings(  # type: ignore[call-arg]
            studylife_shared_secret=None, studylife_internal_api_secret="new-internal-secret"
        ),
    )

    response = await client.post(
        "/internal/register-key",
        json={"user_id": "alice", "ai_api_key": "key-a"},
        headers={SHARED_SECRET_HEADER: "new-internal-secret"},
    )

    assert response.status_code == 200


async def test_register_key_rejects_the_legacy_secret_once_the_new_one_replaces_it(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "studylife_ai.api.internal.get_settings",
        lambda: Settings(  # type: ignore[call-arg]
            studylife_shared_secret=None, studylife_internal_api_secret="new-internal-secret"
        ),
    )

    response = await client.post(
        "/internal/register-key",
        json={"user_id": "alice", "ai_api_key": "key-a"},
        headers={SHARED_SECRET_HEADER: TEST_SHARED_SECRET},
    )

    assert response.status_code == 401


async def test_register_key_accepts_either_value_of_a_comma_separated_internal_api_secret(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    """Rotation (audit A5): both the old and the new value are accepted while both are listed,
    so StudyLife's backend (which always sends only the first value from its own config) can
    switch over without a coordinated cutover."""
    monkeypatch.setattr(
        "studylife_ai.api.internal.get_settings",
        lambda: Settings(  # type: ignore[call-arg]
            studylife_shared_secret=None,
            studylife_internal_api_secret="new-internal-secret,old-internal-secret",
        ),
    )

    for secret in ("new-internal-secret", "old-internal-secret"):
        response = await client.post(
            "/internal/register-key",
            json={"user_id": "alice", "ai_api_key": "key-a"},
            headers={SHARED_SECRET_HEADER: secret},
        )
        assert response.status_code == 200


async def test_register_key_still_accepts_the_legacy_secret_while_it_is_configured(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    """Rollout compatibility (audit A5): the legacy secret keeps working alongside the new one,
    not just as an either/or fallback - so StudyLife's backend and studylife-ai can deploy the
    split independently, in either order."""
    monkeypatch.setattr(
        "studylife_ai.api.internal.get_settings",
        lambda: Settings(  # type: ignore[call-arg]
            studylife_shared_secret=TEST_SHARED_SECRET,
            studylife_internal_api_secret="new-internal-secret",
        ),
    )

    for secret in ("new-internal-secret", TEST_SHARED_SECRET):
        response = await client.post(
            "/internal/register-key",
            json={"user_id": "alice", "ai_api_key": "key-a"},
            headers={SHARED_SECRET_HEADER: secret},
        )
        assert response.status_code == 200


async def test_register_key_returns_503_when_neither_secret_is_configured(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "studylife_ai.api.internal.get_settings",
        lambda: Settings(  # type: ignore[call-arg]
            studylife_shared_secret=None, studylife_internal_api_secret=None
        ),
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


# --- F5/F13: revoke-key now does a full purge (registration + Qdrant partition + checkpoint
# threads), via the same ingestion.sync.purge_user() the sync loop's zombie cleanup uses. ---


async def test_revoke_key_purges_the_users_qdrant_partition(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    fake_store = AsyncMock()
    monkeypatch.setattr(app.state, "qdrant_store", fake_store)

    response = await client.post(
        "/internal/revoke-key",
        json={"user_id": "alice"},
        headers={SHARED_SECRET_HEADER: TEST_SHARED_SECRET},
    )

    assert response.status_code == 200
    fake_store.delete_user.assert_awaited_once_with(user_id="alice")


async def test_revoke_key_purges_only_the_target_users_checkpoint_threads(
    client: AsyncClient, monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.db")) as checkpointer:
        await checkpointer.setup()
        # Mirrors the real f"{user_id}:{uuid4()}" thread_id shape (see api/agent.py) - the exact
        # suffix doesn't matter, only that it's prefixed by the owning user_id.
        await checkpointer.conn.execute(
            "INSERT INTO checkpoints "
            "(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, "
            "metadata) VALUES (?, '', 'chk-1', NULL, 'json', ?, ?)",
            ("alice:thread-1", b"{}", b"{}"),
        )
        await checkpointer.conn.execute(
            "INSERT INTO checkpoints "
            "(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, "
            "metadata) VALUES (?, '', 'chk-1', NULL, 'json', ?, ?)",
            ("bob:thread-1", b"{}", b"{}"),
        )
        await checkpointer.conn.commit()

        monkeypatch.setattr(app.state, "qdrant_store", AsyncMock())
        monkeypatch.setattr(app.state, "agent_checkpointer", checkpointer)

        response = await client.post(
            "/internal/revoke-key",
            json={"user_id": "alice"},
            headers={SHARED_SECRET_HEADER: TEST_SHARED_SECRET},
        )

        assert response.status_code == 200
        async with checkpointer.conn.execute("SELECT thread_id FROM checkpoints") as cursor:
            remaining = {row[0] for row in await cursor.fetchall()}
        assert remaining == {"bob:thread-1"}


async def test_revoke_key_still_deletes_the_registration_when_qdrant_deletion_fails(
    client: AsyncClient, monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """Best-effort purge (F5/F13): a Qdrant hiccup must not leave the key registered - the
    registration row is deleted regardless, after the best-effort deletes are attempted."""
    await app.state.registered_key_store.set("alice", "key-a")
    fake_store = AsyncMock()
    fake_store.delete_user.side_effect = RuntimeError("qdrant unreachable")
    monkeypatch.setattr(app.state, "qdrant_store", fake_store)

    with caplog.at_level(logging.ERROR, logger="studylife_ai.ingestion.sync"):
        response = await client.post(
            "/internal/revoke-key",
            json={"user_id": "alice"},
            headers={SHARED_SECRET_HEADER: TEST_SHARED_SECRET},
        )

    assert response.status_code == 200
    assert await app.state.registered_key_store.get("alice") is None
    assert any("Qdrant" in record.message for record in caplog.records)


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
