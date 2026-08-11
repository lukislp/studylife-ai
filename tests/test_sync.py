from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from pytest import MonkeyPatch

from studylife_ai.config import Settings
from studylife_ai.ingestion import sync as sync_module
from studylife_ai.studylife.models import CourseDto, CourseGoalDto, StudyLifeNote, StudySessionDto
from studylife_ai.studylife.registered_keys import RegisteredKeyStore


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "studylife_api_base_url": "http://studylife.test",
        "embedding_model": "ollama/nomic-embed-text",
        "chunk_size_tokens": 500,
        "chunk_overlap_tokens": 75,
        "qdrant_url": "http://qdrant.test:6333",
        "qdrant_collection": "studylife_notes",
        "studylife_session_history_days": 1825,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


async def _install_registered_users(monkeypatch: MonkeyPatch, users: dict[str, str]) -> None:
    """sync_all() builds its own RegisteredKeyStore internally (from
    settings.registered_keys_db_path) - swap in an in-memory one pre-loaded
    with the given {user_id: ai_api_key} pairs instead of touching a real
    file. sync_all() calls .setup() on it too, which is a harmless no-op
    against an already-set-up store (CREATE TABLE IF NOT EXISTS)."""
    store = RegisteredKeyStore(":memory:")
    await store.setup()
    for user_id, ai_api_key in users.items():
        await store.set(user_id, ai_api_key)
    monkeypatch.setattr(sync_module, "RegisteredKeyStore", lambda db_path: store)


def _note(note_id: int, title: str, content: str) -> StudyLifeNote:
    return StudyLifeNote(
        id=note_id,
        title=title,
        content=content,
        created_at="2026-08-01T10:00:00",  # type: ignore[arg-type]
        updated_at="2026-08-01T10:00:00",  # type: ignore[arg-type]
        course_id=None,
        session_id=None,
    )


def _course(course_id: int, name: str) -> CourseDto:
    return CourseDto(id=course_id, semester=3, name=name, code="X101", topics=[], ects=5)


def _session(session_id: int, course_id: int) -> StudySessionDto:
    return StudySessionDto(
        id=session_id,
        course_id=course_id,
        course_name="Lineare Algebra",
        start_time=datetime(2026, 8, 1, 10, 0),
        end_time=datetime(2026, 8, 1, 11, 30),
        is_completed=False,
    )


def _course_goal(course_id: int) -> CourseGoalDto:
    return CourseGoalDto(course_id=course_id, course_name="Lineare Algebra")


class FakeStudyLifeClient:
    def __init__(
        self,
        notes: list[StudyLifeNote] | None = None,
        courses: list[CourseDto] | None = None,
        sessions: list[StudySessionDto] | None = None,
        course_goals: list[CourseGoalDto] | None = None,
    ) -> None:
        self._notes = notes or []
        self._courses = courses or []
        self._sessions = sessions or []
        self._course_goals = course_goals or []

    async def get_notes(self) -> list[StudyLifeNote]:
        return self._notes

    async def get_courses(self) -> list[CourseDto]:
        return self._courses

    async def get_sessions_history(
        self, *, days: int, only_completed: bool
    ) -> list[StudySessionDto]:
        return self._sessions

    async def get_course_goals(self) -> list[CourseGoalDto]:
        return self._course_goals

    async def __aenter__(self) -> "FakeStudyLifeClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None


class FakeQdrantStore:
    def __init__(self, known: dict[tuple[str, int], str]) -> None:
        self._known = known
        self.ensure_collection = AsyncMock()
        self.replace_entity = AsyncMock()
        self.delete_entity = AsyncMock()
        self.close = AsyncMock()

    async def get_known_fingerprints(self, *, user_id: str) -> dict[tuple[str, int], str]:
        return self._known


async def test_sync_raises_without_studylife_base_url() -> None:
    settings = _settings(studylife_api_base_url=None)

    with pytest.raises(RuntimeError, match="STUDYLIFE_API_BASE_URL"):
        await sync_module.sync_all(settings)


async def test_sync_raises_without_any_registered_users(monkeypatch: MonkeyPatch) -> None:
    await _install_registered_users(monkeypatch, {})

    with pytest.raises(RuntimeError, match="No users registered"):
        await sync_module.sync_all(_settings())


async def _run_sync_all(
    monkeypatch: MonkeyPatch, fake_client: FakeStudyLifeClient, fake_store: FakeQdrantStore
) -> None:
    async def fake_embed_texts(
        texts: list[str], *, model: str, **_kwargs: object
    ) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(sync_module, "StudyLifeClient", lambda **kwargs: fake_client)
    monkeypatch.setattr(sync_module, "QdrantStore", lambda **kwargs: fake_store)
    monkeypatch.setattr(sync_module, "embed_texts", fake_embed_texts)
    await _install_registered_users(monkeypatch, {"primary": "secret"})

    await sync_module.sync_all(_settings())


async def test_sync_ingests_new_note(monkeypatch: MonkeyPatch) -> None:
    note = _note(1, "Linear Algebra", "Eigenvalues are important.")
    fake_client = FakeStudyLifeClient(notes=[note])
    fake_store = FakeQdrantStore(known={})

    await _run_sync_all(monkeypatch, fake_client, fake_store)

    fake_store.ensure_collection.assert_awaited_once_with(vector_size=2)
    fake_store.replace_entity.assert_awaited_once()
    _, kwargs = fake_store.replace_entity.call_args
    assert kwargs["metadata"].content_type == "note"
    assert kwargs["metadata"].entity_id == 1
    assert kwargs["chunks"] == ["Eigenvalues are important."]
    fake_store.delete_entity.assert_not_awaited()
    fake_store.close.assert_awaited_once()


async def test_sync_skips_unchanged_note(monkeypatch: MonkeyPatch) -> None:
    note = _note(1, "Linear Algebra", "Eigenvalues are important.")
    fake_client = FakeStudyLifeClient(notes=[note])
    fake_store = FakeQdrantStore(known={("note", 1): sync_module.fingerprint_note(note)})

    await _run_sync_all(monkeypatch, fake_client, fake_store)

    fake_store.replace_entity.assert_not_awaited()
    fake_store.delete_entity.assert_not_awaited()


async def test_sync_deletes_notes_no_longer_present(monkeypatch: MonkeyPatch) -> None:
    fake_client = FakeStudyLifeClient(notes=[])
    fake_store = FakeQdrantStore(known={("note", 99): "stale-fingerprint"})

    await _run_sync_all(monkeypatch, fake_client, fake_store)

    fake_store.delete_entity.assert_awaited_once_with(
        user_id="primary", content_type="note", entity_id=99
    )
    fake_store.replace_entity.assert_not_awaited()


async def test_sync_ingests_new_course(monkeypatch: MonkeyPatch) -> None:
    course = _course(6, "Lineare Algebra")
    fake_client = FakeStudyLifeClient(courses=[course])
    fake_store = FakeQdrantStore(known={})

    await _run_sync_all(monkeypatch, fake_client, fake_store)

    fake_store.replace_entity.assert_awaited_once()
    _, kwargs = fake_store.replace_entity.call_args
    assert kwargs["metadata"].content_type == "course"
    assert kwargs["metadata"].entity_id == 6
    assert kwargs["metadata"].course_id is None


async def test_sync_ingests_new_session(monkeypatch: MonkeyPatch) -> None:
    session = _session(42, course_id=6)
    fake_client = FakeStudyLifeClient(sessions=[session])
    fake_store = FakeQdrantStore(known={})

    await _run_sync_all(monkeypatch, fake_client, fake_store)

    fake_store.replace_entity.assert_awaited_once()
    _, kwargs = fake_store.replace_entity.call_args
    assert kwargs["metadata"].content_type == "session"
    assert kwargs["metadata"].entity_id == 42
    assert kwargs["metadata"].course_id == 6
    assert kwargs["metadata"].session_start == "2026-08-01T10:00:00"


async def test_sync_reingests_a_session_whose_fingerprint_predates_the_schema_bump(
    monkeypatch: MonkeyPatch,
) -> None:
    """Migration regression test (see docs/decisions.md "Structured session dates"):
    fingerprint_session() was bumped with a schema marker specifically so an
    already-ingested session (whose stored fingerprint is the pre-bump hash) looks "changed" on
    the next sync and gets re-upserted with the new session_start field - otherwise it would
    never get backfilled, since its actual content never changed."""
    import hashlib

    from studylife_ai.ingestion.rendering import render_session

    session = _session(42, course_id=6)
    pre_migration_fingerprint = hashlib.sha256(render_session(session).encode()).hexdigest()
    fake_client = FakeStudyLifeClient(sessions=[session])
    fake_store = FakeQdrantStore(known={("session", 42): pre_migration_fingerprint})

    await _run_sync_all(monkeypatch, fake_client, fake_store)

    fake_store.replace_entity.assert_awaited_once()
    _, kwargs = fake_store.replace_entity.call_args
    assert kwargs["metadata"].session_start == "2026-08-01T10:00:00"


async def test_sync_ingests_new_course_goal_keyed_by_course_id(monkeypatch: MonkeyPatch) -> None:
    goal = _course_goal(course_id=6)
    fake_client = FakeStudyLifeClient(course_goals=[goal])
    fake_store = FakeQdrantStore(known={})

    await _run_sync_all(monkeypatch, fake_client, fake_store)

    fake_store.replace_entity.assert_awaited_once()
    _, kwargs = fake_store.replace_entity.call_args
    assert kwargs["metadata"].content_type == "course_goal"
    assert kwargs["metadata"].entity_id == 6


async def test_sync_warns_when_two_course_goals_share_a_course_id(
    monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # CourseGoalDto has no own id - entity_id falls back to course_id, which
    # assumes the API never returns more than one goal per course. If it
    # ever does, only one survives the upsert; this must at least be logged.
    goal_a = _course_goal(course_id=6)
    goal_b = CourseGoalDto(course_id=6, course_name="Lineare Algebra", tag="duplicate")
    fake_client = FakeStudyLifeClient(course_goals=[goal_a, goal_b])
    fake_store = FakeQdrantStore(known={})

    with caplog.at_level("WARNING", logger="studylife_ai.ingestion.sync"):
        await _run_sync_all(monkeypatch, fake_client, fake_store)

    assert any("share an entity_id" in record.message for record in caplog.records)


async def test_sync_deletes_course_goal_when_its_course_disappears(
    monkeypatch: MonkeyPatch,
) -> None:
    fake_client = FakeStudyLifeClient(course_goals=[])
    fake_store = FakeQdrantStore(known={("course_goal", 6): "stale"})

    await _run_sync_all(monkeypatch, fake_client, fake_store)

    fake_store.delete_entity.assert_awaited_once_with(
        user_id="primary", content_type="course_goal", entity_id=6
    )


async def test_sync_does_not_confuse_a_note_and_a_course_sharing_the_same_id(
    monkeypatch: MonkeyPatch,
) -> None:
    """A note and a course can both have numeric id=5 - the cross-type collision
    regression test: only the note changed, the course must be left untouched."""
    note = _note(5, "Linear Algebra", "New content.")
    course = _course(5, "Lineare Algebra")
    fake_client = FakeStudyLifeClient(notes=[note], courses=[course])
    fake_store = FakeQdrantStore(
        known={
            ("note", 5): "stale-note-fingerprint",
            ("course", 5): sync_module.fingerprint_course(course),
        }
    )

    await _run_sync_all(monkeypatch, fake_client, fake_store)

    assert fake_store.replace_entity.await_count == 1
    _, kwargs = fake_store.replace_entity.call_args
    assert kwargs["metadata"].content_type == "note"
    fake_store.delete_entity.assert_not_awaited()


async def test_sync_all_closes_store_once_even_with_empty_entity_lists(
    monkeypatch: MonkeyPatch,
) -> None:
    fake_client = FakeStudyLifeClient()
    fake_store = FakeQdrantStore(known={})

    await _run_sync_all(monkeypatch, fake_client, fake_store)

    fake_store.close.assert_awaited_once()
    fake_store.replace_entity.assert_not_awaited()
    fake_store.delete_entity.assert_not_awaited()


async def test_sync_all_syncs_every_registered_user_with_their_own_key(
    monkeypatch: MonkeyPatch,
) -> None:
    """Multi-user (see docs/decisions.md "M4.5 Multi-user support"):
    sync_all() must build a separate StudyLifeClient per registered user,
    using that user's own ai_api_key - never one shared credential."""
    constructed_keys: list[object] = []

    def fake_client_factory(**kwargs: object) -> FakeStudyLifeClient:
        constructed_keys.append(kwargs["api_key"])
        return FakeStudyLifeClient()

    fake_store = FakeQdrantStore(known={})

    async def fake_embed_texts(
        texts: list[str], *, model: str, **_kwargs: object
    ) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(sync_module, "StudyLifeClient", fake_client_factory)
    monkeypatch.setattr(sync_module, "QdrantStore", lambda **kwargs: fake_store)
    monkeypatch.setattr(sync_module, "embed_texts", fake_embed_texts)
    await _install_registered_users(monkeypatch, {"alice": "key-a", "bob": "key-b"})

    await sync_module.sync_all(_settings())

    assert sorted(constructed_keys) == ["key-a", "key-b"]
    fake_store.close.assert_awaited_once()


async def test_sync_all_continues_with_other_users_after_one_fails(
    monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A revoked key or StudyLife-side error for one user must not starve
    ingestion for every other registered user."""
    synced_users: list[str] = []

    def fake_client_factory(**kwargs: object) -> FakeStudyLifeClient:
        if kwargs["api_key"] == "key-a":
            raise RuntimeError("401 Unauthorized")
        synced_users.append(str(kwargs["api_key"]))
        return FakeStudyLifeClient()

    fake_store = FakeQdrantStore(known={})

    async def fake_embed_texts(
        texts: list[str], *, model: str, **_kwargs: object
    ) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(sync_module, "StudyLifeClient", fake_client_factory)
    monkeypatch.setattr(sync_module, "QdrantStore", lambda **kwargs: fake_store)
    monkeypatch.setattr(sync_module, "embed_texts", fake_embed_texts)
    await _install_registered_users(monkeypatch, {"alice": "key-a", "bob": "key-b"})

    with caplog.at_level("ERROR", logger="studylife_ai.ingestion.sync"):
        await sync_module.sync_all(_settings())

    assert synced_users == ["key-b"]
    assert any("failed for user_id=alice" in record.message for record in caplog.records)
    fake_store.close.assert_awaited_once()
