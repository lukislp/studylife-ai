from unittest.mock import AsyncMock

import pytest
from pytest import MonkeyPatch

from studylife_ai.config import Settings
from studylife_ai.ingestion import sync as sync_module
from studylife_ai.studylife.models import StudyLifeNote


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "studylife_api_base_url": "http://studylife.test",
        "studylife_api_key": "secret",
        "embedding_model": "ollama/nomic-embed-text",
        "chunk_size_tokens": 500,
        "chunk_overlap_tokens": 75,
        "qdrant_url": "http://qdrant.test:6333",
        "qdrant_collection": "studylife_notes",
        "studylife_user_id": "primary",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


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


class FakeStudyLifeClient:
    def __init__(self, notes: list[StudyLifeNote]) -> None:
        self._notes = notes

    async def get_notes(self) -> list[StudyLifeNote]:
        return self._notes

    async def __aenter__(self) -> "FakeStudyLifeClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None


class FakeQdrantStore:
    def __init__(self, known: dict[int, str]) -> None:
        self._known = known
        self.ensure_collection = AsyncMock()
        self.replace_note = AsyncMock()
        self.delete_note = AsyncMock()
        self.close = AsyncMock()

    async def get_known_fingerprints(self) -> dict[int, str]:
        return self._known


async def test_sync_notes_raises_without_studylife_config() -> None:
    settings = _settings(studylife_api_base_url=None)

    with pytest.raises(RuntimeError):
        await sync_module.sync_notes(settings)


async def test_sync_notes_ingests_new_note(monkeypatch: MonkeyPatch) -> None:
    note = _note(1, "Linear Algebra", "Eigenvalues are important.")
    fake_client = FakeStudyLifeClient([note])
    fake_store = FakeQdrantStore(known={})

    async def fake_embed_texts(texts: list[str], *, model: str) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(sync_module, "StudyLifeClient", lambda **kwargs: fake_client)
    monkeypatch.setattr(sync_module, "QdrantStore", lambda **kwargs: fake_store)
    monkeypatch.setattr(sync_module, "embed_texts", fake_embed_texts)

    await sync_module.sync_notes(_settings())

    fake_store.ensure_collection.assert_awaited_once_with(vector_size=2)
    fake_store.replace_note.assert_awaited_once()
    _, kwargs = fake_store.replace_note.call_args
    assert kwargs["metadata"].note_id == 1
    assert kwargs["chunks"] == ["Eigenvalues are important."]
    fake_store.delete_note.assert_not_awaited()
    fake_store.close.assert_awaited_once()


async def test_sync_notes_skips_unchanged_note(monkeypatch: MonkeyPatch) -> None:
    note = _note(1, "Linear Algebra", "Eigenvalues are important.")
    fake_client = FakeStudyLifeClient([note])
    fake_store = FakeQdrantStore(known={1: sync_module.fingerprint_note(note)})
    embed_calls: list[list[str]] = []

    async def fake_embed_texts(texts: list[str], *, model: str) -> list[list[float]]:
        embed_calls.append(texts)
        return [[0.1] for _ in texts]

    monkeypatch.setattr(sync_module, "StudyLifeClient", lambda **kwargs: fake_client)
    monkeypatch.setattr(sync_module, "QdrantStore", lambda **kwargs: fake_store)
    monkeypatch.setattr(sync_module, "embed_texts", fake_embed_texts)

    await sync_module.sync_notes(_settings())

    assert embed_calls == []
    fake_store.replace_note.assert_not_awaited()
    fake_store.delete_note.assert_not_awaited()


async def test_sync_notes_deletes_notes_no_longer_present(monkeypatch: MonkeyPatch) -> None:
    fake_client = FakeStudyLifeClient([])
    fake_store = FakeQdrantStore(known={99: "stale-fingerprint"})

    async def fake_embed_texts(texts: list[str], *, model: str) -> list[list[float]]:
        return [[0.1] for _ in texts]

    monkeypatch.setattr(sync_module, "StudyLifeClient", lambda **kwargs: fake_client)
    monkeypatch.setattr(sync_module, "QdrantStore", lambda **kwargs: fake_store)
    monkeypatch.setattr(sync_module, "embed_texts", fake_embed_texts)

    await sync_module.sync_notes(_settings())

    fake_store.delete_note.assert_awaited_once_with(99)
    fake_store.replace_note.assert_not_awaited()
