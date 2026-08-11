from pathlib import Path
from unittest.mock import AsyncMock

from pytest import MonkeyPatch

from studylife_ai.config import Settings
from studylife_ai.eval import fixture as fixture_module
from studylife_ai.eval.fixture import FixtureNote, load_fixture_notes, seed_fixture_notes


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "embedding_model": "openai/text-embedding-3-small",
        "chunk_size_tokens": 500,
        "chunk_overlap_tokens": 75,
        "qdrant_url": "http://qdrant.test:6333",
        "qdrant_collection": "studylife_notes",
        "studylife_user_id": "primary",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


class FakeQdrantStore:
    def __init__(self) -> None:
        self.ensure_collection = AsyncMock()
        self.replace_entity = AsyncMock()


def test_load_fixture_notes_parses_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "fixture_notes.jsonl"
    path.write_text(
        '{"id": 1, "title": "Eigenwerte", "content": "det(A - \\u03bbI) = 0", "course_id": 6}\n',
        encoding="utf-8",
    )

    notes = load_fixture_notes(path)

    assert notes == [FixtureNote(id=1, title="Eigenwerte", content="det(A - λI) = 0", course_id=6)]


async def test_seed_fixture_notes_chunks_embeds_and_upserts_each_note(
    monkeypatch: MonkeyPatch,
) -> None:
    notes = [FixtureNote(id=1, title="Eigenwerte", content="det(A - λI) = 0", course_id=6)]
    store = FakeQdrantStore()

    async def fake_embed_texts(texts: list[str], *, model: str) -> list[list[float]]:
        assert model == "openai/text-embedding-3-small"
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(fixture_module, "embed_texts", fake_embed_texts)

    await seed_fixture_notes(notes, settings=_settings(), store=store)  # type: ignore[arg-type]

    store.ensure_collection.assert_awaited_once_with(vector_size=2)
    store.replace_entity.assert_awaited_once()
    _, kwargs = store.replace_entity.call_args
    assert kwargs["chunks"] == ["det(A - λI) = 0"]
    assert kwargs["metadata"].content_type == "note"
    assert kwargs["metadata"].entity_id == 1
    assert kwargs["metadata"].title == "Eigenwerte"
    assert kwargs["metadata"].course_id == 6
    assert kwargs["metadata"].user_id == "primary"
