from pathlib import Path
from unittest.mock import AsyncMock

from pytest import MonkeyPatch

from studylife_ai.config import Settings
from studylife_ai.eval import fixture as fixture_module
from studylife_ai.eval.fixture import (
    FixtureNote,
    load_fixture_course_goals,
    load_fixture_courses,
    load_fixture_notes,
    load_fixture_sessions,
    seed_fixture_course_goals,
    seed_fixture_courses,
    seed_fixture_notes,
    seed_fixture_sessions,
)
from studylife_ai.studylife.models import CourseDto, CourseGoalDto, StudySessionDto


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "embedding_model": "openai/text-embedding-3-small",
        "chunk_size_tokens": 500,
        "chunk_overlap_tokens": 75,
        "qdrant_url": "http://qdrant.test:6333",
        "qdrant_collection": "studylife_notes",
        "eval_user_id": "eval-user",
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

    async def fake_embed_texts(
        texts: list[str], *, model: str, **_kwargs: object
    ) -> list[list[float]]:
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
    assert kwargs["metadata"].user_id == "eval-user"


def test_load_fixture_courses_parses_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "fixture_courses.jsonl"
    path.write_text('{"id": 1, "name": "Analysis", "ects": 8}\n', encoding="utf-8")

    courses = load_fixture_courses(path)

    assert courses == [CourseDto(id=1, name="Analysis", ects=8)]


def test_load_fixture_sessions_parses_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "fixture_sessions.jsonl"
    path.write_text(
        '{"id": 101, "course_id": 1, "course_name": "Algorithmen", '
        '"start_time": "2026-08-11T16:00:00", "end_time": "2026-08-11T17:30:00"}\n',
        encoding="utf-8",
    )

    sessions = load_fixture_sessions(path)

    assert len(sessions) == 1
    assert sessions[0].id == 101
    assert sessions[0].course_name == "Algorithmen"


def test_load_fixture_course_goals_parses_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "fixture_course_goals.jsonl"
    path.write_text(
        '{"course_id": 9, "course_name": "Studium Generale I", "grade": 1.0, '
        '"completed_at": "2026-07-12T10:00:00"}\n',
        encoding="utf-8",
    )

    goals = load_fixture_course_goals(path)

    assert len(goals) == 1
    assert goals[0].course_id == 9
    assert goals[0].grade == 1.0


async def test_seed_fixture_courses_upserts_via_the_real_ingestion_path(
    monkeypatch: MonkeyPatch,
) -> None:
    courses = [CourseDto(id=1, name="Analysis", ects=8)]
    store = FakeQdrantStore()

    async def fake_embed_texts(
        texts: list[str], *, model: str, **_kwargs: object
    ) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr("studylife_ai.ingestion.sync.embed_texts", fake_embed_texts)

    await seed_fixture_courses(courses, settings=_settings(), store=store)  # type: ignore[arg-type]

    store.replace_entity.assert_awaited_once()
    _, kwargs = store.replace_entity.call_args
    assert kwargs["metadata"].content_type == "course"
    assert kwargs["metadata"].entity_id == 1
    assert kwargs["metadata"].title == "Analysis"
    assert kwargs["metadata"].user_id == "eval-user"


async def test_seed_fixture_sessions_titles_by_course_name_and_start_time(
    monkeypatch: MonkeyPatch,
) -> None:
    sessions = [
        StudySessionDto(
            id=101,
            course_id=1,
            course_name="Algorithmen",
            start_time="2026-08-11T16:00:00",  # type: ignore[arg-type]
            end_time="2026-08-11T17:30:00",  # type: ignore[arg-type]
        )
    ]
    store = FakeQdrantStore()

    async def fake_embed_texts(
        texts: list[str], *, model: str, **_kwargs: object
    ) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr("studylife_ai.ingestion.sync.embed_texts", fake_embed_texts)

    await seed_fixture_sessions(sessions, settings=_settings(), store=store)  # type: ignore[arg-type]

    _, kwargs = store.replace_entity.call_args
    assert kwargs["metadata"].content_type == "session"
    assert kwargs["metadata"].title == "Algorithmen, 2026-08-11 16:00"
    assert kwargs["metadata"].course_id == 1


async def test_seed_fixture_course_goals_titles_with_goal_suffix(
    monkeypatch: MonkeyPatch,
) -> None:
    goals = [CourseGoalDto(course_id=9, course_name="Studium Generale I", grade=1.0)]
    store = FakeQdrantStore()

    async def fake_embed_texts(
        texts: list[str], *, model: str, **_kwargs: object
    ) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr("studylife_ai.ingestion.sync.embed_texts", fake_embed_texts)

    await seed_fixture_course_goals(goals, settings=_settings(), store=store)  # type: ignore[arg-type]

    _, kwargs = store.replace_entity.call_args
    assert kwargs["metadata"].content_type == "course_goal"
    assert kwargs["metadata"].title == "Studium Generale I goal"
    assert kwargs["metadata"].entity_id == 9
