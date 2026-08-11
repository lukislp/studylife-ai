from datetime import datetime
from unittest.mock import AsyncMock

from pytest import MonkeyPatch

from studylife_ai.agent import tools as tools_module
from studylife_ai.agent.tools import build_tools
from studylife_ai.config import Settings
from studylife_ai.ingestion.qdrant_store import RetrievedChunk
from studylife_ai.studylife.models import CourseDto, StudyLifeNote, StudySessionDto


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "embedding_model": "ollama/nomic-embed-text",
        "retrieval_top_k": 5,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def _tools_with(
    *,
    studylife: object = None,
    store: object = None,
    settings: Settings | None = None,
    user_id: str = "test-user",
) -> dict[str, object]:
    built = build_tools(
        studylife=studylife or AsyncMock(),
        store=store or AsyncMock(),
        settings=settings or _settings(),
        user_id=user_id,
    )
    return {t.name: t for t in built}


async def test_list_courses_returns_id_name_code_ects() -> None:
    fake_studylife = AsyncMock()
    fake_studylife.get_courses.return_value = [
        CourseDto(id=6, semester=3, name="Lineare Algebra", code="MATH101", ects=5)
    ]
    tool = _tools_with(studylife=fake_studylife)["list_courses"]

    result = await tool.ainvoke({})

    assert result == [{"id": 6, "name": "Lineare Algebra", "code": "MATH101", "ects": 5}]


async def test_search_notes_filters_to_note_content_type(monkeypatch: MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_retrieve_with_rerank(query: str, **kwargs: object) -> list[RetrievedChunk]:
        captured["query"] = query
        captured.update(kwargs)
        return [
            RetrievedChunk(
                content_type="note",
                entity_id=1,
                chunk_index=0,
                content="det(A - λI) = 0",
                title="Eigenwerte",
                course_id=None,
                session_id=None,
                score=0.9,
            )
        ]

    monkeypatch.setattr(tools_module, "retrieve_with_rerank", fake_retrieve_with_rerank)
    tool = _tools_with(user_id="alice")["search_notes"]

    result = await tool.ainvoke({"query": "Eigenwerte"})

    assert captured["content_type"] == "note"
    assert captured["user_id"] == "alice"
    assert result == [{"title": "Eigenwerte", "content": "det(A - λI) = 0"}]


async def test_create_study_session_calls_client_with_zero_id() -> None:
    fake_studylife = AsyncMock()
    fake_studylife.create_session.return_value = StudySessionDto(
        id=99,
        course_id=6,
        course_name="Lineare Algebra",
        start_time=datetime(2026, 8, 12, 10, 0),
        end_time=datetime(2026, 8, 12, 11, 0),
    )
    tool = _tools_with(studylife=fake_studylife)["create_study_session"]

    result = await tool.ainvoke(
        {
            "course_id": 6,
            "course_name": "Lineare Algebra",
            "start_time": "2026-08-12T10:00:00",
            "end_time": "2026-08-12T11:00:00",
            "topic": "Eigenwerte",
        }
    )

    fake_studylife.create_session.assert_awaited_once()
    (sent,), _ = fake_studylife.create_session.call_args
    assert sent.id == 0
    assert sent.course_id == 6
    assert sent.topic == "Eigenwerte"
    assert result == {"id": 99, "course_name": "Lineare Algebra"}


async def test_save_note_calls_client_and_returns_id_and_title() -> None:
    fake_studylife = AsyncMock()
    fake_studylife.create_note.return_value = StudyLifeNote(
        id=55,
        title="Zusammenfassung",
        content="Kurze Zusammenfassung.",
        created_at=datetime(2026, 8, 12, 10, 0),
        updated_at=datetime(2026, 8, 12, 10, 0),
    )
    tool = _tools_with(studylife=fake_studylife)["save_note"]

    result = await tool.ainvoke(
        {"title": "Zusammenfassung", "content": "Kurze Zusammenfassung.", "course_id": 6}
    )

    fake_studylife.create_note.assert_awaited_once_with(
        title="Zusammenfassung", content="Kurze Zusammenfassung.", course_id=6
    )
    assert result == {"id": 55, "title": "Zusammenfassung"}
