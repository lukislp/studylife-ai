"""rag/chat_context.py: which context a chat question gets - a structured listing for
enumeration questions, retrieval for everything else - and how each degrades."""

from datetime import datetime

from pytest import MonkeyPatch

from studylife_ai.config import Settings
from studylife_ai.ingestion.qdrant_store import RetrievedChunk
from studylife_ai.rag import chat_context as chat_context_module
from studylife_ai.rag.chat_context import build_chat_context
from studylife_ai.rag.listing_intent import ListingIntent
from studylife_ai.studylife.models import CourseDto, StudyLifeNote, StudySessionDto

NOW = datetime(2026, 11, 18, 14, 0)


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {"date_parse_model": "openai/gpt-4o"}
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


class FakeSource:
    def __init__(self, *, fail_notes: bool = False) -> None:
        self.fail_notes = fail_notes

    async def get_courses(self) -> list[CourseDto]:
        return [CourseDto(id=1, name="Cloud Computing"), CourseDto(id=2, name="Analysis")]

    async def get_notes(self) -> list[StudyLifeNote]:
        if self.fail_notes:
            raise RuntimeError("StudyLife down")
        return [
            StudyLifeNote(
                id=10,
                title="Older note",
                content="IaaS vs PaaS",
                created_at=datetime(2026, 10, 1),
                updated_at=datetime(2026, 10, 1),
                course_id=1,
            ),
            StudyLifeNote(
                id=11,
                title="Fresh capture",
                content="Serverless overview",
                created_at=datetime(2026, 11, 18, 13, 55),
                updated_at=datetime(2026, 11, 18, 13, 55),
                course_id=1,
            ),
        ]

    async def get_sessions_history(
        self, *, days: int, only_completed: bool
    ) -> list[StudySessionDto]:
        return []


def _note_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        content_type="note",
        entity_id=10,
        chunk_index=0,
        content="IaaS vs PaaS",
        title="Older note",
        course_id=1,
        session_id=None,
        score=0.8,
        session_start=None,
    )


def _past_session_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        content_type="session",
        entity_id=99,
        chunk_index=0,
        content="Study session: Analysis, ...\nStatus: Not completed",
        title="Analysis, 2026-10-22 10:00",
        course_id=2,
        session_id=None,
        score=0.5,
        session_start="2026-10-22T10:00:00",
    )


def _classifier(result: ListingIntent | None, calls: list[str]) -> object:
    async def fake_classify(query: str, **kwargs: object) -> ListingIntent | None:
        calls.append(str(kwargs["model"]))
        return result

    return fake_classify


async def _retrieve() -> list[RetrievedChunk]:
    return [_note_chunk(), _past_session_chunk()]


async def test_enumeration_question_gets_the_listing_not_retrieval(
    monkeypatch: MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        chat_context_module,
        "classify_listing_intent",
        _classifier(ListingIntent(kind="notes", course="cloud computing"), calls),
    )

    context = await build_chat_context(
        "welche notizen hab ich im kurs cloud computing?",
        retrieve=_retrieve,
        source=FakeSource(),
        settings=_settings(),
        now=NOW,
        user_id="u1",
    )

    assert calls == ["openai/gpt-4o"]  # DATE_PARSE_MODEL is the default classifier
    assert [s.title for s in context.sources] == ["Fresh capture", "Older note"]
    assert "<records>" in context.system_message
    assert "[1] Note: Fresh capture (course: Cloud Computing;" in context.system_message
    # The retrieved past session is not offered to the model at all.
    assert "Analysis, 2026-10-22" not in context.system_message
    assert context.listing is not None


async def test_listing_intent_model_overrides_the_classifier(monkeypatch: MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(chat_context_module, "classify_listing_intent", _classifier(None, calls))

    await build_chat_context(
        "q",
        retrieve=_retrieve,
        source=FakeSource(),
        settings=_settings(listing_intent_model="openai/gpt-4o-mini"),
        now=NOW,
        user_id="u1",
    )

    assert calls == ["openai/gpt-4o-mini"]


async def test_content_question_gets_annotated_retrieval_context(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(chat_context_module, "classify_listing_intent", _classifier(None, []))

    context = await build_chat_context(
        "Was ist der Unterschied zwischen IaaS und PaaS?",
        retrieve=_retrieve,
        source=FakeSource(),
        settings=_settings(),
        now=NOW,
        user_id="u1",
    )

    assert "<notes>" in context.system_message
    assert "[1] Note: Older note (course: Cloud Computing)" in context.system_message
    assert "[2] Session: Analysis, 2026-10-22 10:00 [27 days ago] (past)" in context.system_message
    assert [s.entity_id for s in context.sources] == [10, 99]


async def test_listing_failure_falls_back_to_retrieval(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        chat_context_module,
        "classify_listing_intent",
        _classifier(ListingIntent(kind="notes"), []),
    )

    context = await build_chat_context(
        "welche notizen habe ich?",
        retrieve=_retrieve,
        source=FakeSource(fail_notes=True),
        settings=_settings(),
        now=NOW,
        user_id="u1",
    )

    assert "<notes>" in context.system_message
    assert context.listing is None


async def test_no_source_means_no_classification_and_plain_retrieval(
    monkeypatch: MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(chat_context_module, "classify_listing_intent", _classifier(None, calls))

    context = await build_chat_context(
        "q", retrieve=_retrieve, source=None, settings=_settings(), now=NOW, user_id="u1"
    )

    assert calls == []
    assert "[1] Note: Older note\n" in context.system_message


async def test_structured_listings_can_be_switched_off(monkeypatch: MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(chat_context_module, "classify_listing_intent", _classifier(None, calls))

    await build_chat_context(
        "q",
        retrieve=_retrieve,
        source=FakeSource(),
        settings=_settings(structured_listings=False),
        now=NOW,
        user_id="u1",
    )

    assert calls == []


async def test_retrieval_failure_still_answers(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(chat_context_module, "classify_listing_intent", _classifier(None, []))

    async def failing_retrieve() -> list[RetrievedChunk]:
        raise RuntimeError("qdrant down")

    context = await build_chat_context(
        "q", retrieve=failing_retrieve, source=None, settings=_settings(), now=NOW, user_id="u1"
    )

    assert "(no matching notes found)" in context.system_message
    assert context.sources == []
