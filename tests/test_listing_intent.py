from datetime import date

import pytest
from pytest import MonkeyPatch

from studylife_ai.rag.listing_intent import (
    ListingIntent,
    classify_listing_intent,
    detect_recent_notes_listing,
    parse_listing_response,
)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("NONE", None),
        ("", None),
        ("garbage", None),
        ("NOTES|cloud computing|ALL", ListingIntent(kind="notes", course="cloud computing")),
        ("notes | - | all", ListingIntent(kind="notes")),
        ("`NOTES|-|ALL`", ListingIntent(kind="notes")),
        (
            "SESSIONS|Analysis|UPCOMING",
            ListingIntent(kind="sessions", course="Analysis", timeframe="upcoming"),
        ),
        ("SESSIONS|-|THIS_WEEK", ListingIntent(kind="sessions", timeframe="this_week")),
        (
            "SESSIONS|-|DATE:2026-11-20",
            ListingIntent(kind="sessions", timeframe="date", day=date(2026, 11, 20)),
        ),
        ("SESSIONS|-|DATE:2026-13-40", None),
        ("SESSIONS|-|SOMETIME", None),
        ("TASKS|-|ALL", None),
        ("NOTES|a|b|c", None),
    ],
)
def test_parse_listing_response(response: str, expected: ListingIntent | None) -> None:
    assert parse_listing_response(response) == expected


async def test_classify_pins_temperature_and_tags_the_call_site(monkeypatch: MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_complete_chat(messages: list[object], **kwargs: object) -> str:
        captured["prompt"] = messages[0].content  # type: ignore[attr-defined]
        captured.update(kwargs)
        return "NOTES|cloud computing|ALL"

    monkeypatch.setattr("studylife_ai.rag.listing_intent.complete_chat", fake_complete_chat)

    intent = await classify_listing_intent(
        "welche notizen hab ich im kurs cloud computing?",
        model="openai/gpt-4o",
        api_base=None,
        timeout=10.0,
        today=date(2026, 11, 18),
        user_id="u1",
    )

    assert intent == ListingIntent(kind="notes", course="cloud computing")
    assert captured["temperature"] == 0.0
    assert captured["call_site"] == "listing_intent"
    assert captured["user_id"] == "u1"
    assert "2026-11-18" in str(captured["prompt"])


@pytest.mark.parametrize(
    "query",
    [
        # The eval case that the model classified as NONE on main (eval run 44393383199).
        "Was ist meine neueste Notiz?",
        "was ist meine neuste notiz",
        "Zeig mir meine letzten 3 Notizen",
        "Welche ist die zuletzt angelegte Notiz?",
        "Welche Notiz habe ich zuletzt angelegt?",
        "Was war meine jüngste Notiz?",
        "Was ist die neueste Notiz zum Kurs Lineare Algebra?",
        "What is my newest note?",
        "Show me my latest notes",
        "my most recent note",
        "my last two notes",
        "Which note did I create last?",
        "the most recently created note",
    ],
)
def test_detect_recent_notes_listing_catches_recency_phrasing(query: str) -> None:
    assert detect_recent_notes_listing(query) == ListingIntent(kind="notes")


@pytest.mark.parametrize(
    "query",
    [
        "Wie funktioniert das Gram-Schmidt-Verfahren?",
        "Was steht in meinen Notizen zu Eigenwerten?",
        "Was steht in meiner letzten Notiz über Eigenwerte?",
        "What does my latest note on Gram-Schmidt say?",
        "Was war meine letzte Lernsitzung?",
        "What is planned for this week?",
        "Welche Notizen habe ich im Kurs Lineare Algebra?",
        "Die Notation der letzten Vorlesung",
    ],
)
def test_detect_recent_notes_listing_ignores_other_questions(query: str) -> None:
    assert detect_recent_notes_listing(query) is None


def _fake_model(reply: str | Exception) -> object:
    async def fake_complete_chat(*_args: object, **_kwargs: object) -> str:
        if isinstance(reply, Exception):
            raise reply
        return reply

    return fake_complete_chat


async def _classify(query: str) -> ListingIntent | None:
    return await classify_listing_intent(
        query, model="m", api_base=None, timeout=1.0, today=date(2026, 11, 18)
    )


async def test_recency_question_is_a_listing_even_when_the_model_says_none(
    monkeypatch: MonkeyPatch,
) -> None:
    """Regression: gpt-4o-mini replied NONE (one completion token) to the eval's newest-note
    question, so the answer came from similarity retrieval and missed the newest note."""
    monkeypatch.setattr("studylife_ai.rag.listing_intent.complete_chat", _fake_model("NONE"))

    assert await _classify("Was ist meine neueste Notiz?") == ListingIntent(kind="notes")


async def test_recency_question_keeps_the_models_course_when_it_agrees(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "studylife_ai.rag.listing_intent.complete_chat",
        _fake_model("NOTES|Lineare Algebra|ALL"),
    )

    assert await _classify("Meine neueste Notiz in Lineare Algebra?") == ListingIntent(
        kind="notes", course="Lineare Algebra"
    )


async def test_recency_question_overrides_a_sessions_classification(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "studylife_ai.rag.listing_intent.complete_chat", _fake_model("SESSIONS|-|UPCOMING")
    )

    assert await _classify("What is my latest note?") == ListingIntent(kind="notes")


async def test_recency_question_survives_a_model_failure(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "studylife_ai.rag.listing_intent.complete_chat", _fake_model(RuntimeError("down"))
    )

    assert await _classify("Zeig mir meine neuesten Notizen") == ListingIntent(kind="notes")


async def test_other_questions_still_follow_the_model(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("studylife_ai.rag.listing_intent.complete_chat", _fake_model("NONE"))

    assert await _classify("Wie funktioniert ein Hypothesentest?") is None


async def test_classify_never_raises(monkeypatch: MonkeyPatch) -> None:
    async def failing_complete_chat(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("provider down")

    monkeypatch.setattr("studylife_ai.rag.listing_intent.complete_chat", failing_complete_chat)

    assert (
        await classify_listing_intent(
            "x", model="m", api_base=None, timeout=1.0, today=date(2026, 11, 18)
        )
        is None
    )
