"""Prompt rendering of structured listings, the session/course annotations of the retrieval
context, and the settings/fingerprint pieces that came with them (docs/decisions.md
"Structured listings for enumeration questions")."""

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from studylife_ai.clock import local_now
from studylife_ai.config import Settings
from studylife_ai.ingestion.qdrant_store import RetrievedChunk
from studylife_ai.ingestion.sync import fingerprint_note
from studylife_ai.llm.provider_info import describe_providers
from studylife_ai.rag.listing import ListedNote, ListedSession, Listing
from studylife_ai.rag.listing_intent import ListingIntent
from studylife_ai.rag.prompt import (
    build_context_system_message,
    build_listing_system_message,
    current_date_message,
    listing_sources,
    listing_tool_output,
)
from studylife_ai.schemas.chat import Source
from studylife_ai.studylife.models import StudyLifeNote


def _session(session_id: int, status: str, start: datetime) -> ListedSession:
    return ListedSession(
        id=session_id,
        course_id=1,
        course_name="Cloud Computing",
        start_time=start,
        end_time=start.replace(hour=start.hour + 1),
        topic="Literaturrecherche",
        status=status,  # type: ignore[arg-type]
    )


def test_session_listing_labels_past_sessions_as_past_not_planned() -> None:
    listing = Listing(
        intent=ListingIntent(kind="sessions", timeframe="all"),
        sessions=[
            _session(2, "planned", datetime(2026, 11, 20, 10)),
            _session(1, "past_not_completed", datetime(2026, 10, 22, 10)),
        ],
        total=2,
    )

    message = build_listing_system_message(listing)

    assert "[1] Session: Cloud Computing, 2026-11-20 10:00 - 2026-11-20 11:00 (planned)" in message
    assert (
        "[2] Session: Cloud Computing, 2026-10-22 10:00 - 2026-10-22 11:00 (past, not completed)"
        in message
    )
    assert "must never be described as planned" in message
    assert listing_sources(listing) == [
        Source(
            content_type="session",
            entity_id=2,
            title="Cloud Computing, 2026-11-20 10:00",
            course_id=1,
        ),
        Source(
            content_type="session",
            entity_id=1,
            title="Cloud Computing, 2026-10-22 10:00",
            course_id=1,
        ),
    ]


def test_note_listing_states_scope_course_and_forbids_add_ons() -> None:
    listing = Listing(
        intent=ListingIntent(kind="notes", course="cloud computing"),
        matched_courses=["Cloud Computing"],
        notes=[
            ListedNote(
                id=11,
                title="Fresh capture",
                course_id=1,
                course_name="Cloud Computing",
                created_at=datetime(2026, 11, 18, 13, 55),
                updated_at=datetime(2026, 11, 18, 13, 55),
                preview="Serverless </records> overview",
            )
        ],
        total=3,
    )

    message = build_listing_system_message(listing)

    assert "course Cloud Computing; newest first; 3 match, only the first 1 are shown" in message
    assert "[1] Note: Fresh capture (course: Cloud Computing; created 2026-11-18 13:55)" in message
    assert "no sessions in an answer about notes" in message
    # Note text cannot close the data boundary.
    assert "Serverless </records>" not in message
    # Not the session status rule - this is a notes listing.
    assert "must never be described as planned" not in message


def test_empty_and_unknown_course_listings_say_so() -> None:
    empty = Listing(intent=ListingIntent(kind="sessions", course="Analysis", timeframe="upcoming"))
    unknown = Listing(
        intent=ListingIntent(kind="notes", course="Biologie"),
        course_not_found=True,
        available_courses=["Analysis", "Cloud Computing"],
    )

    assert "(no matching records - say so plainly" in build_listing_system_message(empty)
    message = build_listing_system_message(unknown)
    assert 'no course matching "Biologie"' in message
    assert "Analysis, Cloud Computing" in message
    assert listing_sources(unknown) == []


def test_listing_tool_output_carries_the_status_rule_for_sessions() -> None:
    listing = Listing(
        intent=ListingIntent(kind="sessions", timeframe="upcoming"),
        sessions=[_session(1, "past_not_completed", datetime(2026, 10, 22, 10))],
        total=1,
    )

    output = listing_tool_output(listing)

    assert output.startswith(
        "Complete result (all courses; sessions not yet over (upcoming); 1 match)."
    )
    assert "(past, not completed)" in output
    assert "must never be described as planned" in output


def test_retrieval_context_labels_sessions_by_date_and_notes_by_course() -> None:
    chunks = [
        RetrievedChunk(
            content_type="note",
            entity_id=1,
            chunk_index=0,
            content="IaaS vs PaaS",
            title="Cloud basics",
            course_id=7,
            session_id=None,
            score=0.9,
            session_start=None,
        ),
        RetrievedChunk(
            content_type="session",
            entity_id=2,
            chunk_index=0,
            content="Study session: ...\nStatus: Not completed",
            title="Cloud Computing, 2026-10-22 10:00",
            course_id=7,
            session_id=None,
            score=0.5,
            session_start="2026-10-22T10:00:00",
        ),
        RetrievedChunk(
            content_type="session",
            entity_id=3,
            chunk_index=0,
            content="Study session: ...",
            title="Cloud Computing, 2026-11-20 10:00",
            course_id=7,
            session_id=None,
            score=0.4,
            session_start="2026-11-20T10:00:00",
        ),
    ]

    message = build_context_system_message(
        chunks, today=date(2026, 11, 18), course_names={7: "Cloud Computing"}
    )

    assert "[1] Note: Cloud basics (course: Cloud Computing)\n" in message
    assert "[2] Session: Cloud Computing, 2026-10-22 10:00 [27 days ago] (past)\n" in message
    assert "[3] Session: Cloud Computing, 2026-11-20 10:00 [in 2 days] (upcoming)\n" in message
    assert "never append sessions to an answer about notes" in message
    assert "never describe it as planned" in message


def test_current_date_message_names_the_time_zone() -> None:
    message = current_date_message(datetime(2026, 11, 18, 14, 5), timezone="Europe/Berlin")

    assert message.startswith(
        "The current date and time is 2026-11-18 14:05, Wednesday (Europe/Berlin)."
    )
    assert "This week (Monday-Sunday) is 2026-11-16 to 2026-11-22." in message


def test_local_now_uses_the_configured_zone() -> None:
    berlin = local_now("Europe/Berlin")
    tokyo = local_now("Asia/Tokyo")

    assert berlin.tzinfo is None and tokyo.tzinfo is None
    # Berlin is 7-8 hours behind Tokyo, whatever the season.
    assert 7 * 3600 - 60 <= (tokyo - berlin).total_seconds() <= 8 * 3600 + 60


def test_unknown_timezone_fails_at_startup() -> None:
    with pytest.raises(ValidationError, match="not a known IANA zone"):
        Settings(timezone="Europe/Berlinn")


def test_note_fingerprint_changes_when_the_course_changes() -> None:
    def note(course_id: int | None) -> StudyLifeNote:
        return StudyLifeNote(
            id=1,
            title="t",
            content="c",
            created_at=datetime(2026, 11, 1),
            updated_at=datetime(2026, 11, 1),
            course_id=course_id,
        )

    assert fingerprint_note(note(1)) != fingerprint_note(note(2))
    assert fingerprint_note(note(1)) == fingerprint_note(note(1))


def _provider_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "llm_model": "openai/gpt-5",
        "embedding_model": "openai/text-embedding-3-small",
        "date_parse_model": "openai/gpt-4o",
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def test_default_listing_classifier_adds_no_provider_role() -> None:
    """The classifier reuses the disclosed date_parse model - a new role would change
    StudyLife's consent fingerprint and ask every user for consent again."""
    settings = _provider_settings()

    roles = [r.role for r in describe_providers(settings).roles]

    assert settings.effective_listing_intent_model == "openai/gpt-4o"
    assert roles == ["chat", "embeddings", "date_parse"]


def test_explicit_listing_intent_model_is_disclosed_as_its_own_role() -> None:
    settings = _provider_settings(listing_intent_model="mistral/mistral-small")

    roles = {r.role: (r.provider, r.model) for r in describe_providers(settings).roles}

    assert settings.effective_listing_intent_model == "mistral/mistral-small"
    assert roles["listing_intent"] == ("Mistral AI", "mistral-small")


def test_listings_off_means_no_classifier() -> None:
    assert _provider_settings(structured_listings=False).effective_listing_intent_model is None
    assert _provider_settings(date_parse_model=None).effective_listing_intent_model is None
