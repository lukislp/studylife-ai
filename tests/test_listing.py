"""Structured listings (rag/listing.py): the deterministic filter/sort/status logic behind
"which notes do I have in course X" and "what is planned" - regression cover for the 2026-11-18
owner test (newest note missing, a past session presented as planned)."""

from datetime import date, datetime

import pytest

from studylife_ai.rag.listing import (
    fetch_listing,
    match_courses,
    select_notes,
    select_sessions,
    session_status,
)
from studylife_ai.rag.listing_intent import ListingIntent
from studylife_ai.studylife.models import CourseDto, StudyLifeNote, StudySessionDto

NOW = datetime(2026, 11, 18, 14, 0)  # a Wednesday

COURSES = [
    CourseDto(id=1, name="Cloud Computing", code="INF350"),
    CourseDto(id=2, name="Statistik", code="MATH301"),
    CourseDto(id=3, name="Statistik Tutorium", code="MATH301T"),
    CourseDto(id=4, name="Analysis", code="MATH201"),
]


def _note(note_id: int, course_id: int | None, created: datetime, title: str = "") -> StudyLifeNote:
    return StudyLifeNote(
        id=note_id,
        title=title or f"Note {note_id}",
        content=f"content {note_id}",
        created_at=created,
        updated_at=created,
        course_id=course_id,
    )


def _session(
    session_id: int,
    course_id: int,
    start: datetime,
    end: datetime,
    *,
    completed: bool = False,
    course_name: str = "",
) -> StudySessionDto:
    return StudySessionDto(
        id=session_id,
        course_id=course_id,
        course_name=course_name,
        start_time=start,
        end_time=end,
        is_completed=completed,
    )


class FakeSource:
    def __init__(
        self,
        notes: list[StudyLifeNote] | None = None,
        sessions: list[StudySessionDto] | None = None,
    ) -> None:
        self.notes = notes or []
        self.sessions = sessions or []
        self.history_calls: list[tuple[int, bool]] = []

    async def get_notes(self) -> list[StudyLifeNote]:
        return self.notes

    async def get_courses(self) -> list[CourseDto]:
        return COURSES

    async def get_sessions_history(
        self, *, days: int, only_completed: bool
    ) -> list[StudySessionDto]:
        self.history_calls.append((days, only_completed))
        return self.sessions


# --- match_courses ---


@pytest.mark.parametrize(
    ("query", "expected_ids"),
    [
        ("cloud computing", [1]),
        ("Cloud-Computing", [1]),
        ("INF350", [1]),
        ("kurs cloud computing", [1]),
        ("statistik", [2, 3]),
        ("Biologie", []),
        ("", []),
    ],
)
def test_match_courses(query: str, expected_ids: list[int]) -> None:
    assert [c.id for c in match_courses(query, COURSES)] == expected_ids


# --- session_status: a past session is never "planned" ---


def test_session_status_never_calls_a_past_uncompleted_session_planned() -> None:
    past = _session(1, 1, datetime(2026, 10, 22, 10), datetime(2026, 10, 22, 12))
    assert session_status(past, now=NOW) == "past_not_completed"


def test_session_status_covers_completed_running_and_future() -> None:
    done = _session(1, 1, datetime(2026, 10, 22, 10), datetime(2026, 10, 22, 12), completed=True)
    running = _session(2, 1, datetime(2026, 11, 18, 13), datetime(2026, 11, 18, 15))
    future = _session(3, 1, datetime(2026, 11, 20, 10), datetime(2026, 11, 20, 12))
    assert session_status(done, now=NOW) == "completed"
    assert session_status(running, now=NOW) == "in_progress"
    assert session_status(future, now=NOW) == "planned"


# --- select_notes ---


def test_select_notes_filters_by_course_id_and_puts_the_newest_first() -> None:
    notes = [
        _note(1, 1, datetime(2026, 10, 1)),
        _note(2, 2, datetime(2026, 11, 18, 13, 55)),  # newest overall, other course
        _note(3, 1, datetime(2026, 11, 18, 13, 50)),  # newest in course 1, minutes old
        _note(4, 1, datetime(2026, 9, 1)),
        _note(5, None, datetime(2026, 11, 1)),
    ]

    selected = select_notes(notes, course_ids={1})

    assert [n.id for n in selected] == [3, 1, 4]


def test_select_notes_without_course_returns_every_note_newest_first() -> None:
    notes = [_note(1, 1, datetime(2026, 10, 1)), _note(2, None, datetime(2026, 11, 1))]
    assert [n.id for n in select_notes(notes, course_ids=None)] == [2, 1]


# --- select_sessions ---


def _week_sessions() -> list[StudySessionDto]:
    return [
        _session(1, 1, datetime(2026, 10, 22, 10), datetime(2026, 10, 22, 12)),  # past
        _session(2, 1, datetime(2026, 11, 18, 9), datetime(2026, 11, 18, 10)),  # today, over
        _session(3, 1, datetime(2026, 11, 18, 16), datetime(2026, 11, 18, 17)),  # today, later
        _session(4, 1, datetime(2026, 11, 19, 10), datetime(2026, 11, 19, 11)),  # tomorrow
        _session(5, 2, datetime(2026, 11, 20, 10), datetime(2026, 11, 20, 11)),  # other course
        _session(6, 1, datetime(2026, 11, 25, 10), datetime(2026, 11, 25, 11)),  # next week
        _session(7, 1, datetime(2026, 11, 10, 10), datetime(2026, 11, 10, 11)),  # last week
    ]


@pytest.mark.parametrize(
    ("timeframe", "day", "expected_ids"),
    [
        ("upcoming", None, [3, 4, 6]),
        ("today", None, [3, 2]),
        ("tomorrow", None, [4]),
        ("this_week", None, [3, 4, 2]),
        ("next_week", None, [6]),
        ("last_week", None, [7]),
        ("past", None, [2, 7, 1]),
        ("date", date(2026, 10, 22), [1]),
        ("all", None, [3, 4, 6, 2, 7, 1]),
    ],
)
def test_select_sessions_by_timeframe_and_course(
    timeframe: str, day: date | None, expected_ids: list[int]
) -> None:
    selected = select_sessions(
        _week_sessions(),
        course_ids={1},
        timeframe=timeframe,  # type: ignore[arg-type]
        day=day,
        now=NOW,
    )
    assert [s.id for s in selected] == expected_ids


def test_upcoming_never_includes_a_past_session() -> None:
    selected = select_sessions(
        _week_sessions(), course_ids=None, timeframe="upcoming", day=None, now=NOW
    )
    assert all(s.end_time >= NOW for s in selected)
    assert 1 not in {s.id for s in selected}


# --- fetch_listing ---


async def test_fetch_listing_notes_of_a_course_include_the_newest_note_with_catalog_names() -> None:
    source = FakeSource(
        notes=[
            _note(1, 1, datetime(2026, 10, 1), "Old"),
            _note(2, 1, datetime(2026, 11, 18, 13, 55), "Fresh capture"),
            _note(3, 4, datetime(2026, 11, 18, 13, 56), "Analysis note"),
        ]
    )

    listing = await fetch_listing(
        ListingIntent(kind="notes", course="cloud computing"),
        source=source,
        courses=COURSES,
        now=NOW,
        max_items=25,
        session_history_days=1825,
    )

    assert listing.matched_courses == ["Cloud Computing"]
    assert [n.title for n in listing.notes] == ["Fresh capture", "Old"]
    assert {n.course_name for n in listing.notes} == {"Cloud Computing"}
    assert listing.sessions == []
    assert listing.total == 2


async def test_fetch_listing_reports_an_unknown_course_instead_of_guessing() -> None:
    listing = await fetch_listing(
        ListingIntent(kind="notes", course="Biologie"),
        source=FakeSource(notes=[_note(1, 1, datetime(2026, 10, 1))]),
        courses=COURSES,
        now=NOW,
        max_items=25,
        session_history_days=1825,
    )

    assert listing.course_not_found
    assert listing.notes == []
    assert "Cloud Computing" in listing.available_courses


async def test_fetch_listing_caps_items_but_reports_the_real_total() -> None:
    notes = [_note(i, 1, datetime(2026, 11, 1, 10, i)) for i in range(1, 6)]

    listing = await fetch_listing(
        ListingIntent(kind="notes"),
        source=FakeSource(notes=notes),
        courses=COURSES,
        now=NOW,
        max_items=2,
        session_history_days=1825,
    )

    assert [n.id for n in listing.notes] == [5, 4]
    assert listing.total == 5


async def test_fetch_listing_sessions_use_catalog_course_names_and_computed_status() -> None:
    source = FakeSource(
        sessions=[
            _session(
                1, 1, datetime(2026, 10, 22, 10), datetime(2026, 10, 22, 12), course_name="stale"
            ),
            _session(2, 1, datetime(2026, 11, 19, 10), datetime(2026, 11, 19, 11)),
        ]
    )

    listing = await fetch_listing(
        ListingIntent(kind="sessions", course="Cloud Computing", timeframe="all"),
        source=source,
        courses=COURSES,
        now=NOW,
        max_items=25,
        session_history_days=1825,
    )

    assert [(s.id, s.status) for s in listing.sessions] == [
        (2, "planned"),
        (1, "past_not_completed"),
    ]
    assert {s.course_name for s in listing.sessions} == {"Cloud Computing"}
    # Includes not-yet-completed (future) sessions - onlyCompleted=false.
    assert source.history_calls == [(1825, False)]
