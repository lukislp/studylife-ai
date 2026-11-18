"""Structured listings: answers enumeration questions from the live StudyLife API.

"Which notes do I have in course X", "my newest note", "what is planned this week" are not
similarity questions (see docs/decisions.md "Structured listings for enumeration questions"):
they need every matching item, filtered by the item's own `course_id` and ordered by date. This
module does exactly that, deterministically, over whatever `StudyDataSource` it is given - the
real `StudyLifeClient` (scoped to the calling user's own AI key) in /chat and /agent, a fixture
source in the eval. Which questions get here is decided by rag/listing_intent.py.

Everything below the fetch is pure and synchronous, so it is unit-tested without any I/O.
"""

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Literal, Protocol

from studylife_ai.ingestion.rendering import render_note
from studylife_ai.rag.date_parse import week_bounds
from studylife_ai.rag.listing_intent import ListingIntent, Timeframe
from studylife_ai.studylife.models import CourseDto, StudyLifeNote, StudySessionDto

_NOTE_PREVIEW_CHARS = 200

SessionStatus = Literal["completed", "in_progress", "planned", "past_not_completed"]


class StudyDataSource(Protocol):
    """The read side of `StudyLifeClient` a listing needs - a Protocol so the eval (and tests)
    can hand in fixture data without an HTTP server."""

    async def get_notes(self) -> list[StudyLifeNote]: ...

    async def get_courses(self) -> list[CourseDto]: ...

    async def get_sessions_history(
        self, *, days: int, only_completed: bool
    ) -> list[StudySessionDto]: ...


@dataclass(frozen=True)
class ListedNote:
    id: int
    title: str
    course_id: int | None
    course_name: str | None
    created_at: datetime
    updated_at: datetime
    preview: str


@dataclass(frozen=True)
class ListedSession:
    id: int
    course_id: int
    course_name: str
    start_time: datetime
    end_time: datetime
    topic: str | None
    status: SessionStatus


@dataclass(frozen=True)
class Listing:
    """The result of one structured listing. `total` counts every match before the
    `listing_max_items` cut, so the prompt can say how many were left out. `course_not_found`
    means the user named a course that matches nothing in their catalog - `available_courses`
    then lets the answer say so instead of guessing."""

    intent: ListingIntent
    matched_courses: list[str] = field(default_factory=list)
    course_not_found: bool = False
    available_courses: list[str] = field(default_factory=list)
    notes: list[ListedNote] = field(default_factory=list)
    sessions: list[ListedSession] = field(default_factory=list)
    total: int = 0


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[\W_]+", " ", text.casefold()).split())


def match_courses(query: str, courses: list[CourseDto]) -> list[CourseDto]:
    """Courses the user's words plausibly name: equal to the name or code, the words contained
    in the name ("statistik" -> "Statistik", "Statistik Tutorium"), or the whole name contained
    in the words ("kurs cloud computing" -> "Cloud Computing"). Inclusive on purpose - every
    listed item carries its real course name, so a near match is visible in the answer, while a
    missed course would silently drop items."""
    wanted = _normalize(query)
    if not wanted:
        return []
    matches = []
    for course in courses:
        name = _normalize(course.name)
        code = _normalize(course.code)
        if not name:
            continue
        if wanted in {name, code} or wanted in name or name in wanted:
            matches.append(course)
    return matches


def session_status(session: StudySessionDto, *, now: datetime) -> SessionStatus:
    """The session's state relative to `now`. "planned" is reserved for sessions that have not
    ended yet - a past session that was never ticked off is `past_not_completed`, never
    "planned" (the bug behind presenting a session from weeks ago as "geplant")."""
    if session.is_completed:
        return "completed"
    if session.end_time < now:
        return "past_not_completed"
    if session.start_time <= now:
        return "in_progress"
    return "planned"


def _in_timeframe(
    session: StudySessionDto, *, timeframe: Timeframe, day: date | None, now: datetime
) -> bool:
    start_day = session.start_time.date()
    today = now.date()
    if timeframe == "all":
        return True
    if timeframe == "upcoming":
        return session.end_time >= now
    if timeframe == "past":
        return session.end_time < now
    if timeframe == "today":
        return start_day == today
    if timeframe == "tomorrow":
        return start_day == today + timedelta(days=1)
    if timeframe == "date":
        return day is not None and start_day == day
    weeks_ago = {"this_week": 0, "last_week": 1, "next_week": -1}[timeframe]
    week = week_bounds(today, weeks_ago=weeks_ago)
    return week.start <= start_day <= week.end


def select_notes(notes: list[StudyLifeNote], *, course_ids: set[int] | None) -> list[StudyLifeNote]:
    """Notes in `course_ids` (all notes when `None`), newest first by creation time - a note
    created minutes ago is always first, whatever its text is about."""
    selected = [n for n in notes if course_ids is None or n.course_id in course_ids]
    return sorted(selected, key=lambda n: (n.created_at, n.id), reverse=True)


def select_sessions(
    sessions: list[StudySessionDto],
    *,
    course_ids: set[int] | None,
    timeframe: Timeframe,
    day: date | None,
    now: datetime,
) -> list[StudySessionDto]:
    """Sessions in `course_ids` (all when `None`) inside `timeframe`. Sessions that have not
    ended come first, soonest first; past ones follow, most recent first - so a cut by
    `listing_max_items` drops the oldest history, never the next upcoming session."""
    selected = [
        s
        for s in sessions
        if (course_ids is None or s.course_id in course_ids)
        and _in_timeframe(s, timeframe=timeframe, day=day, now=now)
    ]
    upcoming = sorted((s for s in selected if s.end_time >= now), key=lambda s: s.start_time)
    past = sorted(
        (s for s in selected if s.end_time < now), key=lambda s: s.start_time, reverse=True
    )
    return upcoming + past


def _preview(note: StudyLifeNote) -> str:
    text = " ".join(render_note(note).split())
    if len(text) <= _NOTE_PREVIEW_CHARS:
        return text
    return text[:_NOTE_PREVIEW_CHARS].rstrip() + "..."


async def fetch_listing(
    intent: ListingIntent,
    *,
    source: StudyDataSource,
    courses: list[CourseDto],
    now: datetime,
    max_items: int,
    session_history_days: int,
) -> Listing:
    """Runs `intent` against `source`. `courses` is the user's catalog (fetched once by the
    caller, which needs it for course names anyway). Course names always come from that catalog
    via the item's own `course_id`, never from the item's text. Raises whatever `source` raises -
    the caller decides how to degrade."""
    course_names = {c.id: c.name for c in courses}
    course_ids: set[int] | None = None
    matched_names: list[str] = []
    if intent.course:
        matched = match_courses(intent.course, courses)
        if not matched:
            return Listing(
                intent=intent,
                course_not_found=True,
                available_courses=sorted(c.name for c in courses if c.name),
            )
        course_ids = {c.id for c in matched}
        matched_names = [c.name for c in matched]

    if intent.kind == "notes":
        notes = select_notes(await source.get_notes(), course_ids=course_ids)
        return Listing(
            intent=intent,
            matched_courses=matched_names,
            notes=[
                ListedNote(
                    id=n.id,
                    title=n.title,
                    course_id=n.course_id,
                    course_name=course_names.get(n.course_id) if n.course_id else None,
                    created_at=n.created_at,
                    updated_at=n.updated_at,
                    preview=_preview(n),
                )
                for n in notes[:max_items]
            ],
            total=len(notes),
        )

    sessions = select_sessions(
        await source.get_sessions_history(days=session_history_days, only_completed=False),
        course_ids=course_ids,
        timeframe=intent.timeframe,
        day=intent.day,
        now=now,
    )
    return Listing(
        intent=intent,
        matched_courses=matched_names,
        sessions=[
            ListedSession(
                id=s.id,
                course_id=s.course_id,
                course_name=course_names.get(s.course_id) or s.course_name,
                start_time=s.start_time,
                end_time=s.end_time,
                topic=s.topic,
                status=session_status(s, now=now),
            )
            for s in sessions[:max_items]
        ],
        total=len(sessions),
    )
