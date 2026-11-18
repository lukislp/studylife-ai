"""Listing-intent classification: does a chat question enumerate notes or sessions?

Vector similarity answers "what do my notes say about X" well, but it cannot answer "which notes
do I have in course X" or "what is my newest note": the embedded note text does not contain its
course, and similarity has no notion of recency or of "all of them" (see docs/decisions.md
"Structured listings for enumeration questions"). Those questions are answered from the live
StudyLife API instead (rag/listing.py); this module only decides which questions those are.

Same split as rag/date_parse.py: the LLM only classifies (kind, course words, time frame) in a
plain-text one-line format that small local models can follow; every filter, sort and date
boundary is computed in Python afterwards. Never raises - any failure means "no listing", and
the caller falls back to normal retrieval.

"Newest/latest note" questions are additionally recognized by a deterministic phrase check
(`detect_recent_notes_listing`), because the model alone proved unreliable for them: in the
first eval run on main, gpt-4o-mini answered `NONE` to "Was ist meine neueste Notiz?" although
the prompt has the English twin as an example. The phrase check decides *whether* it is a
listing; the model, when it agrees, still contributes the course words.
"""

import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import Literal

from studylife_ai.llm.client import complete_chat
from studylife_ai.schemas.chat import ChatMessage

logger = logging.getLogger(__name__)

ListingKind = Literal["notes", "sessions"]
Timeframe = Literal[
    "all", "upcoming", "past", "today", "tomorrow", "this_week", "next_week", "last_week", "date"
]

_TIMEFRAMES: dict[str, Timeframe] = {
    "ALL": "all",
    "UPCOMING": "upcoming",
    "PAST": "past",
    "TODAY": "today",
    "TOMORROW": "tomorrow",
    "THIS_WEEK": "this_week",
    "NEXT_WEEK": "next_week",
    "LAST_WEEK": "last_week",
}


@dataclass(frozen=True)
class ListingIntent:
    """What to list. `course` is the user's own words for a course (resolved against the real
    catalog later, never trusted as an id); `day` is only set for `timeframe == "date"`."""

    kind: ListingKind
    course: str | None = None
    timeframe: Timeframe = "all"
    day: date | None = None


_PROMPT_TEMPLATE = (
    "Today's date is {today}.\n\n"
    "Decide whether the question below asks to LIST or ENUMERATE the user's own notes or study "
    "sessions (calendar entries) - by course, by recency (newest/latest), or by time - in any "
    'language. Questions about the CONTENT of notes or about a topic (e.g. "how does X '
    'work", "what did I write about X", "explain X") are NOT listings.\n\n'
    "Reply with EXACTLY one line, no other text:\n"
    "- not a listing: NONE\n"
    "- a listing: <KIND>|<COURSE>|<WHEN>\n"
    "  KIND: NOTES or SESSIONS\n"
    "  COURSE: the course name exactly as the user wrote it, or - if no course is named\n"
    "  WHEN: ALL, UPCOMING, PAST, TODAY, TOMORROW, THIS_WEEK, NEXT_WEEK, LAST_WEEK, or "
    "DATE:<YYYY-MM-DD> for one specific day. Use ALL for notes unless the user restricts the "
    'time. For sessions without a time restriction use UPCOMING ("planned", "geplant", '
    '"next", "anstehend" also mean UPCOMING).\n\n'
    "Examples:\n"
    '"welche notizen hab ich im kurs cloud computing?" -> NOTES|cloud computing|ALL\n'
    '"what is my newest note?" -> NOTES|-|ALL\n'
    '"was steht diese Woche an?" -> SESSIONS|-|THIS_WEEK\n'
    '"welche Lernsitzungen habe ich in Analysis geplant?" -> SESSIONS|Analysis|UPCOMING\n'
    '"wie funktioniert ein Hypothesentest?" -> NONE\n\n'
    "Question: {query}"
)

_DATE_RE = re.compile(r"^DATE:(\d{4}-\d{2}-\d{2})$")


def parse_listing_response(response: str) -> ListingIntent | None:
    """Parses the classifier's one-line reply. Anything unrecognizable degrades to `None`
    ("no listing"), same graceful-degradation rule as rag/date_parse.py's parser."""
    parts = [part.strip() for part in response.strip().strip("`").split("|")]
    if len(parts) != 3:
        return None
    kind_raw, course_raw, when = parts[0].upper(), parts[1], parts[2].upper()
    if kind_raw not in {"NOTES", "SESSIONS"}:
        return None
    kind: ListingKind = "notes" if kind_raw == "NOTES" else "sessions"
    course = None if course_raw.upper() in {"", "-", "NONE"} else course_raw
    date_match = _DATE_RE.match(when)
    if date_match:
        try:
            day = date.fromisoformat(date_match.group(1))
        except ValueError:
            return None
        return ListingIntent(kind=kind, course=course, timeframe="date", day=day)
    timeframe = _TIMEFRAMES.get(when)
    if timeframe is None:
        return None
    return ListingIntent(kind=kind, course=course, timeframe=timeframe)


_NOTE_WORD = r"(?:notiz(?:en)?|notes?)"
_RECENCY_ADJECTIVE = (
    r"(?:neu(?:e)?ste[nrs]?|letzte[nrs]?|jüngste[nrs]?"
    r"|zuletzt\s+(?:angelegte|erstellte|geschriebene|hinzugefügte|gespeicherte)[nrs]?"
    r"|newest|latest|most\s+recent(?:ly\s+(?:created|added|written|saved))?"
    r"|last|recent(?:ly\s+(?:created|added|written|saved))?)"
)
# "meine neueste Notiz", "die letzten 3 Notizen", "my latest two notes", "most recent note":
# a recency word, then at most two filler words (an article, a number), then the note word.
_RECENCY_BEFORE_NOTE_RE = re.compile(
    rf"\b{_RECENCY_ADJECTIVE}\s+(?:\S+\s+){{0,2}}?{_NOTE_WORD}\b", re.IGNORECASE
)
# "welche Notiz habe ich zuletzt angelegt?", "which note did I create last?".
_NOTE_BEFORE_RECENCY_RE = re.compile(
    rf"\b{_NOTE_WORD}\b[^.?!]{{0,40}}?"
    r"(?:\bzuletzt\b|\b(?:creat(?:e|ed)|add(?:ed)?|wr(?:ite|ote|itten)|made|make|sav(?:e|ed)"
    r"|t(?:ake|ook))\s+(?:most\s+)?"
    r"(?:last|recently)\b)",
    re.IGNORECASE,
)
# "meine letzte Notiz über Eigenwerte", "my last note on Gram-Schmidt": the question is about a
# topic, so content retrieval (or the model) decides, not the recency phrase check. "zum Kurs X"
# / "on course X" names a course, not a topic, and stays a listing.
_TOPIC_AFTER_NOTE_RE = re.compile(
    rf"\b{_NOTE_WORD}\s+(?:über|ueber|zu|zum|zur|bezüglich|about|on|regarding|concerning)\b"
    r"(?!\s+(?:(?:the|my|dem|den|meinem|meinen)\s+)?(?:kurs|course|fach|modul|module)\b)",
    re.IGNORECASE,
)


def detect_recent_notes_listing(query: str) -> ListingIntent | None:
    """Deterministic check for "newest / latest / most recent note(s)" questions, English and
    German ("neueste", "neuste", "letzte", "jüngste", "zuletzt angelegte"). Returns a notes
    listing without a course (newest first is how every notes listing is ordered), or `None`
    when the phrasing does not clearly ask for recency or narrows the note to a topic."""
    if _TOPIC_AFTER_NOTE_RE.search(query):
        return None
    if _RECENCY_BEFORE_NOTE_RE.search(query) or _NOTE_BEFORE_RECENCY_RE.search(query):
        return ListingIntent(kind="notes")
    return None


async def classify_listing_intent(
    query: str,
    *,
    model: str,
    api_base: str | None,
    timeout: float,
    today: date,
    user_id: str = "unknown",
) -> ListingIntent | None:
    """Classifies `query`; `None` means "answer with normal retrieval". Never raises.
    `temperature=0.0` for the same reason as rag/date_parse.py: a fixed question on a fixed day
    should always take the same path.

    A recency question caught by `detect_recent_notes_listing` is a notes listing whatever the
    model says; the model's reply is only used for its course words when it agrees on notes."""
    recent_notes = detect_recent_notes_listing(query)
    prompt = _PROMPT_TEMPLATE.format(today=today.strftime("%Y-%m-%d, %A"), query=query)
    try:
        response = await complete_chat(
            [ChatMessage(role="user", content=prompt)],
            model=model,
            api_base=api_base,
            timeout=timeout,
            call_site="listing_intent",
            user_id=user_id,
            temperature=0.0,
        )
    except Exception:
        logger.exception("Listing-intent classification failed, falling back to retrieval")
        return recent_notes
    intent = parse_listing_response(response)
    if recent_notes is None:
        return intent
    if intent is not None and intent.kind == "notes":
        return intent
    if intent is None:
        logger.info("Listing intent: recency phrase check overrode a NONE classification")
    return recent_notes
