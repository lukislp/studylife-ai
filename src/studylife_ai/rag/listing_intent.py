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
    should always take the same path."""
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
        return None
    return parse_listing_response(response)
