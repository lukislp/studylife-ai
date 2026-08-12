"""NL date-range resolution: resolves a query's own date/date-range expression (if any) into
an exact calendar range via a dedicated LLM call, using a plain-text prompt (not JSON mode) for
the same reason as rag/rerank.py - so it works with small local Ollama models too, not just API
providers with JSON mode.

This is the escalation path named and deferred in docs/decisions.md "Structured session dates"
(option (2), "full NL date-range resolution"). The fixed ±SESSION_WINDOW_DAYS window plus
rag/rerank.py's `_relative_day_label()` only give single-day relative labels (e.g. "[3 days
ago]") - there is no week/month/range-level framing anywhere in the pipeline, so a question like
"letzte Woche" requires some downstream LLM to infer "a chunk labeled [3 days ago] falls inside
the range called 'last week'", which proved unreliable live (see docs/decisions.md "NL
date-range resolution"). This module moves that one inference out of any downstream LLM into a
single, narrow, deterministic-as-possible upfront call, whose result then drives a real Qdrant
DatetimeRange filter directly (see rag/retrieval.py's `_fetch_session_window`).
"""

import logging
import re
from dataclasses import dataclass
from datetime import date

from studylife_ai.llm.client import complete_chat
from studylife_ai.schemas.chat import ChatMessage

logger = logging.getLogger(__name__)


@dataclass
class DateRange:
    """An inclusive calendar-day range resolved from a query's own date expression - see
    `parse_date_range()`. `start`/`end` are plain `date`s, not `datetime`s; the caller decides
    how to expand them to a full-day `datetime` range for the actual Qdrant filter."""

    start: date
    end: date


_PROMPT_TEMPLATE = (
    "Today's date is {today}.\n\n"
    "Does the question below refer to a specific date, day, or date range - in any language, "
    'however it\'s phrased? Examples: an exact date ("09.08", "August 9th"), a relative single '
    'day ("gestern"/"yesterday", "übermorgen"/"day after tomorrow"), or a relative range '
    '("letzte Woche"/"last week", "diese Woche"/"this week", "im Mai"/"in May", "seit Anfang '
    'des Monats"/"since the start of the month").\n\n'
    "If yes, reply with EXACTLY one line: START|END, using ISO 8601 dates (YYYY-MM-DD) - the "
    "first and last calendar day the question refers to (a single day: START equals END). Use "
    'a Monday-Sunday week: "last week"/"letzte Woche" is the most recently completed Mon-Sun '
    'week before today\'s week; "this week"/"diese Woche" is the Mon-Sun week containing today. '
    'An open-ended phrase like "seit Anfang des Monats"/"since the start of the month" ends at '
    "today, not the end of the month. No other text, no explanation, no reasoning - just the "
    "one line.\n\n"
    "If the question is NOT about any specific date or date range (e.g. it just asks about a "
    "topic, a course, or general study content, with no date expression at all), reply with "
    "EXACTLY the single word: NONE\n\n"
    "Question: {query}"
)


def _build_prompt(query: str, *, today: date) -> str:
    return _PROMPT_TEMPLATE.format(today=today.strftime("%Y-%m-%d, %A"), query=query)


_RANGE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\s*\|\s*(\d{4}-\d{2}-\d{2})")
_SINGLE_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})$")


def _parse_response(response: str) -> DateRange | None:
    """Parses the model's plain-text reply into a `DateRange`. Degrades to `None` - the same
    "no range" signal as an explicit "NONE" reply - for anything that isn't a recoverable
    date/range: "NONE" itself, empty/garbage output, or malformed dates. Same
    graceful-degradation philosophy as rerank.py's `_parse_order()`: never raise, treat
    unparseable output as "no signal" rather than an error.

    Two recovery paths, most to least specific: a START|END pair (tolerant of surrounding
    prose/whitespace, swaps start/end if the model reversed them); a single bare ISO date, for a
    model that skips the "|" format on an obviously single-day question - still fully
    recoverable, not garbage.
    """
    text = response.strip()
    range_match = _RANGE_RE.search(text)
    if range_match:
        try:
            start = date.fromisoformat(range_match.group(1))
            end = date.fromisoformat(range_match.group(2))
        except ValueError:
            return None
        return DateRange(start, end) if start <= end else DateRange(end, start)
    single_match = _SINGLE_DATE_RE.match(text)
    if single_match:
        try:
            day = date.fromisoformat(single_match.group(1))
        except ValueError:
            return None
        return DateRange(start=day, end=day)
    return None


async def parse_date_range(
    query: str,
    *,
    model: str,
    api_base: str | None,
    timeout: float,
    today: date,
    user_id: str = "unknown",
) -> DateRange | None:
    """Resolves `query`'s date/date-range expression, if any, into a `DateRange`. Never raises -
    returns `None` when the question isn't about a specific date/range, when the model's
    response can't be parsed, or on any call failure (timeout, provider error) - a caller always
    treats `None` as "fall back to whatever default behavior applies without this feature" (see
    rag/retrieval.py's `_fetch_sessions()`).

    Pins `temperature=0.0` (see docs/decisions.md "Reranker temperature pinned") - same
    reasoning as rag/rerank.py: a fixed question against a fixed "today" should resolve to the
    same range every time, not vary call to call.
    """
    prompt = _build_prompt(query, today=today)
    try:
        response = await complete_chat(
            [ChatMessage(role="user", content=prompt)],
            model=model,
            api_base=api_base,
            timeout=timeout,
            call_site="date_parse",
            user_id=user_id,
            temperature=0.0,
        )
    except Exception:
        logger.exception("Date-range parsing failed, falling back to fixed session window")
        return None
    return _parse_response(response)
