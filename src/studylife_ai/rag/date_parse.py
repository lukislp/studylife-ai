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

import calendar
import logging
import re
from dataclasses import dataclass
from datetime import date, timedelta

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
    "however it's phrased? Reply with EXACTLY one line, using ONE of these formats:\n\n"
    '- An exact date or a single relative day ("gestern"/"yesterday", "09.08", '
    '"übermorgen"/"day after tomorrow", "next Monday"): DATE:<YYYY-MM-DD>\n'
    '- "this week"/"diese Woche": THIS_WEEK\n'
    '- "last week"/"letzte Woche": LAST_WEEK\n'
    '- "this month"/"diesen Monat": THIS_MONTH\n'
    '- a named month ("im Mai"/"in May"): MONTH:<1-12> (just the month number, e.g. MONTH:5 for '
    "May)\n"
    '- "since the start of the month"/"seit Anfang des Monats": SINCE_MONTH_START\n'
    "- anything else, or no date/day/range reference at all (a topic/course question): NONE\n\n"
    "Do NOT calculate any week/month boundary dates yourself - just classify which category "
    "above the question matches; the exact boundaries are computed separately. For DATE:, only "
    "compute a single day's date. No other text, no explanation, no reasoning - just the one "
    "line.\n\n"
    "Question: {query}"
)


def _build_prompt(query: str, *, today: date) -> str:
    return _PROMPT_TEMPLATE.format(today=today.strftime("%Y-%m-%d, %A"), query=query)


_DATE_RE = re.compile(r"^DATE:(\d{4}-\d{2}-\d{2})$")
_MONTH_RE = re.compile(r"^MONTH:(\d{1,2})$")
_SINGLE_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})$")


def _month_bounds(year: int, month: int) -> DateRange:
    last_day = calendar.monthrange(year, month)[1]
    return DateRange(date(year, month, 1), date(year, month, last_day))


def week_bounds(today: date, *, weeks_ago: int) -> DateRange:
    """The Mon-Sun week `weeks_ago` weeks before the week containing `today` (0 = this week,
    1 = last week). Public - also used by api/chat.py to state this week's/last week's exact
    boundaries as ground truth in the answer-generation prompt (see docs/decisions.md "State
    week boundaries as ground truth for the answering LLM too"), not just for retrieval."""
    this_monday = today - timedelta(days=today.weekday())
    monday = this_monday - timedelta(weeks=weeks_ago)
    return DateRange(monday, monday + timedelta(days=6))


def _resolve_named_range(token: str, *, today: date) -> DateRange | None:
    """Deterministically computes the real calendar boundaries for a category the model only
    had to *classify*, never calculate - same "don't trust the LLM with date arithmetic"
    philosophy as rerank.py's `_relative_day_label()`. Live testing found the LLM's own
    "Monday-Sunday week" boundary math unreliable (off by ~2 days) even when the prompt already
    spelled out the convention explicitly - the fix isn't clearer wording, it's not asking the
    LLM to do the arithmetic at all (see docs/decisions.md "NL date-range resolution: LLM
    classifies, Python computes").

    Month-name resolution ("im Mai") is ambiguous about the year when the named month hasn't
    happened yet this year - resolved to "the most recent occurrence not in the future" (this
    year if the month has started, otherwise last year). A known soft edge, not made fully
    deterministic - flagged in docs/decisions.md, not chased further here.
    """
    if token == "THIS_WEEK":
        return week_bounds(today, weeks_ago=0)
    if token == "LAST_WEEK":
        return week_bounds(today, weeks_ago=1)
    if token == "THIS_MONTH":
        return _month_bounds(today.year, today.month)
    if token == "SINCE_MONTH_START":
        return DateRange(date(today.year, today.month, 1), today)
    month_match = _MONTH_RE.match(token)
    if month_match:
        month = int(month_match.group(1))
        if not 1 <= month <= 12:
            return None
        year = today.year if month <= today.month else today.year - 1
        return _month_bounds(year, month)
    return None


def _parse_response(response: str, *, today: date) -> DateRange | None:
    """Parses the model's plain-text classification into a `DateRange`. Degrades to `None` -
    the same "no range" signal as an explicit "NONE" reply - for anything that isn't a
    recoverable date/category: "NONE" itself, empty/garbage output, or a malformed date. Same
    graceful-degradation philosophy as rerank.py's `_parse_order()`: never raise, treat
    unparseable output as "no signal" rather than an error.

    Three recovery paths, most to least specific: `DATE:<iso>` (the documented format); one of
    the named-range tokens, resolved via `_resolve_named_range()`; a bare ISO date without the
    `DATE:` prefix, for a model that skips the prefix on an obviously single-day question -
    still fully recoverable, not garbage.
    """
    text = response.strip()
    date_match = _DATE_RE.match(text)
    if date_match:
        try:
            day = date.fromisoformat(date_match.group(1))
        except ValueError:
            return None
        return DateRange(start=day, end=day)
    resolved = _resolve_named_range(text, today=today)
    if resolved is not None:
        return resolved
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

    The LLM only classifies which category applies (see `_PROMPT_TEMPLATE`) - it never computes
    week/month boundary dates itself; `_resolve_named_range()` does that deterministically in
    Python. Pins `temperature=0.0` (see docs/decisions.md "Reranker temperature pinned") - same
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
    return _parse_response(response, today=today)
