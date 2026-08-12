"""LLM-based reranking: reorders a candidate pool of retrieved chunks by
relevance to the query, using a plain-text prompt (not JSON mode) so it
works with small local Ollama models too, not just API providers (see
docs/decisions.md "Retrieval quality: reranking + per-content-type quota").
"""

import logging
import re
from datetime import date

from studylife_ai.ingestion.qdrant_store import RetrievedChunk
from studylife_ai.llm.client import complete_chat
from studylife_ai.schemas.chat import ChatMessage

logger = logging.getLogger(__name__)

_CONTENT_PREVIEW_CHARS = 300

_PROMPT_TEMPLATE = (
    "Today's date is {today}.\n\n"
    "Rank the following passages by relevance to the question, most relevant "
    "first. Session passages are pre-labeled in brackets with their EXACT, "
    'already-computed offset from today (e.g. "[tomorrow]", "[in 2 '
    'days]", "[3 days ago]") - trust that label as ground truth, don\'t '
    "recompute it yourself from the date in the passage text. If the "
    "question refers to a specific date or date range - in any language, "
    'and however it\'s phrased, e.g. "today", "tomorrow", "day after '
    'tomorrow"/"übermorgen", "yesterday", "next Monday", "in 3 days", "this '
    'week" - work out which of those bracket labels (or, for passages '
    "without one, which calendar date) actually matches. Direction and "
    'exact offset both matter, not just rough closeness: "day after '
    'tomorrow" means precisely "[in 2 days]", and "[3 days ago]" is NOT a '
    "match for it just because it's nearby in time. A passage whose date "
    "doesn't genuinely match what's being asked is NOT relevant, even if it "
    "shares the same topic or course as a well-matching passage. Reply with "
    "ONLY the passage numbers, comma-separated, most to least relevant - "
    'e.g. "2,0,4,1,3". No other text.\n\n'
    "Question: {query}\n\n"
    "Passages:\n{passages}"
)


def _relative_day_label(session_start: str | None, *, today: date) -> str | None:
    """Deterministic day-offset label for a session's start date relative to `today` (e.g.
    "[tomorrow]", "[in 2 days]", "[3 days ago]") - computed in plain Python, not left for the
    LLM to derive from raw calendar dates while scanning dozens of similar-looking passages
    (see docs/decisions.md "Structured session dates" - three rounds of asking the model to do
    this arithmetic itself proved unreliable). Returns None for a non-session passage (no
    `session_start`) or an unparseable value, so those passages just get no label."""
    if not session_start:
        return None
    try:
        session_date = date.fromisoformat(session_start[:10])
    except ValueError:
        return None
    delta = (session_date - today).days
    if delta == 0:
        return "[today]"
    if delta == 1:
        return "[tomorrow]"
    if delta == -1:
        return "[yesterday]"
    if delta > 0:
        return f"[in {delta} days]"
    return f"[{-delta} days ago]"


def _build_prompt(query: str, chunks: list[RetrievedChunk], *, today: date) -> str:
    def _passage(i: int, chunk: RetrievedChunk) -> str:
        label = _relative_day_label(chunk.session_start, today=today)
        prefix = f"{label} " if label else ""
        return (
            f"[{i}] ({chunk.content_type}) {prefix}{chunk.title}: "
            f"{chunk.content[:_CONTENT_PREVIEW_CHARS]}"
        )

    passages = "\n".join(_passage(i, chunk) for i, chunk in enumerate(chunks))
    return _PROMPT_TEMPLATE.format(
        today=today.strftime("%Y-%m-%d, %A"), query=query, passages=passages
    )


def _parse_order(response: str, num_candidates: int) -> list[int]:
    """Parses a comma/whitespace-separated list of indices. Invalid tokens
    (non-numeric, out of range, duplicates) are skipped rather than raising;
    any index the model didn't mention is appended in its original order -
    an empty or completely unparseable response naturally degrades to the
    unchanged original order, not a special-cased fallback."""
    seen: set[int] = set()
    order: list[int] = []
    for token in re.split(r"[,\s]+", response.strip()):
        if not token:
            continue
        try:
            index = int(token)
        except ValueError:
            continue
        if 0 <= index < num_candidates and index not in seen:
            seen.add(index)
            order.append(index)
    # A non-empty response that still recognized less than half the
    # candidates is a signal worth surfacing (e.g. a model 1-indexing
    # instead of following the 0-indexed prompt) - distinct from a genuinely
    # empty/garbage response, which degrades to the unchanged order silently
    # since that's the expected, harmless fallback path.
    if response.strip() and len(seen) < num_candidates / 2:
        logger.warning(
            "Reranker recognized only %d/%d candidate indices in its response "
            "(possible indexing mismatch): %r",
            len(seen),
            num_candidates,
            response,
        )
    order.extend(i for i in range(num_candidates) if i not in seen)
    return order


async def rerank_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    model: str,
    api_base: str | None,
    timeout: float,
    today: date,
    user_id: str = "unknown",
    reasoning_effort: str | None = None,
) -> list[RetrievedChunk]:
    """Reorders `chunks` most-to-least relevant to `query`. Never raises -
    a bad or missing rerank result degrades to the original vector-search
    order instead of breaking retrieval, matching /chat's existing
    "retrieval failed, continue anyway" fallback philosophy.

    `today` grounds date-relative questions ("what's on today/this week?") -
    found live: without it, pure text similarity ranked same-course sessions
    from months away above the actual session happening today, since nothing
    in the prompt let the reranker compare a passage's own date against the
    real current one (see docs/decisions.md "Retrieval quality"). A `date`,
    not a pre-formatted string, so `_build_prompt()` can also use it for
    exact per-passage day-offset arithmetic (see `_relative_day_label()`) -
    not just the "Today's date is ..." sentence.

    Pins `temperature=0.0` on the underlying completion (see docs/decisions.md
    "Reranker temperature pinned") - found live, even after the deterministic
    date labels above: the identical question asked twice about the same real
    date got different answers (once correct, once not), because nothing kept
    the model's sampling consistent call to call. Reranking a fixed pool by a
    fixed prompt should be a deterministic operation; leaving temperature at
    the provider default made it needlessly a coin flip. EXCEPT for reasoning
    models (`reasoning_effort` set): found live (2026-08-13) that OpenAI's
    gpt-5 family flatly rejects `temperature=0.0` ("only temperature=1 is
    supported"), which made every single rerank call raise and silently fall
    back to plain vector-search order - the temperature-pinning fix and
    reasoning-model support are mutually exclusive for this model family, so
    temperature is omitted entirely (not just set to 1) whenever
    `reasoning_effort` is configured, deferring to the model's own default.

    `reasoning_effort` (only relevant for reasoning models like `gpt-5-mini`)
    is passed straight through to `complete_chat` - `None` (the default) is
    stripped by LiteLLM, correct for non-reasoning `model`s.
    """
    if len(chunks) <= 1:
        return chunks

    prompt = _build_prompt(query, chunks, today=today)
    try:
        response = await complete_chat(
            [ChatMessage(role="user", content=prompt)],
            model=model,
            api_base=api_base,
            timeout=timeout,
            call_site="rerank",
            user_id=user_id,
            temperature=None if reasoning_effort else 0.0,
            reasoning_effort=reasoning_effort,
        )
    except Exception:
        logger.exception("Reranking failed, falling back to vector-search order")
        return chunks

    order = _parse_order(response, len(chunks))
    return [chunks[i] for i in order]
