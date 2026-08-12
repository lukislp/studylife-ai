"""Retrieval: embed a query, fetch matching chunks from Qdrant, optionally rerank.

`retrieve_with_rerank()` is the sole entry point (see docs/decisions.md
"Retrieval quality: reranking + per-content-type quota"): it fetches an even
per-content-type candidate quota instead of one global top-k - without this,
a single popular course's ~90 near-duplicate session chunks can crowd a
genuinely relevant note out of the running entirely, confirmed live - then
optionally reranks the merged pool with an LLM when `Settings.rerank_model`
is set. `session` is the one exception to the quota: instead of one
per-type vector-search slice, it gets two merged pools - a real Qdrant
DatetimeRange window around "today" (`QdrantStore.get_sessions_in_window()`)
for near-term date questions, plus a normal topic-vector search over ALL
sessions for farther-out topic questions (see docs/decisions.md "Structured
session dates"). The window replaced an earlier "fetch every session, let
the reranker sort it out in free text" approach - confirmed live that an LLM
reading exact dates out of dozens of near-identical passages is unreliable
for anything but the most obvious offsets ("today"), while a real database
range filter isn't.
"""

import asyncio
import logging
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from typing import get_args

from studylife_ai.config import Settings
from studylife_ai.ingestion.qdrant_store import ContentType, QdrantStore, RetrievedChunk
from studylife_ai.llm.embeddings import embed_texts
from studylife_ai.rag.date_parse import DateRange, parse_date_range
from studylife_ai.rag.rerank import rerank_chunks

logger = logging.getLogger(__name__)

_CONTENT_TYPES: tuple[ContentType, ...] = get_args(ContentType)


async def _search_by_vector(
    vector: list[float],
    *,
    store: QdrantStore,
    user_id: str,
    top_k: int,
    content_type: ContentType | None,
) -> list[RetrievedChunk]:
    """Never raises - a failed Qdrant search degrades to no results for that
    content type instead of aborting the whole retrieval, matching the same
    "retrieval failure shouldn't break the caller" philosophy already applied
    to embedding and reranking failures below."""
    try:
        return await store.search(
            vector=vector, user_id=user_id, limit=top_k, content_type=content_type
        )
    except Exception:
        logger.exception("Qdrant search failed for content_type=%s", content_type)
        return []


def _days_from_today(session_start: str | None, *, today: date) -> int:
    """Absolute day distance from `today` - a huge value (sorts last) for a missing/unparseable
    date, which `get_sessions_in_window()`'s results should never actually have (its own range
    filter excludes points without `session_start`), but mypy doesn't know that."""
    if not session_start:
        return 10**9
    try:
        return abs((date.fromisoformat(session_start[:10]) - today).days)
    except ValueError:
        return 10**9


async def _fetch_session_window(
    store: QdrantStore,
    *,
    user_id: str,
    window_days: int,
    today: date,
    top_k: int,
    date_range: DateRange | None = None,
) -> list[RetrievedChunk]:
    """Same never-raises contract as `_search_by_vector` above, for `get_sessions_in_window()`.

    Capped to `top_k` and sorted by proximity to `today` (closest first) before truncating -
    `get_sessions_in_window()` itself is otherwise unbounded (`safety_cap=1000` only), and every
    chunk it returns comes back with `score=0.0` (a plain scroll, not a vector search). Once
    merged with the topic-vector pool below and globally sorted by score, an unbounded window
    pool meant EVERY window chunk sorted behind ANY topic-matched chunk with a real similarity
    score - crowding today/tomorrow-relevant sessions out of the reranker's effective attention
    as account history grew (confirmed live, 2026-08-12: a ~30-chunk window let two topically-
    similar sessions from 3 months earlier outrank the one real "tomorrow" session for a "what's
    on tomorrow" query, purely via this position effect). Truncating by proximity, not scroll
    order, means an overflow drops the FARTHEST-out window days first, not the nearest ones that
    matter most for near-term queries.

    `date_range`, when given (settings.date_parse_model resolved it for this specific query -
    see `_fetch_sessions()` and rag/date_parse.py), replaces the fixed +-`window_days`
    computation with that exact range - everything downstream (the scroll call, the
    proximity-to-today sort, the `top_k` cap) is unchanged either way, so a resolved range gets
    the same "closest-to-today survives an overflow" truncation behavior a fixed window already
    had, not a parallel code path.
    """
    if date_range is not None:
        start = datetime.combine(date_range.start, time.min)
        end = datetime.combine(date_range.end, time.max)
    else:
        now = datetime.now()
        start = now - timedelta(days=window_days)
        end = now + timedelta(days=window_days)
    try:
        chunks = await store.get_sessions_in_window(user_id=user_id, start=start, end=end)
    except Exception:
        logger.exception("Qdrant session-window scroll failed")
        return []
    chunks.sort(key=lambda c: _days_from_today(c.session_start, today=today))
    chunks = chunks[:top_k]
    if date_range is not None:
        # Tags every chunk from an exact, date_parse-resolved range as unconditionally relevant
        # (see RetrievedChunk.exact_date_match and retrieve_with_rerank() below) - never done for
        # the fixed +-window_days fallback, which is a proximity heuristic, not an exact filter.
        chunks = [replace(c, exact_date_match=True) for c in chunks]
    return chunks


async def _fetch_sessions(
    query: str,
    vector: list[float],
    *,
    store: QdrantStore,
    user_id: str,
    settings: Settings,
    today: date,
    top_k: int,
) -> list[RetrievedChunk]:
    """Sessions' two-pool fetch: the date-window above, plus a normal topic-vector search over
    ALL sessions (same `top_k` quota every other content type gets) so a question like "what did
    we cover in Analysis last year" - no near-term date in it at all - still finds something.
    Merged and deduped by `entity_id`, window pool first (it's the one actually relevant to
    date-specific questions, which is the common case this whole design targets). The window leg
    uses `settings.session_window_top_k`, not the shared per-type `top_k` passed in here - see
    that setting's docstring for why it needs its own, larger budget.

    When `settings.date_parse_model` is set, also resolves `query`'s date expression (if any)
    via `parse_date_range()` and, when found, feeds it to the window leg in place of the fixed
    +-session_window_days window - the escalation path named and deferred in docs/decisions.md
    "Structured session dates" (option (2)), needed because the fixed window's per-passage
    day-offset labels have no week/month-range framing for a question like "letzte Woche" (see
    docs/decisions.md "NL date-range resolution"). Unset (the default), a `None` resolution
    result, or a parse-call failure all take the exact same path as before this feature existed
    - zero behavior change, same convention as `rerank_model`. Resolved concurrently with the
    topic-vector leg (independent of it) rather than serially before everything, to keep this
    opt-in feature's added latency to "however long the slower of the two legs takes," not
    "date-parse latency + everything else."
    """

    async def _resolve_date_range() -> DateRange | None:
        if not settings.date_parse_model:
            return None
        return await parse_date_range(
            query,
            model=settings.date_parse_model,
            api_base=settings.llm_api_base,
            timeout=settings.llm_request_timeout_seconds,
            today=today,
            user_id=user_id,
        )

    date_range, topic_chunks = await asyncio.gather(
        _resolve_date_range(),
        _search_by_vector(
            vector, store=store, user_id=user_id, top_k=top_k, content_type="session"
        ),
    )
    window_chunks = await _fetch_session_window(
        store,
        user_id=user_id,
        window_days=settings.session_window_days,
        today=today,
        # An exact resolved range gets its own, larger budget (date_range_chunk_cap) - it's not
        # an approximate "nearest N sessions" guess like the fixed-window fallback, so it
        # shouldn't be capped to the same small quota (see RetrievedChunk.exact_date_match).
        top_k=settings.date_range_chunk_cap
        if date_range is not None
        else settings.session_window_top_k,
        date_range=date_range,
    )
    seen_ids = {c.entity_id for c in window_chunks}
    return window_chunks + [c for c in topic_chunks if c.entity_id not in seen_ids]


async def retrieve_with_rerank(
    query: str,
    *,
    store: QdrantStore,
    settings: Settings,
    user_id: str,
    content_type: ContentType | None = None,
) -> list[RetrievedChunk]:
    """Retrieval entry point used by /chat, the eval pipeline, and the
    search_notes agent tool. `user_id` scopes every search to one user's
    Qdrant partition - resolved per-request from headers for /chat and the
    agent, fixed for eval (see docs/decisions.md "M4.5 Multi-user support").

    When `content_type` is given, fetches a single `rerank_candidate_k`-sized
    pool for that type (no cross-type crowding is possible when already
    scoped to one type). When it's `None` (the default, whole-corpus
    case), fetches an even quota from each content type separately and
    merges them - a type with many entities (e.g. sessions) can no longer
    squeeze out a type with few (e.g. notes) purely by outnumbering it in a
    shared top-k. The query is embedded once and reused across every
    per-type search.

    Reranking (LLM-scored reordering of the merged pool) is applied on top
    when `settings.rerank_model` is set; otherwise the merged pool is just
    sorted by vector-similarity score. Either way, the final list is cut
    down to `settings.retrieval_top_k`.
    """
    today = datetime.now().date()
    vectors = await embed_texts(
        [query], model=settings.embedding_model, call_site="retrieval", user_id=user_id
    )
    if not vectors:
        return []
    vector = vectors[0]

    if content_type is not None:
        chunks = await _search_by_vector(
            vector,
            store=store,
            user_id=user_id,
            top_k=settings.rerank_candidate_k,
            content_type=content_type,
        )
    else:
        per_type_k = max(1, settings.rerank_candidate_k // len(_CONTENT_TYPES))
        # "session" gets the two-pool fetch above instead of a plain vector-similarity quota -
        # see QdrantStore.get_sessions_in_window() and _fetch_sessions() docstrings for why a
        # text embedding alone can't be trusted to rank sessions by date proximity to "today".
        results = await asyncio.gather(
            *(
                _fetch_sessions(
                    query,
                    vector,
                    store=store,
                    user_id=user_id,
                    settings=settings,
                    today=today,
                    top_k=per_type_k,
                )
                if ct == "session"
                else _search_by_vector(
                    vector,
                    store=store,
                    user_id=user_id,
                    top_k=per_type_k,
                    content_type=ct,
                )
                for ct in _CONTENT_TYPES
            )
        )
        chunks = [chunk for group in results for chunk in group]
        chunks.sort(key=lambda c: c.score, reverse=True)

    if settings.rerank_model and chunks:
        chunks = await rerank_chunks(
            query,
            chunks,
            model=settings.rerank_model,
            api_base=settings.llm_api_base,
            timeout=settings.llm_request_timeout_seconds,
            today=today,
            user_id=user_id,
        )

    # Chunks from an exact, date_parse-resolved date range (RetrievedChunk.exact_date_match) are
    # unconditionally relevant to the question - retrieval_top_k exists to approximate relevance
    # under uncertainty (vector similarity), which doesn't apply once a range is exact. They get
    # their own, larger date_range_chunk_cap instead, on top of (not counted against) the normal
    # retrieval_top_k budget everything else still gets. Confirmed live 2026-08-12: a real "last
    # week" had 21 matching sessions, silently cut to 8 by the shared retrieval_top_k before this.
    # When nothing is tagged (the common case - no date_range resolved, or the feature is unset),
    # `exact_matches` is empty and this reduces to today's exact `chunks[:retrieval_top_k]`.
    exact_matches = [c for c in chunks if c.exact_date_match]
    rest = [c for c in chunks if not c.exact_date_match]
    return exact_matches[: settings.date_range_chunk_cap] + rest[: settings.retrieval_top_k]
