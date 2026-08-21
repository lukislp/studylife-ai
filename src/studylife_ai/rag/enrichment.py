"""Capture enrichment (studylife-capture browser extension, see docs/decisions.md "Capture
enrichment" and "Capture enrichment: related notes + immediate ingestion"): given a freshly
captured note's text, finds the best-matching existing course and the most similar existing
notes by embedding similarity, generates a short tag list + one-sentence summary via a dedicated
LLM call, and immediately embeds the capture itself into Qdrant instead of waiting for the next
periodic sync. Same "never raises, degrade to a safe default" contract as rag/rerank.py and
rag/date_parse.py - a bad LLM response or an unreachable Qdrant must not fail the enrichment
call, since the note it's enriching has already been created by the time this runs (called from
api/internal.py's /internal/enrich-capture, itself triggered by StudyLife's own background task,
not synchronously from the extension's save request - see StudyLife.Server's
BackgroundTaskService.CaptureEnrichment.cs).
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime

from studylife_ai.config import Settings
from studylife_ai.ingestion.chunking import chunk_text
from studylife_ai.ingestion.qdrant_store import EntityChunkMetadata, QdrantStore
from studylife_ai.ingestion.rendering import render_note
from studylife_ai.ingestion.sync import fingerprint_note
from studylife_ai.llm.client import complete_chat
from studylife_ai.llm.embeddings import embed_texts
from studylife_ai.schemas.chat import ChatMessage
from studylife_ai.studylife.models import StudyLifeNote

logger = logging.getLogger(__name__)

_MAX_TAGS = 5
_MAX_RELATED_NOTES = 3
# Raw chunk candidates fetched before deduping to distinct notes (see _find_related_notes) - a
# long existing note can contribute several chunks to the top results, so this needs headroom
# above _MAX_RELATED_NOTES to still surface that many DISTINCT notes.
_RELATED_NOTES_CANDIDATE_POOL = 10
_CONTENT_PREVIEW_CHARS = 2000

_PROMPT_TEMPLATE = (
    "You are tagging and summarizing a short note a student just saved while reading something "
    "online. Reply with EXACTLY two lines, no other text, no explanation:\n\n"
    "TAGS: comma-separated list of at most {max_tags} short topical keywords (single words or "
    "short phrases), same language as the note\n"
    "SUMMARY: one concise sentence describing what the note is about, same language as the note\n\n"
    "Title: {title}\n\n"
    "Content:\n{content}"
)


@dataclass
class CaptureEnrichment:
    course_id: int | None
    course_confidence: float | None
    tags: list[str]
    summary: str | None
    related_note_ids: list[int]


def _build_prompt(title: str, content: str) -> str:
    return _PROMPT_TEMPLATE.format(
        max_tags=_MAX_TAGS, title=title, content=content[:_CONTENT_PREVIEW_CHARS]
    )


def _parse_response(response: str) -> tuple[list[str], str | None]:
    """Parses the TAGS:/SUMMARY: lines. A missing or malformed line degrades to an empty tag
    list / no summary rather than raising - same fallback philosophy as rag/rerank.py's
    `_parse_order()` and rag/date_parse.py's response parsing."""
    tags: list[str] = []
    summary: str | None = None
    for line in response.splitlines():
        stripped = line.strip()
        tags_match = re.match(r"^TAGS:\s*(.*)$", stripped, re.IGNORECASE)
        if tags_match:
            tags = [t.strip() for t in tags_match.group(1).split(",") if t.strip()][:_MAX_TAGS]
            continue
        summary_match = re.match(r"^SUMMARY:\s*(.*)$", stripped, re.IGNORECASE)
        if summary_match and summary_match.group(1).strip():
            summary = summary_match.group(1).strip()
    return tags, summary


async def _embed_content(content: str, *, user_id: str, settings: Settings) -> list[float] | None:
    """Embeds `content` once, reused by both _match_course and _find_related_notes below (was
    previously embedded separately per call site - wasteful, and the two searches want the exact
    same vector anyway). Never raises - a failure here just means course-matching and
    related-notes both degrade to empty, independent of tag/summary generation."""
    try:
        vectors = await embed_texts(
            [content], model=settings.embedding_model, call_site="capture-enrich", user_id=user_id
        )
        return vectors[0] if vectors else None
    except Exception:
        logger.exception("Capture content embedding failed for user_id=%s", user_id)
        return None


# How many note/session candidates _match_via_related_content fetches per content type - small
# headroom above 1, since the single closest note/session might not have a course_id (a
# course-less general note) even when a slightly-lower-ranked one does.
_COURSE_FALLBACK_CANDIDATE_POOL = 5


async def _match_course(
    vector: list[float] | None,
    *,
    user_id: str,
    note_id: int,
    active_course_ids: list[int],
    settings: Settings,
    store: QdrantStore,
) -> tuple[int | None, float | None]:
    """Resolves a course for the capture in two steps, both scoped to `active_course_ids` (the
    StudyLife-side caller's UserSettingsDto.SelectedCourseIds - found live 2026-08-21 that
    matching against a user's ENTIRE course history, semesters-old and completed courses
    included, made a wrong match measurably more likely purely from topical vocabulary overlap
    with a currently-active course; an empty `active_course_ids` means no active courses to
    match against at all, not "match against everything"). Never raises - any failure (no
    vector, Qdrant unreachable) just leaves the capture unassigned.

    1. Direct match against the course partition (see ingestion/sync.py's `render_course()` -
       courses are embedded the same way notes/sessions/goals are, `entity_id` is the real
       course id). `render_course()` is deliberately sparse (name/code/semester/topics, not
       prose) - a genuinely correct match against a prose-heavy capture can score surprisingly
       low this way (found live 2026-08-21: a real match scored only 0.44), so this step alone
       is too unreliable to be the only signal.
    2. Fallback: if step 1 doesn't clear the threshold, search the user's own existing notes and
       sessions (in that order - notes are closer in "register" to a captured article than a
       session's own structured fields) for the closest match that already has a course
       assigned, and inherit that course. Prose-to-prose comparison against real, already-
       course-tagged content the user wrote themselves - much more reliable than comparing
       against the course's own sparse metadata blurb.

    Both "no course found at all" and "found one but below threshold" return (None, None) - the
    public contract deliberately doesn't distinguish the sub-cases (the caller only cares
    "assign or don't"); the log lines exist purely so the reason is diagnosable from logs alone
    (found live 2026-08-21: `course_id=None confidence=None` in the endpoint's own log line is
    genuinely ambiguous between "zero course results" and "a near-miss just under threshold" -
    operationally very different situations)."""
    if vector is None:
        return None, None
    try:
        direct_id, direct_score = await _best_direct_course_match(
            vector, user_id=user_id, active_course_ids=active_course_ids, store=store
        )
        if direct_score is not None and direct_score >= settings.capture_course_match_threshold:
            return direct_id, direct_score

        fallback_id, fallback_score = await _match_course_via_related_content(
            vector,
            user_id=user_id,
            note_id=note_id,
            active_course_ids=active_course_ids,
            settings=settings,
            store=store,
        )
        if fallback_id is not None:
            return fallback_id, fallback_score

        if direct_id is not None:
            logger.info(
                "Capture course-matching: best direct match below threshold for user_id=%s "
                "(entity_id=%s score=%.4f threshold=%.4f), no note/session fallback matched either",
                user_id,
                direct_id,
                direct_score,
                settings.capture_course_match_threshold,
            )
        else:
            logger.info(
                "Capture course-matching: no course results at all for user_id=%s, "
                "no note/session fallback matched either",
                user_id,
            )
        return None, None
    except Exception:
        logger.exception("Capture course-matching failed for user_id=%s", user_id)
        return None, None


async def _best_direct_course_match(
    vector: list[float], *, user_id: str, active_course_ids: list[int], store: QdrantStore
) -> tuple[int | None, float | None]:
    results = await store.search(
        vector=vector,
        user_id=user_id,
        limit=1,
        content_type="course",
        entity_ids=active_course_ids,
    )
    if not results:
        return None, None
    return results[0].entity_id, results[0].score


async def _match_course_via_related_content(
    vector: list[float],
    *,
    user_id: str,
    note_id: int,
    active_course_ids: list[int],
    settings: Settings,
    store: QdrantStore,
) -> tuple[int | None, float | None]:
    for content_type in ("note", "session"):
        results = await store.search(
            vector=vector,
            user_id=user_id,
            limit=_COURSE_FALLBACK_CANDIDATE_POOL,
            content_type=content_type,
            course_ids=active_course_ids,
        )
        for chunk in results:
            # Excludes the capture being enriched itself, in case it was already immediately-
            # ingested by a previous partial run (same defensive reasoning as
            # _find_related_notes) - only relevant for content_type="note".
            if content_type == "note" and chunk.entity_id == note_id:
                continue
            if (
                chunk.course_id is not None
                and chunk.score >= settings.capture_course_match_threshold
            ):
                logger.info(
                    "Capture course-matching: matched via existing %s (entity_id=%s "
                    "course_id=%s score=%.4f) for user_id=%s",
                    content_type,
                    chunk.entity_id,
                    chunk.course_id,
                    chunk.score,
                    user_id,
                )
                return chunk.course_id, chunk.score
    return None, None


async def _find_related_notes(
    vector: list[float] | None, *, user_id: str, note_id: int, store: QdrantStore
) -> list[int]:
    """Searches the note partition for the most similar EXISTING notes (excluding the capture
    being enriched itself, in case it was already immediately-ingested by a previous partial
    run - see _ingest_note). Results are per-CHUNK, not per-note (ingestion/sync.py's
    entity_id=lambda n: n.id for notes, but a long note can contribute several chunks) - dedupes
    to distinct note ids, keeping the first (highest-scoring, since QdrantStore.search() returns
    Qdrant's own score-sorted order) occurrence of each. Never raises - a Qdrant outage just
    means no related notes are suggested, independent of every other enrichment step."""
    if vector is None:
        return []
    try:
        results = await store.search(
            vector=vector, user_id=user_id, limit=_RELATED_NOTES_CANDIDATE_POOL, content_type="note"
        )
        related: list[int] = []
        seen = {note_id}
        for chunk in results:
            if chunk.entity_id in seen:
                continue
            seen.add(chunk.entity_id)
            related.append(chunk.entity_id)
            if len(related) >= _MAX_RELATED_NOTES:
                break
        return related
    except Exception:
        logger.exception("Capture related-notes search failed for user_id=%s", user_id)
        return []


async def _generate_tags_and_summary(
    title: str, content: str, *, user_id: str, settings: Settings
) -> tuple[list[str], str | None]:
    """Same model-resolution convention as rag/retrieval.py's reranking call: a small/fast
    model suffices for this (tagging/summarizing, not answer generation), independent of
    `llm_model` - falls back to it only when no dedicated `rerank_model` is configured."""
    if settings.rerank_model:
        model, reasoning_effort = settings.rerank_model, settings.rerank_reasoning_effort
    else:
        model, reasoning_effort = settings.llm_model, settings.llm_reasoning_effort

    try:
        response = await complete_chat(
            [ChatMessage(role="user", content=_build_prompt(title, content))],
            model=model,
            api_base=settings.llm_api_base,
            timeout=settings.llm_request_timeout_seconds,
            call_site="capture-enrich",
            user_id=user_id,
            # Same reasoning-model exemption as rag/rerank.py's temperature pinning: OpenAI's
            # gpt-5 family rejects temperature=0.0 outright when reasoning_effort is set.
            temperature=None if reasoning_effort else 0.0,
            reasoning_effort=reasoning_effort,
        )
    except Exception:
        logger.exception("Capture tag/summary generation failed for user_id=%s", user_id)
        return [], None
    return _parse_response(response)


async def _ingest_note(
    note_id: int,
    title: str,
    content: str,
    *,
    user_id: str,
    settings: Settings,
    store: QdrantStore,
    course_id: int | None,
) -> None:
    """Immediately embeds+upserts the just-created capture note into Qdrant instead of waiting
    for the next periodic sync_all() pass (up to ingestion_sync_interval_seconds later, see
    ingestion/scheduler.py) - so it's searchable right away, both for a future capture's
    related-notes search above and for /chat and /agent. Safe to run every time a capture is
    enriched even though the next periodic sync will see this note again and write it a second
    time: QdrantStore.replace_entity() always deletes-then-inserts (see its own docstring), so no
    duplicate points are ever possible regardless of fingerprint drift - at worst a fingerprint
    mismatch costs one harmless extra re-embed on the next sync tick, never a correctness issue.

    Builds a minimal placeholder StudyLifeNote purely to reuse fingerprint_note()/render_note()
    instead of duplicating their formulas here - is_markdown=False is exact, not a guess (the
    studylife-capture extension always sends isMarkdown: false, see its api.ts), so the
    fingerprint this writes matches what the next real sync_all() pass computes from StudyLife's
    own API response, as long as the note is unedited in between (the common case).
    """
    try:
        placeholder = StudyLifeNote(
            id=note_id,
            title=title,
            content=content,
            is_markdown=False,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            course_id=course_id,
            session_id=None,
        )
        rendered = render_note(placeholder)
        chunks = chunk_text(
            rendered,
            chunk_size_tokens=settings.chunk_size_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        vectors = (
            await embed_texts(
                chunks,
                model=settings.embedding_model,
                call_site="capture-enrich-ingest",
                user_id=user_id,
            )
            if chunks
            else []
        )
        if vectors:
            await store.ensure_collection(vector_size=len(vectors[0]))
        await store.replace_entity(
            chunks=chunks,
            vectors=vectors,
            metadata=EntityChunkMetadata(
                content_type="note",
                entity_id=note_id,
                title=title,
                course_id=course_id,
                session_id=None,
                user_id=user_id,
                fingerprint=fingerprint_note(placeholder),
                session_start=None,
            ),
        )
    except Exception:
        logger.exception("Immediate ingestion failed for user_id=%s note_id=%d", user_id, note_id)


async def enrich_capture(
    note_id: int,
    title: str,
    content: str,
    *,
    user_id: str,
    active_course_ids: list[int],
    settings: Settings,
    store: QdrantStore,
) -> CaptureEnrichment:
    """Never raises - every sub-step degrades to a safe default independently, so e.g. a Qdrant
    outage doesn't also block tag/summary generation, and vice versa.

    `active_course_ids` scopes course-matching to the caller's currently-active courses only
    (see _match_course's docstring) - the caller (StudyLife.Server) is the source of truth for
    which courses are active (UserSettingsDto.SelectedCourseIds), studylife-ai has no notion of
    "active" on its own.

    Course-matching, related-notes search, and tag/summary generation all run concurrently
    (independent of each other - the first two share one embedding call, the third doesn't need
    a vector at all); immediate ingestion runs last, after course_id is known, so the ingested
    note's own Qdrant payload carries the correct course_id from the start rather than None.
    """
    vector = await _embed_content(content, user_id=user_id, settings=settings)
    (course_id, confidence), related_note_ids, (tags, summary) = await asyncio.gather(
        _match_course(
            vector,
            user_id=user_id,
            note_id=note_id,
            active_course_ids=active_course_ids,
            settings=settings,
            store=store,
        ),
        _find_related_notes(vector, user_id=user_id, note_id=note_id, store=store),
        _generate_tags_and_summary(title, content, user_id=user_id, settings=settings),
    )
    await _ingest_note(
        note_id,
        title,
        content,
        user_id=user_id,
        settings=settings,
        store=store,
        course_id=course_id,
    )
    return CaptureEnrichment(
        course_id=course_id,
        course_confidence=confidence,
        tags=tags,
        summary=summary,
        related_note_ids=related_note_ids,
    )
