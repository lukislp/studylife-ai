"""Capture enrichment (studylife-capture browser extension, see docs/decisions.md "Capture
enrichment"): given a freshly captured note's text, finds the best-matching existing course by
embedding similarity and generates a short tag list + one-sentence summary via a dedicated LLM
call. Same "never raises, degrade to a safe default" contract as rag/rerank.py and
rag/date_parse.py - a bad LLM response or an unreachable Qdrant must not fail the enrichment
call, since the note it's enriching has already been created by the time this runs (called from
api/internal.py's /internal/enrich-capture, itself triggered by StudyLife's own background task,
not synchronously from the extension's save request - see StudyLife.Server's
BackgroundTaskService.CaptureEnrichment.cs).
"""

import logging
import re
from dataclasses import dataclass

from studylife_ai.config import Settings
from studylife_ai.ingestion.qdrant_store import QdrantStore
from studylife_ai.llm.client import complete_chat
from studylife_ai.llm.embeddings import embed_texts
from studylife_ai.schemas.chat import ChatMessage

logger = logging.getLogger(__name__)

_MAX_TAGS = 5
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


async def _match_course(
    content: str, *, user_id: str, settings: Settings, store: QdrantStore
) -> tuple[int | None, float | None]:
    """Embeds `content` and searches the course partition of the shared Qdrant collection for
    the closest match (see ingestion/sync.py's `render_course()` - courses are embedded the same
    way notes/sessions/goals are, `entity_id` is the real course id). Never raises - any failure
    (embedding call, Qdrant unreachable) just leaves the capture unassigned."""
    try:
        vectors = await embed_texts(
            [content], model=settings.embedding_model, call_site="capture-enrich", user_id=user_id
        )
        if not vectors:
            return None, None
        results = await store.search(
            vector=vectors[0], user_id=user_id, limit=1, content_type="course"
        )
        if not results or results[0].score < settings.capture_course_match_threshold:
            return None, None
        return results[0].entity_id, results[0].score
    except Exception:
        logger.exception("Capture course-matching failed for user_id=%s", user_id)
        return None, None


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


async def enrich_capture(
    title: str, content: str, *, user_id: str, settings: Settings, store: QdrantStore
) -> CaptureEnrichment:
    """Never raises - course-matching and tag/summary generation degrade to safe defaults
    independently, so e.g. a Qdrant outage doesn't also block tag/summary generation."""
    course_id, confidence = await _match_course(
        content, user_id=user_id, settings=settings, store=store
    )
    tags, summary = await _generate_tags_and_summary(
        title, content, user_id=user_id, settings=settings
    )
    return CaptureEnrichment(
        course_id=course_id, course_confidence=confidence, tags=tags, summary=summary
    )
