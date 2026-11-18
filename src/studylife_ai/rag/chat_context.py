"""Builds the context system message for one chat question - shared by /chat and the eval.

Two paths (see docs/decisions.md "Structured listings for enumeration questions"):

- an enumeration question ("which notes do I have in course X", "what's on this week") gets a
  structured listing read live from the StudyLife API (rag/listing.py), filtered by the item's
  own course id and ordered by date - complete and current, no index lag, no top-k cut;
- every other question gets the existing retrieval-augmented context (rag/retrieval.py), now
  annotated with each note's real course and each session's computed past/upcoming label.

Classification, retrieval and the course-catalog fetch run concurrently, so a content question
pays no extra latency for the classification step. Never raises: every leg degrades on its own,
and any listing failure falls back to the retrieval context.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime

from studylife_ai.config import Settings
from studylife_ai.ingestion.qdrant_store import RetrievedChunk
from studylife_ai.rag.listing import Listing, StudyDataSource, fetch_listing
from studylife_ai.rag.listing_intent import ListingIntent, classify_listing_intent
from studylife_ai.rag.prompt import (
    build_context_system_message,
    build_listing_system_message,
    listing_record_texts,
    listing_sources,
    sources_payload,
)
from studylife_ai.schemas.chat import Source
from studylife_ai.studylife.models import CourseDto

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatContext:
    """`contexts` are the plain texts the answer is grounded in (RAGAS' retrieved_contexts)."""

    system_message: str
    sources: list[Source]
    contexts: list[str] = field(default_factory=list)
    listing: Listing | None = None


async def build_chat_context(
    query: str,
    *,
    retrieve: Callable[[], Awaitable[list[RetrievedChunk]]],
    source: StudyDataSource | None,
    settings: Settings,
    now: datetime,
    user_id: str,
) -> ChatContext:
    """`retrieve` runs the vector retrieval for `query`; `source` is the calling user's own
    StudyLife data (their AI key) or `None` when it isn't reachable, which disables listings and
    course annotations but never the answer itself."""

    async def _chunks() -> list[RetrievedChunk]:
        try:
            return await retrieve()
        except Exception:
            logger.exception("Retrieval failed, continuing without note context")
            return []

    async def _intent() -> ListingIntent | None:
        model = settings.effective_listing_intent_model
        if source is None or not model:
            return None
        return await classify_listing_intent(
            query,
            model=model,
            api_base=settings.llm_api_base,
            timeout=settings.llm_request_timeout_seconds,
            today=now.date(),
            user_id=user_id,
        )

    async def _courses() -> list[CourseDto] | None:
        if source is None:
            return None
        try:
            return await source.get_courses()
        except Exception:
            logger.exception("Course catalog fetch failed, continuing without course names")
            return None

    chunks, intent, courses = await asyncio.gather(_chunks(), _intent(), _courses())

    if intent is not None and source is not None and courses is not None:
        try:
            listing = await fetch_listing(
                intent,
                source=source,
                courses=courses,
                now=now,
                max_items=settings.listing_max_items,
                session_history_days=settings.studylife_session_history_days,
            )
        except Exception:
            logger.exception("Structured listing failed, falling back to retrieval context")
        else:
            return ChatContext(
                system_message=build_listing_system_message(listing),
                sources=listing_sources(listing),
                contexts=listing_record_texts(listing),
                listing=listing,
            )

    return ChatContext(
        system_message=build_context_system_message(
            chunks,
            today=now.date(),
            course_names={c.id: c.name for c in courses or []},
        ),
        sources=sources_payload(chunks),
        contexts=[chunk.content for chunk in chunks],
    )
