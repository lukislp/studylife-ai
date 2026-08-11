"""Orchestrates one ingestion sync run.

Full-list diff against Qdrant's own state (no separate manifest store —
see docs/decisions.md "Incremental sync"): fetch all entities, compare each
against its last known fingerprint, chunk+embed+upsert what's new or
changed, delete what's gone. Runs across all four content types (notes,
courses, sessions, course goals — see docs/decisions.md "Ingestion scope
expansion") through one shared helper, since the diff/chunk/embed/upsert/
delete control flow is identical for all of them.
"""

import asyncio
import hashlib
import logging
from collections.abc import Callable, Sequence

from studylife_ai.config import Settings
from studylife_ai.ingestion.chunking import chunk_text
from studylife_ai.ingestion.qdrant_store import ContentType, EntityChunkMetadata, QdrantStore
from studylife_ai.ingestion.rendering import (
    DATETIME_FORMAT,
    render_course,
    render_course_goal,
    render_session,
)
from studylife_ai.llm.embeddings import embed_texts
from studylife_ai.studylife.client import StudyLifeClient
from studylife_ai.studylife.models import CourseDto, CourseGoalDto, StudyLifeNote, StudySessionDto

logger = logging.getLogger(__name__)


def fingerprint_note(note: StudyLifeNote) -> str:
    """Content hash used to detect note changes.

    Deliberately not `note.updated_at`: StudyLife sets it via a server-local
    `DateTime.Now`, not UTC (see docs/decisions.md) — a content hash sidesteps
    that ambiguity entirely and also catches edits if the timestamp somehow
    wasn't bumped. Hashes title+content (not just content) since a title-only
    edit should still refresh stored metadata.
    """
    digest = hashlib.sha256(f"{note.title}\n{note.content}".encode())
    return digest.hexdigest()


def fingerprint_course(course: CourseDto) -> str:
    return hashlib.sha256(render_course(course).encode()).hexdigest()


def fingerprint_session(session: StudySessionDto) -> str:
    return hashlib.sha256(render_session(session).encode()).hexdigest()


def fingerprint_course_goal(goal: CourseGoalDto) -> str:
    return hashlib.sha256(render_course_goal(goal).encode()).hexdigest()


async def _sync_content_type[T](
    *,
    store: QdrantStore,
    settings: Settings,
    known: dict[tuple[str, int], str],
    entities: Sequence[T],
    content_type: ContentType,
    entity_id: Callable[[T], int],
    fingerprint: Callable[[T], str],
    render_text: Callable[[T], str],
    title: Callable[[T], str],
    course_id: Callable[[T], int | None],
    session_id: Callable[[T], int | None],
) -> None:
    current_ids = {entity_id(e) for e in entities}
    if len(current_ids) != len(entities):
        # Two entities mapped to the same entity_id (e.g. more than one
        # course goal for the same course, which entity_id=course_id
        # assumes never happens) - only the last-diffed one survives the
        # upsert below, silently. Surfacing it here beats a silent data loss.
        logger.warning(
            "Sync[%s]: %d entities share an entity_id with another - only one will be kept",
            content_type,
            len(entities) - len(current_ids),
        )
    known_ids = {eid for (ctype, eid) in known if ctype == content_type}
    deleted_ids = known_ids - current_ids
    changed = [e for e in entities if known.get((content_type, entity_id(e))) != fingerprint(e)]

    logger.info(
        "Sync[%s]: %d total, %d new/changed, %d deleted",
        content_type,
        len(entities),
        len(changed),
        len(deleted_ids),
    )

    for entity in changed:
        chunks = chunk_text(
            render_text(entity),
            chunk_size_tokens=settings.chunk_size_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        vectors = await embed_texts(chunks, model=settings.embedding_model) if chunks else []
        if vectors:
            await store.ensure_collection(vector_size=len(vectors[0]))
        await store.replace_entity(
            chunks=chunks,
            vectors=vectors,
            metadata=EntityChunkMetadata(
                content_type=content_type,
                entity_id=entity_id(entity),
                title=title(entity),
                course_id=course_id(entity),
                session_id=session_id(entity),
                user_id=settings.studylife_user_id,
                fingerprint=fingerprint(entity),
            ),
        )

    for eid in deleted_ids:
        await store.delete_entity(content_type=content_type, entity_id=eid)


async def sync_all(settings: Settings) -> None:
    if not settings.studylife_api_base_url or not settings.studylife_api_key:
        raise RuntimeError(
            "STUDYLIFE_API_BASE_URL and STUDYLIFE_API_KEY must be set to run ingestion."
        )

    store = QdrantStore(url=settings.qdrant_url, collection=settings.qdrant_collection)
    try:
        known = await store.get_known_fingerprints()

        async with StudyLifeClient(
            base_url=settings.studylife_api_base_url,
            api_key=settings.studylife_api_key,
        ) as studylife:
            # Four independent endpoints - fetch concurrently rather than
            # paying each round-trip in series.
            notes, courses, sessions, course_goals = await asyncio.gather(
                studylife.get_notes(),
                studylife.get_courses(),
                studylife.get_sessions_history(
                    days=settings.studylife_session_history_days, only_completed=False
                ),
                studylife.get_course_goals(),
            )

        await _sync_content_type(
            store=store,
            settings=settings,
            known=known,
            entities=notes,
            content_type="note",
            entity_id=lambda n: n.id,
            fingerprint=fingerprint_note,
            render_text=lambda n: n.content,
            title=lambda n: n.title,
            course_id=lambda n: n.course_id,
            session_id=lambda n: n.session_id,
        )

        await _sync_content_type(
            store=store,
            settings=settings,
            known=known,
            entities=courses,
            content_type="course",
            entity_id=lambda c: c.id,
            fingerprint=fingerprint_course,
            render_text=render_course,
            title=lambda c: c.name,
            course_id=lambda _c: None,
            session_id=lambda _c: None,
        )

        await _sync_content_type(
            store=store,
            settings=settings,
            known=known,
            entities=sessions,
            content_type="session",
            entity_id=lambda s: s.id,
            fingerprint=fingerprint_session,
            render_text=render_session,
            title=lambda s: f"{s.course_name}, {s.start_time.strftime(DATETIME_FORMAT)}",
            course_id=lambda s: s.course_id,
            session_id=lambda _s: None,
        )

        # CourseGoalDto has no own id - course_id is its natural unique key
        # (the API only ever returns one goal per course). If that ever
        # changes, this silently keeps only the last-diffed goal per course.
        await _sync_content_type(
            store=store,
            settings=settings,
            known=known,
            entities=course_goals,
            content_type="course_goal",
            entity_id=lambda g: g.course_id,
            fingerprint=fingerprint_course_goal,
            render_text=render_course_goal,
            title=lambda g: f"{g.course_name} goal",
            course_id=lambda g: g.course_id,
            session_id=lambda _g: None,
        )
    finally:
        await store.close()
