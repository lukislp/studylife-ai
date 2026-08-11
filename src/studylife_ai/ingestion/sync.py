"""Orchestrates one ingestion sync run, across all configured users.

Full-list diff against Qdrant's own state (no separate manifest store —
see docs/decisions.md "Incremental sync"): fetch all entities, compare each
against its last known fingerprint, chunk+embed+upsert what's new or
changed, delete what's gone. Runs across all four content types (notes,
courses, sessions, course goals — see docs/decisions.md "Ingestion scope
expansion") through one shared helper, since the diff/chunk/embed/upsert/
delete control flow is identical for all of them. `sync_all()` then repeats
this per StudyLife account registered in `RegisteredKeyStore` (see
docs/decisions.md "M4.5 Multi-user support" - populated by StudyLife's
registration callback, not a manually-maintained list), each into its own
Qdrant partition.
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
from studylife_ai.studylife.registered_keys import RegisteredKeyStore

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
    # "|schema=2" is a one-time migration bump (see docs/decisions.md "Structured session
    # dates"): adding the session_start payload field didn't change render_session()'s text, so
    # without this every already-ingested session would look "unchanged" forever and never get
    # re-upserted with the new field - the diff in sync_content_type() is purely a fingerprint
    # comparison, it has no other way to know the payload shape itself changed. Bumping the
    # fingerprint makes every session look "changed" exactly once, on the next sync tick after
    # this deploys; safe to remove this suffix (or reuse the trick with a new number) the next
    # time such a payload-only migration is needed.
    return hashlib.sha256((render_session(session) + "|schema=2").encode()).hexdigest()


def fingerprint_course_goal(goal: CourseGoalDto) -> str:
    return hashlib.sha256(render_course_goal(goal).encode()).hexdigest()


async def sync_content_type[T](
    *,
    store: QdrantStore,
    settings: Settings,
    user_id: str,
    known: dict[tuple[str, int], str],
    entities: Sequence[T],
    content_type: ContentType,
    entity_id: Callable[[T], int],
    fingerprint: Callable[[T], str],
    render_text: Callable[[T], str],
    title: Callable[[T], str],
    course_id: Callable[[T], int | None],
    session_id: Callable[[T], int | None],
    session_start: Callable[[T], str | None],
) -> None:
    """Diffs `entities` against `known` and chunks/embeds/upserts/deletes as needed.

    Public (not `_`-prefixed): reused by `eval/fixture.py`'s course/session/course_goal seeding
    (with `known={}`, so every fixture entity is treated as new) - the exact same chunk+embed+
    upsert mechanics real ingestion uses, not a separate parallel implementation.
    """
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
        vectors = (
            await embed_texts(chunks, model=settings.embedding_model, call_site="ingestion")
            if chunks
            else []
        )
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
                user_id=user_id,
                fingerprint=fingerprint(entity),
                session_start=session_start(entity),
            ),
        )

    for eid in deleted_ids:
        await store.delete_entity(user_id=user_id, content_type=content_type, entity_id=eid)


async def sync_user(
    *, user_id: str, ai_api_key: str, settings: Settings, store: QdrantStore
) -> None:
    """Sync one user's notes/courses/sessions/course goals into their own Qdrant partition.

    Public (not `_`-prefixed): called both by `sync_all()` below (looping over every
    registered user) and directly by `api/internal.py`'s registration handler (a single user,
    right after they register their key - see docs/decisions.md "Auto-ingestion on register").
    Takes an already-open `store` rather than opening its own - the caller owns that lifecycle,
    since `api/internal.py`'s caller reuses the app-lifetime `app.state.qdrant_store` instead of
    opening a new connection per registration.
    """
    known = await store.get_known_fingerprints(user_id=user_id)

    async with StudyLifeClient(
        base_url=settings.studylife_api_base_url,  # type: ignore[arg-type]
        api_key=ai_api_key,
        ca_cert_path=settings.studylife_ca_cert_path,
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

    await sync_content_type(
        store=store,
        settings=settings,
        user_id=user_id,
        known=known,
        entities=notes,
        content_type="note",
        entity_id=lambda n: n.id,
        fingerprint=fingerprint_note,
        render_text=lambda n: n.content,
        title=lambda n: n.title,
        course_id=lambda n: n.course_id,
        session_id=lambda n: n.session_id,
        session_start=lambda _n: None,
    )

    await sync_content_type(
        store=store,
        settings=settings,
        user_id=user_id,
        known=known,
        entities=courses,
        content_type="course",
        entity_id=lambda c: c.id,
        fingerprint=fingerprint_course,
        render_text=render_course,
        title=lambda c: c.name,
        course_id=lambda _c: None,
        session_id=lambda _c: None,
        session_start=lambda _c: None,
    )

    await sync_content_type(
        store=store,
        settings=settings,
        user_id=user_id,
        known=known,
        entities=sessions,
        content_type="session",
        entity_id=lambda s: s.id,
        fingerprint=fingerprint_session,
        render_text=render_session,
        title=lambda s: f"{s.course_name}, {s.start_time.strftime(DATETIME_FORMAT)}",
        course_id=lambda s: s.course_id,
        session_id=lambda _s: None,
        session_start=lambda s: s.start_time.isoformat(),
    )

    # CourseGoalDto has no own id - course_id is its natural unique key
    # (the API only ever returns one goal per course). If that ever
    # changes, this silently keeps only the last-diffed goal per course.
    await sync_content_type(
        store=store,
        settings=settings,
        user_id=user_id,
        known=known,
        entities=course_goals,
        content_type="course_goal",
        entity_id=lambda g: g.course_id,
        fingerprint=fingerprint_course_goal,
        render_text=render_course_goal,
        title=lambda g: f"{g.course_name} goal",
        course_id=lambda g: g.course_id,
        session_id=lambda _g: None,
        session_start=lambda _g: None,
    )


async def sync_all(settings: Settings) -> None:
    """Syncs every registered StudyLife account (see docs/decisions.md
    "M4.5 Multi-user support") into its own Qdrant partition, one after
    another - not concurrently, since each user's own four-endpoint fetch is
    already the parallel unit of work, and running multiple users at once
    would multiply StudyLife API load for no benefit at ingestion's scale.

    One user's failure (e.g. a revoked ai_api_key, a StudyLife-side error) is
    logged and skipped rather than aborting the whole run - otherwise a
    single broken account would silently starve ingestion for every other
    registered user until fixed.
    """
    if not settings.studylife_api_base_url:
        raise RuntimeError("STUDYLIFE_API_BASE_URL must be set to run ingestion.")

    store = QdrantStore(url=settings.qdrant_url, collection=settings.qdrant_collection)
    registered_keys = RegisteredKeyStore(settings.registered_keys_db_path)
    try:
        await registered_keys.setup()
        user_ids = await registered_keys.list_user_ids()
        if not user_ids:
            raise RuntimeError(
                "No users registered - generate an AiApiKey in StudyLife's settings first."
            )
        for user_id in user_ids:
            logger.info("Sync: starting user_id=%s", user_id)
            ai_api_key = await registered_keys.get(user_id)
            if ai_api_key is None:
                # Revoked between list_user_ids() and here - skip rather than
                # sync with a stale/missing credential.
                logger.info("Sync: user_id=%s was revoked mid-run, skipping", user_id)
                continue
            try:
                await sync_user(
                    user_id=user_id, ai_api_key=ai_api_key, settings=settings, store=store
                )
            except Exception:
                logger.exception("Sync: failed for user_id=%s, skipping", user_id)
    finally:
        await store.close()
        await registered_keys.close()
