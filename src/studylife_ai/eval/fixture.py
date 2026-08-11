"""Seeds a small, fixed corpus into Qdrant for CI eval runs.

CI has no access to a real StudyLife instance (local-only, dev-machine
bound) and no local Ollama, so the eval job needs a self-contained stand-in
for both the ingestion source and the embedding backend. This seeds
eval/fixture_notes.jsonl (and its course/session/course_goal siblings) - the
same corpus used to build and locally validate eval/dataset.jsonl - through
the same chunk+embed+upsert path ingestion.sync uses (`sync_content_type()`,
reused directly rather than reimplemented here), so CI measures against the
real pipeline mechanics.

Courses/sessions/course goals use StudyLife's own DTOs directly (not a
separate Fixture* model per type, unlike notes below) specifically so the
existing `fingerprint_*`/`render_*` functions in `ingestion.sync` apply
unchanged - one seeding path, not a parallel one that could drift from real
ingestion's actual behavior.
"""

from pathlib import Path

from pydantic import BaseModel

from studylife_ai.config import Settings
from studylife_ai.ingestion.chunking import chunk_text
from studylife_ai.ingestion.qdrant_store import EntityChunkMetadata, QdrantStore
from studylife_ai.ingestion.rendering import render_course, render_course_goal, render_session
from studylife_ai.ingestion.sync import (
    fingerprint_course,
    fingerprint_course_goal,
    fingerprint_session,
    sync_content_type,
)
from studylife_ai.llm.embeddings import embed_texts
from studylife_ai.studylife.models import CourseDto, CourseGoalDto, StudySessionDto

DEFAULT_NOTES_PATH = Path("eval/fixture_notes.jsonl")
DEFAULT_COURSES_PATH = Path("eval/fixture_courses.jsonl")
DEFAULT_SESSIONS_PATH = Path("eval/fixture_sessions.jsonl")
DEFAULT_COURSE_GOALS_PATH = Path("eval/fixture_course_goals.jsonl")


class FixtureNote(BaseModel):
    id: int
    title: str
    content: str
    course_id: int | None = None


def load_fixture_notes(path: Path = DEFAULT_NOTES_PATH) -> list[FixtureNote]:
    with path.open(encoding="utf-8") as f:
        return [FixtureNote.model_validate_json(line) for line in f if line.strip()]


def load_fixture_courses(path: Path = DEFAULT_COURSES_PATH) -> list[CourseDto]:
    with path.open(encoding="utf-8") as f:
        return [CourseDto.model_validate_json(line) for line in f if line.strip()]


def load_fixture_sessions(path: Path = DEFAULT_SESSIONS_PATH) -> list[StudySessionDto]:
    with path.open(encoding="utf-8") as f:
        return [StudySessionDto.model_validate_json(line) for line in f if line.strip()]


def load_fixture_course_goals(path: Path = DEFAULT_COURSE_GOALS_PATH) -> list[CourseGoalDto]:
    with path.open(encoding="utf-8") as f:
        return [CourseGoalDto.model_validate_json(line) for line in f if line.strip()]


async def seed_fixture_notes(
    notes: list[FixtureNote], *, settings: Settings, store: QdrantStore
) -> None:
    for note in notes:
        chunks = chunk_text(
            note.content,
            chunk_size_tokens=settings.chunk_size_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        vectors = (
            await embed_texts(chunks, model=settings.embedding_model, call_site="eval-fixture")
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
                entity_id=note.id,
                title=note.title,
                course_id=note.course_id,
                session_id=None,
                user_id=settings.eval_user_id,
                fingerprint="fixture",
                session_start=None,
            ),
        )


async def seed_fixture_courses(
    courses: list[CourseDto], *, settings: Settings, store: QdrantStore
) -> None:
    await sync_content_type(
        store=store,
        settings=settings,
        user_id=settings.eval_user_id,
        known={},
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


async def seed_fixture_sessions(
    sessions: list[StudySessionDto], *, settings: Settings, store: QdrantStore
) -> None:
    await sync_content_type(
        store=store,
        settings=settings,
        user_id=settings.eval_user_id,
        known={},
        entities=sessions,
        content_type="session",
        entity_id=lambda s: s.id,
        fingerprint=fingerprint_session,
        render_text=render_session,
        title=lambda s: f"{s.course_name}, {s.start_time.strftime('%Y-%m-%d %H:%M')}",
        course_id=lambda s: s.course_id,
        session_id=lambda _s: None,
        session_start=lambda s: s.start_time.isoformat(),
    )


async def seed_fixture_course_goals(
    goals: list[CourseGoalDto], *, settings: Settings, store: QdrantStore
) -> None:
    await sync_content_type(
        store=store,
        settings=settings,
        user_id=settings.eval_user_id,
        known={},
        entities=goals,
        content_type="course_goal",
        entity_id=lambda g: g.course_id,
        fingerprint=fingerprint_course_goal,
        render_text=render_course_goal,
        title=lambda g: f"{g.course_name} goal",
        course_id=lambda g: g.course_id,
        session_id=lambda _g: None,
        session_start=lambda _g: None,
    )
