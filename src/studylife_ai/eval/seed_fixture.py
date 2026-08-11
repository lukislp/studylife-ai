"""Standalone seeding entrypoint: `uv run python -m studylife_ai.eval.seed_fixture`.

Run once before `python -m studylife_ai.eval` against a fresh/empty Qdrant
(as in CI - see .github/workflows/ci.yml). Not needed locally, where the
real dev Qdrant is already populated by ingestion.
"""

import asyncio
import logging

from studylife_ai.config import get_settings
from studylife_ai.eval.fixture import (
    load_fixture_course_goals,
    load_fixture_courses,
    load_fixture_notes,
    load_fixture_sessions,
    seed_fixture_course_goals,
    seed_fixture_courses,
    seed_fixture_notes,
    seed_fixture_sessions,
)
from studylife_ai.ingestion.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


async def _main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    notes = load_fixture_notes()
    courses = load_fixture_courses()
    sessions = load_fixture_sessions()
    goals = load_fixture_course_goals()
    store = QdrantStore(url=settings.qdrant_url, collection=settings.qdrant_collection)
    try:
        await seed_fixture_notes(notes, settings=settings, store=store)
        await seed_fixture_courses(courses, settings=settings, store=store)
        await seed_fixture_sessions(sessions, settings=settings, store=store)
        await seed_fixture_course_goals(goals, settings=settings, store=store)
    finally:
        await store.close()

    print(
        f"Seeded {len(notes)} notes, {len(courses)} courses, {len(sessions)} sessions, "
        f"{len(goals)} course goals into Qdrant."
    )


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
