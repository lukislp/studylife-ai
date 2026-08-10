"""Standalone seeding entrypoint: `uv run python -m studylife_ai.eval.seed_fixture`.

Run once before `python -m studylife_ai.eval` against a fresh/empty Qdrant
(as in CI - see .github/workflows/ci.yml). Not needed locally, where the
real dev Qdrant is already populated by ingestion.
"""

import asyncio
import logging

from studylife_ai.config import get_settings
from studylife_ai.eval.fixture import load_fixture_notes, seed_fixture_notes
from studylife_ai.ingestion.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


async def _main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    notes = load_fixture_notes()
    store = QdrantStore(url=settings.qdrant_url, collection=settings.qdrant_collection)
    try:
        await seed_fixture_notes(notes, settings=settings, store=store)
    finally:
        await store.close()

    print(f"Seeded {len(notes)} fixture notes into Qdrant.")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
