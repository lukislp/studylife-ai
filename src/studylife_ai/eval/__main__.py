"""Standalone eval entrypoint: `uv run python -m studylife_ai.eval`."""

import asyncio
import logging

from studylife_ai.config import get_settings
from studylife_ai.eval.dataset import load_eval_cases
from studylife_ai.eval.runner import run_eval
from studylife_ai.ingestion.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


async def _main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    cases = load_eval_cases()
    store = QdrantStore(url=settings.qdrant_url, collection=settings.qdrant_collection)
    try:
        report = await run_eval(cases, settings=settings, store=store)
    finally:
        await store.close()

    print(f"\n{len(cases)} eval cases, note-match rate: {report.note_match_rate:.0%}\n")
    for result in report.note_matches:
        marker = "OK" if result.matched else "MISS"
        print(f"  [{marker}] {result.case_id}: expected={result.expected_titles or '{}'}")

    print("\nRAGAS scores (mean across all cases):")
    print(report.ragas_scores.mean(numeric_only=True))


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
