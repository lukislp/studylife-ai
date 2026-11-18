"""Standalone eval entrypoint: `uv run python -m studylife_ai.eval`."""

import asyncio
import logging

from studylife_ai.config import get_settings
from studylife_ai.eval.dataset import load_eval_cases
from studylife_ai.eval.fixture import (
    FixtureDataSource,
    load_fixture_courses,
    load_fixture_notes,
    load_fixture_sessions,
)
from studylife_ai.eval.runner import run_eval
from studylife_ai.ingestion.qdrant_store import QdrantStore
from studylife_ai.llm.logging import configure_llm_usage_logging

logger = logging.getLogger(__name__)


async def _main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    configure_llm_usage_logging()

    cases = load_eval_cases()
    # The same fixture corpus seed_fixture.py puts into Qdrant, standing in for the StudyLife
    # API that structured listings read (see rag/listing.py).
    source = FixtureDataSource(
        notes=load_fixture_notes(),
        courses=load_fixture_courses(),
        sessions=load_fixture_sessions(),
    )
    store = QdrantStore(url=settings.qdrant_url, collection=settings.qdrant_collection)
    try:
        report = await run_eval(cases, settings=settings, store=store, source=source)
    finally:
        await store.close()

    print(f"\n{len(cases)} eval cases, note-match rate: {report.note_match_rate:.0%}\n")
    for result in report.note_matches:
        marker = "OK" if result.matched else "MISS"
        print(f"  [{marker}] {result.case_id}: expected={result.expected_titles or '{}'}")

    passed = sum(r.passed for r in report.answer_terms)
    print(f"\nAnswer term checks: {passed}/{len(report.answer_terms)} passed")
    for term_result in report.answer_terms:
        if not term_result.passed:
            print(
                f"  [FAIL] {term_result.case_id}: missing={term_result.missing_terms or '{}'} "
                f"forbidden_found={term_result.forbidden_terms_found or '{}'}"
            )

    print("\nRAGAS scores (mean across all cases):")
    print(report.ragas_scores.mean(numeric_only=True))


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
