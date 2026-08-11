"""Standalone ingestion entrypoint: `uv run python -m studylife_ai.ingestion`."""

import asyncio
import logging

from studylife_ai.config import get_settings
from studylife_ai.ingestion.sync import sync_all


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    asyncio.run(sync_all(settings))


if __name__ == "__main__":
    main()
