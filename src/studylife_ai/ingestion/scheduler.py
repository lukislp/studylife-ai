"""Periodic ingestion sync: keeps Qdrant in step with StudyLife's own data
without a separate CronJob or a manually-run `python -m studylife_ai.ingestion`
(see docs/decisions.md "Periodic ingestion sync" for why an in-process loop
was chosen over a CronJob at a 60s interval, and why a full sync_all() every
tick is affordable - sync_content_type's per-entity fingerprint diff already
skips anything unchanged, so each tick only ever pays for what actually
changed since the last one).
"""

import asyncio
import logging

from studylife_ai.config import Settings
from studylife_ai.ingestion.sync import sync_all

logger = logging.getLogger(__name__)


async def run_periodic_sync(settings: Settings) -> None:
    """Runs sync_all() every `ingestion_sync_interval_seconds`, forever, until
    the task is cancelled (see main.py's lifespan, which cancels it on
    shutdown). A RuntimeError from sync_all is the expected "not configured
    yet" / "no users registered yet" state, not a failure - logged at info
    level so the loop doesn't fill the log with tracebacks every tick before
    the first user ever registers. Any other exception is logged with a full
    traceback but likewise never stops the loop - one bad tick shouldn't
    permanently end periodic syncing.
    """
    while True:
        try:
            await sync_all(settings)
        except RuntimeError as exc:
            logger.info("Periodic sync skipped: %s", exc)
        except Exception:
            logger.exception("Periodic sync run failed")
        await asyncio.sleep(settings.ingestion_sync_interval_seconds)
