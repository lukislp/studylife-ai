"""Periodic sweep of stale agent-checkpoint threads (audit A13/F14 rest, 2026-08-26).

`AGENT_CHECKPOINT_DB_PATH` grows one thread per `/agent` call (`f"{user_id}:{uuid4()}"`, see
api/agent.py) - both completed turns and turns that proposed a write action but were never
confirmed/rejected ("pending"). Nothing else ever deletes a thread except an explicit
`/internal/revoke-key` purge (ingestion/sync.py's `purge_user`, one user at a time, only on
revoke) or a failed agent run (api/agent.py's `_invoke_and_handle_failure`, that one thread
only) - so on an active, never-revoked account, threads accumulate in this SQLite file forever.

This sweep deletes any thread (completed or still-pending, deliberately not distinguished -
see `sweep_stale_checkpoint_threads`) whose most recent checkpoint is older than
`Settings.agent_checkpoint_ttl_days`, run on a fixed interval by `run_periodic_checkpoint_cleanup`
- same loop/cancel/log-don't-crash shape as `ingestion.scheduler.run_periodic_sync`, wired up
in main.py's lifespan.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from studylife_ai.config import Settings

logger = logging.getLogger(__name__)


async def _latest_checkpoint_ts(checkpointer: AsyncSqliteSaver, thread_id: str) -> datetime | None:
    """The `ts` (ISO 8601, UTC - see langgraph's `Checkpoint` TypedDict) of `thread_id`'s most
    recent checkpoint row, ordered the same way `AsyncSqliteSaver.aget_tuple` picks "latest"
    (`checkpoint_id DESC` - checkpoint ids are time-ordered uuid6s). None if the thread
    somehow has no row (shouldn't happen, a sweep must not crash on it regardless) or the
    stored blob fails to deserialize (defensive - e.g. a foreign/old serializer version)."""
    async with checkpointer.conn.execute(
        "SELECT type, checkpoint FROM checkpoints WHERE thread_id = ? "
        "ORDER BY checkpoint_id DESC LIMIT 1",
        (thread_id,),
    ) as cursor:
        row = await cursor.fetchone()
    if row is None:
        return None
    checkpoint_type, checkpoint_blob = row
    try:
        checkpoint = checkpointer.serde.loads_typed((checkpoint_type, checkpoint_blob))
    except Exception:
        logger.exception(
            "Checkpoint cleanup: failed to deserialize latest checkpoint for thread_id=%s",
            thread_id,
        )
        return None
    ts = checkpoint.get("ts")
    if not ts:
        return None
    parsed = datetime.fromisoformat(ts)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


async def sweep_stale_checkpoint_threads(checkpointer: AsyncSqliteSaver, ttl_days: int) -> int:
    """Deletes every thread whose most recent checkpoint is older than `ttl_days` - completed
    turns and never-confirmed pending-action turns alike (a proposed write nobody confirmed in
    `ttl_days` is exactly as stale as a finished conversation; neither is worth ever resuming,
    and `/agent/confirm` against a deleted thread_id already returns a clean 404, same as any
    other unknown thread_id). Returns the number of threads deleted, for the caller's per-sweep
    summary log line."""
    async with checkpointer.conn.execute("SELECT DISTINCT thread_id FROM checkpoints") as cursor:
        thread_ids = [row[0] for row in await cursor.fetchall()]

    cutoff = datetime.now(UTC) - timedelta(days=ttl_days)
    deleted = 0
    for thread_id in thread_ids:
        ts = await _latest_checkpoint_ts(checkpointer, thread_id)
        if ts is not None and ts < cutoff:
            await checkpointer.adelete_thread(thread_id)
            deleted += 1
    return deleted


async def run_periodic_checkpoint_cleanup(
    settings: Settings, checkpointer: AsyncSqliteSaver
) -> None:
    """Runs `sweep_stale_checkpoint_threads` every `agent_checkpoint_cleanup_interval_seconds`,
    forever, until cancelled (see main.py's lifespan) - same loop/cancel/never-die-on-one-bad-
    tick shape as `ingestion.scheduler.run_periodic_sync`."""
    while True:
        try:
            deleted = await sweep_stale_checkpoint_threads(
                checkpointer, settings.agent_checkpoint_ttl_days
            )
            logger.info(
                "Checkpoint cleanup: swept %d stale thread(s) (TTL=%dd)",
                deleted,
                settings.agent_checkpoint_ttl_days,
            )
        except Exception:
            logger.exception("Checkpoint cleanup sweep failed")
        await asyncio.sleep(settings.agent_checkpoint_cleanup_interval_seconds)
