"""Tests for the checkpoint TTL sweep (audit A13/F14 rest, see
studylife_ai.agent.checkpoint_cleanup)."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from studylife_ai.agent.checkpoint_cleanup import (
    sweep_stale_checkpoint_threads,
)
from studylife_ai.config import Settings


def _checkpoint(checkpoint_id: str, ts: datetime) -> dict[str, Any]:
    """A minimal, real-shape `Checkpoint` (langgraph's `Checkpoint` TypedDict) - enough for
    `checkpointer.serde.dumps_typed`/`loads_typed` to round-trip it, which is all the sweep
    actually needs (it only ever reads back `ts`)."""
    return {
        "v": 1,
        "id": checkpoint_id,
        "ts": ts.isoformat(),
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }


async def _insert_checkpoint(
    checkpointer: AsyncSqliteSaver, thread_id: str, checkpoint_id: str, ts: datetime
) -> None:
    """Writes one checkpoint row directly (bypassing `aput`'s full config/metadata plumbing,
    same raw-insert style as test_internal.py's own checkpoint-table tests) - `checkpoint_id`
    controls ordering within a thread (`ORDER BY checkpoint_id DESC`, mirrors real uuid6 ids
    being lexicographically time-ordered), `ts` is what the sweep actually reads."""
    type_, blob = checkpointer.serde.dumps_typed(_checkpoint(checkpoint_id, ts))
    await checkpointer.conn.execute(
        "INSERT INTO checkpoints "
        "(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, "
        "metadata) VALUES (?, '', ?, NULL, ?, ?, ?)",
        (thread_id, checkpoint_id, type_, blob, b"{}"),
    )
    await checkpointer.conn.commit()


async def _thread_ids(checkpointer: AsyncSqliteSaver) -> set[str]:
    async with checkpointer.conn.execute("SELECT DISTINCT thread_id FROM checkpoints") as cursor:
        return {row[0] for row in await cursor.fetchall()}


async def test_sweep_deletes_a_thread_whose_latest_checkpoint_is_older_than_the_ttl(
    tmp_path: Path,
) -> None:
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.db")) as checkpointer:
        await checkpointer.setup()
        stale_ts = datetime.now(UTC) - timedelta(days=31)
        await _insert_checkpoint(checkpointer, "alice:thread-1", "0001", stale_ts)

        deleted = await sweep_stale_checkpoint_threads(checkpointer, ttl_days=30)

        assert deleted == 1
        assert await _thread_ids(checkpointer) == set()


async def test_sweep_keeps_a_thread_whose_latest_checkpoint_is_within_the_ttl(
    tmp_path: Path,
) -> None:
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.db")) as checkpointer:
        await checkpointer.setup()
        fresh_ts = datetime.now(UTC) - timedelta(days=1)
        await _insert_checkpoint(checkpointer, "alice:thread-1", "0001", fresh_ts)

        deleted = await sweep_stale_checkpoint_threads(checkpointer, ttl_days=30)

        assert deleted == 0
        assert await _thread_ids(checkpointer) == {"alice:thread-1"}


async def test_sweep_only_deletes_stale_threads_not_fresh_ones_in_the_same_run(
    tmp_path: Path,
) -> None:
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.db")) as checkpointer:
        await checkpointer.setup()
        stale_ts = datetime.now(UTC) - timedelta(days=31)
        fresh_ts = datetime.now(UTC) - timedelta(days=1)
        await _insert_checkpoint(checkpointer, "alice:thread-1", "0001", stale_ts)
        await _insert_checkpoint(checkpointer, "bob:thread-1", "0001", fresh_ts)

        deleted = await sweep_stale_checkpoint_threads(checkpointer, ttl_days=30)

        assert deleted == 1
        assert await _thread_ids(checkpointer) == {"bob:thread-1"}


async def test_sweep_uses_the_most_recent_checkpoint_per_thread_not_the_oldest(
    tmp_path: Path,
) -> None:
    """A thread with several checkpoints (a real agent turn writes more than one) must be
    judged by its LATEST checkpoint, not an old intermediate one - otherwise an active thread
    with any checkpoint older than the TTL would be wrongly swept."""
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.db")) as checkpointer:
        await checkpointer.setup()
        old_ts = datetime.now(UTC) - timedelta(days=60)
        recent_ts = datetime.now(UTC) - timedelta(days=1)
        # "0001" < "0002" lexicographically - same ordering behavior as real time-ordered
        # uuid6 checkpoint ids.
        await _insert_checkpoint(checkpointer, "alice:thread-1", "0001", old_ts)
        await _insert_checkpoint(checkpointer, "alice:thread-1", "0002", recent_ts)

        deleted = await sweep_stale_checkpoint_threads(checkpointer, ttl_days=30)

        assert deleted == 0
        assert await _thread_ids(checkpointer) == {"alice:thread-1"}


async def test_sweep_deletes_a_pending_unconfirmed_thread_past_the_ttl_the_same_as_a_completed_one(
    tmp_path: Path,
) -> None:
    """Explicitly covers "pending proposals past TTL": a thread's completed/pending status
    isn't tracked separately here - both age out purely by their latest checkpoint's `ts`, and
    a proposed write nobody confirmed in `ttl_days` is exactly as stale as a finished turn."""
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.db")) as checkpointer:
        await checkpointer.setup()
        stale_ts = datetime.now(UTC) - timedelta(days=45)
        # A "pending" thread in reality just has an interrupted/paused checkpoint - from the
        # sweep's perspective (it only reads `ts`) that's indistinguishable from a completed
        # one, which is exactly the point.
        await _insert_checkpoint(checkpointer, "alice:pending-thread", "0001", stale_ts)

        deleted = await sweep_stale_checkpoint_threads(checkpointer, ttl_days=30)

        assert deleted == 1


async def test_sweep_deletes_every_stale_thread_across_multiple_users(tmp_path: Path) -> None:
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.db")) as checkpointer:
        await checkpointer.setup()
        stale_ts = datetime.now(UTC) - timedelta(days=31)
        await _insert_checkpoint(checkpointer, "alice:thread-1", "0001", stale_ts)
        await _insert_checkpoint(checkpointer, "bob:thread-1", "0001", stale_ts)
        await _insert_checkpoint(checkpointer, "bob:thread-2", "0001", stale_ts)

        deleted = await sweep_stale_checkpoint_threads(checkpointer, ttl_days=30)

        assert deleted == 3
        assert await _thread_ids(checkpointer) == set()


async def test_sweep_with_no_checkpoints_at_all_deletes_nothing(tmp_path: Path) -> None:
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.db")) as checkpointer:
        await checkpointer.setup()

        deleted = await sweep_stale_checkpoint_threads(checkpointer, ttl_days=30)

        assert deleted == 0


async def test_default_ttl_is_thirty_days() -> None:
    assert Settings().agent_checkpoint_ttl_days == 30
