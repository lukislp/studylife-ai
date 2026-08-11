"""Persistent store of per-user StudyLife AiApiKeys, registered by StudyLife
itself (see docs/decisions.md "M4.5 Multi-user support").

StudyLife's backend calls POST /internal/register-key the moment a user
generates their AiApiKey (the plaintext exists only in that instant - see
StudyLife's SettingsController.GenerateAiApiKey) and POST /internal/revoke-key
when they revoke it. This store is what `/agent` looks up a real,
usable credential from (the signed proxy token only proves *who* is asking,
not a StudyLife-API-usable credential) and what `ingestion.sync_all()` reads
its list of accounts to sync from - no more manually-maintained user list.
"""

import aiosqlite


class RegisteredKeyStore:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._connection: aiosqlite.Connection | None = None

    async def setup(self) -> None:
        """Idempotent - a second call is a no-op. Matters beyond tidiness:
        for a `:memory:` path (used by tests), reconnecting via
        `aiosqlite.connect()` again would silently open a brand new, empty
        database - in-memory SQLite databases aren't shared across
        connections, unlike a real file path where reconnecting preserves
        data."""
        if self._connection is not None:
            return
        self._connection = await aiosqlite.connect(self._db_path)
        await self._connection.execute(
            "CREATE TABLE IF NOT EXISTS registered_keys ("
            "user_id TEXT PRIMARY KEY, "
            "ai_api_key TEXT NOT NULL, "
            "registered_at TEXT NOT NULL"
            ")"
        )
        await self._connection.commit()

    @property
    def _conn(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("RegisteredKeyStore.setup() must be called before use.")
        return self._connection

    async def get(self, user_id: str) -> str | None:
        async with self._conn.execute(
            "SELECT ai_api_key FROM registered_keys WHERE user_id = ?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return row[0] if row else None

    async def set(self, user_id: str, ai_api_key: str) -> None:
        await self._conn.execute(
            "INSERT INTO registered_keys (user_id, ai_api_key, registered_at) "
            "VALUES (?, ?, datetime('now')) "
            "ON CONFLICT(user_id) DO UPDATE SET "
            "ai_api_key = excluded.ai_api_key, registered_at = excluded.registered_at",
            (user_id, ai_api_key),
        )
        await self._conn.commit()

    async def delete(self, user_id: str) -> None:
        await self._conn.execute("DELETE FROM registered_keys WHERE user_id = ?", (user_id,))
        await self._conn.commit()

    async def list_user_ids(self) -> list[str]:
        async with self._conn.execute("SELECT user_id FROM registered_keys") as cursor:
            rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
