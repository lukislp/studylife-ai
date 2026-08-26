"""Persistent store of per-user StudyLife AiApiKeys, registered by StudyLife
itself (see docs/decisions.md "M4.5 Multi-user support").

StudyLife's backend calls POST /internal/register-key the moment a user
generates their AiApiKey (the plaintext exists only in that instant - see
StudyLife's SettingsController.GenerateAiApiKey) and POST /internal/revoke-key
when they revoke it. This store is what `/agent` looks up a real,
usable credential from (the signed proxy token only proves *who* is asking,
not a StudyLife-API-usable credential) and what `ingestion.sync_all()` reads
its list of accounts to sync from - no more manually-maintained user list.

ai_api_key is encrypted at rest with Fernet (audit finding A4, 2026-08-25) -
each row is a full, usable StudyLife account credential, unlike StudyLife's
own hash-only key storage, so plaintext SQLite storage was a real exposure.
See config.py's `ai_key_encryption_key` for the key itself and
studylife-mcp's oauth_store.py for the sibling project's identical pattern.
"""

import logging
import sqlite3

import aiosqlite
from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

_GENERATE_KEY_HINT = (
    'python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"'
)


def _build_fernet(encryption_key: str | None) -> Fernet:
    """Fails loudly and actionably (A4) rather than letting the service start with plaintext
    storage, or crash later on cryptography's own less specific ValueError. Called from
    RegisteredKeyStore.__init__, which every real entrypoint (main.py's app lifespan,
    ingestion.sync.sync_all()) constructs at startup - so this is effectively a startup check,
    without needing its own separate validation pass."""
    if not encryption_key:
        raise RuntimeError(
            "AI_KEY_ENCRYPTION_KEY is not set. Every registered AiApiKey is a full, usable "
            "StudyLife account credential and must be encrypted at rest (see config.py's "
            "ai_key_encryption_key). Generate one with:\n"
            f"  {_GENERATE_KEY_HINT}\n"
            "and set it as AI_KEY_ENCRYPTION_KEY (see .env.example)."
        )
    try:
        return Fernet(encryption_key.encode())
    except ValueError as exc:
        raise RuntimeError(
            "AI_KEY_ENCRYPTION_KEY is not a valid Fernet key (must be 32 url-safe "
            "base64-encoded bytes). Generate one with:\n"
            f"  {_GENERATE_KEY_HINT}"
        ) from exc


class RegisteredKeyStore:
    def __init__(self, db_path: str, encryption_key: str | None) -> None:
        self._db_path = db_path
        self._fernet = _build_fernet(encryption_key)
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
        conn = await aiosqlite.connect(self._db_path)
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS registered_keys ("
            "user_id TEXT PRIMARY KEY, "
            "ai_api_key TEXT NOT NULL, "
            "registered_at TEXT NOT NULL"
            ")"
        )
        await conn.commit()
        self._connection = conn
        await self._migrate_plaintext_keys()
        await self._ensure_consecutive_401_column()

    async def _ensure_consecutive_401_column(self) -> None:
        """Additive schema change (audit F5/F13, identity-contract-v1.md section 4): tracks how
        many sync_all() runs in a row got a 401 for this user, so a key that died without a
        revoke call (e.g. StudyLife regenerated it) can be detected and purged instead of
        401ing forever every ingestion_sync_interval_seconds. SQLite has no
        `ADD COLUMN IF NOT EXISTS`, so this uses the standard idiom instead: attempt the ALTER
        TABLE and ignore the "duplicate column" error on a repeat call - safe to run
        unconditionally on every setup()/service start, same convention as
        `_migrate_plaintext_keys` above."""
        try:
            await self._conn.execute(
                "ALTER TABLE registered_keys "
                "ADD COLUMN consecutive_401_count INTEGER NOT NULL DEFAULT 0"
            )
            await self._conn.commit()
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise

    async def _migrate_plaintext_keys(self) -> None:
        """One-time, in-place migration for A4: older rows stored ai_api_key as plaintext.
        For each row, a value that decrypts successfully is already encrypted (left alone); a
        value that fails to decrypt (InvalidToken) is treated as legacy plaintext and
        re-encrypted in place. Idempotent - a fully-migrated table does zero UPDATEs on a
        repeat call, so this is safe to run unconditionally on every setup()/service start."""
        async with self._conn.execute("SELECT user_id, ai_api_key FROM registered_keys") as cur:
            rows = list(await cur.fetchall())
        migrated = 0
        for user_id, stored_value in rows:
            try:
                self._fernet.decrypt(stored_value.encode())
            except InvalidToken:
                encrypted = self._fernet.encrypt(stored_value.encode()).decode()
                await self._conn.execute(
                    "UPDATE registered_keys SET ai_api_key = ? WHERE user_id = ?",
                    (encrypted, user_id),
                )
                migrated += 1
        if migrated:
            await self._conn.commit()
        logger.info(
            "RegisteredKeyStore migration: re-encrypted %d/%d legacy plaintext row(s)",
            migrated,
            len(rows),
        )

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
        if row is None:
            return None
        return self._fernet.decrypt(row[0].encode()).decode()

    async def set(self, user_id: str, ai_api_key: str) -> None:
        encrypted = self._fernet.encrypt(ai_api_key.encode()).decode()
        await self._conn.execute(
            "INSERT INTO registered_keys (user_id, ai_api_key, registered_at) "
            "VALUES (?, ?, datetime('now')) "
            "ON CONFLICT(user_id) DO UPDATE SET "
            "ai_api_key = excluded.ai_api_key, registered_at = excluded.registered_at",
            (user_id, encrypted),
        )
        await self._conn.commit()

    async def delete(self, user_id: str) -> None:
        await self._conn.execute("DELETE FROM registered_keys WHERE user_id = ?", (user_id,))
        await self._conn.commit()

    async def record_sync_success(self, user_id: str) -> None:
        """Resets the consecutive-401 counter (see `_ensure_consecutive_401_column`) - a
        successful sync means the registered key is still good."""
        await self._conn.execute(
            "UPDATE registered_keys SET consecutive_401_count = 0 WHERE user_id = ?", (user_id,)
        )
        await self._conn.commit()

    async def record_sync_401(self, user_id: str) -> int:
        """Increments the consecutive-401 counter and returns its new value - used by
        `ingestion.sync.sync_all()` to detect a zombie registration (see
        `_ensure_consecutive_401_column`). No-op returning 0 if the user was revoked in the
        meantime (row no longer exists)."""
        await self._conn.execute(
            "UPDATE registered_keys SET consecutive_401_count = consecutive_401_count + 1 "
            "WHERE user_id = ?",
            (user_id,),
        )
        await self._conn.commit()
        async with self._conn.execute(
            "SELECT consecutive_401_count FROM registered_keys WHERE user_id = ?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return row[0] if row else 0

    async def list_user_ids(self) -> list[str]:
        async with self._conn.execute("SELECT user_id FROM registered_keys") as cursor:
            rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
