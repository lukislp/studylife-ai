import sqlite3
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from studylife_ai.studylife.registered_keys import RegisteredKeyStore
from tests.conftest import TEST_AI_KEY_ENCRYPTION_KEY


async def _store(tmp_path: Path) -> RegisteredKeyStore:
    store = RegisteredKeyStore(str(tmp_path / "registered_keys.db"), TEST_AI_KEY_ENCRYPTION_KEY)
    await store.setup()
    return store


async def test_get_returns_none_for_unknown_user(tmp_path: Path) -> None:
    store = await _store(tmp_path)

    assert await store.get("alice") is None

    await store.close()


async def test_set_then_get_roundtrips(tmp_path: Path) -> None:
    store = await _store(tmp_path)

    await store.set("alice", "key-a")

    assert await store.get("alice") == "key-a"

    await store.close()


async def test_set_overwrites_an_existing_entry(tmp_path: Path) -> None:
    store = await _store(tmp_path)

    await store.set("alice", "key-a")
    await store.set("alice", "key-a-rotated")

    assert await store.get("alice") == "key-a-rotated"

    await store.close()


async def test_delete_removes_the_entry(tmp_path: Path) -> None:
    store = await _store(tmp_path)
    await store.set("alice", "key-a")

    await store.delete("alice")

    assert await store.get("alice") is None

    await store.close()


async def test_delete_is_a_noop_for_an_unknown_user(tmp_path: Path) -> None:
    store = await _store(tmp_path)

    await store.delete("does-not-exist")  # must not raise

    await store.close()


async def test_list_user_ids_returns_every_registered_user(tmp_path: Path) -> None:
    store = await _store(tmp_path)
    await store.set("alice", "key-a")
    await store.set("bob", "key-b")

    assert sorted(await store.list_user_ids()) == ["alice", "bob"]

    await store.close()


async def test_data_persists_across_reconnecting_to_the_same_file(tmp_path: Path) -> None:
    db_path = str(tmp_path / "registered_keys.db")
    store1 = RegisteredKeyStore(db_path, TEST_AI_KEY_ENCRYPTION_KEY)
    await store1.setup()
    await store1.set("alice", "key-a")
    await store1.close()

    store2 = RegisteredKeyStore(db_path, TEST_AI_KEY_ENCRYPTION_KEY)
    await store2.setup()

    assert await store2.get("alice") == "key-a"

    await store2.close()


async def test_setup_is_idempotent_and_does_not_wipe_an_in_memory_store() -> None:
    """Regression test: an in-memory (":memory:") database is NOT shared
    across connections, so a naive setup() that reconnects unconditionally
    would silently discard all data on a second call - exactly what
    sync_all() does when a caller (e.g. a test) has already set up and
    populated a store it then hands to sync_all() via a monkeypatch."""
    store = RegisteredKeyStore(":memory:", TEST_AI_KEY_ENCRYPTION_KEY)
    await store.setup()
    await store.set("alice", "key-a")

    await store.setup()  # second call must be a no-op

    assert await store.get("alice") == "key-a"

    await store.close()


# --- A4: encryption at rest ---


def test_construction_fails_without_an_encryption_key(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="AI_KEY_ENCRYPTION_KEY is not set"):
        RegisteredKeyStore(str(tmp_path / "registered_keys.db"), None)


def test_construction_fails_with_an_invalid_encryption_key(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not a valid Fernet key"):
        RegisteredKeyStore(str(tmp_path / "registered_keys.db"), "not-a-valid-fernet-key")


async def test_ai_api_key_is_stored_encrypted_not_plaintext(tmp_path: Path) -> None:
    db_path = tmp_path / "registered_keys.db"
    store = await _store(tmp_path)
    await store.set("alice", "key-a")
    await store.close()

    conn = sqlite3.connect(db_path)
    try:
        raw_value = conn.execute(
            "SELECT ai_api_key FROM registered_keys WHERE user_id = ?", ("alice",)
        ).fetchone()[0]
    finally:
        conn.close()

    assert raw_value != "key-a"
    # A valid Fernet token for the same key, round-tripping back to the real plaintext.
    assert Fernet(TEST_AI_KEY_ENCRYPTION_KEY.encode()).decrypt(raw_value.encode()) == b"key-a"


async def test_legacy_plaintext_rows_are_transparently_readable_and_migrated_on_setup(
    tmp_path: Path,
) -> None:
    """Regression test for the A4 in-place migration: a row written before Fernet encryption
    existed (plain INSERT via raw sqlite3, bypassing the store entirely - mirrors real
    pre-migration data) must (1) still be readable via get() without any manual step, and (2)
    end up re-encrypted on disk once a store has setup() against that file."""
    db_path = tmp_path / "registered_keys.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE registered_keys ("
            "user_id TEXT PRIMARY KEY, ai_api_key TEXT NOT NULL, registered_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO registered_keys (user_id, ai_api_key, registered_at) "
            "VALUES ('alice', 'legacy-plaintext-key', datetime('now'))"
        )
        conn.commit()
    finally:
        conn.close()

    store = RegisteredKeyStore(str(db_path), TEST_AI_KEY_ENCRYPTION_KEY)
    await store.setup()  # must transparently migrate the plaintext row

    assert await store.get("alice") == "legacy-plaintext-key"

    await store.close()

    conn = sqlite3.connect(db_path)
    try:
        raw_value = conn.execute(
            "SELECT ai_api_key FROM registered_keys WHERE user_id = ?", ("alice",)
        ).fetchone()[0]
    finally:
        conn.close()

    assert raw_value != "legacy-plaintext-key"
    assert (
        Fernet(TEST_AI_KEY_ENCRYPTION_KEY.encode()).decrypt(raw_value.encode())
        == b"legacy-plaintext-key"
    )


async def test_migration_does_not_touch_an_already_encrypted_row(tmp_path: Path) -> None:
    """Idempotency: a row already encrypted by a prior setup() must not be re-encrypted (and
    thus not rewritten - Fernet tokens embed a fresh nonce/timestamp per encrypt() call, so an
    unwanted re-encryption would change the stored ciphertext even though it still decrypts to
    the same plaintext)."""
    db_path = tmp_path / "registered_keys.db"
    store1 = RegisteredKeyStore(str(db_path), TEST_AI_KEY_ENCRYPTION_KEY)
    await store1.setup()
    await store1.set("alice", "key-a")
    await store1.close()

    conn = sqlite3.connect(db_path)
    try:
        raw_before = conn.execute(
            "SELECT ai_api_key FROM registered_keys WHERE user_id = ?", ("alice",)
        ).fetchone()[0]
    finally:
        conn.close()

    store2 = RegisteredKeyStore(str(db_path), TEST_AI_KEY_ENCRYPTION_KEY)
    await store2.setup()  # migration pass must be a no-op for an already-encrypted row
    await store2.close()

    conn = sqlite3.connect(db_path)
    try:
        raw_after = conn.execute(
            "SELECT ai_api_key FROM registered_keys WHERE user_id = ?", ("alice",)
        ).fetchone()[0]
    finally:
        conn.close()

    assert raw_after == raw_before
