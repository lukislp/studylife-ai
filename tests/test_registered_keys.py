from pathlib import Path

from studylife_ai.studylife.registered_keys import RegisteredKeyStore


async def _store(tmp_path: Path) -> RegisteredKeyStore:
    store = RegisteredKeyStore(str(tmp_path / "registered_keys.db"))
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
    store1 = RegisteredKeyStore(db_path)
    await store1.setup()
    await store1.set("alice", "key-a")
    await store1.close()

    store2 = RegisteredKeyStore(db_path)
    await store2.setup()

    assert await store2.get("alice") == "key-a"

    await store2.close()


async def test_setup_is_idempotent_and_does_not_wipe_an_in_memory_store() -> None:
    """Regression test: an in-memory (":memory:") database is NOT shared
    across connections, so a naive setup() that reconnects unconditionally
    would silently discard all data on a second call - exactly what
    sync_all() does when a caller (e.g. a test) has already set up and
    populated a store it then hands to sync_all() via a monkeypatch."""
    store = RegisteredKeyStore(":memory:")
    await store.setup()
    await store.set("alice", "key-a")

    await store.setup()  # second call must be a no-op

    assert await store.get("alice") == "key-a"

    await store.close()
