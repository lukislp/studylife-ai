from types import SimpleNamespace
from unittest.mock import AsyncMock

from studylife_ai.ingestion.qdrant_store import NoteChunkMetadata, QdrantStore


def _make_store() -> QdrantStore:
    return QdrantStore(url="http://qdrant.test:6333", collection="studylife_notes")


async def test_ensure_collection_creates_when_missing() -> None:
    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=False)
    store._client.create_collection = AsyncMock()

    await store.ensure_collection(vector_size=768)

    store._client.create_collection.assert_awaited_once()
    _, kwargs = store._client.create_collection.call_args
    assert kwargs["collection_name"] == "studylife_notes"
    assert kwargs["vectors_config"].size == 768


async def test_ensure_collection_skips_when_already_exists() -> None:
    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=True)
    store._client.create_collection = AsyncMock()

    await store.ensure_collection(vector_size=768)

    store._client.create_collection.assert_not_awaited()


async def test_get_known_fingerprints_returns_empty_when_collection_missing() -> None:
    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=False)

    assert await store.get_known_fingerprints() == {}


async def test_get_known_fingerprints_paginates_and_dedupes_by_note_id() -> None:
    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=True)

    page1 = (
        [
            SimpleNamespace(payload={"note_id": 1, "fingerprint": "a"}),
            SimpleNamespace(payload={"note_id": 1, "fingerprint": "a"}),
        ],
        "next-offset",
    )
    page2 = ([SimpleNamespace(payload={"note_id": 2, "fingerprint": "b"})], None)
    store._client.scroll = AsyncMock(side_effect=[page1, page2])

    result = await store.get_known_fingerprints()

    assert result == {1: "a", 2: "b"}
    assert store._client.scroll.await_count == 2


async def test_replace_note_deletes_existing_then_upserts_new_chunks() -> None:
    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=True)
    store._client.delete = AsyncMock()
    store._client.upsert = AsyncMock()

    metadata = NoteChunkMetadata(
        note_id=7,
        title="Linear Algebra",
        course_id=3,
        session_id=None,
        user_id="primary",
        fingerprint="hash123",
    )

    await store.replace_note(
        chunks=["chunk one", "chunk two"], vectors=[[0.1], [0.2]], metadata=metadata
    )

    store._client.delete.assert_awaited_once()
    store._client.upsert.assert_awaited_once()
    _, kwargs = store._client.upsert.call_args
    points = kwargs["points"]
    assert len(points) == 2
    assert [p.payload["chunk_index"] for p in points] == [0, 1]
    assert all(p.payload["note_id"] == 7 for p in points)
    assert all(p.payload["fingerprint"] == "hash123" for p in points)


async def test_replace_note_with_no_chunks_only_deletes() -> None:
    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=True)
    store._client.delete = AsyncMock()
    store._client.upsert = AsyncMock()

    metadata = NoteChunkMetadata(
        note_id=7,
        title="Empty note",
        course_id=None,
        session_id=None,
        user_id="primary",
        fingerprint="hash000",
    )

    await store.replace_note(chunks=[], vectors=[], metadata=metadata)

    store._client.delete.assert_awaited_once()
    store._client.upsert.assert_not_awaited()


async def test_delete_note_filters_by_note_id() -> None:
    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=True)
    store._client.delete = AsyncMock()

    await store.delete_note(42)

    store._client.delete.assert_awaited_once()
    _, kwargs = store._client.delete.call_args
    condition = kwargs["points_selector"].filter.must[0]
    assert condition.key == "note_id"
    assert condition.match.value == 42


async def test_delete_note_is_noop_when_collection_missing() -> None:
    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=False)
    store._client.delete = AsyncMock()

    await store.delete_note(42)

    store._client.delete.assert_not_awaited()
