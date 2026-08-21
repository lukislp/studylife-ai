from types import SimpleNamespace
from unittest.mock import AsyncMock

from studylife_ai.ingestion.qdrant_store import EntityChunkMetadata, QdrantStore, RetrievedChunk


def _make_store() -> QdrantStore:
    return QdrantStore(url="http://qdrant.test:6333", collection="studylife_notes")


async def test_ensure_collection_creates_when_missing() -> None:
    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=False)
    store._client.create_collection = AsyncMock()
    store._client.create_payload_index = AsyncMock()

    await store.ensure_collection(vector_size=768)

    store._client.create_collection.assert_awaited_once()
    _, kwargs = store._client.create_collection.call_args
    assert kwargs["collection_name"] == "studylife_notes"
    assert kwargs["vectors_config"].size == 768


async def test_ensure_collection_skips_when_already_exists() -> None:
    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=True)
    store._client.create_collection = AsyncMock()
    store._client.create_payload_index = AsyncMock()

    await store.ensure_collection(vector_size=768)

    store._client.create_collection.assert_not_awaited()


async def test_ensure_collection_always_ensures_the_session_start_index() -> None:
    """Idempotent by design (see qdrant_store.py) - runs every call, not just on first
    creation, so an already-existing collection (every real deployment predating this field)
    also gets indexed the first time this runs after upgrading."""
    from studylife_ai.ingestion.qdrant_store import SESSION_START_FIELD

    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=True)
    store._client.create_collection = AsyncMock()
    store._client.create_payload_index = AsyncMock()

    await store.ensure_collection(vector_size=768)

    store._client.create_payload_index.assert_awaited_once()
    _, kwargs = store._client.create_payload_index.call_args
    assert kwargs["collection_name"] == "studylife_notes"
    assert kwargs["field_name"] == SESSION_START_FIELD


async def test_get_known_fingerprints_returns_empty_when_collection_missing() -> None:
    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=False)

    assert await store.get_known_fingerprints(user_id="primary") == {}


async def test_get_known_fingerprints_paginates_and_dedupes_by_content_type_and_entity_id() -> None:
    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=True)

    page1 = (
        [
            SimpleNamespace(payload={"content_type": "note", "entity_id": 1, "fingerprint": "a"}),
            SimpleNamespace(payload={"content_type": "note", "entity_id": 1, "fingerprint": "a"}),
        ],
        "next-offset",
    )
    page2 = (
        [SimpleNamespace(payload={"content_type": "course", "entity_id": 1, "fingerprint": "b"})],
        None,
    )
    store._client.scroll = AsyncMock(side_effect=[page1, page2])

    result = await store.get_known_fingerprints(user_id="primary")

    assert result == {("note", 1): "a", ("course", 1): "b"}
    assert store._client.scroll.await_count == 2
    _, kwargs = store._client.scroll.call_args
    condition = kwargs["scroll_filter"].must[0]
    assert condition.key == "user_id"
    assert condition.match.value == "primary"


async def test_get_all_chunks_returns_empty_when_collection_missing() -> None:
    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=False)

    assert await store.get_all_chunks(user_id="primary", content_type="session") == []


async def test_get_all_chunks_paginates_filters_by_type_and_scores_zero() -> None:
    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=True)

    def _point(entity_id: int, title: str) -> SimpleNamespace:
        return SimpleNamespace(
            payload={
                "content_type": "session",
                "entity_id": entity_id,
                "chunk_index": 0,
                "content": f"content-{entity_id}",
                "title": title,
                "course_id": None,
                "session_id": None,
            }
        )

    page1 = ([_point(1, "Session A")], "next-offset")
    page2 = ([_point(2, "Session B")], None)
    store._client.scroll = AsyncMock(side_effect=[page1, page2])

    result = await store.get_all_chunks(user_id="primary", content_type="session")

    assert [c.title for c in result] == ["Session A", "Session B"]
    assert all(c.score == 0.0 for c in result)
    assert store._client.scroll.await_count == 2
    _, kwargs = store._client.scroll.call_args
    conditions = kwargs["scroll_filter"].must
    assert conditions[0].key == "user_id"
    assert conditions[0].match.value == "primary"
    assert conditions[1].key == "content_type"
    assert conditions[1].match.value == "session"


async def test_get_all_chunks_reads_session_start_when_present() -> None:
    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=True)
    point = SimpleNamespace(
        payload={
            "content_type": "session",
            "entity_id": 1,
            "chunk_index": 0,
            "content": "content-1",
            "title": "Session A",
            "course_id": None,
            "session_id": None,
            "session_start": "2026-08-13T18:00:00",
        }
    )
    store._client.scroll = AsyncMock(return_value=([point], None))

    result = await store.get_all_chunks(user_id="primary", content_type="session")

    assert result[0].session_start == "2026-08-13T18:00:00"


async def test_get_sessions_in_window_returns_empty_when_collection_missing() -> None:
    from datetime import datetime

    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=False)

    result = await store.get_sessions_in_window(
        user_id="primary", start=datetime(2026, 8, 1), end=datetime(2026, 8, 28)
    )

    assert result == []


async def test_get_sessions_in_window_filters_by_user_type_and_datetime_range() -> None:
    from datetime import datetime

    from studylife_ai.ingestion.qdrant_store import SESSION_START_FIELD

    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=True)
    point = SimpleNamespace(
        payload={
            "content_type": "session",
            "entity_id": 1,
            "chunk_index": 0,
            "content": "content-1",
            "title": "Session in window",
            "course_id": None,
            "session_id": None,
            "session_start": "2026-08-12T16:00:00",
        }
    )
    store._client.scroll = AsyncMock(return_value=([point], None))

    start = datetime(2026, 7, 28)
    end = datetime(2026, 8, 25)
    result = await store.get_sessions_in_window(user_id="primary", start=start, end=end)

    assert [c.title for c in result] == ["Session in window"]
    _, kwargs = store._client.scroll.call_args
    conditions = kwargs["scroll_filter"].must
    assert conditions[0].key == "user_id"
    assert conditions[0].match.value == "primary"
    assert conditions[1].key == "content_type"
    assert conditions[1].match.value == "session"
    assert conditions[2].key == SESSION_START_FIELD
    assert conditions[2].range.gte == start
    assert conditions[2].range.lte == end


async def test_get_sessions_in_window_stops_at_safety_cap() -> None:
    from datetime import datetime

    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=True)

    def _page(offset: str | None) -> tuple[list[SimpleNamespace], str | None]:
        points = [
            SimpleNamespace(
                payload={
                    "content_type": "session",
                    "entity_id": i,
                    "chunk_index": 0,
                    "content": "x",
                    "title": f"s{i}",
                    "course_id": None,
                    "session_id": None,
                    "session_start": "2026-08-12T16:00:00",
                }
            )
            for i in range(256)
        ]
        return points, "more"

    store._client.scroll = AsyncMock(side_effect=lambda **kwargs: _page(kwargs.get("offset")))

    result = await store.get_sessions_in_window(
        user_id="primary",
        start=datetime(2026, 7, 28),
        end=datetime(2026, 8, 25),
        safety_cap=300,
    )

    assert len(result) == 300


async def test_get_all_chunks_stops_at_safety_cap() -> None:
    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=True)

    def _page(offset: str | None) -> tuple[list[SimpleNamespace], str | None]:
        points = [
            SimpleNamespace(
                payload={
                    "content_type": "session",
                    "entity_id": i,
                    "chunk_index": 0,
                    "content": "x",
                    "title": f"s{i}",
                    "course_id": None,
                    "session_id": None,
                }
            )
            for i in range(256)
        ]
        return points, "more"

    store._client.scroll = AsyncMock(side_effect=lambda **kwargs: _page(kwargs.get("offset")))

    result = await store.get_all_chunks(user_id="primary", content_type="session", safety_cap=300)

    assert len(result) == 300


async def test_replace_entity_deletes_existing_then_upserts_new_chunks() -> None:
    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=True)
    store._client.delete = AsyncMock()
    store._client.upsert = AsyncMock()

    metadata = EntityChunkMetadata(
        content_type="session",
        entity_id=7,
        title="Linear Algebra",
        course_id=3,
        session_id=None,
        user_id="primary",
        fingerprint="hash123",
        session_start="2026-08-13T18:00:00",
    )

    await store.replace_entity(
        chunks=["chunk one", "chunk two"], vectors=[[0.1], [0.2]], metadata=metadata
    )

    store._client.delete.assert_awaited_once()
    store._client.upsert.assert_awaited_once()
    _, kwargs = store._client.upsert.call_args
    points = kwargs["points"]
    assert len(points) == 2
    assert [p.payload["chunk_index"] for p in points] == [0, 1]
    assert all(p.payload["content_type"] == "session" for p in points)
    assert all(p.payload["entity_id"] == 7 for p in points)
    assert all(p.payload["fingerprint"] == "hash123" for p in points)
    assert all(p.payload["session_start"] == "2026-08-13T18:00:00" for p in points)


async def test_replace_entity_with_no_chunks_only_deletes() -> None:
    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=True)
    store._client.delete = AsyncMock()
    store._client.upsert = AsyncMock()

    metadata = EntityChunkMetadata(
        content_type="note",
        entity_id=7,
        title="Empty note",
        course_id=None,
        session_id=None,
        user_id="primary",
        fingerprint="hash000",
        session_start=None,
    )

    await store.replace_entity(chunks=[], vectors=[], metadata=metadata)

    store._client.delete.assert_awaited_once()
    store._client.upsert.assert_not_awaited()


async def test_delete_entity_filters_by_user_id_content_type_and_entity_id() -> None:
    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=True)
    store._client.delete = AsyncMock()

    await store.delete_entity(user_id="primary", content_type="course", entity_id=42)

    store._client.delete.assert_awaited_once()
    _, kwargs = store._client.delete.call_args
    conditions = kwargs["points_selector"].filter.must
    assert {c.key for c in conditions} == {"user_id", "content_type", "entity_id"}
    assert {c.match.value for c in conditions} == {"primary", "course", 42}


async def test_delete_entity_is_noop_when_collection_missing() -> None:
    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=False)
    store._client.delete = AsyncMock()

    await store.delete_entity(user_id="primary", content_type="note", entity_id=42)

    store._client.delete.assert_not_awaited()


async def test_search_returns_empty_when_collection_missing() -> None:
    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=False)

    result = await store.search(vector=[0.1, 0.2], user_id="primary", limit=5)

    assert result == []


async def test_search_filters_by_user_id_and_maps_results() -> None:
    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=True)
    fake_point = SimpleNamespace(
        score=0.87,
        payload={
            "content_type": "note",
            "entity_id": 7,
            "chunk_index": 1,
            "content": "Eigenvalues are important.",
            "title": "Linear Algebra",
            "course_id": 3,
            "session_id": None,
            "user_id": "primary",
            "fingerprint": "hash123",
            "session_start": None,
        },
    )
    store._client.query_points = AsyncMock(return_value=SimpleNamespace(points=[fake_point]))

    result = await store.search(vector=[0.1, 0.2], user_id="primary", limit=5)

    assert result == [
        RetrievedChunk(
            content_type="note",
            entity_id=7,
            chunk_index=1,
            content="Eigenvalues are important.",
            title="Linear Algebra",
            course_id=3,
            session_id=None,
            score=0.87,
            session_start=None,
        )
    ]
    _, kwargs = store._client.query_points.call_args
    assert kwargs["query"] == [0.1, 0.2]
    assert kwargs["limit"] == 5
    condition = kwargs["query_filter"].must[0]
    assert condition.key == "user_id"
    assert condition.match.value == "primary"


async def test_search_with_course_ids_adds_a_course_id_match_any_filter() -> None:
    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=True)
    store._client.query_points = AsyncMock(return_value=SimpleNamespace(points=[]))

    await store.search(
        vector=[0.1, 0.2], user_id="primary", limit=5, content_type="note", course_ids=[1, 2, 3]
    )

    _, kwargs = store._client.query_points.call_args
    conditions = kwargs["query_filter"].must
    course_condition = next(c for c in conditions if c.key == "course_id")
    assert course_condition.match.any == [1, 2, 3]


async def test_search_with_entity_ids_adds_an_entity_id_match_any_filter() -> None:
    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=True)
    store._client.query_points = AsyncMock(return_value=SimpleNamespace(points=[]))

    await store.search(
        vector=[0.1, 0.2], user_id="primary", limit=1, content_type="course", entity_ids=[10, 20]
    )

    _, kwargs = store._client.query_points.call_args
    conditions = kwargs["query_filter"].must
    entity_condition = next(c for c in conditions if c.key == "entity_id")
    assert entity_condition.match.any == [10, 20]


async def test_search_without_course_ids_or_entity_ids_omits_those_filters() -> None:
    store = _make_store()
    store._client.collection_exists = AsyncMock(return_value=True)
    store._client.query_points = AsyncMock(return_value=SimpleNamespace(points=[]))

    await store.search(vector=[0.1, 0.2], user_id="primary", limit=5)

    _, kwargs = store._client.query_points.call_args
    keys = [c.key for c in kwargs["query_filter"].must]
    assert "course_id" not in keys
    assert "entity_id" not in keys
