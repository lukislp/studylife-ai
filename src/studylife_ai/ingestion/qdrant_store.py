"""Qdrant wrapper: collection lifecycle, entity-scoped upsert/delete, known-state scroll.

Callers must call `ensure_collection()` before `replace_entity()`/`delete_entity()`.
`get_known_fingerprints()` is safe to call first (returns `{}` if the
collection doesn't exist yet) — that's the intended order for a sync run,
since the vector size needed by `ensure_collection()` is only known after
the first real embedding call.

Notes, courses, sessions, and course goals all share one collection,
disambiguated by `content_type` (see docs/decisions.md "Ingestion scope
expansion"). `user_id` + `content_type` + `entity_id` together are an
entity's real identity — a course and a note can both have id=5 without
colliding, and so can two different users' entities (see docs/decisions.md
"M4.5 Multi-user support" — StudyLife ids are per-account, not global).
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from qdrant_client import AsyncQdrantClient, models

ContentType = Literal["note", "course", "session", "course_goal"]

# Payload field name for a session's start time (see docs/decisions.md
# "Structured session dates"). Indexed as DATETIME in ensure_collection() so
# get_sessions_in_window() can filter by a real Qdrant range query instead of
# relying on an LLM to read dates out of free text.
SESSION_START_FIELD = "session_start"


@dataclass
class EntityChunkMetadata:
    content_type: ContentType
    entity_id: int
    title: str
    course_id: int | None
    session_id: int | None
    user_id: str
    fingerprint: str
    # ISO 8601 (naive local time, matching StudyLife's own session timestamps
    # - see docs/decisions.md), set only for content_type="session". None for
    # every other content type.
    session_start: str | None


@dataclass
class RetrievedChunk:
    content_type: ContentType
    entity_id: int
    chunk_index: int
    content: str
    title: str
    course_id: int | None
    session_id: int | None
    score: float
    session_start: str | None


def _user_id_condition(user_id: str) -> models.FieldCondition:
    return models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))


def _chunk_from_payload(payload: dict[str, object], *, score: float) -> RetrievedChunk:
    """Shared field mapping for the three read paths below (get_all_chunks,
    get_sessions_in_window, search) - `score` is the one field each computes
    differently (0.0 for a plain scroll, a real similarity score for vector
    search)."""
    return RetrievedChunk(
        content_type=payload["content_type"],  # type: ignore[arg-type]
        entity_id=payload["entity_id"],  # type: ignore[arg-type]
        chunk_index=payload["chunk_index"],  # type: ignore[arg-type]
        content=payload["content"],  # type: ignore[arg-type]
        title=payload["title"],  # type: ignore[arg-type]
        course_id=payload["course_id"],  # type: ignore[arg-type]
        session_id=payload["session_id"],  # type: ignore[arg-type]
        score=score,
        # .get(), not [] - a point ingested before this field existed won't have it until its
        # next sync (see fingerprint_session()'s migration bump in ingestion/sync.py).
        session_start=payload.get(SESSION_START_FIELD),  # type: ignore[arg-type]
    )


class QdrantStore:
    def __init__(self, *, url: str, collection: str) -> None:
        # check_compatibility=False: skip the server-version handshake on
        # construction — it needlessly costs a round-trip (and warns loudly
        # if unreachable) every time a store is created, including in tests.
        self._client = AsyncQdrantClient(url=url, check_compatibility=False)
        self._collection = collection

    async def collection_exists(self) -> bool:
        return await self._client.collection_exists(self._collection)

    async def ensure_collection(self, vector_size: int) -> None:
        if not await self.collection_exists():
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )
        # Idempotent - safe to call on every sync, not just at collection creation, so an
        # already-existing collection (every real deployment, pre-dating this field) also gets
        # the index the first time ensure_collection() runs after upgrading.
        await self._client.create_payload_index(
            collection_name=self._collection,
            field_name=SESSION_START_FIELD,
            field_schema=models.PayloadSchemaType.DATETIME,
        )

    async def get_known_fingerprints(self, *, user_id: str) -> dict[tuple[str, int], str]:
        """Last known fingerprint per (content_type, entity_id), scoped to one user.

        All chunks of an entity share the same fingerprint, so the last one
        seen per (content_type, entity_id) while scrolling is sufficient.
        Scoped by `user_id` - without it, two users' entities that happen to
        share a (content_type, entity_id) pair (e.g. both have a note with
        id=5, since StudyLife ids are per-account) would collide, corrupting
        the diff for both.
        """
        if not await self.collection_exists():
            return {}

        known: dict[tuple[str, int], str] = {}
        offset = None
        while True:
            points, offset = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=models.Filter(must=[_user_id_condition(user_id)]),
                with_payload=["content_type", "entity_id", "fingerprint"],
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            for point in points:
                payload = point.payload or {}
                known[(payload["content_type"], payload["entity_id"])] = payload["fingerprint"]
            if offset is None:
                break
        return known

    async def get_all_chunks(
        self, *, user_id: str, content_type: ContentType, safety_cap: int = 1000
    ) -> list[RetrievedChunk]:
        """Every chunk of one content type for one user, unordered (no vector search - a plain
        scroll, same pagination pattern as `get_known_fingerprints()`).

        For content types where relevance genuinely depends on something a text embedding can't
        see - `session`'s "is this near today?" - the vector-similarity top-k in `search()` can
        starve out the correct answer entirely before an LLM reranker ever gets a chance to look
        at it (confirmed live: the same 5 textually-similar-but-months-old sessions always won
        the embedding-similarity race over the two sessions actually happening today). This
        trades that away for "the reranker sees everything, decides what's actually relevant" -
        correct by construction, at the cost of a larger rerank prompt. `safety_cap` exists only
        as a defensive bound against a pathological account history, not as an intended limit at
        personal scale (see docs/decisions.md "Retrieval design").
        """
        if not await self.collection_exists():
            return []
        chunks: list[RetrievedChunk] = []
        offset = None
        while len(chunks) < safety_cap:
            points, offset = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=models.Filter(
                    must=[
                        _user_id_condition(user_id),
                        models.FieldCondition(
                            key="content_type", match=models.MatchValue(value=content_type)
                        ),
                    ]
                ),
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            for point in points:
                payload = point.payload or {}
                if not payload:
                    continue
                # No similarity score for a scroll fetch - 0.0 sorts these last in the naive
                # score-only fallback ordering used when rerank_model is unset; harmless there
                # (that path never claimed date-awareness either) and irrelevant once reranking
                # is active, which is what this exists for.
                chunks.append(_chunk_from_payload(payload, score=0.0))
            if offset is None:
                break
        return chunks[:safety_cap]

    async def get_sessions_in_window(
        self, *, user_id: str, start: datetime, end: datetime, safety_cap: int = 1000
    ) -> list[RetrievedChunk]:
        """Every session chunk whose `session_start` falls within `[start, end]`, via a real
        Qdrant DatetimeRange filter on the indexed field - not text the reranker has to
        interpret (see docs/decisions.md "Structured session dates"). `retrieve_with_rerank()`
        calls this with a window centered on "today" for the common near-term case ("what's on
        tomorrow/this week"), merged with a plain topic-vector search over ALL sessions so
        farther-out topic questions ("what did we cover in Analysis last year") still work -
        this method alone only ever returns what's inside the window, by design.

        Points from before this field existed (`session_start` missing entirely) never match a
        DatetimeRange filter and are silently excluded here - they'll appear once their next
        sync backfills the field (see fingerprint_session()'s migration bump).
        """
        if not await self.collection_exists():
            return []
        chunks: list[RetrievedChunk] = []
        offset = None
        while len(chunks) < safety_cap:
            points, offset = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=models.Filter(
                    must=[
                        _user_id_condition(user_id),
                        models.FieldCondition(
                            key="content_type", match=models.MatchValue(value="session")
                        ),
                        models.FieldCondition(
                            key=SESSION_START_FIELD,
                            range=models.DatetimeRange(gte=start, lte=end),
                        ),
                    ]
                ),
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            for point in points:
                payload = point.payload or {}
                if not payload:
                    continue
                chunks.append(_chunk_from_payload(payload, score=0.0))
            if offset is None:
                break
        return chunks[:safety_cap]

    async def replace_entity(
        self,
        *,
        chunks: list[str],
        vectors: list[list[float]],
        metadata: EntityChunkMetadata,
    ) -> None:
        """Replace all chunks of an entity: delete whatever exists, insert the given chunks."""
        await self.delete_entity(
            user_id=metadata.user_id,
            content_type=metadata.content_type,
            entity_id=metadata.entity_id,
        )
        if not chunks:
            return
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "content_type": metadata.content_type,
                    "entity_id": metadata.entity_id,
                    "chunk_index": index,
                    "content": chunk,
                    "title": metadata.title,
                    "course_id": metadata.course_id,
                    "session_id": metadata.session_id,
                    "user_id": metadata.user_id,
                    "fingerprint": metadata.fingerprint,
                    SESSION_START_FIELD: metadata.session_start,
                },
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        await self._client.upsert(collection_name=self._collection, points=points)

    async def search(
        self,
        *,
        vector: list[float],
        user_id: str,
        limit: int,
        content_type: ContentType | None = None,
    ) -> list[RetrievedChunk]:
        """Vector search scoped to a single user (see docs/decisions.md "Retrieval design").

        `content_type` is an optional narrowing filter. `rag/retrieval.py`'s
        `retrieve_with_rerank()` is the only caller that matters in practice:
        it always passes a concrete `content_type` (see docs/decisions.md
        "Retrieval quality: reranking + per-content-type quota") - one call
        per content type when scoping the whole corpus, or a single call
        already scoped to one type (e.g. the M4 `search_notes` agent tool,
        which needs notes only, not courses/sessions mixed in).
        """
        if not await self.collection_exists():
            return []
        must: list[models.Condition] = [_user_id_condition(user_id)]
        if content_type is not None:
            must.append(
                models.FieldCondition(
                    key="content_type", match=models.MatchValue(value=content_type)
                )
            )
        response = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=models.Filter(must=must),
            limit=limit,
            with_payload=True,
        )
        return [
            _chunk_from_payload(point.payload, score=point.score)
            for point in response.points
            if point.payload is not None
        ]

    async def delete_entity(
        self, *, user_id: str, content_type: ContentType, entity_id: int
    ) -> None:
        """No-op if the collection doesn't exist yet — nothing to delete.

        Filters on `user_id`, `content_type`, AND `entity_id` - a course and
        a note can share a numeric id, and two different users' entities can
        too (StudyLife ids are per-account) - `content_type`+`entity_id`
        alone would risk deleting the wrong entity, or another user's.
        """
        if not await self.collection_exists():
            return
        await self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        _user_id_condition(user_id),
                        models.FieldCondition(
                            key="content_type", match=models.MatchValue(value=content_type)
                        ),
                        models.FieldCondition(
                            key="entity_id", match=models.MatchValue(value=entity_id)
                        ),
                    ]
                )
            ),
        )

    async def close(self) -> None:
        await self._client.close()
