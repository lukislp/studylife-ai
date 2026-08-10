"""Qdrant wrapper: collection lifecycle, note-scoped upsert/delete, known-state scroll.

Callers must call `ensure_collection()` before `replace_note()`/`delete_note()`.
`get_known_fingerprints()` is safe to call first (returns `{}` if the
collection doesn't exist yet) — that's the intended order for a sync run,
since the vector size needed by `ensure_collection()` is only known after
the first real embedding call.
"""

import uuid
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient, models


@dataclass
class NoteChunkMetadata:
    note_id: int
    title: str
    course_id: int | None
    session_id: int | None
    user_id: str
    fingerprint: str


@dataclass
class RetrievedChunk:
    note_id: int
    chunk_index: int
    content: str
    title: str
    course_id: int | None
    session_id: int | None
    score: float


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
        if await self.collection_exists():
            return
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    async def get_known_fingerprints(self) -> dict[int, str]:
        """Last known fingerprint per note_id, derived from stored chunks.

        All chunks of a note share the same fingerprint, so the last one
        seen per note_id while scrolling is sufficient.
        """
        if not await self.collection_exists():
            return {}

        known: dict[int, str] = {}
        offset = None
        while True:
            points, offset = await self._client.scroll(
                collection_name=self._collection,
                with_payload=["note_id", "fingerprint"],
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            for point in points:
                payload = point.payload or {}
                known[payload["note_id"]] = payload["fingerprint"]
            if offset is None:
                break
        return known

    async def replace_note(
        self,
        *,
        chunks: list[str],
        vectors: list[list[float]],
        metadata: NoteChunkMetadata,
    ) -> None:
        """Replace all chunks of a note: delete whatever exists, insert the given chunks."""
        await self.delete_note(metadata.note_id)
        if not chunks:
            return
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "note_id": metadata.note_id,
                    "chunk_index": index,
                    "content": chunk,
                    "title": metadata.title,
                    "course_id": metadata.course_id,
                    "session_id": metadata.session_id,
                    "user_id": metadata.user_id,
                    "fingerprint": metadata.fingerprint,
                },
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        await self._client.upsert(collection_name=self._collection, points=points)

    async def search(
        self, *, vector: list[float], user_id: str, limit: int
    ) -> list[RetrievedChunk]:
        """Vector search scoped to a single user (see docs/decisions.md "Retrieval design")."""
        if not await self.collection_exists():
            return []
        response = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=models.Filter(
                must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
            ),
            limit=limit,
            with_payload=True,
        )
        return [
            RetrievedChunk(
                note_id=point.payload["note_id"],
                chunk_index=point.payload["chunk_index"],
                content=point.payload["content"],
                title=point.payload["title"],
                course_id=point.payload["course_id"],
                session_id=point.payload["session_id"],
                score=point.score,
            )
            for point in response.points
            if point.payload is not None
        ]

    async def delete_note(self, note_id: int) -> None:
        """No-op if the collection doesn't exist yet — nothing to delete."""
        if not await self.collection_exists():
            return
        await self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="note_id", match=models.MatchValue(value=note_id))
                    ]
                )
            ),
        )

    async def close(self) -> None:
        await self._client.close()
