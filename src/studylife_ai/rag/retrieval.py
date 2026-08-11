"""Retrieval: embed a query, fetch matching chunks from Qdrant, optionally rerank.

`retrieve_with_rerank()` is the sole entry point (see docs/decisions.md
"Retrieval quality: reranking + per-content-type quota"): it fetches an even
per-content-type candidate quota instead of one global top-k - without this,
a single popular course's ~90 near-duplicate session chunks can crowd a
genuinely relevant note out of the running entirely, confirmed live - then
optionally reranks the merged pool with an LLM when `Settings.rerank_model`
is set.
"""

import asyncio
import logging
from datetime import datetime
from typing import get_args

from studylife_ai.config import Settings
from studylife_ai.ingestion.qdrant_store import ContentType, QdrantStore, RetrievedChunk
from studylife_ai.llm.embeddings import embed_texts
from studylife_ai.rag.rerank import rerank_chunks

logger = logging.getLogger(__name__)

_CONTENT_TYPES: tuple[ContentType, ...] = get_args(ContentType)


async def _search_by_vector(
    vector: list[float],
    *,
    store: QdrantStore,
    user_id: str,
    top_k: int,
    content_type: ContentType | None,
) -> list[RetrievedChunk]:
    """Never raises - a failed Qdrant search degrades to no results for that
    content type instead of aborting the whole retrieval, matching the same
    "retrieval failure shouldn't break the caller" philosophy already applied
    to embedding and reranking failures below."""
    try:
        return await store.search(
            vector=vector, user_id=user_id, limit=top_k, content_type=content_type
        )
    except Exception:
        logger.exception("Qdrant search failed for content_type=%s", content_type)
        return []


async def retrieve_with_rerank(
    query: str,
    *,
    store: QdrantStore,
    settings: Settings,
    user_id: str,
    content_type: ContentType | None = None,
) -> list[RetrievedChunk]:
    """Retrieval entry point used by /chat, the eval pipeline, and the
    search_notes agent tool. `user_id` scopes every search to one user's
    Qdrant partition - resolved per-request from headers for /chat and the
    agent, fixed for eval (see docs/decisions.md "M4.5 Multi-user support").

    When `content_type` is given, fetches a single `rerank_candidate_k`-sized
    pool for that type (no cross-type crowding is possible when already
    scoped to one type). When it's `None` (the default, whole-corpus
    case), fetches an even quota from each content type separately and
    merges them - a type with many entities (e.g. sessions) can no longer
    squeeze out a type with few (e.g. notes) purely by outnumbering it in a
    shared top-k. The query is embedded once and reused across every
    per-type search.

    Reranking (LLM-scored reordering of the merged pool) is applied on top
    when `settings.rerank_model` is set; otherwise the merged pool is just
    sorted by vector-similarity score. Either way, the final list is cut
    down to `settings.retrieval_top_k`.
    """
    vectors = await embed_texts([query], model=settings.embedding_model, call_site="retrieval")
    if not vectors:
        return []
    vector = vectors[0]

    if content_type is not None:
        chunks = await _search_by_vector(
            vector,
            store=store,
            user_id=user_id,
            top_k=settings.rerank_candidate_k,
            content_type=content_type,
        )
    else:
        per_type_k = max(1, settings.rerank_candidate_k // len(_CONTENT_TYPES))
        results = await asyncio.gather(
            *(
                _search_by_vector(
                    vector,
                    store=store,
                    user_id=user_id,
                    top_k=per_type_k,
                    content_type=ct,
                )
                for ct in _CONTENT_TYPES
            )
        )
        chunks = [chunk for group in results for chunk in group]
        chunks.sort(key=lambda c: c.score, reverse=True)

    if settings.rerank_model and chunks:
        chunks = await rerank_chunks(
            query,
            chunks,
            model=settings.rerank_model,
            api_base=settings.llm_api_base,
            timeout=settings.llm_request_timeout_seconds,
            today=datetime.now().strftime("%Y-%m-%d, %A"),
        )
    return chunks[: settings.retrieval_top_k]
