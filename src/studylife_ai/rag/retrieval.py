"""Retrieval v1: embed a query, fetch top-k matching chunks from Qdrant.

Scope per docs/decisions.md ("Retrieval design"): fixed top-k, pure dense
vector search, no reranking, filtered only by `user_id`. Decides *which*
chunks to retrieve — not how they're formatted into an LLM prompt, that's
the separate, not-yet-decided "Prompt-Design" area, so this module has no
/chat wiring yet.
"""

from studylife_ai.ingestion.qdrant_store import QdrantStore, RetrievedChunk
from studylife_ai.llm.embeddings import embed_texts


async def retrieve_chunks(
    query: str,
    *,
    store: QdrantStore,
    embedding_model: str,
    user_id: str,
    top_k: int,
) -> list[RetrievedChunk]:
    """Return up to `top_k` chunks belonging to `user_id`, ranked by similarity to `query`."""
    vectors = await embed_texts([query], model=embedding_model)
    if not vectors:
        return []
    return await store.search(vector=vectors[0], user_id=user_id, limit=top_k)
