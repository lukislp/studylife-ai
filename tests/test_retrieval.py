from unittest.mock import AsyncMock

from pytest import MonkeyPatch

from studylife_ai.ingestion.qdrant_store import QdrantStore, RetrievedChunk
from studylife_ai.rag import retrieval as retrieval_module
from studylife_ai.rag.retrieval import retrieve_chunks


async def test_retrieve_chunks_embeds_query_and_searches_by_user(monkeypatch: MonkeyPatch) -> None:
    store = QdrantStore(url="http://qdrant.test:6333", collection="studylife_notes")
    expected = [
        RetrievedChunk(
            content_type="note",
            entity_id=7,
            chunk_index=0,
            content="Eigenvalues are important.",
            title="Linear Algebra",
            course_id=3,
            session_id=None,
            score=0.9,
        )
    ]
    store.search = AsyncMock(return_value=expected)  # type: ignore[method-assign]

    embed_calls: list[list[str]] = []

    async def fake_embed_texts(texts: list[str], *, model: str) -> list[list[float]]:
        embed_calls.append(texts)
        assert model == "ollama/nomic-embed-text"
        return [[0.1, 0.2]]

    monkeypatch.setattr(retrieval_module, "embed_texts", fake_embed_texts)

    result = await retrieve_chunks(
        "What did I write about eigenvalues?",
        store=store,
        embedding_model="ollama/nomic-embed-text",
        user_id="primary",
        top_k=5,
    )

    assert result == expected
    assert embed_calls == [["What did I write about eigenvalues?"]]
    store.search.assert_awaited_once_with(vector=[0.1, 0.2], user_id="primary", limit=5)


async def test_retrieve_chunks_returns_empty_when_embedding_fails_to_produce_a_vector(
    monkeypatch: MonkeyPatch,
) -> None:
    store = QdrantStore(url="http://qdrant.test:6333", collection="studylife_notes")
    store.search = AsyncMock()  # type: ignore[method-assign]

    async def fake_embed_texts(texts: list[str], *, model: str) -> list[list[float]]:
        return []

    monkeypatch.setattr(retrieval_module, "embed_texts", fake_embed_texts)

    result = await retrieve_chunks(
        "irrelevant", store=store, embedding_model="x", user_id="primary", top_k=5
    )

    assert result == []
    store.search.assert_not_awaited()
