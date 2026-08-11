from unittest.mock import AsyncMock

from pytest import MonkeyPatch

from studylife_ai.config import Settings
from studylife_ai.ingestion.qdrant_store import QdrantStore, RetrievedChunk
from studylife_ai.rag import retrieval as retrieval_module
from studylife_ai.rag.retrieval import retrieve_with_rerank


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "embedding_model": "ollama/nomic-embed-text",
        "retrieval_top_k": 5,
        "rerank_candidate_k": 20,
        "rerank_model": None,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def _chunk_of_type(entity_id: int, title: str, content_type: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        content_type=content_type,  # type: ignore[arg-type]
        entity_id=entity_id,
        chunk_index=0,
        content="...",
        title=title,
        course_id=None,
        session_id=None,
        score=score,
    )


async def test_retrieve_with_rerank_fetches_an_even_quota_per_content_type_except_session(
    monkeypatch: MonkeyPatch,
) -> None:
    fetched_types: list[object] = []

    async def fake_search_by_vector(vector: list[float], **kwargs: object) -> list[RetrievedChunk]:
        fetched_types.append(kwargs["content_type"])
        assert kwargs["top_k"] == 5  # rerank_candidate_k=20 // 4 types
        return []

    async def fake_embed_texts(
        texts: list[str], *, model: str, **_kwargs: object
    ) -> list[list[float]]:
        return [[0.1, 0.2]]

    monkeypatch.setattr(retrieval_module, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(retrieval_module, "_search_by_vector", fake_search_by_vector)

    await retrieve_with_rerank(
        "query", store=AsyncMock(), settings=_settings(rerank_model=None), user_id="primary"
    )

    # session deliberately does NOT go through the per-type vector-similarity quota (see
    # retrieval.py's module docstring / QdrantStore.get_all_chunks) - covered separately below.
    assert set(fetched_types) == {"note", "course", "course_goal"}
    assert len(fetched_types) == 3


async def test_retrieve_with_rerank_fetches_every_session_via_get_all_chunks(
    monkeypatch: MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    async def fake_search_by_vector(vector: list[float], **kwargs: object) -> list[RetrievedChunk]:
        return []

    async def fake_get_all_chunks(**kwargs: object) -> list[RetrievedChunk]:
        calls.append(kwargs)
        return [_chunk_of_type(1, "today's session", "session", 0.0)]

    async def fake_embed_texts(
        texts: list[str], *, model: str, **_kwargs: object
    ) -> list[list[float]]:
        return [[0.1, 0.2]]

    monkeypatch.setattr(retrieval_module, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(retrieval_module, "_search_by_vector", fake_search_by_vector)
    store = AsyncMock()
    store.get_all_chunks = fake_get_all_chunks

    result = await retrieve_with_rerank(
        "query", store=store, settings=_settings(rerank_model=None), user_id="primary"
    )

    assert calls == [{"user_id": "primary", "content_type": "session"}]
    assert [c.title for c in result] == ["today's session"]


async def test_retrieve_with_rerank_without_model_sorts_merged_pool_by_score(
    monkeypatch: MonkeyPatch,
) -> None:
    per_type_results = {
        "note": [_chunk_of_type(1, "note-hi", "note", 0.5)],
        "course": [_chunk_of_type(2, "course-hi", "course", 0.9)],
        "course_goal": [_chunk_of_type(4, "goal-mid", "course_goal", 0.6)],
    }

    async def fake_search_by_vector(vector: list[float], **kwargs: object) -> list[RetrievedChunk]:
        return per_type_results[kwargs["content_type"]]

    async def fake_get_all_chunks(**kwargs: object) -> list[RetrievedChunk]:
        return [_chunk_of_type(3, "session-lo", "session", 0.1)]

    async def fake_embed_texts(
        texts: list[str], *, model: str, **_kwargs: object
    ) -> list[list[float]]:
        return [[0.1, 0.2]]

    async def fake_rerank_chunks(*args: object, **kwargs: object) -> list[RetrievedChunk]:
        raise AssertionError("rerank_chunks must not be called when rerank_model is unset")

    monkeypatch.setattr(retrieval_module, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(retrieval_module, "_search_by_vector", fake_search_by_vector)
    monkeypatch.setattr(retrieval_module, "rerank_chunks", fake_rerank_chunks)
    store = AsyncMock()
    store.get_all_chunks = fake_get_all_chunks

    result = await retrieve_with_rerank(
        "query",
        store=store,
        settings=_settings(rerank_model=None, retrieval_top_k=4),
        user_id="primary",
    )

    assert [c.title for c in result] == ["course-hi", "goal-mid", "note-hi", "session-lo"]


async def test_retrieve_with_rerank_reranks_merged_pool_when_model_set(
    monkeypatch: MonkeyPatch,
) -> None:
    candidates = [_chunk_of_type(i, f"chunk-{i}", "note", 0.5) for i in range(4)]

    async def fake_search_by_vector(vector: list[float], **kwargs: object) -> list[RetrievedChunk]:
        return [candidates.pop(0)] if candidates else []

    async def fake_get_all_chunks(**kwargs: object) -> list[RetrievedChunk]:
        return [candidates.pop(0)] if candidates else []

    async def fake_embed_texts(
        texts: list[str], *, model: str, **_kwargs: object
    ) -> list[list[float]]:
        return [[0.1, 0.2]]

    async def fake_rerank_chunks(
        query: str, chunks: list[RetrievedChunk], **kwargs: object
    ) -> list[RetrievedChunk]:
        return list(reversed(chunks))

    monkeypatch.setattr(retrieval_module, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(retrieval_module, "_search_by_vector", fake_search_by_vector)
    monkeypatch.setattr(retrieval_module, "rerank_chunks", fake_rerank_chunks)
    store = AsyncMock()
    store.get_all_chunks = fake_get_all_chunks

    result = await retrieve_with_rerank(
        "query",
        store=store,
        settings=_settings(rerank_model="ollama/llama3.2", retrieval_top_k=4),
        user_id="primary",
    )

    assert [c.title for c in result] == ["chunk-3", "chunk-2", "chunk-1", "chunk-0"]


async def test_retrieve_with_rerank_with_explicit_content_type_skips_per_type_split(
    monkeypatch: MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_search_by_vector(vector: list[float], **kwargs: object) -> list[RetrievedChunk]:
        captured.update(kwargs)
        return [_chunk_of_type(1, "note-1", "note", 0.9)]

    async def fake_embed_texts(
        texts: list[str], *, model: str, **_kwargs: object
    ) -> list[list[float]]:
        return [[0.1, 0.2]]

    monkeypatch.setattr(retrieval_module, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(retrieval_module, "_search_by_vector", fake_search_by_vector)

    result = await retrieve_with_rerank(
        "query",
        store=AsyncMock(),
        settings=_settings(rerank_model=None),
        user_id="primary",
        content_type="note",
    )

    assert captured["content_type"] == "note"
    assert captured["top_k"] == 20  # full rerank_candidate_k, not split across types
    assert [c.title for c in result] == ["note-1"]


async def test_retrieve_with_rerank_returns_empty_when_embedding_fails(
    monkeypatch: MonkeyPatch,
) -> None:
    async def fake_embed_texts(
        texts: list[str], *, model: str, **_kwargs: object
    ) -> list[list[float]]:
        return []

    monkeypatch.setattr(retrieval_module, "embed_texts", fake_embed_texts)

    result = await retrieve_with_rerank(
        "query", store=AsyncMock(), settings=_settings(), user_id="primary"
    )

    assert result == []


async def test_a_single_content_types_search_failure_does_not_abort_the_others(
    monkeypatch: MonkeyPatch,
) -> None:
    store = QdrantStore(url="http://qdrant.test:6333", collection="studylife_notes")

    async def fake_search(**kwargs: object) -> list[RetrievedChunk]:
        if kwargs["content_type"] == "session":
            raise RuntimeError("Qdrant timeout")
        return [
            _chunk_of_type(1, f"{kwargs['content_type']}-hit", str(kwargs["content_type"]), 0.5)
        ]

    store.search = AsyncMock(side_effect=fake_search)  # type: ignore[method-assign]

    async def fake_embed_texts(
        texts: list[str], *, model: str, **_kwargs: object
    ) -> list[list[float]]:
        return [[0.1, 0.2]]

    monkeypatch.setattr(retrieval_module, "embed_texts", fake_embed_texts)

    result = await retrieve_with_rerank(
        "query", store=store, settings=_settings(rerank_model=None), user_id="primary"
    )

    assert {c.content_type for c in result} == {"note", "course", "course_goal"}
