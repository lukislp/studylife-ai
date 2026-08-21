from unittest.mock import AsyncMock

from pytest import MonkeyPatch

from studylife_ai.config import Settings
from studylife_ai.ingestion.qdrant_store import RetrievedChunk
from studylife_ai.rag import enrichment as enrichment_module
from studylife_ai.rag.enrichment import (
    CaptureEnrichment,
    _build_prompt,
    _embed_content,
    _find_related_notes,
    _generate_tags_and_summary,
    _ingest_note,
    _match_course,
    _parse_response,
    enrich_capture,
)


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "embedding_model": "ollama/nomic-embed-text",
        "llm_model": "ollama/llama3.2",
        "llm_api_base": "http://localhost:11434",
        "llm_request_timeout_seconds": 60.0,
        "llm_reasoning_effort": None,
        "rerank_model": None,
        "rerank_reasoning_effort": None,
        "capture_course_match_threshold": 0.75,
        "chunk_size_tokens": 500,
        "chunk_overlap_tokens": 75,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def _chunk(entity_id: int, score: float, content_type: str = "course") -> RetrievedChunk:
    return RetrievedChunk(
        content_type=content_type,  # type: ignore[arg-type]
        entity_id=entity_id,
        chunk_index=0,
        content="Some content",
        title="Some title",
        course_id=None,
        session_id=None,
        score=score,
        session_start=None,
    )


def test_build_prompt_includes_title_and_truncates_long_content() -> None:
    long_content = "x" * 3000

    prompt = _build_prompt("My note", long_content)

    assert "My note" in prompt
    assert "x" * 2000 in prompt
    assert "x" * 2001 not in prompt


def test_parse_response_extracts_tags_and_summary() -> None:
    tags, summary = _parse_response(
        "TAGS: linear algebra, eigenvalues, matrices\nSUMMARY: Notes on eigenvalue computation."
    )

    assert tags == ["linear algebra", "eigenvalues", "matrices"]
    assert summary == "Notes on eigenvalue computation."


def test_parse_response_caps_tags_at_five() -> None:
    tags, _ = _parse_response("TAGS: a, b, c, d, e, f, g")

    assert tags == ["a", "b", "c", "d", "e"]


def test_parse_response_degrades_to_empty_on_garbage() -> None:
    tags, summary = _parse_response("not the expected format at all")

    assert tags == []
    assert summary is None


def test_parse_response_is_case_insensitive_and_ignores_blank_lines() -> None:
    tags, summary = _parse_response("\ntags: a, b\n\nsummary: A short summary.\n")

    assert tags == ["a", "b"]
    assert summary == "A short summary."


async def test_embed_content_returns_first_vector(monkeypatch: MonkeyPatch) -> None:
    async def fake_embed_texts(texts: list[str], **_kwargs: object) -> list[list[float]]:
        assert texts == ["some content"]
        return [[0.1, 0.2]]

    monkeypatch.setattr(enrichment_module, "embed_texts", fake_embed_texts)

    vector = await _embed_content("some content", user_id="alice", settings=_settings())

    assert vector == [0.1, 0.2]


async def test_embed_content_degrades_to_none_on_failure(monkeypatch: MonkeyPatch) -> None:
    async def failing_embed_texts(texts: list[str], **_kwargs: object) -> list[list[float]]:
        raise RuntimeError("embedding service unreachable")

    monkeypatch.setattr(enrichment_module, "embed_texts", failing_embed_texts)

    vector = await _embed_content("content", user_id="alice", settings=_settings())

    assert vector is None


async def test_match_course_returns_best_match_above_threshold() -> None:
    store = AsyncMock()
    store.search.return_value = [_chunk(42, 0.9)]

    course_id, confidence = await _match_course(
        [0.1, 0.2], user_id="alice", settings=_settings(), store=store
    )

    assert course_id == 42
    assert confidence == 0.9
    assert store.search.await_args.kwargs["content_type"] == "course"


async def test_match_course_returns_none_below_threshold() -> None:
    store = AsyncMock()
    store.search.return_value = [_chunk(42, 0.5)]

    course_id, confidence = await _match_course(
        [0.1, 0.2],
        user_id="alice",
        settings=_settings(capture_course_match_threshold=0.75),
        store=store,
    )

    assert course_id is None
    assert confidence is None


async def test_match_course_returns_none_when_no_results() -> None:
    store = AsyncMock()
    store.search.return_value = []

    course_id, confidence = await _match_course(
        [0.1, 0.2], user_id="alice", settings=_settings(), store=store
    )

    assert course_id is None
    assert confidence is None


async def test_match_course_returns_none_without_a_vector() -> None:
    store = AsyncMock()

    course_id, confidence = await _match_course(
        None, user_id="alice", settings=_settings(), store=store
    )

    assert course_id is None
    assert confidence is None
    store.search.assert_not_awaited()


async def test_match_course_degrades_to_none_on_search_failure() -> None:
    store = AsyncMock()
    store.search.side_effect = RuntimeError("qdrant unreachable")

    course_id, confidence = await _match_course(
        [0.1, 0.2], user_id="alice", settings=_settings(), store=store
    )

    assert course_id is None
    assert confidence is None


async def test_find_related_notes_returns_distinct_entity_ids_in_score_order() -> None:
    store = AsyncMock()
    # entity_id=5 appears twice (two chunks of the same long note) - must be deduped.
    store.search.return_value = [
        _chunk(5, 0.95, content_type="note"),
        _chunk(5, 0.9, content_type="note"),
        _chunk(3, 0.8, content_type="note"),
        _chunk(9, 0.7, content_type="note"),
    ]

    related = await _find_related_notes([0.1, 0.2], user_id="alice", note_id=99, store=store)

    assert related == [5, 3, 9]
    assert store.search.await_args.kwargs["content_type"] == "note"


async def test_find_related_notes_excludes_the_note_being_enriched() -> None:
    store = AsyncMock()
    store.search.return_value = [
        _chunk(99, 0.95, content_type="note"),
        _chunk(3, 0.8, content_type="note"),
    ]

    related = await _find_related_notes([0.1, 0.2], user_id="alice", note_id=99, store=store)

    assert related == [3]


async def test_find_related_notes_caps_at_three() -> None:
    store = AsyncMock()
    store.search.return_value = [_chunk(i, 1.0 - i / 10, content_type="note") for i in range(1, 8)]

    related = await _find_related_notes([0.1, 0.2], user_id="alice", note_id=99, store=store)

    assert related == [1, 2, 3]


async def test_find_related_notes_returns_empty_without_a_vector() -> None:
    store = AsyncMock()

    related = await _find_related_notes(None, user_id="alice", note_id=99, store=store)

    assert related == []
    store.search.assert_not_awaited()


async def test_find_related_notes_degrades_to_empty_on_search_failure() -> None:
    store = AsyncMock()
    store.search.side_effect = RuntimeError("qdrant unreachable")

    related = await _find_related_notes([0.1, 0.2], user_id="alice", note_id=99, store=store)

    assert related == []


async def test_generate_tags_and_summary_uses_rerank_model_when_configured(
    monkeypatch: MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_complete_chat(messages: list[object], **kwargs: object) -> str:
        captured.update(kwargs)
        return "TAGS: a, b\nSUMMARY: A summary."

    monkeypatch.setattr(enrichment_module, "complete_chat", fake_complete_chat)

    tags, summary = await _generate_tags_and_summary(
        "Title",
        "Content",
        user_id="alice",
        settings=_settings(rerank_model="openai/gpt-4o-mini"),
    )

    assert tags == ["a", "b"]
    assert summary == "A summary."
    assert captured["model"] == "openai/gpt-4o-mini"
    assert captured["temperature"] == 0.0


async def test_generate_tags_and_summary_falls_back_to_llm_model(
    monkeypatch: MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_complete_chat(messages: list[object], **kwargs: object) -> str:
        captured.update(kwargs)
        return "TAGS: a\nSUMMARY: Fine."

    monkeypatch.setattr(enrichment_module, "complete_chat", fake_complete_chat)

    await _generate_tags_and_summary(
        "Title", "Content", user_id="alice", settings=_settings(rerank_model=None)
    )

    assert captured["model"] == "ollama/llama3.2"


async def test_generate_tags_and_summary_omits_temperature_for_reasoning_models(
    monkeypatch: MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_complete_chat(messages: list[object], **kwargs: object) -> str:
        captured.update(kwargs)
        return "TAGS: a\nSUMMARY: Fine."

    monkeypatch.setattr(enrichment_module, "complete_chat", fake_complete_chat)

    await _generate_tags_and_summary(
        "Title",
        "Content",
        user_id="alice",
        settings=_settings(rerank_model="openai/gpt-5-mini", rerank_reasoning_effort="minimal"),
    )

    assert captured["temperature"] is None
    assert captured["reasoning_effort"] == "minimal"


async def test_generate_tags_and_summary_degrades_to_empty_on_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    async def failing_complete_chat(*args: object, **kwargs: object) -> str:
        raise RuntimeError("llm unreachable")

    monkeypatch.setattr(enrichment_module, "complete_chat", failing_complete_chat)

    tags, summary = await _generate_tags_and_summary(
        "Title", "Content", user_id="alice", settings=_settings()
    )

    assert tags == []
    assert summary is None


async def test_ingest_note_chunks_embeds_and_replaces_entity(monkeypatch: MonkeyPatch) -> None:
    async def fake_embed_texts(texts: list[str], **_kwargs: object) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(enrichment_module, "embed_texts", fake_embed_texts)
    store = AsyncMock()

    await _ingest_note(
        99,
        "Title",
        "Some captured content",
        user_id="alice",
        settings=_settings(),
        store=store,
        course_id=7,
    )

    store.ensure_collection.assert_awaited_once()
    store.replace_entity.assert_awaited_once()
    metadata = store.replace_entity.await_args.kwargs["metadata"]
    assert metadata.content_type == "note"
    assert metadata.entity_id == 99
    assert metadata.title == "Title"
    assert metadata.course_id == 7
    assert metadata.user_id == "alice"
    assert metadata.fingerprint  # non-empty hash


async def test_ingest_note_never_raises_on_failure(monkeypatch: MonkeyPatch) -> None:
    async def failing_embed_texts(texts: list[str], **_kwargs: object) -> list[list[float]]:
        raise RuntimeError("embedding service unreachable")

    monkeypatch.setattr(enrichment_module, "embed_texts", failing_embed_texts)
    store = AsyncMock()

    # Must not raise.
    await _ingest_note(
        99, "Title", "Content", user_id="alice", settings=_settings(), store=store, course_id=None
    )

    store.replace_entity.assert_not_awaited()


async def test_enrich_capture_combines_all_sub_steps(monkeypatch: MonkeyPatch) -> None:
    async def fake_embed_content(content: str, **_kwargs: object) -> list[float] | None:
        return [0.1, 0.2]

    async def fake_match_course(
        vector: object, **_kwargs: object
    ) -> tuple[int | None, float | None]:
        return 7, 0.88

    async def fake_find_related(vector: object, **_kwargs: object) -> list[int]:
        return [3, 5]

    async def fake_generate(
        title: str, content: str, **_kwargs: object
    ) -> tuple[list[str], str | None]:
        return ["tag"], "A summary."

    ingest_calls = []

    async def fake_ingest(note_id: int, title: str, content: str, **kwargs: object) -> None:
        ingest_calls.append({"note_id": note_id, "title": title, "content": content, **kwargs})

    monkeypatch.setattr(enrichment_module, "_embed_content", fake_embed_content)
    monkeypatch.setattr(enrichment_module, "_match_course", fake_match_course)
    monkeypatch.setattr(enrichment_module, "_find_related_notes", fake_find_related)
    monkeypatch.setattr(enrichment_module, "_generate_tags_and_summary", fake_generate)
    monkeypatch.setattr(enrichment_module, "_ingest_note", fake_ingest)

    result = await enrich_capture(
        99, "Title", "Content", user_id="alice", settings=_settings(), store=AsyncMock()
    )

    assert result == CaptureEnrichment(
        course_id=7,
        course_confidence=0.88,
        tags=["tag"],
        summary="A summary.",
        related_note_ids=[3, 5],
    )
    # Ingestion runs with the resolved course_id, not None - so the ingested note's own Qdrant
    # payload carries the matched course from the start.
    assert len(ingest_calls) == 1
    assert ingest_calls[0]["note_id"] == 99
    assert ingest_calls[0]["course_id"] == 7
