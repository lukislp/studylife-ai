from unittest.mock import AsyncMock

from pytest import MonkeyPatch

from studylife_ai.config import Settings
from studylife_ai.ingestion.qdrant_store import RetrievedChunk
from studylife_ai.rag import enrichment as enrichment_module
from studylife_ai.rag.enrichment import (
    CaptureEnrichment,
    _build_prompt,
    _generate_tags_and_summary,
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
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def _course_chunk(entity_id: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        content_type="course",
        entity_id=entity_id,
        chunk_index=0,
        content="Course: Linear Algebra",
        title="Linear Algebra",
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


async def test_match_course_returns_best_match_above_threshold(monkeypatch: MonkeyPatch) -> None:
    async def fake_embed_texts(texts: list[str], **_kwargs: object) -> list[list[float]]:
        return [[0.1, 0.2]]

    monkeypatch.setattr(enrichment_module, "embed_texts", fake_embed_texts)
    store = AsyncMock()
    store.search.return_value = [_course_chunk(42, 0.9)]

    course_id, confidence = await _match_course(
        "some note content", user_id="alice", settings=_settings(), store=store
    )

    assert course_id == 42
    assert confidence == 0.9
    assert store.search.await_args.kwargs["content_type"] == "course"


async def test_match_course_returns_none_below_threshold(monkeypatch: MonkeyPatch) -> None:
    async def fake_embed_texts(texts: list[str], **_kwargs: object) -> list[list[float]]:
        return [[0.1, 0.2]]

    monkeypatch.setattr(enrichment_module, "embed_texts", fake_embed_texts)
    store = AsyncMock()
    store.search.return_value = [_course_chunk(42, 0.5)]

    course_id, confidence = await _match_course(
        "some note content",
        user_id="alice",
        settings=_settings(capture_course_match_threshold=0.75),
        store=store,
    )

    assert course_id is None
    assert confidence is None


async def test_match_course_returns_none_when_no_results(monkeypatch: MonkeyPatch) -> None:
    async def fake_embed_texts(texts: list[str], **_kwargs: object) -> list[list[float]]:
        return [[0.1, 0.2]]

    monkeypatch.setattr(enrichment_module, "embed_texts", fake_embed_texts)
    store = AsyncMock()
    store.search.return_value = []

    course_id, confidence = await _match_course(
        "content", user_id="alice", settings=_settings(), store=store
    )

    assert course_id is None
    assert confidence is None


async def test_match_course_degrades_to_none_on_embedding_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    async def failing_embed_texts(texts: list[str], **_kwargs: object) -> list[list[float]]:
        raise RuntimeError("embedding service unreachable")

    monkeypatch.setattr(enrichment_module, "embed_texts", failing_embed_texts)
    store = AsyncMock()

    course_id, confidence = await _match_course(
        "content", user_id="alice", settings=_settings(), store=store
    )

    assert course_id is None
    assert confidence is None
    store.search.assert_not_awaited()


async def test_match_course_degrades_to_none_on_search_failure(monkeypatch: MonkeyPatch) -> None:
    async def fake_embed_texts(texts: list[str], **_kwargs: object) -> list[list[float]]:
        return [[0.1, 0.2]]

    monkeypatch.setattr(enrichment_module, "embed_texts", fake_embed_texts)
    store = AsyncMock()
    store.search.side_effect = RuntimeError("qdrant unreachable")

    course_id, confidence = await _match_course(
        "content", user_id="alice", settings=_settings(), store=store
    )

    assert course_id is None
    assert confidence is None


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


async def test_enrich_capture_combines_course_match_and_tags_summary(
    monkeypatch: MonkeyPatch,
) -> None:
    async def fake_match_course(
        content: str, **_kwargs: object
    ) -> tuple[int | None, float | None]:
        return 7, 0.88

    async def fake_generate(
        title: str, content: str, **_kwargs: object
    ) -> tuple[list[str], str | None]:
        return ["tag"], "A summary."

    monkeypatch.setattr(enrichment_module, "_match_course", fake_match_course)
    monkeypatch.setattr(enrichment_module, "_generate_tags_and_summary", fake_generate)

    result = await enrich_capture(
        "Title", "Content", user_id="alice", settings=_settings(), store=AsyncMock()
    )

    assert result == CaptureEnrichment(
        course_id=7, course_confidence=0.88, tags=["tag"], summary="A summary."
    )
