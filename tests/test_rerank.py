from pytest import MonkeyPatch

from studylife_ai.ingestion.qdrant_store import RetrievedChunk
from studylife_ai.rag import rerank as rerank_module
from studylife_ai.rag.rerank import _build_prompt, _parse_order, rerank_chunks


def _chunk(entity_id: int, title: str, content: str) -> RetrievedChunk:
    return RetrievedChunk(
        content_type="note",
        entity_id=entity_id,
        chunk_index=0,
        content=content,
        title=title,
        course_id=None,
        session_id=None,
        score=0.9,
        session_start=None,
    )


def test_build_prompt_includes_query_and_truncates_long_content() -> None:
    long_content = "x" * 500
    chunks = [_chunk(1, "Eigenwerte", long_content)]

    prompt = _build_prompt("Was sind Eigenwerte?", chunks, today="2026-08-11, Tuesday")

    assert "Was sind Eigenwerte?" in prompt
    assert "[0] (note) Eigenwerte:" in prompt
    assert "x" * 300 in prompt
    assert "x" * 301 not in prompt


def test_build_prompt_includes_todays_date_and_temporal_relevance_instruction() -> None:
    prompt = _build_prompt(
        "Was steht heute an?", [_chunk(1, "A", "...")], today="2026-08-11, Tuesday"
    )

    assert "2026-08-11, Tuesday" in prompt
    assert "NOT relevant" in prompt


def test_build_prompt_instructs_exact_offset_not_just_direction_or_proximity() -> None:
    """Regression test for the "übermorgen" bug (docs/decisions.md "Bug found live: reranker
    matched 'nearby' dates"): the prompt must tell the model to resolve an EXACT date for any
    relative expression, in any language - not just distinguish past from future, and not just
    give English examples."""
    prompt = _build_prompt(
        "Welche Session haben wir übermorgen?", [_chunk(1, "A", "...")], today="2026-08-11"
    )

    assert "EXACT" in prompt
    assert "übermorgen" in prompt
    assert "any language" in prompt


def test_parse_order_reorders_by_well_formed_response() -> None:
    assert _parse_order("2,0,1", num_candidates=3) == [2, 0, 1]


def test_parse_order_appends_omitted_indices_in_original_order() -> None:
    # model only mentioned index 2 - 0 and 1 must still appear, in order
    assert _parse_order("2", num_candidates=3) == [2, 0, 1]


def test_parse_order_ignores_invalid_and_out_of_range_tokens() -> None:
    assert _parse_order("2, banana, 99, 0", num_candidates=3) == [2, 0, 1]


def test_parse_order_falls_back_to_original_order_for_empty_response() -> None:
    assert _parse_order("", num_candidates=3) == [0, 1, 2]


def test_parse_order_dedupes_repeated_indices() -> None:
    assert _parse_order("1,1,1,0", num_candidates=3) == [1, 0, 2]


async def test_rerank_chunks_reorders_using_model_response(monkeypatch: MonkeyPatch) -> None:
    chunks = [_chunk(1, "A", "..."), _chunk(2, "B", "..."), _chunk(3, "C", "...")]

    async def fake_complete_chat(messages: list[object], **kwargs: object) -> str:
        return "2,0,1"

    monkeypatch.setattr(rerank_module, "complete_chat", fake_complete_chat)

    result = await rerank_chunks(
        "query", chunks, model="ollama/llama3.2", api_base=None, timeout=30.0, today="2026-08-11"
    )

    assert [c.title for c in result] == ["C", "A", "B"]


async def test_rerank_chunks_returns_unchanged_for_single_or_no_chunk(
    monkeypatch: MonkeyPatch,
) -> None:
    called = False

    async def fake_complete_chat(*args: object, **kwargs: object) -> str:
        nonlocal called
        called = True
        return "0"

    monkeypatch.setattr(rerank_module, "complete_chat", fake_complete_chat)

    assert (
        await rerank_chunks("q", [], model="m", api_base=None, timeout=30.0, today="2026-08-11")
        == []
    )
    one = [_chunk(1, "A", "...")]
    assert (
        await rerank_chunks("q", one, model="m", api_base=None, timeout=30.0, today="2026-08-11")
        == one
    )
    assert called is False


async def test_rerank_chunks_falls_back_to_original_order_on_llm_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    chunks = [_chunk(1, "A", "..."), _chunk(2, "B", "...")]

    async def failing_complete_chat(*args: object, **kwargs: object) -> str:
        raise RuntimeError("network error")

    monkeypatch.setattr(rerank_module, "complete_chat", failing_complete_chat)

    result = await rerank_chunks(
        "query", chunks, model="ollama/llama3.2", api_base=None, timeout=30.0, today="2026-08-11"
    )

    assert result == chunks
