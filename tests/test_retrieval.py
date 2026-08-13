from datetime import date, datetime
from unittest.mock import AsyncMock

from pytest import MonkeyPatch

from studylife_ai.config import Settings
from studylife_ai.ingestion.qdrant_store import QdrantStore, RetrievedChunk
from studylife_ai.rag import retrieval as retrieval_module
from studylife_ai.rag.date_parse import DateRange
from studylife_ai.rag.retrieval import _fetch_session_window, _fetch_sessions, retrieve_with_rerank


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "embedding_model": "ollama/nomic-embed-text",
        "retrieval_top_k": 5,
        "rerank_candidate_k": 20,
        "rerank_model": None,
        "session_window_days": 14,
        "session_window_top_k": 20,
        "date_parse_model": None,
        "date_range_chunk_cap": 60,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def _chunk_of_type(
    entity_id: int,
    title: str,
    content_type: str,
    score: float,
    session_start: str | None = None,
    exact_date_match: bool = False,
) -> RetrievedChunk:
    return RetrievedChunk(
        content_type=content_type,  # type: ignore[arg-type]
        entity_id=entity_id,
        chunk_index=0,
        content="...",
        title=title,
        course_id=None,
        session_id=None,
        score=score,
        session_start=session_start,
        exact_date_match=exact_date_match,
    )


async def test_retrieve_with_rerank_fetches_an_even_quota_per_content_type(
    monkeypatch: MonkeyPatch,
) -> None:
    fetched_types: list[object] = []

    async def fake_search_by_vector(vector: list[float], **kwargs: object) -> list[RetrievedChunk]:
        fetched_types.append(kwargs["content_type"])
        assert kwargs["top_k"] == 5  # rerank_candidate_k=20 // 4 types
        return []

    async def fake_get_sessions_in_window(**kwargs: object) -> list[RetrievedChunk]:
        return []

    async def fake_embed_texts(
        texts: list[str], *, model: str, **_kwargs: object
    ) -> list[list[float]]:
        return [[0.1, 0.2]]

    monkeypatch.setattr(retrieval_module, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(retrieval_module, "_search_by_vector", fake_search_by_vector)
    store = AsyncMock()
    store.get_sessions_in_window = fake_get_sessions_in_window

    await retrieve_with_rerank(
        "query", store=store, settings=_settings(rerank_model=None), user_id="primary"
    )

    # Every content type, including session's topic-fallback pool, goes through the same
    # per-type vector-similarity quota - session additionally gets a date-window fetch,
    # covered separately below (see docs/decisions.md "Structured session dates").
    assert set(fetched_types) == {"note", "course", "course_goal", "session"}
    assert len(fetched_types) == 4


async def test_fetch_sessions_merges_window_and_topic_pools_deduped_by_entity_id(
    monkeypatch: MonkeyPatch,
) -> None:
    window_calls: list[dict[str, object]] = []

    async def fake_search_by_vector(vector: list[float], **kwargs: object) -> list[RetrievedChunk]:
        # entity_id=1 also appears in the window pool below - must not be duplicated.
        return [
            _chunk_of_type(1, "window session (topic-matched too)", "session", 0.8),
            _chunk_of_type(2, "topic-only session", "session", 0.5),
        ]

    async def fake_get_sessions_in_window(**kwargs: object) -> list[RetrievedChunk]:
        window_calls.append(kwargs)
        return [_chunk_of_type(1, "window session", "session", 0.0)]

    monkeypatch.setattr(retrieval_module, "_search_by_vector", fake_search_by_vector)
    store = AsyncMock()
    store.get_sessions_in_window = fake_get_sessions_in_window

    result = await _fetch_sessions(
        "query",
        [0.1, 0.2],
        store=store,
        user_id="primary",
        settings=_settings(),
        today=date(2026, 8, 12),
        top_k=5,
    )

    # entity_id=1: window pool wins (listed first, its title survives) over the topic pool's
    # duplicate. entity_id=2 (topic-only) is still included.
    assert [(c.entity_id, c.title) for c in result] == [
        (1, "window session"),
        (2, "topic-only session"),
    ]
    assert len(window_calls) == 1
    assert window_calls[0]["user_id"] == "primary"
    start, end = window_calls[0]["start"], window_calls[0]["end"]
    assert isinstance(start, datetime) and isinstance(end, datetime)
    assert (end - start).days == 28  # 2 * session_window_days=14


async def test_fetch_session_window_caps_to_top_k_keeping_the_nearest_days(
    monkeypatch: MonkeyPatch,
) -> None:
    """Regression test for the 2026-08-12 bug: get_sessions_in_window() itself is unbounded
    (safety_cap=1000 only) and every chunk it returns has score=0.0, so an overflowing window
    used to let the reranker's candidate pool balloon and let far-window sessions crowd out
    near-term ones purely by position once merged with scored topic-vector hits. Fixed by
    capping to top_k here, sorted by proximity to `today` so an overflow drops the FARTHEST
    days, not the nearest ones "today/tomorrow"-style queries actually care about."""
    today = date(2026, 8, 12)
    # Deliberately unsorted and outnumbering top_k=2, spanning both directions from today.
    unsorted_chunks = [
        _chunk_of_type(1, "5 days out", "session", 0.0, session_start="2026-08-17"),
        _chunk_of_type(2, "today", "session", 0.0, session_start="2026-08-12"),
        _chunk_of_type(3, "3 days ago", "session", 0.0, session_start="2026-08-09"),
        _chunk_of_type(4, "tomorrow", "session", 0.0, session_start="2026-08-13"),
    ]

    async def fake_get_sessions_in_window(**kwargs: object) -> list[RetrievedChunk]:
        return list(unsorted_chunks)

    store = AsyncMock()
    store.get_sessions_in_window = fake_get_sessions_in_window

    result = await _fetch_session_window(
        store, user_id="primary", window_days=14, today=today, top_k=2
    )

    # Only the 2 nearest-to-today survive (today, tomorrow) - "3 days ago" and "5 days out" are
    # dropped, not kept, even though "3 days ago" arrived earlier in the unsorted input.
    assert [c.title for c in result] == ["today", "tomorrow"]


async def test_fetch_session_window_tags_chunks_only_when_an_exact_range_was_given(
    monkeypatch: MonkeyPatch,
) -> None:
    """RetrievedChunk.exact_date_match must be True only for a resolved, exact date_range - the
    fixed +-window_days fallback is a proximity heuristic, not an exact filter, so its chunks
    must NOT be exempted from retrieve_with_rerank()'s normal retrieval_top_k cut."""
    today = date(2026, 8, 12)

    async def fake_get_sessions_in_window(**kwargs: object) -> list[RetrievedChunk]:
        return [_chunk_of_type(1, "session", "session", 0.0, session_start="2026-08-12")]

    store = AsyncMock()
    store.get_sessions_in_window = fake_get_sessions_in_window

    fixed_window_result = await _fetch_session_window(
        store, user_id="primary", window_days=14, today=today, top_k=5
    )
    assert fixed_window_result[0].exact_date_match is False

    exact_range_result = await _fetch_session_window(
        store,
        user_id="primary",
        window_days=14,
        today=today,
        top_k=5,
        date_range=DateRange(date(2026, 8, 12), date(2026, 8, 12)),
    )
    assert exact_range_result[0].exact_date_match is True


async def test_fetch_sessions_uses_session_window_top_k_not_the_shared_per_type_quota(
    monkeypatch: MonkeyPatch,
) -> None:
    """Regression test: the window leg must use settings.session_window_top_k, not the smaller
    per-type `top_k` passed in for the topic-vector leg - otherwise a busy nearby day can still
    fill the shared quota and starve an equally-near day (see session_window_top_k's docstring)."""
    captured: dict[str, object] = {}

    async def fake_fetch_session_window(store: object, **kwargs: object) -> list[RetrievedChunk]:
        captured.update(kwargs)
        return []

    async def fake_search_by_vector(vector: list[float], **kwargs: object) -> list[RetrievedChunk]:
        return []

    monkeypatch.setattr(retrieval_module, "_fetch_session_window", fake_fetch_session_window)
    monkeypatch.setattr(retrieval_module, "_search_by_vector", fake_search_by_vector)
    store = AsyncMock()

    await _fetch_sessions(
        "query",
        [0.1, 0.2],
        store=store,
        user_id="primary",
        settings=_settings(session_window_top_k=20),
        today=date(2026, 8, 12),
        top_k=5,  # the shared per-type quota - must NOT be what the window leg receives
    )

    assert captured["top_k"] == 20


async def test_fetch_sessions_uses_parsed_date_range_instead_of_fixed_window(
    monkeypatch: MonkeyPatch,
) -> None:
    """When date_parse_model is set and parse_date_range() resolves a range, the window leg must
    use that exact range instead of the fixed +-session_window_days window - see docs/decisions.md
    "NL date-range resolution"."""
    window_calls: list[dict[str, object]] = []

    async def fake_get_sessions_in_window(**kwargs: object) -> list[RetrievedChunk]:
        window_calls.append(kwargs)
        return []

    async def fake_search_by_vector(vector: list[float], **kwargs: object) -> list[RetrievedChunk]:
        return []

    async def fake_parse_date_range(query: str, **kwargs: object) -> DateRange:
        return DateRange(date(2026, 8, 3), date(2026, 8, 9))

    monkeypatch.setattr(retrieval_module, "_search_by_vector", fake_search_by_vector)
    monkeypatch.setattr(retrieval_module, "parse_date_range", fake_parse_date_range)
    store = AsyncMock()
    store.get_sessions_in_window = fake_get_sessions_in_window

    await _fetch_sessions(
        "letzte Woche",
        [0.1, 0.2],
        store=store,
        user_id="primary",
        settings=_settings(date_parse_model="openai/gpt-4o"),
        today=date(2026, 8, 12),
        top_k=5,
    )

    assert len(window_calls) == 1
    assert window_calls[0]["start"] == datetime(2026, 8, 3, 0, 0, 0)
    assert window_calls[0]["end"] == datetime(2026, 8, 9, 23, 59, 59, 999999)


async def test_fetch_sessions_uses_date_range_chunk_cap_when_range_resolved(
    monkeypatch: MonkeyPatch,
) -> None:
    """Regression test for the 2026-08-12 "21 real sessions in a week, only 8 survived" bug: an
    exact resolved range must get date_range_chunk_cap's larger budget, not the smaller
    session_window_top_k meant for the fixed-window fallback's proximity guessing."""
    captured: dict[str, object] = {}

    async def fake_fetch_session_window(store: object, **kwargs: object) -> list[RetrievedChunk]:
        captured.update(kwargs)
        return []

    async def fake_search_by_vector(vector: list[float], **kwargs: object) -> list[RetrievedChunk]:
        return []

    async def fake_parse_date_range(query: str, **kwargs: object) -> DateRange:
        return DateRange(date(2026, 8, 3), date(2026, 8, 9))

    monkeypatch.setattr(retrieval_module, "_fetch_session_window", fake_fetch_session_window)
    monkeypatch.setattr(retrieval_module, "_search_by_vector", fake_search_by_vector)
    monkeypatch.setattr(retrieval_module, "parse_date_range", fake_parse_date_range)
    store = AsyncMock()

    await _fetch_sessions(
        "letzte Woche",
        [0.1, 0.2],
        store=store,
        user_id="primary",
        settings=_settings(
            date_parse_model="openai/gpt-4o", session_window_top_k=20, date_range_chunk_cap=60
        ),
        today=date(2026, 8, 12),
        top_k=5,
    )

    assert captured["top_k"] == 60


async def test_fetch_sessions_skips_date_parsing_when_date_parse_model_unset(
    monkeypatch: MonkeyPatch,
) -> None:
    async def fake_get_sessions_in_window(**kwargs: object) -> list[RetrievedChunk]:
        return []

    async def fake_search_by_vector(vector: list[float], **kwargs: object) -> list[RetrievedChunk]:
        return []

    async def fake_parse_date_range(query: str, **kwargs: object) -> DateRange:
        raise AssertionError("parse_date_range must not be called when date_parse_model is unset")

    monkeypatch.setattr(retrieval_module, "_search_by_vector", fake_search_by_vector)
    monkeypatch.setattr(retrieval_module, "parse_date_range", fake_parse_date_range)
    store = AsyncMock()
    store.get_sessions_in_window = fake_get_sessions_in_window

    await _fetch_sessions(
        "letzte Woche",
        [0.1, 0.2],
        store=store,
        user_id="primary",
        settings=_settings(date_parse_model=None),
        today=date(2026, 8, 12),
        top_k=5,
    )


async def test_fetch_sessions_falls_back_to_fixed_window_when_parsing_returns_none(
    monkeypatch: MonkeyPatch,
) -> None:
    window_calls: list[dict[str, object]] = []

    async def fake_get_sessions_in_window(**kwargs: object) -> list[RetrievedChunk]:
        window_calls.append(kwargs)
        return []

    async def fake_search_by_vector(vector: list[float], **kwargs: object) -> list[RetrievedChunk]:
        return []

    async def fake_parse_date_range(query: str, **kwargs: object) -> DateRange | None:
        return None

    monkeypatch.setattr(retrieval_module, "_search_by_vector", fake_search_by_vector)
    monkeypatch.setattr(retrieval_module, "parse_date_range", fake_parse_date_range)
    store = AsyncMock()
    store.get_sessions_in_window = fake_get_sessions_in_window

    await _fetch_sessions(
        "was haben wir in Analysis behandelt?",
        [0.1, 0.2],
        store=store,
        user_id="primary",
        settings=_settings(date_parse_model="openai/gpt-4o", session_window_days=14),
        today=date(2026, 8, 12),
        top_k=5,
    )

    assert len(window_calls) == 1
    start, end = window_calls[0]["start"], window_calls[0]["end"]
    assert isinstance(start, datetime) and isinstance(end, datetime)
    assert (end - start).days == 28


async def test_retrieve_with_rerank_exempts_exact_date_matches_from_retrieval_top_k(
    monkeypatch: MonkeyPatch,
) -> None:
    """Regression test for the 2026-08-12 "21 real sessions in a week, only 8 survived" bug:
    exact_date_match=True chunks must bypass retrieval_top_k, capped only by
    date_range_chunk_cap - while ordinary chunks (exact_date_match=False) still respect
    retrieval_top_k as before."""
    exact_matches = [
        _chunk_of_type(i, f"exact-{i}", "session", 0.0, exact_date_match=True) for i in range(7)
    ]
    ordinary = [_chunk_of_type(100 + i, f"note-{i}", "note", 0.9 - i * 0.1) for i in range(3)]

    async def fake_search_by_vector(vector: list[float], **kwargs: object) -> list[RetrievedChunk]:
        return ordinary if kwargs["content_type"] == "note" else []

    async def fake_get_sessions_in_window(**kwargs: object) -> list[RetrievedChunk]:
        return exact_matches

    async def fake_embed_texts(
        texts: list[str], *, model: str, **_kwargs: object
    ) -> list[list[float]]:
        return [[0.1, 0.2]]

    monkeypatch.setattr(retrieval_module, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(retrieval_module, "_search_by_vector", fake_search_by_vector)
    store = AsyncMock()
    store.get_sessions_in_window = fake_get_sessions_in_window

    result = await retrieve_with_rerank(
        "query",
        store=store,
        settings=_settings(rerank_model=None, retrieval_top_k=2, date_range_chunk_cap=60),
        user_id="primary",
    )

    # All 7 exact matches survive (well under the cap of 60) despite retrieval_top_k=2 - plus up
    # to 2 ordinary chunks (retrieval_top_k still applies to those, unaffected).
    result_titles = {c.title for c in result}
    assert {c.title for c in exact_matches} <= result_titles
    assert len([t for t in result_titles if t.startswith("note-")]) == 2


async def test_retrieve_with_rerank_excludes_exact_date_matches_from_reranking(
    monkeypatch: MonkeyPatch,
) -> None:
    """Regression test for a 2026-08-13 production incident: exact_date_match chunks used to be
    reranked along with everything else - a "letzte Woche" query merged 21 near-duplicate exact-
    match session chunks into the same rerank call as the chunks that actually needed ranking,
    and every model tried (gpt-4o, gpt-5, gpt-5-mini) truncated its response partway through the
    near-duplicate block, starving the reranker's attention away from the chunks whose order
    actually mattered. Their inclusion is already unconditional (see the test above) - there's
    no relevance decision left for a reranker to make, so they must never reach it."""
    exact_matches = [
        _chunk_of_type(i, f"exact-{i}", "session", 0.0, exact_date_match=True) for i in range(21)
    ]
    ordinary = [_chunk_of_type(100 + i, f"note-{i}", "note", 0.9 - i * 0.1) for i in range(3)]
    captured_rerank_input: list[RetrievedChunk] = []

    async def fake_search_by_vector(vector: list[float], **kwargs: object) -> list[RetrievedChunk]:
        return ordinary if kwargs["content_type"] == "note" else []

    async def fake_get_sessions_in_window(**kwargs: object) -> list[RetrievedChunk]:
        return exact_matches

    async def fake_embed_texts(
        texts: list[str], *, model: str, **_kwargs: object
    ) -> list[list[float]]:
        return [[0.1, 0.2]]

    async def fake_rerank_chunks(
        query: str, chunks: list[RetrievedChunk], **kwargs: object
    ) -> list[RetrievedChunk]:
        captured_rerank_input.extend(chunks)
        return chunks

    monkeypatch.setattr(retrieval_module, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(retrieval_module, "_search_by_vector", fake_search_by_vector)
    monkeypatch.setattr(retrieval_module, "rerank_chunks", fake_rerank_chunks)
    store = AsyncMock()
    store.get_sessions_in_window = fake_get_sessions_in_window

    await retrieve_with_rerank(
        "query",
        store=store,
        settings=_settings(
            rerank_model="openai/gpt-4o", retrieval_top_k=8, date_range_chunk_cap=60
        ),
        user_id="primary",
    )

    assert len(captured_rerank_input) == 3
    assert all(not c.exact_date_match for c in captured_rerank_input)


async def test_retrieve_with_rerank_without_model_sorts_merged_pool_by_score(
    monkeypatch: MonkeyPatch,
) -> None:
    per_type_results = {
        "note": [_chunk_of_type(1, "note-hi", "note", 0.5)],
        "course": [_chunk_of_type(2, "course-hi", "course", 0.9)],
        "course_goal": [_chunk_of_type(4, "goal-mid", "course_goal", 0.6)],
        "session": [],
    }

    async def fake_search_by_vector(vector: list[float], **kwargs: object) -> list[RetrievedChunk]:
        return per_type_results[kwargs["content_type"]]

    async def fake_get_sessions_in_window(**kwargs: object) -> list[RetrievedChunk]:
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
    store.get_sessions_in_window = fake_get_sessions_in_window

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
    per_type_results = {
        "note": [_chunk_of_type(0, "chunk-0", "note", 0.5)],
        "course": [_chunk_of_type(1, "chunk-1", "course", 0.5)],
        "course_goal": [_chunk_of_type(2, "chunk-2", "course_goal", 0.5)],
        "session": [],
    }

    async def fake_search_by_vector(vector: list[float], **kwargs: object) -> list[RetrievedChunk]:
        return per_type_results[kwargs["content_type"]]

    async def fake_get_sessions_in_window(**kwargs: object) -> list[RetrievedChunk]:
        return [_chunk_of_type(3, "chunk-3", "session", 0.0)]

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
    store.get_sessions_in_window = fake_get_sessions_in_window

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
    store.get_sessions_in_window = AsyncMock(return_value=[])  # type: ignore[method-assign]

    async def fake_embed_texts(
        texts: list[str], *, model: str, **_kwargs: object
    ) -> list[list[float]]:
        return [[0.1, 0.2]]

    monkeypatch.setattr(retrieval_module, "embed_texts", fake_embed_texts)

    result = await retrieve_with_rerank(
        "query", store=store, settings=_settings(rerank_model=None), user_id="primary"
    )

    # session's own vector-search leg fails (like every other type's would), but that's caught
    # by _search_by_vector's own never-raises contract - the window leg (mocked empty here)
    # still runs fine, so session contributes nothing rather than aborting the whole retrieval.
    assert {c.content_type for c in result} == {"note", "course", "course_goal"}
