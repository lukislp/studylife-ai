from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from pytest import MonkeyPatch
from ragas.dataset_schema import EvaluationResult

from studylife_ai.config import Settings
from studylife_ai.eval import runner as runner_module
from studylife_ai.eval.dataset import EvalCase
from studylife_ai.eval.runner import EvalReport, NoteMatchResult, check_answer_terms, run_eval
from studylife_ai.ingestion.qdrant_store import QdrantStore, RetrievedChunk
from studylife_ai.rag.chat_context import ChatContext
from studylife_ai.rag.prompt import sources_payload


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "llm_model": "ollama/llama3.2",
        "llm_api_base": "http://ollama.test:11434",
        "llm_request_timeout_seconds": 30.0,
        "embedding_model": "ollama/nomic-embed-text",
        "eval_user_id": "eval-user",
        "retrieval_top_k": 5,
        "eval_judge_model": "openai/gpt-4o-mini",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


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


class TestNoteMatchResult:
    def test_matched_when_all_expected_titles_were_retrieved(self) -> None:
        result = NoteMatchResult(
            case_id="a",
            expected_titles={"Eigenwerte"},
            retrieved_titles={"Eigenwerte", "Matrizen"},
        )

        assert result.matched is True

    def test_not_matched_when_an_expected_title_is_missing(self) -> None:
        result = NoteMatchResult(
            case_id="a", expected_titles={"Eigenwerte"}, retrieved_titles={"Matrizen"}
        )

        assert result.matched is False

    def test_matched_when_nothing_is_expected_regardless_of_what_was_retrieved(self) -> None:
        result = NoteMatchResult(case_id="a", expected_titles=set(), retrieved_titles={"Matrizen"})

        assert result.matched is True


class TestEvalReport:
    def test_note_match_rate_is_zero_for_no_cases(self) -> None:
        report = EvalReport(ragas_scores=pd.DataFrame())

        assert report.note_match_rate == 0.0

    def test_note_match_rate_is_the_fraction_of_matched_cases(self) -> None:
        report = EvalReport(
            ragas_scores=pd.DataFrame(),
            note_matches=[
                NoteMatchResult(case_id="a", expected_titles={"X"}, retrieved_titles={"X"}),
                NoteMatchResult(case_id="b", expected_titles={"X"}, retrieved_titles=set()),
            ],
        )

        assert report.note_match_rate == 0.5


async def test_generate_answer_retrieves_context_and_joins_streamed_deltas(
    monkeypatch: MonkeyPatch,
) -> None:
    chunk = _chunk(1, "Eigenwerte", "det(A - λI) = 0")

    async def fake_retrieve_with_rerank(
        query: str, *, store: object, settings: Settings, user_id: str
    ) -> list[RetrievedChunk]:
        assert query == "Was sind Eigenwerte?"
        assert settings.embedding_model == "ollama/nomic-embed-text"
        assert user_id == "eval-user"
        assert settings.retrieval_top_k == 5
        return [chunk]

    monkeypatch.setattr(runner_module, "retrieve_with_rerank", fake_retrieve_with_rerank)

    async def fake_acompletion(*_args: object, **_kwargs: object) -> object:
        async def stream() -> object:
            for text in ["Eigenwerte ", "sind..."]:
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
                )

        return stream()

    monkeypatch.setattr("studylife_ai.llm.client.litellm.acompletion", fake_acompletion)

    answer, context = await runner_module._generate_answer(
        "Was sind Eigenwerte?", _settings(), Mock(spec=QdrantStore)
    )

    assert answer == "Eigenwerte sind..."
    assert context.contexts == ["det(A - λI) = 0"]
    assert [s.title for s in context.sources] == ["Eigenwerte"]


def test_build_judge_raises_when_judge_model_is_not_set() -> None:
    with pytest.raises(RuntimeError, match="EVAL_JUDGE_MODEL"):
        runner_module._build_judge(_settings(eval_judge_model=None))


async def test_run_eval_scores_generated_answers_and_reports_note_matches(
    monkeypatch: MonkeyPatch,
) -> None:
    cases = [
        EvalCase(
            id="a",
            question="Was sind Eigenwerte?",
            expected_titles=["Eigenwerte"],
            category="basic",
        ),
        EvalCase(
            id="b",
            question="Was ist Cloud Computing?",
            expected_titles=["Cloud"],
            category="no-match",
        ),
    ]
    chunk = _chunk(1, "Eigenwerte", "det(A - λI) = 0")

    async def fake_generate_answer(
        question: str, settings: Settings, store: QdrantStore, source: object = None
    ) -> tuple[str, ChatContext]:
        if question == "Was sind Eigenwerte?":
            return "Eigenwerte sind...", ChatContext(
                system_message="",
                sources=sources_payload([chunk]),
                contexts=[chunk.content],
            )
        return "Ich habe dazu nichts gefunden.", ChatContext(system_message="", sources=[])

    monkeypatch.setattr(runner_module, "_generate_answer", fake_generate_answer)

    judge_llm, judge_embeddings = object(), object()
    monkeypatch.setattr(
        runner_module, "_build_judge", lambda settings: (judge_llm, judge_embeddings)
    )

    fake_result = Mock(spec=EvaluationResult)
    fake_result.to_pandas.return_value = pd.DataFrame({"faithfulness": [1.0, 0.0]})
    captured: dict[str, object] = {}

    def fake_evaluate(**kwargs: object) -> EvaluationResult:
        captured.update(kwargs)
        return fake_result  # type: ignore[return-value]

    monkeypatch.setattr(runner_module, "evaluate", fake_evaluate)

    report = await run_eval(cases, settings=_settings(), store=Mock(spec=QdrantStore))

    assert captured["llm"] is judge_llm
    assert captured["embeddings"] is judge_embeddings
    assert len(captured["metrics"]) == 3  # type: ignore[arg-type]  # Faithfulness, AnswerRelevancy, LLMContextPrecisionWithoutReference
    dataset = captured["dataset"]
    assert len(dataset.samples) == 2  # type: ignore[attr-defined]
    assert dataset.samples[0].retrieved_contexts == ["det(A - λI) = 0"]  # type: ignore[attr-defined]
    assert dataset.samples[1].retrieved_contexts == []  # type: ignore[attr-defined]

    assert report.ragas_scores is fake_result.to_pandas.return_value
    assert report.note_matches == [
        NoteMatchResult(
            case_id="a", expected_titles={"Eigenwerte"}, retrieved_titles={"Eigenwerte"}
        ),
        NoteMatchResult(case_id="b", expected_titles={"Cloud"}, retrieved_titles=set()),
    ]
    assert report.note_match_rate == 0.5


def test_check_answer_terms_is_case_insensitive_both_ways() -> None:
    case = EvalCase(
        id="c",
        question="q",
        expected_titles=[],
        category="listing-notes",
        required_answer_terms=["Gram-Schmidt"],
        forbidden_answer_terms=["Sitzung", "geplant"],
    )

    ok = check_answer_terms(case, "Deine neueste Notiz ist GRAM-SCHMIDT Kurzfassung [1].")
    bad = check_answer_terms(case, "Ausserdem ist eine Lernsitzung geplant.")

    assert ok.passed
    assert not bad.passed
    assert bad.missing_terms == {"Gram-Schmidt"}
    assert bad.forbidden_terms_found == {"Sitzung", "geplant"}


async def test_fixture_data_source_serves_the_seeded_corpus_with_the_newest_note() -> None:
    from studylife_ai.eval.dataset import load_eval_cases
    from studylife_ai.eval.fixture import (
        FixtureDataSource,
        load_fixture_course_goals,
        load_fixture_courses,
        load_fixture_notes,
        load_fixture_sessions,
    )

    source = FixtureDataSource(
        notes=load_fixture_notes(),
        courses=load_fixture_courses(),
        sessions=load_fixture_sessions(),
    )
    notes = await source.get_notes()
    newest = max(notes, key=lambda n: n.created_at)

    assert newest.title == "Gram-Schmidt Kurzfassung"
    assert newest.course_id == 6

    # Every expected title of every eval case names something in the fixture corpus - a typo
    # would make a case fail forever for the wrong reason.
    sessions = await source.get_sessions_history(days=1825, only_completed=False)
    known = (
        {n.title for n in notes}
        | {c.name for c in await source.get_courses()}
        | {f"{s.course_name}, {s.start_time:%Y-%m-%d %H:%M}" for s in sessions}
        | {f"{g.course_name} goal" for g in load_fixture_course_goals()}
    )
    for case in load_eval_cases():
        assert set(case.expected_titles) <= known, case.id
