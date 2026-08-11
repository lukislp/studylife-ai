"""RAGAS eval runner: replays eval/dataset.jsonl through the real RAG
pipeline (retrieval + prompt + generation) and scores it.

Judge model is separate from the answer-generation model (docs/decisions.md
"Eval design") - a stronger, independent model is required for trustworthy
scores. `settings.eval_judge_model` must be set; there's no default, since
the provider choice is deliberately not made yet.
"""

from dataclasses import dataclass, field

import litellm
import pandas as pd
from langchain_community.chat_models import ChatLiteLLM
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import EvaluationResult, MultiTurnSample, SingleTurnSample
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import BaseRagasLLM, LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness, LLMContextPrecisionWithoutReference
from ragas.run_config import RunConfig

from studylife_ai.config import Settings
from studylife_ai.eval.dataset import EvalCase
from studylife_ai.ingestion.qdrant_store import QdrantStore, RetrievedChunk
from studylife_ai.llm.client import stream_chat_completion
from studylife_ai.rag.prompt import build_context_system_message
from studylife_ai.rag.retrieval import retrieve_with_rerank
from studylife_ai.schemas.chat import ChatMessage


@dataclass
class NoteMatchResult:
    """Whether retrieval found (at least) the notes an eval case expects.

    A lightweight, deterministic check independent of RAGAS' own (LLM-judged)
    metrics - useful because it's exact and free, unlike Context Precision.
    """

    case_id: str
    expected_titles: set[str]
    retrieved_titles: set[str]

    @property
    def matched(self) -> bool:
        return self.expected_titles.issubset(self.retrieved_titles)


@dataclass
class EvalReport:
    ragas_scores: pd.DataFrame
    note_matches: list[NoteMatchResult] = field(default_factory=list)

    @property
    def note_match_rate(self) -> float:
        if not self.note_matches:
            return 0.0
        return sum(r.matched for r in self.note_matches) / len(self.note_matches)


async def _generate_answer(
    question: str, settings: Settings, store: QdrantStore
) -> tuple[str, list[RetrievedChunk]]:
    """Same retrieval + prompt-construction + generation as /chat, but returns
    the full joined answer instead of an SSE stream."""
    chunks = await retrieve_with_rerank(
        question, store=store, settings=settings, user_id=settings.eval_user_id
    )
    context_message = ChatMessage(role="system", content=build_context_system_message(chunks))
    messages = [context_message, ChatMessage(role="user", content=question)]
    deltas = [
        delta
        async for delta in stream_chat_completion(
            messages,
            model=settings.llm_model,
            api_base=settings.llm_api_base,
            timeout=settings.llm_request_timeout_seconds,
        )
    ]
    return "".join(deltas), chunks


class _JudgeEmbeddings(BaseRagasEmbeddings):
    """Minimal BaseRagasEmbeddings adapter over LiteLLM's embedding call.

    Not ragas' own LiteLLMEmbeddings (ragas.embeddings.litellm_provider): that
    class implements the newer embed_text/embed_texts interface, not
    BaseRagasEmbeddings' embed_query/embed_documents - AnswerRelevancy (a
    legacy metric) calls .embed_query() and fails with AttributeError against
    it. Confirmed by inspecting both classes directly, same mismatch as
    llm_factory() vs BaseRagasLLM above.
    """

    def __init__(self, model: str) -> None:
        super().__init__()
        self._model = model

    def embed_query(self, text: str) -> list[float]:
        response = litellm.embedding(model=self._model, input=[text])
        return list(response.data[0]["embedding"])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = litellm.embedding(model=self._model, input=texts)
        ordered = sorted(response.data, key=lambda item: item["index"])
        return [item["embedding"] for item in ordered]

    async def aembed_query(self, text: str) -> list[float]:
        response = await litellm.aembedding(model=self._model, input=[text])
        return list(response.data[0]["embedding"])

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        response = await litellm.aembedding(model=self._model, input=texts)
        ordered = sorted(response.data, key=lambda item: item["index"])
        return [item["embedding"] for item in ordered]


def _build_judge(settings: Settings) -> tuple[BaseRagasLLM, BaseRagasEmbeddings]:
    if not settings.eval_judge_model:
        raise RuntimeError(
            "EVAL_JUDGE_MODEL must be set to run the eval - a judge model "
            "independent of the answer model is required for trustworthy "
            "scores (see docs/decisions.md 'Eval design')."
        )
    # LangchainLLMWrapper, not ragas' newer llm_factory(): the legacy metrics used
    # here (Faithfulness, AnswerRelevancy, ContextPrecision) expect a BaseRagasLLM
    # (.agenerate_text), which llm_factory()'s InstructorBaseRagasLLM does not
    # implement (.agenerate only) - confirmed by inspecting both classes directly.
    # bypass_n=False (default): tried True to fix ChatLiteLLM's n= support
    # silently returning 1 generation instead of the requested 3 for
    # Faithfulness' statement-generation step, but that triples Faithfulness'
    # call volume - in CI that pushed 26/36 sub-calls into RunConfig's 180s
    # timeout (see docs/decisions.md). A degraded-but-present score (n=1) beats
    # a reliably-NaN one (timed out at n=3), especially with no CI thresholds
    # depending on the exact value yet.
    judge_llm = LangchainLLMWrapper(ChatLiteLLM(model=settings.eval_judge_model))
    judge_embeddings = _JudgeEmbeddings(model=settings.embedding_model)
    return judge_llm, judge_embeddings


async def run_eval(cases: list[EvalCase], *, settings: Settings, store: QdrantStore) -> EvalReport:
    judge_llm, judge_embeddings = _build_judge(settings)

    samples: list[SingleTurnSample | MultiTurnSample] = []
    note_matches: list[NoteMatchResult] = []
    for case in cases:
        answer, chunks = await _generate_answer(case.question, settings, store)
        samples.append(
            SingleTurnSample(
                user_input=case.question,
                response=answer,
                retrieved_contexts=[chunk.content for chunk in chunks],
            )
        )
        note_matches.append(
            NoteMatchResult(
                case_id=case.id,
                expected_titles=set(case.expected_note_titles),
                retrieved_titles={chunk.title for chunk in chunks},
            )
        )

    dataset = EvaluationDataset(samples=samples)
    result = evaluate(
        dataset=dataset,
        # LLMContextPrecisionWithoutReference, not ContextPrecision (=
        # LLMContextPrecisionWithReference): our eval cases have no
        # ground-truth reference answer, only expected_note_titles for the
        # separate, non-LLM note-match check below. The "without reference"
        # variant judges context precision from the generated response
        # itself, which fits the fields our SingleTurnSamples carry.
        metrics=[Faithfulness(), AnswerRelevancy(), LLMContextPrecisionWithoutReference()],
        llm=judge_llm,
        embeddings=judge_embeddings,
        # Lower concurrency, longer per-job timeout than RunConfig's defaults
        # (max_workers=16, timeout=180s): CI (GitHub Actions -> OpenAI) hit
        # repeated executor-level TimeoutErrors at the defaults, mainly on
        # Faithfulness (the most LLM-call-heavy metric per case) - see
        # docs/decisions.md. Fewer in-flight requests and more slack per
        # request costs wall-clock time, which this job doesn't need to
        # minimize.
        run_config=RunConfig(timeout=300, max_workers=4),
    )
    # evaluate() is typed to also allow an Executor, but that only happens when
    # return_executor=True is passed - we never do, so this is always an EvaluationResult.
    assert isinstance(result, EvaluationResult)
    return EvalReport(ragas_scores=result.to_pandas(), note_matches=note_matches)
