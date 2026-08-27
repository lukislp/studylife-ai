from types import SimpleNamespace

from litellm.exceptions import Timeout as LiteLLMTimeout
from pytest import MonkeyPatch

from studylife_ai.llm import retry as retry_module
from studylife_ai.llm.embeddings import embed_texts


async def test_embed_texts_returns_vectors_in_input_order(monkeypatch: MonkeyPatch) -> None:
    async def fake_aembedding(*, model: str, input: list[str], **_kwargs: object) -> object:
        assert model == "ollama/nomic-embed-text"
        # Return out of order to verify embed_texts re-sorts by index.
        return SimpleNamespace(
            data=[
                {"embedding": [0.2, 0.2], "index": 1},
                {"embedding": [0.1, 0.1], "index": 0},
            ]
        )

    monkeypatch.setattr("studylife_ai.llm.embeddings.litellm.aembedding", fake_aembedding)

    vectors = await embed_texts(["first", "second"], model="ollama/nomic-embed-text")

    assert vectors == [[0.1, 0.1], [0.2, 0.2]]


async def test_embed_texts_returns_empty_list_for_no_input() -> None:
    assert await embed_texts([], model="ollama/nomic-embed-text") == []


async def test_embed_texts_retries_a_transient_failure_then_succeeds(
    monkeypatch: MonkeyPatch,
) -> None:
    """See docs/decisions.md "LLM call retry" - same shared llm/retry.py mechanism as
    llm/client.py's complete_chat/stream_chat_completion."""

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(retry_module.asyncio, "sleep", _no_sleep)
    attempts = 0

    async def fake_aembedding(*, model: str, input: list[str], **_kwargs: object) -> object:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise LiteLLMTimeout(message="timed out", model=model, llm_provider="p")
        return SimpleNamespace(data=[{"embedding": [0.1, 0.1], "index": 0}])

    monkeypatch.setattr("studylife_ai.llm.embeddings.litellm.aembedding", fake_aembedding)

    vectors = await embed_texts(["only"], model="ollama/nomic-embed-text")

    assert vectors == [[0.1, 0.1]]
    assert attempts == 2
