from types import SimpleNamespace

from pytest import MonkeyPatch

from studylife_ai.llm.embeddings import embed_texts


async def test_embed_texts_returns_vectors_in_input_order(monkeypatch: MonkeyPatch) -> None:
    async def fake_aembedding(*, model: str, input: list[str]) -> object:
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
