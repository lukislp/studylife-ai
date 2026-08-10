"""Async wrapper around LiteLLM for text embeddings.

Same provider-agnostic convention as `llm/client.py`: the `model` string
selects the provider (e.g. "openai/text-embedding-3-small" or
"ollama/nomic-embed-text").
"""

import litellm


async def embed_texts(texts: list[str], *, model: str) -> list[list[float]]:
    """Embed a batch of texts, returning one vector per input in input order."""
    if not texts:
        return []
    response = await litellm.aembedding(model=model, input=texts)
    ordered = sorted(response.data, key=lambda item: item["index"])
    return [item["embedding"] for item in ordered]
