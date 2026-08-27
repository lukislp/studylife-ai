"""Async wrapper around LiteLLM for text embeddings.

Same provider-agnostic convention as `llm/client.py`: the `model` string
selects the provider (e.g. "openai/text-embedding-3-small" or
"ollama/nomic-embed-text").
"""

import litellm

from studylife_ai.llm.retry import with_retry


async def embed_texts(
    texts: list[str], *, model: str, call_site: str = "unknown", user_id: str = "unknown"
) -> list[list[float]]:
    """Embed a batch of texts, returning one vector per input in input order.

    `call_site`/`user_id` are pure logging metadata (see `llm/logging.py`
    and `llm/metrics.py`) - neither ever reaches the model.

    Retries a transient failure (timeout/429/5xx) up to twice with backoff before giving up
    (see `llm/retry.py`), same mechanism as `llm/client.py`'s `complete_chat`.
    """
    if not texts:
        return []
    response = await with_retry(
        lambda: litellm.aembedding(
            model=model, input=texts, metadata={"call_site": call_site, "user_id": user_id}
        ),
        call_site=call_site,
    )
    ordered = sorted(response.data, key=lambda item: item["index"])
    return [item["embedding"] for item in ordered]
