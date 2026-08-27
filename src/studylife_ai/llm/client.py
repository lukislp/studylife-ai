"""Thin async wrapper around LiteLLM for streaming chat completions.

LiteLLM gives us a single call interface across API providers (OpenAI,
Anthropic, ...) and local models served via Ollama, selected purely by the
`model` string (e.g. "openai/gpt-4o-mini" vs "ollama/llama3.1"). Provider
API keys are read by LiteLLM directly from the environment.
"""

from collections.abc import AsyncIterator

import litellm

from studylife_ai.llm.retry import with_retry
from studylife_ai.schemas.chat import ChatMessage


async def stream_chat_completion(
    messages: list[ChatMessage],
    *,
    model: str,
    api_base: str | None,
    timeout: float,
    call_site: str = "unknown",
    user_id: str = "unknown",
    reasoning_effort: str | None = None,
) -> AsyncIterator[str]:
    """Yield text deltas for a streaming chat completion.

    Empty deltas (e.g. the final chunk carrying only a finish_reason) are
    skipped so callers only ever see actual content. `call_site`/`user_id`
    are pure logging metadata (see `llm/logging.py` and `llm/metrics.py`) -
    neither ever reaches the model. `reasoning_effort=None` (the default)
    omits the parameter entirely - LiteLLM strips `None` completion kwargs
    before sending the request, same as `complete_chat`'s `temperature` -
    correct for non-reasoning models, which don't accept this parameter.

    Retries (see `llm/retry.py`) only cover establishing the stream itself - a transient
    failure (timeout/429/5xx) before the first chunk is retried; once chunks have started
    arriving, a failure mid-stream propagates directly, since some content may already have
    been yielded to the caller and can't be un-sent.
    """
    response = await with_retry(
        lambda: litellm.acompletion(
            model=model,
            messages=[m.model_dump() for m in messages],
            api_base=api_base,
            timeout=timeout,
            stream=True,
            reasoning_effort=reasoning_effort,
            metadata={"call_site": call_site, "user_id": user_id},
        ),
        call_site=call_site,
    )
    async for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def complete_chat(
    messages: list[ChatMessage],
    *,
    model: str,
    api_base: str | None,
    timeout: float,
    call_site: str = "unknown",
    user_id: str = "unknown",
    temperature: float | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Non-streaming chat completion - returns the full response text at
    once. Used for reranking (rag/rerank.py) and date-range parsing
    (rag/date_parse.py), which need one parseable response, not a token
    stream. `call_site`/`user_id` are pure logging metadata (see
    `llm/logging.py` and `llm/metrics.py`) - neither ever reaches the
    model. `temperature=None`/`reasoning_effort=None` (the defaults) omit
    those parameters entirely, leaving the provider's own default in place -
    LiteLLM strips `None` completion kwargs before sending the request, so
    this is equivalent to not passing them at all. `reasoning_effort` only
    matters for reasoning models (e.g. `RERANK_MODEL=openai/gpt-5-mini`) -
    see `Settings.llm_reasoning_effort`'s docstring for why it's needed.

    Retries a transient failure (timeout/429/5xx) up to twice with backoff before giving up
    (see `llm/retry.py`) - safe to retry the whole call here, unlike `stream_chat_completion`,
    since nothing is returned to the caller until the full response is in."""
    response = await with_retry(
        lambda: litellm.acompletion(
            model=model,
            messages=[m.model_dump() for m in messages],
            api_base=api_base,
            timeout=timeout,
            stream=False,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            metadata={"call_site": call_site, "user_id": user_id},
        ),
        call_site=call_site,
    )
    return response.choices[0].message.content or ""
