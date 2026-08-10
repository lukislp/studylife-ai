"""Thin async wrapper around LiteLLM for streaming chat completions.

LiteLLM gives us a single call interface across API providers (OpenAI,
Anthropic, ...) and local models served via Ollama, selected purely by the
`model` string (e.g. "openai/gpt-4o-mini" vs "ollama/llama3.1"). Provider
API keys are read by LiteLLM directly from the environment.
"""

from collections.abc import AsyncIterator

import litellm

from studylife_ai.schemas.chat import ChatMessage


async def stream_chat_completion(
    messages: list[ChatMessage],
    *,
    model: str,
    api_base: str | None,
    timeout: float,
) -> AsyncIterator[str]:
    """Yield text deltas for a streaming chat completion.

    Empty deltas (e.g. the final chunk carrying only a finish_reason) are
    skipped so callers only ever see actual content.
    """
    response = await litellm.acompletion(
        model=model,
        messages=[m.model_dump() for m in messages],
        api_base=api_base,
        timeout=timeout,
        stream=True,
    )
    async for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
