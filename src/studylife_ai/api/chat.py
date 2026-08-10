"""Chat endpoint: streams an LLM completion back to the client via SSE.

No RAG or tool calling yet (that's M2/M4) — this just proxies the given
message history through LiteLLM and streams text deltas as Server-Sent
Events. Each event is a JSON object `{"delta": "..."}`; the stream ends
with a literal `data: [DONE]` event, mirroring the OpenAI streaming
convention so the eventual Blazor frontend can reuse an off-the-shelf
SSE client.
"""

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from studylife_ai.config import get_settings
from studylife_ai.llm.client import stream_chat_completion
from studylife_ai.schemas.chat import ChatRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


async def _sse_event_stream(request: ChatRequest) -> AsyncIterator[str]:
    settings = get_settings()
    model = request.model or settings.llm_model
    try:
        async for delta in stream_chat_completion(
            request.messages,
            model=model,
            api_base=settings.llm_api_base,
            timeout=settings.llm_request_timeout_seconds,
        ):
            yield f"data: {json.dumps({'delta': delta})}\n\n"
    except Exception:
        logger.exception("LLM streaming failed for model=%s", model)
        yield f"data: {json.dumps({'error': 'LLM request failed'})}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(_sse_event_stream(request), media_type="text/event-stream")
