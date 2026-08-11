"""LLM-based reranking: reorders a candidate pool of retrieved chunks by
relevance to the query, using a plain-text prompt (not JSON mode) so it
works with small local Ollama models too, not just API providers (see
docs/decisions.md "Retrieval quality: reranking + per-content-type quota").
"""

import logging
import re

from studylife_ai.ingestion.qdrant_store import RetrievedChunk
from studylife_ai.llm.client import complete_chat
from studylife_ai.schemas.chat import ChatMessage

logger = logging.getLogger(__name__)

_CONTENT_PREVIEW_CHARS = 300

_PROMPT_TEMPLATE = (
    "Rank the following passages by relevance to the question, most relevant "
    "first. Reply with ONLY the passage numbers, comma-separated, most to "
    'least relevant - e.g. "2,0,4,1,3". No other text.\n\n'
    "Question: {query}\n\n"
    "Passages:\n{passages}"
)


def _build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    passages = "\n".join(
        f"[{i}] ({chunk.content_type}) {chunk.title}: {chunk.content[:_CONTENT_PREVIEW_CHARS]}"
        for i, chunk in enumerate(chunks)
    )
    return _PROMPT_TEMPLATE.format(query=query, passages=passages)


def _parse_order(response: str, num_candidates: int) -> list[int]:
    """Parses a comma/whitespace-separated list of indices. Invalid tokens
    (non-numeric, out of range, duplicates) are skipped rather than raising;
    any index the model didn't mention is appended in its original order -
    an empty or completely unparseable response naturally degrades to the
    unchanged original order, not a special-cased fallback."""
    seen: set[int] = set()
    order: list[int] = []
    for token in re.split(r"[,\s]+", response.strip()):
        if not token:
            continue
        try:
            index = int(token)
        except ValueError:
            continue
        if 0 <= index < num_candidates and index not in seen:
            seen.add(index)
            order.append(index)
    # A non-empty response that still recognized less than half the
    # candidates is a signal worth surfacing (e.g. a model 1-indexing
    # instead of following the 0-indexed prompt) - distinct from a genuinely
    # empty/garbage response, which degrades to the unchanged order silently
    # since that's the expected, harmless fallback path.
    if response.strip() and len(seen) < num_candidates / 2:
        logger.warning(
            "Reranker recognized only %d/%d candidate indices in its response "
            "(possible indexing mismatch): %r",
            len(seen),
            num_candidates,
            response,
        )
    order.extend(i for i in range(num_candidates) if i not in seen)
    return order


async def rerank_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    model: str,
    api_base: str | None,
    timeout: float,
) -> list[RetrievedChunk]:
    """Reorders `chunks` most-to-least relevant to `query`. Never raises -
    a bad or missing rerank result degrades to the original vector-search
    order instead of breaking retrieval, matching /chat's existing
    "retrieval failed, continue anyway" fallback philosophy."""
    if len(chunks) <= 1:
        return chunks

    prompt = _build_prompt(query, chunks)
    try:
        response = await complete_chat(
            [ChatMessage(role="user", content=prompt)],
            model=model,
            api_base=api_base,
            timeout=timeout,
        )
    except Exception:
        logger.exception("Reranking failed, falling back to vector-search order")
        return chunks

    order = _parse_order(response, len(chunks))
    return [chunks[i] for i in order]
