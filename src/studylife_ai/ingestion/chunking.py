"""Fixed-size, sentence-aligned chunking with overlap.

See docs/decisions.md ("Chunking strategy") for why this shape was chosen:
note content is unstructured plain text (StudyLife's note editor is a bare
`<textarea>`, no Markdown/rich-text), so there's no reliable structure
(headers, etc.) to split on — only sentence and token boundaries.

Token size is measured via `tiktoken`'s `cl100k_base` encoding as a
provider-independent approximation; it won't exactly match every embedding
model's own tokenizer, but is close enough to size chunks sensibly
regardless of which embedding model is configured.
"""

import re

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


def chunk_text(text: str, *, chunk_size_tokens: int = 500, overlap_tokens: int = 75) -> list[str]:
    """Split text into overlapping, sentence-aligned token windows.

    Sentences are packed greedily up to `chunk_size_tokens`; a single
    sentence longer than that is hard-split at the token level as a
    fallback. Consecutive chunks share roughly `overlap_tokens` of trailing
    context so retrieval doesn't lose meaning at a chunk boundary.
    """
    text = text.strip()
    if not text:
        return []

    sentences: list[str] = []
    for sentence in _split_sentences(text):
        if _token_count(sentence) > chunk_size_tokens:
            sentences.extend(_hard_split_by_tokens(sentence, chunk_size_tokens))
        else:
            sentences.append(sentence)

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = _token_count(sentence)
        if current and current_tokens + sentence_tokens > chunk_size_tokens:
            chunks.append(" ".join(current))
            current, current_tokens = _carry_overlap(current, overlap_tokens)
        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]


def _token_count(text: str) -> int:
    return len(_ENCODING.encode(text))


def _hard_split_by_tokens(text: str, max_tokens: int) -> list[str]:
    tokens = _ENCODING.encode(text)
    return [_ENCODING.decode(tokens[i : i + max_tokens]) for i in range(0, len(tokens), max_tokens)]


def _carry_overlap(sentences: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    """Return the trailing sentences of `sentences` totalling ~overlap_tokens.

    A single sentence larger than `overlap_tokens` on its own (e.g. a
    hard-split fallback piece) is not carried at all — better to lose
    overlap at that one boundary than to blow the next chunk's budget.
    """
    carried: list[str] = []
    carried_tokens = 0
    for sentence in reversed(sentences):
        sentence_tokens = _token_count(sentence)
        if carried_tokens + sentence_tokens > overlap_tokens:
            break
        carried.insert(0, sentence)
        carried_tokens += sentence_tokens
    return carried, carried_tokens
