from studylife_ai.ingestion.chunking import _token_count, chunk_text


def test_chunk_text_empty_returns_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_chunk_text_short_text_is_a_single_chunk() -> None:
    text = "Eigenvalues describe how a linear map scales its eigenvectors."

    chunks = chunk_text(text, chunk_size_tokens=500, overlap_tokens=75)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_splits_long_text_and_keeps_chunks_within_budget() -> None:
    sentences = [f"This is sentence number {i} about linear algebra." for i in range(200)]
    text = " ".join(sentences)

    chunks = chunk_text(text, chunk_size_tokens=100, overlap_tokens=20)

    assert len(chunks) > 1
    for chunk in chunks:
        # A little slack: one sentence may push slightly past the budget
        # since sentences are never split mid-way in the normal path.
        assert _token_count(chunk) <= 100 + 20


def test_chunk_text_overlaps_consecutive_chunks() -> None:
    sentences = [f"This is sentence number {i} about linear algebra." for i in range(200)]
    text = " ".join(sentences)

    chunks = chunk_text(text, chunk_size_tokens=100, overlap_tokens=20)

    first_chunk_sentences = chunks[0].split(". ")
    second_chunk_sentences = chunks[1].split(". ")
    assert set(first_chunk_sentences) & set(second_chunk_sentences)


def test_chunk_text_hard_splits_a_single_oversized_sentence() -> None:
    # One giant "sentence" (no punctuation) far longer than the chunk budget.
    text = " ".join(f"word{i}" for i in range(2000))

    chunks = chunk_text(text, chunk_size_tokens=100, overlap_tokens=20)

    assert len(chunks) > 1
    for chunk in chunks:
        assert _token_count(chunk) <= 100
