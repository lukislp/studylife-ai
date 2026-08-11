"""Seeds a small, fixed note corpus into Qdrant for CI eval runs.

CI has no access to a real StudyLife instance (local-only, dev-machine
bound) and no local Ollama, so the eval job needs a self-contained stand-in
for both the ingestion source and the embedding backend. This seeds
eval/fixture_notes.jsonl - the same notes used to build and locally
validate eval/dataset.jsonl - through the same chunk+embed+upsert path
ingestion.sync uses, so CI measures against the real pipeline mechanics.
"""

from pathlib import Path

from pydantic import BaseModel

from studylife_ai.config import Settings
from studylife_ai.ingestion.chunking import chunk_text
from studylife_ai.ingestion.qdrant_store import EntityChunkMetadata, QdrantStore
from studylife_ai.llm.embeddings import embed_texts

DEFAULT_FIXTURE_PATH = Path("eval/fixture_notes.jsonl")


class FixtureNote(BaseModel):
    id: int
    title: str
    content: str
    course_id: int | None = None


def load_fixture_notes(path: Path = DEFAULT_FIXTURE_PATH) -> list[FixtureNote]:
    with path.open(encoding="utf-8") as f:
        return [FixtureNote.model_validate_json(line) for line in f if line.strip()]


async def seed_fixture_notes(
    notes: list[FixtureNote], *, settings: Settings, store: QdrantStore
) -> None:
    for note in notes:
        chunks = chunk_text(
            note.content,
            chunk_size_tokens=settings.chunk_size_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        vectors = (
            await embed_texts(chunks, model=settings.embedding_model, call_site="eval-fixture")
            if chunks
            else []
        )
        if vectors:
            await store.ensure_collection(vector_size=len(vectors[0]))
        await store.replace_entity(
            chunks=chunks,
            vectors=vectors,
            metadata=EntityChunkMetadata(
                content_type="note",
                entity_id=note.id,
                title=note.title,
                course_id=note.course_id,
                session_id=None,
                user_id=settings.eval_user_id,
                fingerprint="fixture",
            ),
        )
