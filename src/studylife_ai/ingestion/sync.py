"""Orchestrates one ingestion sync run.

Full-list diff against Qdrant's own state (no separate manifest store —
see docs/decisions.md "Incremental sync"): fetch all notes, compare each
against its last known fingerprint, chunk+embed+upsert what's new or
changed, delete what's gone.
"""

import hashlib
import logging

from studylife_ai.config import Settings
from studylife_ai.ingestion.chunking import chunk_text
from studylife_ai.ingestion.qdrant_store import NoteChunkMetadata, QdrantStore
from studylife_ai.llm.embeddings import embed_texts
from studylife_ai.studylife.client import StudyLifeClient
from studylife_ai.studylife.models import StudyLifeNote

logger = logging.getLogger(__name__)


def fingerprint_note(note: StudyLifeNote) -> str:
    """Content hash used to detect note changes.

    Deliberately not `note.updated_at`: StudyLife sets it via a server-local
    `DateTime.Now`, not UTC (see docs/decisions.md) — a content hash sidesteps
    that ambiguity entirely and also catches edits if the timestamp somehow
    wasn't bumped.
    """
    digest = hashlib.sha256(f"{note.title}\n{note.content}".encode())
    return digest.hexdigest()


async def sync_notes(settings: Settings) -> None:
    if not settings.studylife_api_base_url or not settings.studylife_api_key:
        raise RuntimeError(
            "STUDYLIFE_API_BASE_URL and STUDYLIFE_API_KEY must be set to run ingestion."
        )

    store = QdrantStore(url=settings.qdrant_url, collection=settings.qdrant_collection)
    try:
        known = await store.get_known_fingerprints()

        async with StudyLifeClient(
            base_url=settings.studylife_api_base_url,
            api_key=settings.studylife_api_key,
        ) as studylife:
            notes = await studylife.get_notes()

        current_ids = {note.id for note in notes}
        deleted_ids = set(known) - current_ids
        changed_notes = [note for note in notes if known.get(note.id) != fingerprint_note(note)]

        logger.info(
            "Sync: %d notes total, %d new/changed, %d deleted",
            len(notes),
            len(changed_notes),
            len(deleted_ids),
        )

        for note in changed_notes:
            chunks = chunk_text(
                note.content,
                chunk_size_tokens=settings.chunk_size_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
            )
            vectors = await embed_texts(chunks, model=settings.embedding_model) if chunks else []

            if vectors:
                await store.ensure_collection(vector_size=len(vectors[0]))

            await store.replace_note(
                chunks=chunks,
                vectors=vectors,
                metadata=NoteChunkMetadata(
                    note_id=note.id,
                    title=note.title,
                    course_id=note.course_id,
                    session_id=note.session_id,
                    user_id=settings.studylife_user_id,
                    fingerprint=fingerprint_note(note),
                ),
            )

        for note_id in deleted_ids:
            await store.delete_note(note_id)
    finally:
        await store.close()
