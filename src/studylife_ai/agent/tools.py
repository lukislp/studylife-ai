"""Agent tools: two read tools (execute immediately) and two write tools
(never execute directly - HumanInTheLoopMiddleware pauses before them, see
`agent/graph.py`). Built as closures over already-app-lifetime resources
(`StudyLifeClient`, `QdrantStore`) via a factory, so tools don't need their
own dependency-injection machinery.
"""

from datetime import datetime

from langchain_core.tools import BaseTool, tool

from studylife_ai.config import Settings
from studylife_ai.ingestion.qdrant_store import QdrantStore
from studylife_ai.rag.retrieval import retrieve_chunks
from studylife_ai.studylife.client import StudyLifeClient
from studylife_ai.studylife.models import StudySessionDto


def build_tools(
    *, studylife: StudyLifeClient, store: QdrantStore, settings: Settings
) -> list[BaseTool]:
    @tool
    async def list_courses() -> list[dict[str, object]]:
        """List all courses with their id, name, code, and ECTS.

        StudyLife's API does NOT validate course_id when creating a session -
        any positive integer is silently accepted, even for a nonexistent
        course. Always call this first to resolve a course name to its real
        id before calling create_study_session - never guess or invent one.
        """
        courses = await studylife.get_courses()
        return [{"id": c.id, "name": c.name, "code": c.code, "ects": c.ects} for c in courses]

    @tool
    async def search_notes(query: str) -> list[dict[str, object]]:
        """Search the user's notes by meaning, for summarization.

        Args:
            query: What to search for, e.g. "Statistik Hypothesentests".
        """
        chunks = await retrieve_chunks(
            query,
            store=store,
            embedding_model=settings.embedding_model,
            user_id=settings.studylife_user_id,
            top_k=settings.retrieval_top_k,
            content_type="note",
        )
        return [{"title": c.title, "content": c.content} for c in chunks]

    @tool
    async def create_study_session(
        course_id: int,
        course_name: str,
        start_time: datetime,
        end_time: datetime,
        topic: str | None = None,
        notes: str | None = None,
    ) -> dict[str, object]:
        """Propose creating a new study session in StudyLife's calendar.

        Requires human confirmation before it actually executes - calling
        this only proposes the session, it does not create it yet.

        Args:
            course_id: The real course id, from list_courses - never guessed.
            course_name: The course's exact name, from list_courses.
            start_time: When the session starts.
            end_time: When the session ends - must be after start_time, at
                most 24 hours later (StudyLife rejects anything longer).
            topic: What the session covers, if known.
            notes: Any additional notes for the session.
        """
        created = await studylife.create_session(
            StudySessionDto(
                id=0,
                course_id=course_id,
                course_name=course_name,
                start_time=start_time,
                end_time=end_time,
                topic=topic,
                notes=notes,
            )
        )
        return {"id": created.id, "course_name": created.course_name}

    @tool
    async def save_note(
        title: str, content: str, course_id: int | None = None
    ) -> dict[str, object]:
        """Propose saving a new note in StudyLife (e.g. a summary you wrote).

        Requires human confirmation before it actually executes.

        Args:
            title: A short title for the note.
            content: The note's full text - e.g. a summary you generated
                from search_notes results.
            course_id: The real course id to link this note to, from
                list_courses, if relevant - never guessed.
        """
        created = await studylife.create_note(title=title, content=content, course_id=course_id)
        return {"id": created.id, "title": created.title}

    return [list_courses, search_notes, create_study_session, save_note]
