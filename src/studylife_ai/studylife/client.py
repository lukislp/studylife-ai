"""Async HTTP client for the StudyLife REST API.

Auth is a static `X-Api-Key` header (StudyLife's custom middleware — see
`Program.cs` in the StudyLife repo), minted once via the StudyLife setup UI.
There is no OpenAPI spec to generate a client from; this wraps only the
endpoints ingestion currently needs.
"""

import ssl
from types import TracebackType

import httpx

from studylife_ai.studylife.models import (
    CourseDto,
    CourseGoalDto,
    StudyLifeNote,
    StudySessionDto,
)


class StudyLifeClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        ca_cert_path: str | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"X-Api-Key": api_key},
            timeout=timeout,
            transport=transport,
            # A path here REPLACES httpx's default (certifi) trust store with just this CA, not
            # in addition to it - fine since this client only ever talks to StudyLife's own API
            # (a private cert-manager CA, e.g. in the k3s deployment), never any other host.
            # `True` (unset) keeps httpx's normal default trust store, matching local dev where
            # StudyLife runs on plain HTTP/localhost. ssl.create_default_context(cafile=...),
            # not the bare path string - httpx deprecated passing verify= a raw string path.
            verify=ssl.create_default_context(cafile=ca_cert_path) if ca_cert_path else True,
        )

    async def get_notes(self) -> list[StudyLifeNote]:
        """Fetch all notes.

        StudyLife has no pagination/incremental filter on this endpoint —
        see docs/decisions.md for why ingestion diffs the full list
        client-side instead.
        """
        response = await self._client.get("/api/notes")
        response.raise_for_status()
        return [StudyLifeNote.model_validate(item) for item in response.json()]

    async def get_courses(self) -> list[CourseDto]:
        """Fetch the active study program's course catalog."""
        response = await self._client.get("/api/courses")
        response.raise_for_status()
        return [CourseDto.model_validate(item) for item in response.json()]

    async def get_sessions_history(
        self, *, days: int, only_completed: bool
    ) -> list[StudySessionDto]:
        """Fetch study sessions (calendar entries) from the last `days` days.

        `GET /api/sessions` is hard-capped to a fixed -7d/+90d window; this
        endpoint is the only one with a configurable lookback.
        """
        response = await self._client.get(
            "/api/sessions/history",
            # Explicit lowercase string, not the raw bool - don't rely on
            # httpx's implicit bool query-param encoding matching ASP.NET
            # Core's model binder by luck.
            params={"days": days, "onlyCompleted": str(only_completed).lower()},
        )
        response.raise_for_status()
        return [StudySessionDto.model_validate(item) for item in response.json()]

    async def get_course_goals(self) -> list[CourseGoalDto]:
        """Fetch all course goals (one per course, at most)."""
        response = await self._client.get("/api/coursegoals")
        response.raise_for_status()
        return [CourseGoalDto.model_validate(item) for item in response.json()]

    async def create_session(self, session: StudySessionDto) -> StudySessionDto:
        """Create a study session (calendar entry).

        `session.id` is ignored - the server always assigns a fresh one
        (pass 0). The server validates `course_id` against the calling
        user's own course catalog (built-in + custom) - an unknown id gets
        `400` with `"CourseId {id} does not exist."`, and `course_name`
        (plus course color, for calendar display) is derived server-side
        from the resolved course rather than trusted from this request. It
        also validates `end_time > start_time` and duration <= 24h. Callers
        should still resolve a real `course_id` via `get_courses()` first
        rather than rely on this `400` alone, so a bad guess fails before a
        session is even proposed to the user. On invalid input this raises
        `httpx.HTTPStatusError` with a *plain-text* body (not JSON), e.g.
        via `exc.response.text`.
        """
        response = await self._client.post(
            "/api/sessions", json=session.model_dump(by_alias=True, mode="json")
        )
        response.raise_for_status()
        return StudySessionDto.model_validate(response.json())

    async def create_note(
        self,
        *,
        title: str,
        content: str,
        course_id: int | None = None,
        session_id: int | None = None,
    ) -> StudyLifeNote:
        """Create a note. `id`/`created_at`/`updated_at` are server-assigned -
        unlike `create_session`, plain parameters are taken instead of a full
        `StudyLifeNote` (which requires real timestamps a new note doesn't
        have yet). StudyLife validates `course_id` against the user's own
        catalog the same way `create_session` does (`400` with `"CourseId
        {id} does not exist."` for an unknown id) - it still accepts empty
        title/content and any `session_id` without checking it exists.
        """
        response = await self._client.post(
            "/api/notes",
            json={
                "title": title,
                "content": content,
                "courseId": course_id,
                "sessionId": session_id,
            },
        )
        response.raise_for_status()
        return StudyLifeNote.model_validate(response.json())

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "StudyLifeClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()
