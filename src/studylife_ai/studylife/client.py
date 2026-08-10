"""Async HTTP client for the StudyLife REST API.

Auth is a static `X-Api-Key` header (StudyLife's custom middleware — see
`Program.cs` in the StudyLife repo), minted once via the StudyLife setup UI.
There is no OpenAPI spec to generate a client from; this wraps only the
endpoints ingestion currently needs.
"""

from types import TracebackType

import httpx

from studylife_ai.studylife.models import StudyLifeNote


class StudyLifeClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"X-Api-Key": api_key},
            timeout=timeout,
            transport=transport,
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
