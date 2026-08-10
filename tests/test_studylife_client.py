import httpx
import pytest

from studylife_ai.studylife.client import StudyLifeClient

NOTES_PAYLOAD = [
    {
        "id": 1,
        "title": "Linear Algebra",
        "content": "Eigenvalues and eigenvectors...",
        "createdAt": "2026-08-01T10:00:00",
        "updatedAt": "2026-08-05T12:30:00",
        "courseId": 3,
        "sessionId": None,
    },
    {
        "id": 2,
        "title": "Quick thought",
        "content": "Ask professor about the exam format.",
        "createdAt": "2026-08-02T09:00:00",
        "updatedAt": "2026-08-02T09:00:00",
        "courseId": None,
        "sessionId": 42,
    },
]


def _make_client(handler: object) -> StudyLifeClient:
    return StudyLifeClient(
        base_url="http://studylife.test",
        api_key="secret",
        transport=httpx.MockTransport(handler),
    )


async def test_get_notes_parses_camel_case_response_and_sends_api_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/notes"
        assert request.headers["x-api-key"] == "secret"
        return httpx.Response(200, json=NOTES_PAYLOAD)

    async with _make_client(handler) as client:
        notes = await client.get_notes()

    assert len(notes) == 2
    assert notes[0].id == 1
    assert notes[0].course_id == 3
    assert notes[0].session_id is None
    assert notes[1].course_id is None
    assert notes[1].session_id == 42


async def test_get_notes_raises_on_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    async with _make_client(handler) as client:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client.get_notes()

    assert exc_info.value.response.status_code == 401
