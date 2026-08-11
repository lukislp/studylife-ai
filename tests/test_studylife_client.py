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


COURSES_PAYLOAD = [
    {
        "id": 6,
        "semester": 3,
        "name": "Lineare Algebra",
        "code": "MATH101",
        "topics": ["Eigenwerte", "Matrizen"],
        "ects": 5,
        "group": None,
    }
]

SESSIONS_PAYLOAD = [
    {
        "id": 42,
        "courseId": 6,
        "courseName": "Lineare Algebra",
        "startTime": "2026-08-01T10:00:00",
        "endTime": "2026-08-01T11:30:00",
        "topic": "Eigenwerte",
        "notes": None,
        "isCompleted": True,
    }
]

COURSE_GOALS_PAYLOAD = [
    {
        "courseId": 6,
        "courseName": "Lineare Algebra",
        "targetDate": "2026-09-01T00:00:00",
        "completionNote": None,
        "completedAt": None,
        "grade": None,
        "completedTopics": "",
        "tag": None,
    }
]


async def test_get_courses_parses_camel_case_response_and_sends_api_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/courses"
        assert request.headers["x-api-key"] == "secret"
        return httpx.Response(200, json=COURSES_PAYLOAD)

    async with _make_client(handler) as client:
        courses = await client.get_courses()

    assert len(courses) == 1
    assert courses[0].id == 6
    assert courses[0].topics == ["Eigenwerte", "Matrizen"]


async def test_get_courses_raises_on_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    async with _make_client(handler) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.get_courses()


async def test_get_sessions_history_sends_days_and_only_completed_as_query_params() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/sessions/history"
        assert request.url.params["days"] == "1825"
        assert request.url.params["onlyCompleted"] == "false"
        return httpx.Response(200, json=SESSIONS_PAYLOAD)

    async with _make_client(handler) as client:
        sessions = await client.get_sessions_history(days=1825, only_completed=False)

    assert len(sessions) == 1
    assert sessions[0].course_id == 6
    assert sessions[0].is_completed is True


async def test_get_sessions_history_raises_on_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    async with _make_client(handler) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.get_sessions_history(days=1825, only_completed=False)


async def test_get_course_goals_parses_camel_case_response_and_sends_api_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/coursegoals"
        assert request.headers["x-api-key"] == "secret"
        return httpx.Response(200, json=COURSE_GOALS_PAYLOAD)

    async with _make_client(handler) as client:
        goals = await client.get_course_goals()

    assert len(goals) == 1
    assert goals[0].course_id == 6
    assert goals[0].grade is None


async def test_get_course_goals_raises_on_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    async with _make_client(handler) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.get_course_goals()
