from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException
from starlette.datastructures import Headers
from starlette.requests import Request

from studylife_ai.api.identity import resolve_identity, verify_identity


def _request(headers: dict[str, str]) -> Request:
    scope = {
        "type": "http",
        "headers": Headers(headers).raw,
    }
    return Request(scope)


def test_resolve_identity_returns_headers_when_both_present() -> None:
    identity = resolve_identity(
        _request({"X-StudyLife-User-Id": "1", "X-StudyLife-Ai-Key": "secret"})
    )

    assert identity.user_id == "1"
    assert identity.ai_api_key == "secret"


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"X-StudyLife-User-Id": "1"},
        {"X-StudyLife-Ai-Key": "secret"},
    ],
)
def test_resolve_identity_raises_401_when_a_header_is_missing(headers: dict[str, str]) -> None:
    with pytest.raises(HTTPException) as exc_info:
        resolve_identity(_request(headers))

    assert exc_info.value.status_code == 401


async def test_verify_identity_passes_when_the_call_succeeds() -> None:
    fake_client = AsyncMock()
    fake_client.get_courses.return_value = []

    await verify_identity(fake_client)

    fake_client.get_courses.assert_awaited_once()


async def test_verify_identity_raises_401_on_a_401_response() -> None:
    fake_client = AsyncMock()
    response = httpx.Response(
        401, request=httpx.Request("GET", "http://studylife.test/api/courses")
    )
    fake_client.get_courses.side_effect = httpx.HTTPStatusError(
        "unauthorized", request=response.request, response=response
    )

    with pytest.raises(HTTPException) as exc_info:
        await verify_identity(fake_client)

    assert exc_info.value.status_code == 401


async def test_verify_identity_reraises_non_401_http_errors() -> None:
    fake_client = AsyncMock()
    response = httpx.Response(
        500, request=httpx.Request("GET", "http://studylife.test/api/courses")
    )
    fake_client.get_courses.side_effect = httpx.HTTPStatusError(
        "server error", request=response.request, response=response
    )

    with pytest.raises(httpx.HTTPStatusError):
        await verify_identity(fake_client)
