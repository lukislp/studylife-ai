from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from pytest import MonkeyPatch

from studylife_ai.api.identity import AI_API_KEY_HEADER, USER_ID_HEADER
from studylife_ai.config import Settings
from studylife_ai.main import app

TEST_USER_ID = "test-user"
TEST_AI_API_KEY = "test-ai-api-key"


async def _fake_verify_identity(*args: object, **kwargs: object) -> None:
    return None


def _fake_get_settings() -> Settings:
    return Settings(studylife_api_base_url="http://studylife.test")


@pytest.fixture
async def client(monkeypatch: MonkeyPatch) -> AsyncIterator[AsyncClient]:
    # Default so most tests don't need to think about the key-validation
    # call /chat and /agent now make (see docs/decisions.md "Key validity
    # check") - it would otherwise depend on either a real StudyLife
    # instance or whatever STUDYLIFE_API_BASE_URL happens to be in the
    # local/CI environment. Tests exercising validation specifically
    # (invalid key, not configured) override per-test.
    monkeypatch.setattr("studylife_ai.api.chat.verify_identity", _fake_verify_identity)
    monkeypatch.setattr("studylife_ai.api.agent.verify_identity", _fake_verify_identity)
    monkeypatch.setattr("studylife_ai.api.chat.get_settings", _fake_get_settings)
    monkeypatch.setattr("studylife_ai.api.agent.get_settings", _fake_get_settings)

    # Runs the app's lifespan (startup/shutdown) — ASGITransport alone doesn't,
    # and /chat needs app.state.qdrant_store, which lifespan sets up.
    # Default identity headers so most tests don't need to think about
    # multi-user identity - tests exercising that specifically (missing
    # headers, thread ownership) override per-request.
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            headers={USER_ID_HEADER: TEST_USER_ID, AI_API_KEY_HEADER: TEST_AI_API_KEY},
        ) as ac:
            yield ac
