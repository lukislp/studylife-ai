import time
from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from pytest import MonkeyPatch

from studylife_ai.api.identity import PROXY_TOKEN_HEADER, _sign
from studylife_ai.config import Settings
from studylife_ai.main import app
from studylife_ai.studylife.registered_keys import RegisteredKeyStore

TEST_USER_ID = "test-user"
TEST_SHARED_SECRET = "test-shared-secret"


def make_proxy_token(
    user_id: str, *, secret: str = TEST_SHARED_SECRET, expires_in: int = 60
) -> str:
    payload = f"{user_id}.{int(time.time()) + expires_in}"
    return f"{payload}.{_sign(payload, secret)}"


def _fake_get_settings() -> Settings:
    return Settings(
        studylife_api_base_url="http://studylife.test",
        studylife_shared_secret=TEST_SHARED_SECRET,
    )


@pytest.fixture
async def client(monkeypatch: MonkeyPatch) -> AsyncIterator[AsyncClient]:
    # Default so most tests don't need to think about the shared secret
    # (see docs/decisions.md "M4.5 Multi-user support") - it would otherwise
    # depend on whatever STUDYLIFE_SHARED_SECRET happens to be in the
    # local/CI environment. Tests exercising that specifically override
    # per-test.
    for module in ("chat", "agent", "identity", "internal"):
        monkeypatch.setattr(f"studylife_ai.api.{module}.get_settings", _fake_get_settings)

    # Runs the app's lifespan (startup/shutdown) — ASGITransport alone doesn't,
    # and /chat needs app.state.qdrant_store, which lifespan sets up.
    async with app.router.lifespan_context(app):
        # The real lifespan already opened a RegisteredKeyStore against
        # whatever settings.registered_keys_db_path resolves to in this
        # environment (same accepted trade-off as the SQLite agent
        # checkpointer - see docs/decisions.md) - swap in an isolated
        # in-memory one so tests never touch real data and each test starts
        # with a clean, empty registry.
        real_store = app.state.registered_key_store
        test_store = RegisteredKeyStore(":memory:")
        await test_store.setup()
        app.state.registered_key_store = test_store
        try:
            # Default identity: a valid, unexpired proxy token for
            # TEST_USER_ID - tests exercising identity specifically
            # (missing/invalid/expired token, thread ownership) override
            # per-request.
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport,
                base_url="http://test",
                headers={PROXY_TOKEN_HEADER: make_proxy_token(TEST_USER_ID)},
            ) as ac:
                yield ac
        finally:
            await test_store.close()
            app.state.registered_key_store = real_store
