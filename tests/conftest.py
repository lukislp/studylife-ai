import os
import time
from collections.abc import AsyncIterator

import pytest
from cryptography.fernet import Fernet
from httpx import ASGITransport, AsyncClient
from pytest import MonkeyPatch

# A4: AI_KEY_ENCRYPTION_KEY is required for RegisteredKeyStore to construct at all (see
# config.py, registered_keys.py) - must be set before `studylife_ai.main` is imported anywhere,
# since importing it calls create_app() -> get_settings() at module load time, which the real
# app lifespan (see the `client` fixture below) feeds straight into RegisteredKeyStore. conftest
# modules are always collected before any test module in the same directory, so setting this
# here (module level, not inside a fixture) guarantees a valid key is already present by the
# time any test file does `from studylife_ai.main import app`. setdefault, not a plain
# assignment, so a real key already set in the environment/.env for local dev is respected
# rather than overridden.
os.environ.setdefault("AI_KEY_ENCRYPTION_KEY", Fernet.generate_key().decode())

from studylife_ai.api.identity import PROXY_TOKEN_HEADER, _sign
from studylife_ai.config import Settings
from studylife_ai.main import app
from studylife_ai.studylife.registered_keys import RegisteredKeyStore

TEST_USER_ID = "test-user"
TEST_SHARED_SECRET = "test-shared-secret"
# Whatever AI_KEY_ENCRYPTION_KEY resolved to above (either a real one from the environment/.env,
# or the freshly generated fallback) - reused by tests that construct their own RegisteredKeyStore
# directly (this file's `client` fixture below, tests/test_registered_keys.py).
TEST_AI_KEY_ENCRYPTION_KEY = os.environ["AI_KEY_ENCRYPTION_KEY"]


@pytest.fixture(autouse=True)
def _reset_rate_limit_windows() -> None:
    """rate_limit._windows is module-level state, deliberately (it has to persist across
    requests for the process lifetime) - but that means it'd otherwise leak between test
    cases too, since pytest runs them in the same process. Reset before every test, not just
    ones that use `client`, in case a future test imports the module directly."""
    from studylife_ai.api import rate_limit

    rate_limit._windows.clear()


@pytest.fixture(autouse=True)
def _reset_legacy_signing_fallback_warning() -> None:
    """identity._warned_legacy_signing_fallback is module-level "warn once" state (audit A5) -
    same leak-between-tests reasoning as `_reset_rate_limit_windows` above. Every test using
    `client` mints a legacy 3-part token by default (see `make_proxy_token`), so without this
    reset only the very first test in the whole run would ever observe the warning."""
    from studylife_ai.api import identity

    identity._warned_legacy_signing_fallback = False


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
    for module in ("chat", "agent", "identity", "internal", "rate_limit"):
        monkeypatch.setattr(f"studylife_ai.api.{module}.get_settings", _fake_get_settings)

    # Default: /internal/register-key's post-response auto-ingestion background task (see
    # docs/decisions.md "Auto-ingestion on register") is a no-op unless a test specifically
    # wants to exercise it - otherwise every test touching that endpoint would also attempt a
    # real (and here, failing) sync against "http://studylife.test".
    async def _noop_sync_user(**_kwargs: object) -> None:
        return None

    monkeypatch.setattr("studylife_ai.api.internal.sync_user", _noop_sync_user)

    # Default: the periodic ingestion sync loop (see docs/decisions.md "Periodic
    # ingestion sync") is a no-op in tests, same reasoning as _noop_sync_user
    # above - main.py's lifespan reads real, un-monkeypatched settings (it calls
    # studylife_ai.config.get_settings() directly, not any of the per-module
    # ones patched above), so a local .env with STUDYLIFE_API_BASE_URL set would
    # otherwise start a real background task hitting it on every test using
    # this fixture.
    async def _noop_run_periodic_sync(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr("studylife_ai.main.run_periodic_sync", _noop_run_periodic_sync)

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
        test_store = RegisteredKeyStore(":memory:", TEST_AI_KEY_ENCRYPTION_KEY)
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
