"""Audit O6-ai: /internal/* served on its own dedicated port (internal_server.py), in addition
to the shared public port (main.py), for one transition release.

Unlike the rest of the suite, `test_internal_app_is_reachable_on_a_real_second_port` and
`test_serve_internal_app_stops_gracefully_when_should_exit_is_set` below deliberately don't use
the `client`/ASGITransport fixture - the whole point of this feature is a second REAL listening
socket, which ASGITransport (an in-process ASGI-to-HTTP adapter with no actual network stack)
can't exercise. `port=0` lets the OS pick a free ephemeral port, read back via
`server.servers[0].sockets[0].getsockname()`.
"""

import asyncio
import logging
from collections.abc import AsyncIterator

import pytest
import uvicorn
from httpx import AsyncClient
from pytest import LogCaptureFixture, MonkeyPatch

from studylife_ai.config import Settings
from studylife_ai.internal_server import (
    build_internal_server,
    create_internal_app,
    serve_internal_app,
)
from studylife_ai.studylife.registered_keys import RegisteredKeyStore
from tests.conftest import TEST_AI_KEY_ENCRYPTION_KEY, TEST_SHARED_SECRET

RunningInternalServer = tuple[uvicorn.Server, str]


async def _wait_until_started(server: uvicorn.Server, *, timeout: float = 5.0) -> None:
    async def _poll() -> None:
        while not server.started:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_poll(), timeout=timeout)


@pytest.fixture
async def running_internal_server(
    monkeypatch: MonkeyPatch,
) -> AsyncIterator[RunningInternalServer]:
    """Builds a real internal_app (only /internal/*, backed by a fresh in-memory
    RegisteredKeyStore) and serves it on a real, OS-assigned ephemeral port, for the duration of
    one test. Yields (server, base_url)."""
    monkeypatch.setattr(
        "studylife_ai.api.internal.get_settings",
        lambda: Settings(studylife_internal_api_secret=TEST_SHARED_SECRET),  # type: ignore[call-arg]
    )

    key_store = RegisteredKeyStore(":memory:", TEST_AI_KEY_ENCRYPTION_KEY)
    await key_store.setup()

    internal_app = create_internal_app()
    internal_app.state.registered_key_store = key_store

    server = build_internal_server(internal_app, Settings(internal_api_port=0))
    task = asyncio.create_task(serve_internal_app(server))
    await _wait_until_started(server)
    assert server.servers[0].sockets is not None
    port = server.servers[0].sockets[0].getsockname()[1]

    try:
        yield server, f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5.0)
        await key_store.close()


async def test_internal_app_is_reachable_on_a_real_second_port(
    running_internal_server: RunningInternalServer,
) -> None:
    _server, base_url = running_internal_server

    async with AsyncClient(base_url=base_url) as ac:
        response = await ac.post(
            "/internal/register-key",
            json={"user_id": "alice", "ai_api_key": "key-a"},
            headers={"X-StudyLife-Shared-Secret": TEST_SHARED_SECRET},
        )

    assert response.status_code == 200


async def test_internal_app_still_enforces_the_shared_secret_on_the_dedicated_port(
    running_internal_server: RunningInternalServer,
) -> None:
    """The port split doesn't relax the existing auth - see api/internal.py."""
    _server, base_url = running_internal_server

    async with AsyncClient(base_url=base_url) as ac:
        response = await ac.post(
            "/internal/register-key",
            json={"user_id": "alice", "ai_api_key": "key-a"},
            headers={"X-StudyLife-Shared-Secret": "wrong-secret"},
        )

    assert response.status_code == 401


async def test_internal_app_only_exposes_internal_routes(
    running_internal_server: RunningInternalServer,
) -> None:
    """The dedicated internal_app carries ONLY /internal/* - no /chat, /agent, /health,
    /metrics - so a NetworkPolicy scoped to this port can't reach anything else."""
    _server, base_url = running_internal_server

    async with AsyncClient(base_url=base_url) as ac:
        for path, method in (
            ("/chat", "post"),
            ("/agent", "post"),
            ("/health", "get"),
            ("/metrics", "get"),
        ):
            response = await getattr(ac, method)(path)
            assert response.status_code == 404, f"{method.upper()} {path} should not exist"


async def test_hitting_internal_routes_on_the_dedicated_port_does_not_log_the_deprecation_warning(
    running_internal_server: RunningInternalServer, caplog: LogCaptureFixture
) -> None:
    _server, base_url = running_internal_server

    with caplog.at_level(logging.WARNING):
        async with AsyncClient(base_url=base_url) as ac:
            await ac.post(
                "/internal/register-key",
                json={"user_id": "alice", "ai_api_key": "key-a"},
                headers={"X-StudyLife-Shared-Secret": TEST_SHARED_SECRET},
            )

    assert "Deprecated" not in caplog.text


async def test_serve_internal_app_logs_and_returns_instead_of_crashing_on_a_bind_failure(
    running_internal_server: RunningInternalServer, caplog: LogCaptureFixture
) -> None:
    """A port conflict must degrade to "the dedicated port never came up", not take down the
    whole process (see serve_internal_app's own docstring) - /internal/* would still be
    reachable on the main port regardless. `running_internal_server` is already bound to a real
    port; building a second server for that exact same port and starting it reproduces a
    genuine uvicorn.Server.startup() bind failure (OSError -> sys.exit internally), not a mock."""
    _server, base_url = running_internal_server
    port = int(base_url.rsplit(":", 1)[1])

    internal_app = create_internal_app()
    internal_app.state.registered_key_store = RegisteredKeyStore(
        ":memory:", TEST_AI_KEY_ENCRYPTION_KEY
    )
    await internal_app.state.registered_key_store.setup()
    colliding_server = build_internal_server(internal_app, Settings(internal_api_port=port))

    with caplog.at_level(logging.ERROR):
        await asyncio.wait_for(serve_internal_app(colliding_server), timeout=5.0)

    assert "Failed to start" in caplog.text
    # The original server, on its own port, is unaffected - proving the collision didn't take
    # the process (or even the other server) down with it.
    async with AsyncClient(base_url=base_url) as ac:
        response = await ac.post(
            "/internal/register-key",
            json={"user_id": "alice", "ai_api_key": "key-a"},
            headers={"X-StudyLife-Shared-Secret": TEST_SHARED_SECRET},
        )
    assert response.status_code == 200
    await internal_app.state.registered_key_store.close()


async def test_serve_internal_app_stops_gracefully_when_should_exit_is_set(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "studylife_ai.api.internal.get_settings",
        lambda: Settings(studylife_internal_api_secret=TEST_SHARED_SECRET),  # type: ignore[call-arg]
    )
    internal_app = create_internal_app()
    internal_app.state.registered_key_store = RegisteredKeyStore(
        ":memory:", TEST_AI_KEY_ENCRYPTION_KEY
    )
    await internal_app.state.registered_key_store.setup()

    server = build_internal_server(internal_app, Settings(internal_api_port=0))
    task = asyncio.create_task(serve_internal_app(server))
    await _wait_until_started(server)

    server.should_exit = True
    # main_loop() polls should_exit every 0.1s (see uvicorn.Server.main_loop) - this must
    # resolve quickly, not hang, for main.py's own lifespan shutdown to stay responsive.
    await asyncio.wait_for(task, timeout=2.0)

    assert task.done()
    assert task.exception() is None
    await internal_app.state.registered_key_store.close()


# --- Main app: /internal/* still served on the shared public port too, with a deprecation log
# (main.py's include_router(internal.router, dependencies=[Depends(log_deprecated_...)])). ---


async def test_hitting_internal_routes_on_the_main_app_logs_a_deprecation_warning(
    client: AsyncClient, caplog: LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="studylife_ai.internal_server"):
        response = await client.post(
            "/internal/register-key",
            json={"user_id": "alice", "ai_api_key": "key-a"},
            headers={"X-StudyLife-Shared-Secret": TEST_SHARED_SECRET},
        )

    assert response.status_code == 200
    assert "Deprecated" in caplog.text
    assert "/internal/register-key" in caplog.text


async def test_main_app_internal_routes_still_work_exactly_as_before_the_split(
    client: AsyncClient,
) -> None:
    """Existing functionality unchanged: the main app's own /internal/* behavior (already
    covered in depth by tests/test_internal.py) is untouched by the port split - this is just a
    sanity check that the extra dependency doesn't otherwise interfere."""
    response = await client.post(
        "/internal/register-key",
        json={"user_id": "alice", "ai_api_key": "key-a"},
        headers={"X-StudyLife-Shared-Secret": "wrong-secret"},
    )

    assert response.status_code == 401
