from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from studylife_ai.main import app


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    # Runs the app's lifespan (startup/shutdown) — ASGITransport alone doesn't,
    # and /chat needs app.state.qdrant_store, which lifespan sets up.
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
