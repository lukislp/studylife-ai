"""FastAPI application entrypoint."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from studylife_ai.api import chat, health
from studylife_ai.config import get_settings
from studylife_ai.ingestion.qdrant_store import QdrantStore


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    # One QdrantStore for the app's lifetime instead of one per /chat request —
    # avoids opening/closing a new HTTP client (and its connection pool) on every call.
    app.state.qdrant_store = QdrantStore(
        url=settings.qdrant_url, collection=settings.qdrant_collection
    )
    try:
        yield
    finally:
        await app.state.qdrant_store.close()


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    app = FastAPI(title=settings.app_name, lifespan=_lifespan)
    app.include_router(health.router)
    app.include_router(chat.router)
    return app


app = create_app()
