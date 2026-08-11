"""FastAPI application entrypoint."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from studylife_ai.api import agent, chat, health, internal
from studylife_ai.config import get_settings
from studylife_ai.ingestion.qdrant_store import QdrantStore
from studylife_ai.studylife.registered_keys import RegisteredKeyStore


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    # One QdrantStore for the app's lifetime instead of one per /chat request —
    # avoids opening/closing a new HTTP client (and its connection pool) on every call.
    app.state.qdrant_store = QdrantStore(
        url=settings.qdrant_url, collection=settings.qdrant_collection
    )

    registered_key_store = RegisteredKeyStore(settings.registered_keys_db_path)
    await registered_key_store.setup()
    app.state.registered_key_store = registered_key_store

    # The agent graph itself is no longer built here - it's rebuilt per
    # /agent request with the calling user's own StudyLifeClient (see
    # docs/decisions.md "M4.5 Multi-user support" - "Agent graph: rebuilt
    # per request"). Only the checkpointer is app-lifetime: it's not
    # user-specific (partitioned by thread_id, which itself embeds the
    # owning user_id), and a pending write action must survive a service
    # restart between propose and confirm regardless of whether any
    # StudyLife credentials are configured at all.
    async with AsyncSqliteSaver.from_conn_string(settings.agent_checkpoint_db_path) as checkpointer:
        await checkpointer.setup()
        app.state.agent_checkpointer = checkpointer

        try:
            yield
        finally:
            await app.state.qdrant_store.close()
            await registered_key_store.close()


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    app = FastAPI(title=settings.app_name, lifespan=_lifespan)
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(agent.router)
    app.include_router(internal.router)
    return app


app = create_app()
