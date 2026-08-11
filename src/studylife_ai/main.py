"""FastAPI application entrypoint."""

import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from studylife_ai.agent.graph import build_agent
from studylife_ai.agent.tools import build_tools
from studylife_ai.api import agent, chat, health
from studylife_ai.config import get_settings
from studylife_ai.ingestion.qdrant_store import QdrantStore
from studylife_ai.studylife.client import StudyLifeClient


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    # One QdrantStore for the app's lifetime instead of one per /chat request —
    # avoids opening/closing a new HTTP client (and its connection pool) on every call.
    app.state.qdrant_store = QdrantStore(
        url=settings.qdrant_url, collection=settings.qdrant_collection
    )

    async with AsyncExitStack() as stack:
        # Agent (M4) needs real StudyLife credentials to do anything - unlike
        # ingestion (a separate, optional CLI step), /agent is always
        # registered, so without credentials it's built as None here and the
        # endpoint itself returns a clear error, rather than failing the
        # whole app's startup (which would also break /chat and /health for
        # anyone not using the agent).
        app.state.agent = None
        app.state.agent_checkpointer = None
        if settings.studylife_api_base_url and settings.studylife_api_key:
            studylife_client = StudyLifeClient(
                base_url=settings.studylife_api_base_url, api_key=settings.studylife_api_key
            )
            stack.push_async_callback(studylife_client.aclose)

            checkpointer = await stack.enter_async_context(
                AsyncSqliteSaver.from_conn_string(settings.agent_checkpoint_db_path)
            )
            await checkpointer.setup()
            # Kept alongside the compiled agent, not just inside it - api/agent.py
            # needs direct access to delete a thread's checkpoint after a
            # failed tool run (see api/agent.py's _invoke_and_handle_failure).
            app.state.agent_checkpointer = checkpointer

            tools = build_tools(
                studylife=studylife_client, store=app.state.qdrant_store, settings=settings
            )
            app.state.agent = build_agent(tools=tools, checkpointer=checkpointer, settings=settings)

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
    app.include_router(agent.router)
    return app


app = create_app()
