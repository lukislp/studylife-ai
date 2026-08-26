"""FastAPI application entrypoint."""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from prometheus_fastapi_instrumentator import Instrumentator

from studylife_ai.agent.checkpoint_cleanup import run_periodic_checkpoint_cleanup
from studylife_ai.api import agent, chat, health, internal
from studylife_ai.config import get_settings
from studylife_ai.ingestion.qdrant_store import QdrantStore
from studylife_ai.ingestion.scheduler import run_periodic_sync
from studylife_ai.llm.logging import configure_llm_usage_logging
from studylife_ai.studylife.registered_keys import RegisteredKeyStore


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    # One QdrantStore for the app's lifetime instead of one per /chat request —
    # avoids opening/closing a new HTTP client (and its connection pool) on every call.
    app.state.qdrant_store = QdrantStore(
        url=settings.qdrant_url, collection=settings.qdrant_collection
    )

    registered_key_store = RegisteredKeyStore(
        settings.registered_keys_db_path, settings.ai_key_encryption_key
    )
    await registered_key_store.setup()
    app.state.registered_key_store = registered_key_store

    # Keeps Qdrant in step with StudyLife's own data on an ongoing basis (see
    # docs/decisions.md "Periodic ingestion sync") - only started when
    # ingestion is actually configured, same guard as the register-key
    # auto-sync in api/internal.py, so tests/CI without STUDYLIFE_API_BASE_URL
    # never spin up a loop that would just log "not configured" every tick.
    sync_task: asyncio.Task[None] | None = None
    if settings.studylife_api_base_url:
        sync_task = asyncio.create_task(run_periodic_sync(settings))

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

        # Sweeps stale (completed or never-confirmed) agent-checkpoint threads (audit A13/F14
        # rest, see agent/checkpoint_cleanup.py) - unlike the ingestion sync loop above, this
        # always runs: checkpoints accumulate regardless of whether STUDYLIFE_API_BASE_URL is
        # configured at all.
        cleanup_task = asyncio.create_task(run_periodic_checkpoint_cleanup(settings, checkpointer))

        try:
            yield
        finally:
            if sync_task is not None:
                sync_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sync_task
            cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cleanup_task
            await app.state.qdrant_store.close()
            await registered_key_store.close()


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    configure_llm_usage_logging()

    app = FastAPI(title=settings.app_name, lifespan=_lifespan)
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(agent.router)
    app.include_router(internal.router)
    # HTTP request rate/latency/status per endpoint, exposed at /metrics for Prometheus to
    # scrape (see docs/decisions.md "Metrics dashboard") - on top of the hand-registered LLM
    # cost/latency/token counters in llm/metrics.py, which cover the actual model-call cost
    # this auto-instrumentation can't see.
    Instrumentator().instrument(app).expose(app)
    return app


app = create_app()
