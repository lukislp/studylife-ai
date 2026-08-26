"""FastAPI application entrypoint."""

import asyncio
import contextlib
import hmac
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from prometheus_fastapi_instrumentator import Instrumentator

from studylife_ai.agent.checkpoint_cleanup import run_periodic_checkpoint_cleanup
from studylife_ai.api import agent, chat, health, internal
from studylife_ai.config import get_settings
from studylife_ai.ingestion.qdrant_store import QdrantStore
from studylife_ai.ingestion.scheduler import run_periodic_sync
from studylife_ai.internal_server import (
    build_internal_server,
    create_internal_app,
    log_deprecated_main_port_access,
    serve_internal_app,
)
from studylife_ai.llm.logging import configure_llm_usage_logging
from studylife_ai.studylife.registered_keys import RegisteredKeyStore


def _require_metrics_token(request: Request) -> None:
    """FastAPI dependency guarding GET /metrics (audit O6-ai, 2026-08-26): unauthenticated by
    default (`Settings.metrics_token` unset) - a genuine no-op, identical to the endpoint's
    behavior before this existed. Once an operator sets `METRICS_TOKEN`, every scrape must send
    a matching `Authorization: Bearer <token>` header (constant-time compare) or gets 401 - see
    README.md "Observability" for the corresponding Prometheus scrape-config change."""
    settings = get_settings()
    if not settings.metrics_token:
        return
    provided = request.headers.get("Authorization", "")
    if not hmac.compare_digest(provided, f"Bearer {settings.metrics_token}"):
        raise HTTPException(status_code=401, detail="Invalid or missing metrics bearer token.")


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

        # Audit O6-ai (2026-08-26, see internal_server.py): /internal/* also served on its own
        # port, via a second uvicorn.Server task on this same event loop - sharing this app's
        # own qdrant_store/registered_key_store/agent_checkpointer (all built above) rather than
        # opening a second, redundant set of connections to the same backing stores.
        internal_app = create_internal_app()
        internal_app.state.qdrant_store = app.state.qdrant_store
        internal_app.state.registered_key_store = registered_key_store
        internal_app.state.agent_checkpointer = checkpointer
        internal_server = build_internal_server(internal_app, settings)
        internal_server_task = asyncio.create_task(serve_internal_app(internal_server))

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
            # Graceful, not a bare cancel: lets serve_internal_app's own shutdown() run to
            # completion (closing the listening socket cleanly) instead of being cut off
            # mid-await - main_loop() polls should_exit every 0.1s, so this resolves quickly.
            internal_server.should_exit = True
            with contextlib.suppress(asyncio.CancelledError):
                await internal_server_task
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
    # Audit O6-ai (2026-08-26): still served here too, for one transition release (see
    # internal_server.py's module docstring for the full reasoning) - `dependencies=` on
    # include_router only applies to THIS inclusion, not to internal.router itself, so
    # internal_server.create_internal_app()'s own inclusion of the same router (on the
    # dedicated internal port) never logs this warning.
    app.include_router(internal.router, dependencies=[Depends(log_deprecated_main_port_access)])
    # HTTP request rate/latency/status per endpoint, exposed at /metrics for Prometheus to
    # scrape (see docs/decisions.md "Metrics dashboard") - on top of the hand-registered LLM
    # cost/latency/token counters in llm/metrics.py, which cover the actual model-call cost
    # this auto-instrumentation can't see. Optional bearer-token gate (audit O6-ai) via
    # `_require_metrics_token` - off by default, see its own docstring.
    Instrumentator().instrument(app).expose(app, dependencies=[Depends(_require_metrics_token)])
    return app


app = create_app()
