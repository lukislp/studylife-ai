"""Serves `api/internal.py`'s routes on a second, dedicated port (audit O6-ai, 2026-08-26 - see
docs/decisions.md "F15/O6-ai: chat model allowlist, metrics token gate, /internal port split").

`/internal/*` (register-key, revoke-key, enrich-capture) shares the public port with `/chat`,
`/agent`, and `/metrics` today - the only guards are the `X-StudyLife-Shared-Secret` bearer check
in `api/internal.py` and a k8s `NetworkPolicy` that, because it's scoped to the whole shared
port, necessarily also admits every other caller of anything else on that port (e.g. Prometheus,
which only needs `/metrics` - see `k8s/05-network-policies.yaml`'s `allow-prometheus-to-app`).
Splitting `/internal/*` onto its own port lets a `NetworkPolicy` scope ingress to just
StudyLife's own backend pods (`studylife-web`/`studylife-worker`), tighter than what the shared
port can express, without a second process/container/image/Deployment.

`create_internal_app()` builds a second, minimal FastAPI app carrying ONLY `internal.router` -
no `/chat`/`/agent`/`/metrics`/`/health`, so a `NetworkPolicy` scoped to this port can't reach
anything else even if `internal.router` itself ever grew a bug that widened what it exposes. It
has no lifespan of its own: `main.py`'s own `_lifespan` builds it, points its `.state` at the
SAME `qdrant_store`/`registered_key_store`/`agent_checkpointer` objects the main app already
built (not fresh copies - two independent SQLite connections to the same file, in the same
process, would be redundant, not safer), and runs it via `build_internal_server` +
`serve_internal_app` as a second `uvicorn.Server` task on the same event loop.

Transition period: `internal.router` is ALSO still included on the main app (see `main.py`),
with an extra dependency (`log_deprecated_main_port_access` below) that only that inclusion
gets - `include_router(router, dependencies=[...])` applies the given dependencies to just that
one inclusion, not to the router object itself, so `create_internal_app()`'s own inclusion above
never logs anything. This keeps StudyLife's backend working unchanged regardless of whether it
(or this service's own k8s manifests) have switched over to `INTERNAL_API_PORT` yet, in either
order. Drop the second `include_router` call on the main app (and this module's deprecation
dependency) once that warning has gone quiet in production logs for a full StudyLife release
cycle.
"""

import logging

import uvicorn
from fastapi import FastAPI, Request

from studylife_ai.api import internal
from studylife_ai.config import Settings

logger = logging.getLogger(__name__)


def create_internal_app() -> FastAPI:
    """A second, minimal ASGI app carrying only `/internal/*` - see module docstring. Callers
    (main.py's `_lifespan`) must set `.state.qdrant_store` / `.state.registered_key_store` /
    `.state.agent_checkpointer` on the returned app before serving it - `internal.router`'s
    handlers read those off `http_request.app.state`, same as on the main app."""
    internal_app = FastAPI(title="StudyLife AI (internal)")
    internal_app.include_router(internal.router)
    return internal_app


def build_internal_server(internal_app: FastAPI, settings: Settings) -> uvicorn.Server:
    """Builds (but does not start) the `uvicorn.Server` that serves `internal_app` on
    `settings.internal_api_port`. Split out from `serve_internal_app()` below so a caller - both
    `main.py`'s lifespan and this module's own tests - can hold onto the `Server` instance: to
    read back the actual bound port (useful with `internal_api_port=0` in tests), and to request
    a graceful stop via `server.should_exit = True` rather than only a bare task cancellation."""
    config = uvicorn.Config(
        internal_app,
        # Bound inside a k8s Pod (or the docker-compose network) - same all-interfaces bind as
        # the main app's own `CMD` in the Dockerfile; there is no narrower interface to bind to
        # in either environment.
        host="0.0.0.0",
        port=settings.internal_api_port,
        log_level=settings.log_level.lower(),
    )
    if not config.loaded:
        config.load()
    server = uvicorn.Server(config)
    # `Server.__init__` doesn't build this - normally done inside `uvicorn.Server._serve()`,
    # which `serve_internal_app()` below deliberately never calls into (see its own docstring
    # for why). `config.lifespan_class(config)` is the exact same expression `_serve()` itself
    # uses, so this is equivalent to what a normal `uvicorn studylife_ai.main:app` invocation
    # would do for the main app - just reached without going through `serve()`.
    server.lifespan = config.lifespan_class(config)
    return server


async def serve_internal_app(server: uvicorn.Server) -> None:
    """Runs `server` (see `build_internal_server`) until `server.should_exit` is set (main.py's
    lifespan does this on shutdown) or this task is cancelled.

    Deliberately calls `startup()` / `main_loop()` / `shutdown()` directly instead of
    `uvicorn.Server.serve()` (or `.run()`, which wraps `serve()` in its own `asyncio.run()`):
    `serve()` wraps those three in `with self.capture_signals():`, which installs OS
    SIGINT/SIGTERM handlers via `signal.signal()` - process-wide, not per-`Server`-instance. The
    OUTER uvicorn process running the MAIN app (this coroutine runs as a background task on that
    SAME process/event loop - see main.py's `_lifespan` - not a second process) already installs
    its own via that exact mechanism. A second `capture_signals()` call here would silently
    steal SIGTERM/SIGINT away from the main app's own graceful-shutdown path - which matters in
    practice, since this Deployment's `Recreate` strategy (k8s/04-app.yaml) relies on the main
    app shutting down cleanly on `SIGTERM` before the replacement Pod starts (both use the same
    RWO-mounted SQLite files). Skipping `serve()`'s wrapper avoids that collision entirely -
    ordinary asyncio task cancellation, or `server.should_exit = True`, both still exit
    `main_loop()` cleanly and reach `shutdown()` via the `finally` below; verified live in this
    module's own tests (`tests/test_internal_server.py`), not just assumed.

    A bind failure (e.g. `settings.internal_api_port` already in use) is caught and logged, not
    propagated: `Server.startup()` handles a socket-level `OSError` itself by calling
    `sys.exit()` - since this coroutine runs as a background `asyncio.Task` (see main.py's
    `_lifespan`), an uncaught `SystemExit` there would otherwise be free to crash the WHOLE
    process, taking `/chat`/`/agent` on the main port down with it over a problem confined to
    the dedicated internal port - which is still serving `/internal/*` regardless (see
    `main.py::create_app`'s transition-period `include_router`). Degrading to "the dedicated
    port never came up, main port still works" is strictly better than a hard crash here.
    """
    try:
        await server.startup()
    except SystemExit:
        logger.error(
            "Failed to start the dedicated internal-api server on port %d - /internal/* stays "
            "reachable on the main port for the rest of this process (see main.py); this is "
            "not retried until the next restart.",
            server.config.port,
        )
        return
    try:
        await server.main_loop()
    finally:
        await server.shutdown()


def log_deprecated_main_port_access(request: Request) -> None:
    """FastAPI dependency added ONLY where the main app includes `internal.router` (see
    `main.py`) - `create_internal_app()`'s own inclusion above never gets this, so a request
    served on the dedicated internal port never logs it. Logs every hit, not just once: an
    operator watching for "has StudyLife's backend actually switched over to
    INTERNAL_API_PORT yet" wants to see this line stop appearing entirely, not merely appear
    once at process start."""
    logger.warning(
        "Deprecated: %s %s served on the shared public port - StudyLife's backend should call "
        "the dedicated internal port instead (see INTERNAL_API_PORT in README.md).",
        request.method,
        request.url.path,
    )
