"""Optional HTTPS listeners for the StudyLife -> studylife-ai hop.

StudyLife's backend calls this service with a shared secret or a short-lived HMAC token, so the
hop has to be encrypted (OpenSSF `crypto_used_network`, StudyLife's docs/SECURITY-ASSURANCE.md
section 8.3). StudyLife validates the certificate against its own internal CA, host name
included (its `AddonTls:*` settings), so the certificate here is the cert-manager one issued by
`studylife-internal-ca-issuer` (k8s/04-app.yaml).

When `TLS_CERT_FILE`/`TLS_KEY_FILE` are set, `main.py`'s lifespan starts two more
`uvicorn.Server` instances on the same event loop, next to the existing plain ones:

- the main app on `TLS_PORT` (default 8443): /chat, /agent, /health, /metrics
- the internal app on `INTERNAL_API_TLS_PORT` (default 8444): /internal/*

The plain ports (8000/8001) keep running for the kubelet probes and Prometheus, so the TLS
listeners can be rolled out before StudyLife switches its URLs (see README.md "TLS").

Failure is loud on purpose: a certificate that cannot be loaded or a port that cannot be bound
raises during startup, which makes uvicorn exit instead of running without the TLS listener that
StudyLife expects.

uvicorn never reloads certificates on its own. `watch_certificates` polls the two files and, when
cert-manager has rewritten them, loads the new pair into the live `SSLContext`s, so every new
handshake uses the renewed certificate without a restart.
"""

import asyncio
import contextlib
import logging
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TlsFiles:
    """The PEM certificate chain and private key a TLS listener serves."""

    cert_file: Path
    key_file: Path


def tls_files(cert_file: str | None, key_file: str | None) -> TlsFiles | None:
    """`None` when TLS is not configured (both unset). Raises when only one is set or a file is
    missing, so a half-configured TLS setup stops the start instead of serving plain HTTP."""
    if not cert_file and not key_file:
        return None
    if not cert_file or not key_file:
        raise ValueError("TLS_CERT_FILE and TLS_KEY_FILE must be set together")
    files = TlsFiles(Path(cert_file), Path(key_file))
    for path in (files.cert_file, files.key_file):
        if not path.is_file():
            raise ValueError(f"TLS file {path} does not exist")
    return files


def build_tls_server(app: Any, port: int, files: TlsFiles, *, log_level: str) -> uvicorn.Server:
    """A `uvicorn.Server` that serves `app` over HTTPS on `port` (0 = OS-assigned, for tests).

    `lifespan="off"`: the app's lifespan already runs once on the plain server that started the
    process; a second run would build a second set of stores and start this server again.
    TLS 1.2 is the minimum, the rest are Python's `ssl` defaults (no compression, no
    renegotiation, secure cipher list)."""
    config = uvicorn.Config(
        app,
        # Bound inside the pod/container network, same as the main app's CMD in the Dockerfile.
        host="0.0.0.0",
        port=port,
        log_level=log_level.lower(),
        lifespan="off",
        ssl_certfile=str(files.cert_file),
        ssl_keyfile=str(files.key_file),
    )
    config.load()
    assert config.ssl is not None  # set by load() because ssl_certfile is given
    config.ssl.minimum_version = ssl.TLSVersion.TLSv1_2
    server = uvicorn.Server(config)
    server.lifespan = config.lifespan_class(config)
    return server


async def start_tls_server(server: uvicorn.Server) -> asyncio.Task[None]:
    """Binds `server` now (raising on failure) and returns the task that serves it until
    `server.should_exit` is set. Like `internal_server.serve_internal_app`, this skips
    `Server.serve()` so no second set of process-wide signal handlers is installed."""
    try:
        await server.startup()
    except SystemExit as exc:
        raise RuntimeError(
            f"Could not start the HTTPS listener on port {server.config.port}"
        ) from exc
    if server.should_exit:
        raise RuntimeError(f"Could not start the HTTPS listener on port {server.config.port}")

    async def _run() -> None:
        try:
            await server.main_loop()
        finally:
            await server.shutdown()

    return asyncio.create_task(_run())


async def stop_tls_servers(servers: list[uvicorn.Server], tasks: list[asyncio.Task[None]]) -> None:
    """Graceful stop, same as the internal plain server in main.py."""
    for server in servers:
        server.should_exit = True
    for task in tasks:
        with contextlib.suppress(asyncio.CancelledError):
            await task


def bound_port(server: uvicorn.Server) -> int:
    """The port a started server actually listens on (useful with port 0)."""
    sockets = server.servers[0].sockets
    return int(sockets[0].getsockname()[1])


def _fingerprint(files: TlsFiles) -> tuple[tuple[int, int, int], ...]:
    # A Kubernetes Secret volume swaps a symlink on update; stat() follows it, so inode, size or
    # mtime change when cert-manager writes the renewed pair.
    result = []
    for path in (files.cert_file, files.key_file):
        st = path.stat()
        result.append((st.st_ino, st.st_size, st.st_mtime_ns))
    return tuple(result)


def reload_certificates(servers: list[uvicorn.Server], files: TlsFiles) -> bool:
    """Loads the current certificate/key pair into every server's `SSLContext`. Returns False
    (and keeps serving the old pair) when the files cannot be loaded, e.g. mid-update."""
    try:
        for server in servers:
            assert server.config.ssl is not None
            server.config.ssl.load_cert_chain(str(files.cert_file), str(files.key_file))
    except (OSError, ssl.SSLError) as exc:
        logger.warning("TLS certificate reload failed, keeping the previous certificate: %s", exc)
        return False
    logger.info("TLS certificate reloaded from %s", files.cert_file)
    return True


async def watch_certificates(
    servers: list[uvicorn.Server], files: TlsFiles, interval_seconds: float
) -> None:
    """Runs until cancelled: reloads the certificate whenever the files change."""
    last = _fingerprint(files)
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            current = _fingerprint(files)
        except OSError as exc:
            logger.warning("TLS certificate files not readable: %s", exc)
            continue
        if current != last and reload_certificates(servers, files):
            last = current
