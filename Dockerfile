# uv is taken from Astral's own image (pinned by tag + digest, both kept current by
# Dependabot's docker updates) instead of an unpinned `pip install uv`.
FROM ghcr.io/astral-sh/uv:0.13.3@sha256:bee0a2a458fd956c5df7b8b4c5fed2257b0af34866e3da7b51db0be3acb375d1 AS uv

FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Pull in Debian's security updates on every build: the digest-pinned base image lags behind
# the security archive (three fixed CRITICAL perl-base CVEs - CVE-2026-13221, CVE-2026-42496,
# CVE-2026-8376 - would block the release chain's Trivy gate) and a rebuild is cheaper than
# waiting for the next python:3.12-slim digest. HTTPS mirror because plain-http port 80 is
# blocked on some build hosts.
RUN sed -i 's#http://deb.debian.org#https://deb.debian.org#' /etc/apt/sources.list.d/debian.sources     && apt-get update     && apt-get -y --no-install-recommends upgrade     && apt-get clean     && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies first so this layer is cached as long as
# pyproject.toml / uv.lock don't change (source changes shouldn't
# trigger a full dependency reinstall).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# /app/data must exist (and be owned by appuser) before the named volume
# mounts over it - otherwise Docker auto-creates the mount point as root,
# and the non-root appuser below can't open its sqlite files there.
RUN mkdir -p /app/data && useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "studylife_ai.main:app", "--host", "0.0.0.0", "--port", "8000"]
