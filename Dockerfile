# uv is taken from Astral's own image (pinned by tag + digest, both kept current by
# Dependabot's docker updates) instead of an unpinned `pip install uv`.
FROM ghcr.io/astral-sh/uv:0.12.13@sha256:b485bd65cc2cf1c9a93b3554012c9c3778cf7b1b5fd3d3096ce9e1226c97e1e6 AS uv

FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

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
