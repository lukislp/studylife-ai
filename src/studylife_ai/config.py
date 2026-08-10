"""Application configuration, loaded from environment variables / .env."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the StudyLife AI service.

    All values can be overridden via environment variables or a local
    `.env` file (see `.env.example`). Provider-specific LLM credentials
    (e.g. `OPENAI_API_KEY`) are read directly from the environment by
    LiteLLM and are intentionally not modeled here.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "StudyLife AI"
    environment: Literal["local", "staging", "production"] = "local"
    log_level: str = "INFO"

    # LiteLLM model identifier, e.g. "openai/gpt-4o-mini" or "ollama/llama3.2".
    # Defaults to a local Ollama model so the service runs without an API key
    # out of the box via `docker compose up`.
    llm_model: str = "ollama/llama3.2"
    llm_api_base: str | None = "http://localhost:11434"
    llm_request_timeout_seconds: float = 60.0

    # StudyLife REST API (ingestion source, M2). No default base URL/key —
    # ingestion fails loudly if unset rather than silently pointing nowhere.
    studylife_api_base_url: str | None = None
    studylife_api_key: str | None = None
    # Label for the single StudyLife user this instance ingests for. Not a
    # StudyLife-internal ID (the API doesn't expose one) — just a stable tag
    # stored on every Qdrant chunk so a future multi-user setup doesn't need
    # to re-ingest everything to add user scoping.
    studylife_user_id: str = "primary"

    # LiteLLM embedding model identifier, same provider-agnostic convention
    # as llm_model. Defaults to a local Ollama embedding model.
    embedding_model: str = "ollama/nomic-embed-text"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "studylife_notes"

    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 75


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (env is read once per process)."""
    return Settings()
