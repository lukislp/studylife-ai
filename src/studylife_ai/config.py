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

    # LiteLLM model identifier, e.g. "openai/gpt-4o-mini" or "ollama/llama3.1".
    # Defaults to a local Ollama model so the service runs without an API key
    # out of the box via `docker compose up`.
    llm_model: str = "ollama/llama3.1"
    llm_api_base: str | None = "http://localhost:11434"
    llm_request_timeout_seconds: float = 60.0


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (env is read once per process)."""
    return Settings()
