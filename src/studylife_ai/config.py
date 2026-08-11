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
    # Lookback window for GET /api/sessions/history (see docs/decisions.md
    # "Ingestion scope expansion"). ~5 years, measured from "now" every sync
    # run - it IS a rolling window: a session older than this eventually
    # falls out of it and gets deleted from the Qdrant index (indistinguishable
    # from a session that was actually deleted in StudyLife), not just
    # excluded from a one-time fetch. Acceptable for a personal RAG index
    # (5-year-old sessions have little retrieval value anyway), but worth
    # knowing before ever lowering this value, which purges the difference
    # immediately. Not a real deployment knob otherwise (onlyCompleted stays
    # hardcoded False in sync.py), but config-driven like every other
    # pipeline tunable here for consistency and testability.
    studylife_session_history_days: int = 1825

    # LiteLLM embedding model identifier, same provider-agnostic convention
    # as llm_model. Defaults to a local Ollama embedding model.
    embedding_model: str = "ollama/nomic-embed-text"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "studylife_notes"

    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 75

    # Retrieval (see docs/decisions.md "Retrieval design" and "Retrieval
    # quality: reranking + per-content-type quota"): final number of chunks
    # handed to the LLM. retrieve_with_rerank() (rag/retrieval.py) always
    # fetches an even per-content-type candidate quota first (see
    # rerank_candidate_k) - a popular course's many session chunks can no
    # longer crowd out a note just by outnumbering it, independent of
    # whether rerank_model is set.
    retrieval_top_k: int = 5

    # Target candidate pool retrieve_with_rerank() fetches before narrowing
    # down to retrieval_top_k, split evenly across the 4 content types (see
    # retrieval_top_k above) - always applied, not just when reranking. A
    # target, not an exact count: each type fetches at least 1 candidate, so
    # values below 4 fetch more than configured (4 total); values not evenly
    # divisible by 4 are floored per type, so the actual pool can be
    # slightly under the configured number.
    rerank_candidate_k: int = 20

    # Optional LLM-based reranking of that candidate pool (see
    # docs/decisions.md "Retrieval quality: reranking + per-content-type
    # quota"): unset by default - without it, the candidate pool is just
    # sorted by vector-similarity score. Deliberately independent of
    # llm_model (a small/fast model suffices for scoring, doesn't need to
    # match the answer model).
    rerank_model: str | None = None

    # RAGAS eval judge (M3, see docs/decisions.md "Eval design"): a LiteLLM
    # model string, deliberately independent of llm_model. No default -
    # running the eval should fail loudly rather than silently falling back
    # to the answer model as its own judge (rejected as unreliable during
    # that decision). Set to "openai/gpt-4o-mini" in .env.
    eval_judge_model: str | None = None

    # M4 agent (see docs/decisions.md "M4 agent stack"): SQLite file backing
    # the LangGraph checkpointer that holds a proposed-but-not-yet-confirmed
    # write action's paused state. Persists across a service restart between
    # /agent (propose) and /agent/confirm - an in-memory checkpointer would
    # lose a pending action on restart, which a write-confirmation flow
    # specifically must not do.
    agent_checkpoint_db_path: str = "agent_checkpoints.db"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (env is read once per process)."""
    return Settings()
