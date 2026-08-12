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

    # StudyLife REST API. One shared instance URL for all users (see
    # docs/decisions.md "M4.5 Multi-user support") — no default, ingestion
    # and /agent fail loudly if unset rather than silently pointing nowhere.
    studylife_api_base_url: str | None = None
    # Path to a CA certificate bundle to trust for `studylife_api_base_url`, in addition to the
    # system default trust store. Unset by default (local dev: StudyLife runs on a normal
    # publicly-trusted-or-plain-HTTP localhost setup). Set in the k3s deployment, where
    # studylife-ai reaches StudyLife over cluster-internal HTTPS signed by StudyLife's own
    # private cert-manager CA (see k8s/04-app.yaml) - found live: httpx's default trust store
    # (certifi) has no reason to know a private, cluster-only CA.
    studylife_ca_cert_path: str | None = None
    # Shared secret StudyLife signs per-request proxy tokens with (see
    # api/identity.py, docs/decisions.md "M4.5 Multi-user support" - "Auth
    # flow, take two"). Must match the same value configured on the
    # StudyLife side exactly - a mismatch makes every /chat and /agent
    # request fail with 401. Also authenticates POST /internal/register-key
    # and /internal/revoke-key (a simpler constant-time bearer-secret check,
    # not the signed-token scheme - those aren't per-user requests).
    studylife_shared_secret: str | None = None
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
    # How often the in-process background loop re-runs sync_all() (see
    # ingestion/scheduler.py, docs/decisions.md "Periodic ingestion sync").
    # Cheap to run often: sync_content_type's per-entity fingerprint diff
    # already skips anything unchanged, so a 60s full-catalog pass only ever
    # pays for what actually changed since the last tick.
    ingestion_sync_interval_seconds: int = 60
    # Half-width (in days) of the "near today" session date-window retrieve_with_rerank() fetches
    # via a real Qdrant DatetimeRange filter (see docs/decisions.md "Structured session dates") -
    # comfortably covers today/tomorrow/day-after-tomorrow/yesterday/day-before-yesterday and
    # this/next/last week without handing the reranker the full session history to sort through
    # in free text. Doesn't limit topic-based session questions ("what did we cover in Analysis
    # last year") - those still go through the separate, unwindowed vector-search fallback.
    session_window_days: int = 14

    # Cap on how many session chunks _fetch_session_window() keeps from that date-window scroll
    # (see docs/decisions.md "Session window capacity"). Deliberately its own setting, not the
    # shared per-content-type quota (rerank_candidate_k // 4) - a busy day (several sessions) was
    # able to fill that shared quota entirely and starve an equally-near day, confirmed live
    # 2026-08-12 ("vorgestern" lost out to "gestern"). Set well above what the shared quota would
    # give (was effectively 5) so several busy nearby days can all fit; only affects the window
    # leg's candidate count, not the topic-vector fallback or the other content types' quotas.
    session_window_top_k: int = 20

    # Optional NL date-range resolution for session retrieval (see docs/decisions.md "NL
    # date-range resolution" - the escalation path (2) named and deferred in "Structured session
    # dates"). Unset by default (opt-in, same convention as rerank_model/eval_judge_model): when
    # unset, session retrieval keeps its current fixed +-session_window_days window unchanged for
    # every query. When set, a dedicated LLM call (rag/date_parse.py) tries to resolve the
    # query's own date/date-range expression (e.g. "letzte Woche") before retrieval; when it
    # finds one, that exact range replaces the fixed window for THAT query only - queries with no
    # date expression, or when this is unset, are unaffected. Deliberately independent of
    # llm_model/rerank_model, same pattern as every other model concern here.
    date_parse_model: str | None = None

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
    # whether rerank_model is set. Raised from 5 to 8 (see docs/decisions.md
    # "Retrieval top-k raised") - even with the quota above, all 4 content
    # types still get merged into one shared final cut, and a fixed 5-slot
    # cut for 4 competing types was too tight to survive the reranker's
    # normal (non-deterministic) run-to-run variance, confirmed via a real
    # CI eval regression.
    retrieval_top_k: int = 8

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
    # Fixed Qdrant partition the eval pipeline reads/writes under (fixture
    # seeding + real dev-corpus runs) - decoupled from any real StudyLife
    # user id, since eval doesn't go through the per-request identity flow
    # /chat and /agent use.
    eval_user_id: str = "eval-user"

    # M4 agent (see docs/decisions.md "M4 agent stack"): SQLite file backing
    # the LangGraph checkpointer that holds a proposed-but-not-yet-confirmed
    # write action's paused state. Persists across a service restart between
    # /agent (propose) and /agent/confirm - an in-memory checkpointer would
    # lose a pending action on restart, which a write-confirmation flow
    # specifically must not do.
    agent_checkpoint_db_path: str = "agent_checkpoints.db"

    # M4.5 (see docs/decisions.md "M4.5 Multi-user support"): SQLite file
    # backing the per-user AiApiKey registry - populated by StudyLife's
    # POST /internal/register-key/revoke-key calls, read by /agent (to build
    # a real StudyLifeClient for the calling user) and by ingestion's
    # sync_all() (which syncs every registered user, replacing the earlier
    # manually-maintained INGESTION_USERS list).
    registered_keys_db_path: str = "registered_keys.db"

    # Rate limiting (see docs/decisions.md "Rate limiting"): a fixed-window counter per
    # resolved user_id, guarding /chat and /agent /agent-confirm - the endpoints that incur
    # real LLM cost. Defends against a leaked token or a client-side bug (e.g. a retry loop)
    # running up cost or starving the service, not against many distinct attackers - this is a
    # personal-scale, single-replica deployment (see k8s/04-app.yaml's `Recreate` strategy), not
    # a public API, so an in-memory counter (no Redis/shared state) is enough.
    rate_limit_requests: int = 20
    rate_limit_window_seconds: int = 60


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (env is read once per process)."""
    return Settings()
