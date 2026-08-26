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

    # Audit F15 (2026-08-26): ChatRequest.model (schemas/chat.py) lets a caller override
    # llm_model per-request - the server, not the caller, pays for whatever model gets named,
    # so an unrestricted override is a real cost-control gap even though nothing deployed
    # currently sends it (the Blazor chat client never sets it, and /agent never accepted the
    # equivalent - see schemas/agent.py). Comma-separated list of ADDITIONALLY allowed LiteLLM
    # model strings; llm_model itself is always implicitly allowed regardless of this setting,
    # so the default here (empty) means exactly "only the configured default model" - the
    # tightest useful default, and a no-op for every existing caller (none of which ever set
    # `model` in the first place). A request naming anything outside this set gets a 400 before
    # any LLM call is made. Kept as its own setting (not folded into llm_model) so the knob
    # itself - and ChatRequest.model - can stay for legitimate forward compat (e.g. a future
    # per-user model picker in the Blazor UI) without reopening the unrestricted version of this
    # gap. See docs/decisions.md "F15/O6-ai: chat model allowlist, metrics token gate, /internal
    # port split".
    allowed_chat_models: str = ""
    llm_api_base: str | None = "http://localhost:11434"
    llm_request_timeout_seconds: float = 60.0
    # For reasoning models only (e.g. OpenAI's gpt-5 family) - "minimal"/"low"/"medium"/"high".
    # Unset (the default) omits the parameter entirely, which is correct for non-reasoning models
    # (ollama/llama3.2, gpt-4o, ...) - passing it there would just be ignored/rejected by most
    # providers. Found live (2026-08-12): a reasoning model with no reasoning_effort set spends
    # real, billed completion tokens "thinking" before any visible output - a trivial one-word
    # reply burned 64 hidden reasoning tokens - "minimal" eliminates that overhead entirely.
    llm_reasoning_effort: str | None = None

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
    # LEGACY, audit A5 (2026-08-26): the original single shared secret, used for BOTH
    # verifying per-request proxy tokens (api/identity.py) AND authenticating
    # /internal/register-key /revoke-key /enrich-capture (api/internal.py) - split into
    # `studylife_token_signing_secret` / `studylife_internal_api_secret` below because anyone
    # holding this one value could mint a proxy token for ANY user_id *and* administer the
    # registry, with no key-id to support rotation without a simultaneous-redeploy 401 window.
    # Kept as a fallback ONLY: while either new setting is unset, this is used in its place (a
    # one-time deprecation warning is logged) so StudyLife's backend and studylife-ai can
    # deploy the split independently, in either order - see docs/decisions.md "Split the
    # shared secret (audit A5)". Also still accepted, verbatim, as a legacy 3-part
    # (un-keyed) proxy-token format while it's configured, regardless of whether the new
    # signing secret is also set - same rollout-order reasoning.
    studylife_shared_secret: str | None = None
    # Verifies per-request proxy tokens StudyLife's backend signs (see api/identity.py,
    # docs/decisions.md "M4.5 Multi-user support" - "Auth flow, take two", and "Split the
    # shared secret (audit A5)"). One or more comma-separated `kid:secret` entries, e.g.
    # `v1:abc...,v2:def...` - every entry is a valid verification key (looked up by the `kid`
    # embedded in the token, `{user_id}.{expiry}.{kid}.{sig}`), so an older `kid` can still be
    # verified while StudyLife's backend has already rotated to signing with a newer one.
    # StudyLife's own `StudyLifeAi:TokenSigningSecret` always SIGNS with the first entry - this
    # side just needs every currently-valid entry to be listed here too. No default: falls
    # back to `studylife_shared_secret` (legacy, un-keyed 3-part format) while unset.
    studylife_token_signing_secret: str | None = None
    # Authenticates POST /internal/register-key, /internal/revoke-key, /internal/enrich-capture
    # (constant-time bearer-secret check against `X-StudyLife-Shared-Secret` - a simpler check
    # than the signed-token scheme above, since these aren't per-user requests). May itself be
    # a comma-separated list of ACCEPTED values (unlike the signing secret above, there's no
    # key-id here - it's a plain static bearer) - rotate by listing both the old and the new
    # value here while StudyLife's backend (which always sends only the first value configured
    # on its own `StudyLifeAi:InternalApiSecret`) is switched over, then drop the old one. No
    # default: falls back to `studylife_shared_secret` (legacy) while unset.
    studylife_internal_api_secret: str | None = None
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

    # Cap on session chunks kept when date_parse_model resolved an exact date range for this
    # query (see docs/decisions.md "Exempt exact date-range matches from the shared top-k").
    # Used two ways: (1) how many chunks _fetch_session_window() keeps from that exact range
    # (proximity-to-today sorted, like session_window_top_k), and (2) how many of those
    # exact-match chunks retrieve_with_rerank() lets bypass the normal retrieval_top_k final cut
    # - confirmed live 2026-08-12: a real "last week" had 21 matching sessions, but the shared
    # retrieval_top_k=8 final cut (meant for uncertain vector-similarity relevance across ALL
    # content types) silently dropped more than half of them, even though every one of them is
    # unconditionally relevant once the date range itself is exact. Set well above a typical
    # week's session count; still a hard cap, not a real fix for month/year-scale ranges - very
    # large ranges are a known, flagged soft edge (see docs/decisions.md), not solved here.
    date_range_chunk_cap: int = 60

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

    # For reasoning models only (e.g. RERANK_MODEL=openai/gpt-5-mini) - same convention and
    # reasoning as llm_reasoning_effort above, kept as its own setting rather than reused
    # because rerank_model is itself independent of llm_model (a reasoning-capable rerank
    # model doesn't imply the answer model is one too, or vice versa).
    rerank_reasoning_effort: str | None = None

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

    # Checkpoint TTL sweep (audit A13/F14 rest, 2026-08-26): completed AND never-confirmed
    # ("pending", proposed but neither /agent/confirm'd nor rejected) agent threads otherwise
    # accumulate in `agent_checkpoint_db_path` forever - nothing else ever deletes a thread
    # except an explicit /internal/revoke-key purge (ingestion/sync.py's purge_user) or a
    # failed agent run (api/agent.py's _invoke_and_handle_failure). See
    # agent/checkpoint_cleanup.py and docs/decisions.md.
    agent_checkpoint_ttl_days: int = 30
    # How often the in-process background loop (main.py's lifespan, same pattern as
    # ingestion/scheduler.py's run_periodic_sync) sweeps for threads past the TTL above.
    agent_checkpoint_cleanup_interval_seconds: int = 3600

    # M4.5 (see docs/decisions.md "M4.5 Multi-user support"): SQLite file
    # backing the per-user AiApiKey registry - populated by StudyLife's
    # POST /internal/register-key/revoke-key calls, read by /agent (to build
    # a real StudyLifeClient for the calling user) and by ingestion's
    # sync_all() (which syncs every registered user, replacing the earlier
    # manually-maintained INGESTION_USERS list).
    registered_keys_db_path: str = "registered_keys.db"

    # Fernet key (32 url-safe base64-encoded bytes, e.g. via
    # `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`)
    # encrypting each registered AiApiKey at rest in registered_keys_db_path (audit finding A4,
    # 2026-08-25) - every row there is a full, usable StudyLife account credential (unlike
    # StudyLife's own hash-only key storage), so plaintext SQLite storage was a real exposure.
    # No default on purpose: RegisteredKeyStore.__init__ fails loudly at startup (both the app's
    # lifespan and ingestion's sync_all() entrypoint) if this is missing or not a valid Fernet
    # key, same "fail loud rather than silently point nowhere" convention as
    # studylife_shared_secret - see docs/decisions.md.
    ai_key_encryption_key: str | None = None

    # Rate limiting (see docs/decisions.md "Rate limiting"): a fixed-window counter per
    # resolved user_id, guarding /chat and /agent /agent-confirm - the endpoints that incur
    # real LLM cost. Defends against a leaked token or a client-side bug (e.g. a retry loop)
    # running up cost or starving the service, not against many distinct attackers - this is a
    # personal-scale, single-replica deployment (see k8s/04-app.yaml's `Recreate` strategy), not
    # a public API, so an in-memory counter (no Redis/shared state) is enough.
    rate_limit_requests: int = 20
    rate_limit_window_seconds: int = 60

    # Audit O6-ai (2026-08-26, see docs/decisions.md): GET /metrics (main.py, via
    # prometheus-fastapi-instrumentator) is unauthenticated, and its labels include per-user_id
    # LLM cost/token/latency data (llm/metrics.py) - anyone who can reach the port can read
    # every user's spend. Optional static bearer: unset (the default) is a genuine no-op, the
    # endpoint stays exactly as unauthenticated as it is today, since the operator's Prometheus
    # may not be reconfigured yet. When set, GET /metrics requires an `Authorization: Bearer
    # <token>` header matching this value (constant-time compare, same convention as
    # studylife_internal_api_secret) or responds 401 - the operator must then add
    # `authorization: {credentials: <token>}` to Prometheus's own scrape config for the
    # "studylife-ai" job (see README.md "Observability"). Chosen over dropping the user_id
    # label: per-user cost attribution is the whole point of that label (see llm/metrics.py) and
    # cardinality is already bounded (a handful of real users, not a public multi-tenant
    # service) - gating the endpoint closes the actual leak (unauthenticated network access)
    # without losing that data.
    metrics_token: str | None = None

    # Audit O6-ai (2026-08-26, see docs/decisions.md): /internal/* (api/internal.py) shares the
    # public port with /chat, /agent, and /metrics - a k8s NetworkPolicy scoped to that port
    # necessarily also has to admit every other caller of anything else on it (see
    # k8s/05-network-policies.yaml's allow-prometheus-to-app, which only needs /metrics but
    # currently gets network-level access to /internal/* too). Port `internal_server.py`'s
    # second FastAPI app (only /internal/*) listens on, via a second uvicorn.Server task in
    # main.py's lifespan - not a second process/container. /internal/* is ALSO still served on
    # the main port for one transition release (main.py logs a deprecation warning each time
    # that path is actually hit) so StudyLife's backend keeps working unchanged regardless of
    # which port it's configured to call, or which side of the split deploys first.
    internal_api_port: int = 8001

    # Capture enrichment (studylife-capture browser extension, see docs/decisions.md "Capture
    # enrichment"): cosine-similarity threshold (rag/enrichment.py's course vector search,
    # Qdrant's COSINE distance) above which a captured note's courseId gets auto-assigned. Below
    # this, the capture is left unassigned rather than risk a wrong guess - a wrong course
    # assignment silently mis-files a note where the user won't think to look for it, worse than
    # leaving it unsorted for manual filing.
    capture_course_match_threshold: float = 0.75


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (env is read once per process)."""
    return Settings()
