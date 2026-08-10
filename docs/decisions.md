# Design Decisions

Log of notable decisions: what was decided, alternatives considered, and why. Marked as **[owner: assistant]** for boilerplate/infra calls made per CLAUDE.md, or **[owner: user]** for calls in the assist-only areas (chunking, retrieval, prompts, agent loop, eval design, security design).

## M1 — Repository scaffold

### 2026-08-10 — Package manager: uv **[owner: assistant]**
- **Decision:** Use `uv` with a standard PEP 621 `pyproject.toml` for dependency management, instead of Poetry or plain pip/venv.
- **Alternatives:** Poetry (mature, but its own lockfile format and slower); plain pip + requirements.txt (no lockfile, weaker reproducibility).
- **Why:** Already installed locally, fast, and works with standard `pyproject.toml` rather than a tool-specific format — lower lock-in.

### 2026-08-10 — src-layout **[owner: assistant]**
- **Decision:** Application code lives under `src/studylife_ai/`, not a flat top-level package.
- **Why:** Standard practice for installable Python packages; prevents accidentally importing the working-tree copy instead of the installed package during tests.

### 2026-08-10 — SSE without `sse-starlette` **[owner: assistant]**
- **Decision:** Implement Server-Sent Events manually via Starlette's `StreamingResponse` and hand-formatted `data: ...\n\n` lines, instead of adding the `sse-starlette` dependency.
- **Why:** The format needed (OpenAI-style `data: {json}` + `data: [DONE]`) is a few lines; adding a dependency for it isn't justified yet. Revisit if SSE needs (retry hints, event IDs, per-connection heartbeats) grow.

### 2026-08-10 — Default LLM model: local Ollama **[owner: assistant]**
- **Decision:** `LLM_MODEL` defaults to `ollama/llama3.1`, `LLM_API_BASE` to `http://localhost:11434`.
- **Why:** Lets `docker compose up` produce a working `/chat` endpoint with zero API keys. Provider-specific keys (e.g. `OPENAI_API_KEY`) are intentionally *not* modeled in `Settings` — LiteLLM reads them directly from the environment based on the `LLM_MODEL` prefix, avoiding a second source of truth for credentials.

### 2026-08-10 — docker-compose includes Qdrant + Ollama from M1 **[owner: assistant]**
- **Decision:** `docker-compose.yml` starts `qdrant` and `ollama` services already, even though the app doesn't use Qdrant until M2.
- **Why:** Per CLAUDE.md, the full local dev compose (service + Qdrant + Ollama) is boilerplate owned end-to-end; starting it now means M2 doesn't need an infra change, just application code.

### 2026-08-10 — StudyLife API integration deferred **[owner: user via CLAUDE.md]**
- **Decision:** No StudyLife REST API client built in M1; API base URL, auth scheme (bearer token vs. API key), and relevant endpoints (notes, sessions, calendar) are still unknown to this assistant.
- **Why:** M1's `/chat` endpoint doesn't call the StudyLife API — only LiteLLM. Needed for M2 (ingestion reads from the StudyLife DB directly, not the API) and concretely for M4 (agent tools call the StudyLife REST API). Will ask for OpenAPI spec / endpoint details / auth scheme when M4 starts, per CLAUDE.md instruction not to assume.

### 2026-08-10 — Decision log location: `docs/decisions.md`, tracked in the repo **[owner: user]**
- **Decision:** The decision log lives at `docs/decisions.md` and is committed to the repo (not gitignored). Originally drafted as `NOTES.md` at repo root; briefly gitignored as private-only, then reverted back to tracked-and-public under the new path/name.
- **Why:** User's call — `decisions.md` names the content more precisely than `notes.md`, and `docs/` groups it with other documentation.

### 2026-08-10 — License: AGPL-3.0 **[owner: user]**
- **Decision:** `LICENSE` is AGPL-3.0, matching the main [StudyLife](https://github.com/lukislp/studylife) repo (copyright: Lukas Koerber, 2026). Replaces the unfilled GPL-3.0 template GitHub had created by default when the `studylife-ai` repo was set up.
- **Why:** User's call — consistency with the main platform's license.

### 2026-08-10 — Default local model: `ollama/llama3.2` **[owner: assistant]**
- **Decision:** Switched the default `LLM_MODEL` from `ollama/llama3.1` to `ollama/llama3.2` (README, `.env.example`, `docker-compose.yml`, `config.py`).
- **Why:** Newer small Ollama model, same class of hardware requirements; no reason to default to the older one. Purely a friendlier out-of-the-box default — any model/provider is still swappable via `LLM_MODEL`.

### 2026-08-10 — CI badge in README **[owner: assistant]**
- **Decision:** Added a GitHub Actions CI status badge to the top of README.md, linking to the workflow.
- **Why:** Makes CI status visible at a glance instead of requiring a trip to the Actions tab; standard practice for a portfolio repo. Confirmed via the GitHub Actions API that the M1 commit's CI run (`run_number: 1`) completed with `conclusion: success` before adding the badge — an unverified badge would be worse than none.

### 2026-08-10 — Badge row matches the main StudyLife repo's style, minus fabricated metrics **[owner: assistant]**
- **Decision:** Added License (AGPL-3.0) and Python-version badges alongside the CI badge, mirroring the badge row style used in the main [StudyLife](https://github.com/lukislp/studylife) README (which has CI/CD, Release, License, .NET-version, and Coverage badges).
- **Why:** Deliberately did *not* add a Release badge (no tags/release process exist yet in this repo) or a Coverage badge (no coverage measurement wired into CI yet) — both would render as empty/misleading or effectively fabricate a metric, which CLAUDE.md explicitly forbids. Add them once the underlying capability (semantic-release/tags, `pytest --cov` + a coverage badge job) actually exists, not just for visual parity.

## M2 — Ingestion architecture (planning, ahead of implementation)

### 2026-08-10 — Ingestion reads via the StudyLife REST API, not direct DB access **[owner: user]**
- **Decision:** The M2 ingestion worker will call the StudyLife REST API to pull notes/courses/calendar data, rather than reading the StudyLife database directly as originally specified in CLAUDE.md's architecture table ("Ingestion: Python-Worker, liest Notizen aus der StudyLife-DB").
- **Alternatives:** Direct DB access (original CLAUDE.md spec) — faster, no extra HTTP hop, but couples `studylife-ai` to StudyLife's internal schema and bypasses its access-control logic; breaks silently on StudyLife schema migrations.
- **Why:** Decoupling from the internal schema and reusing StudyLife's existing access logic/auth outweighs the minor performance cost of going through the API. This overrides the CLAUDE.md architecture table — flagged as a deliberate architecture change, not a silent one.
- **Follow-up:** CLAUDE.md's architecture table should be updated to reflect this once M2 concretely starts. Concrete API shape (notes endpoint, incremental-sync support, auth scheme) still being investigated against the StudyLife source at `C:\Users\koerb\OneDrive\Documentos\Code\repos\studylife\studylife`.

### Open questions for later milestones
- StudyLife API: concrete notes/sessions/calendar endpoints, incremental-sync support (e.g. updated-since filtering), and auth scheme — being investigated directly against the StudyLife source now that the ingestion-via-API decision is made.
