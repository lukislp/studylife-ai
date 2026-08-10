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

### 2026-08-10 — Badge row matches the main StudyLife repo's style **[owner: user]**
- **Decision:** User asked to look at the main [StudyLife](https://github.com/lukislp/studylife) repo's badge row and mirror the style. Added License (AGPL-3.0) and Python-version badges alongside the CI badge (StudyLife's row: CI/CD, Release, License, .NET-version, Coverage).
- **Why:** User's call for visual/branding consistency across both repos. Assistant's judgment call within that instruction: deliberately did *not* add a Release badge (no tags/release process exist yet in this repo) or a Coverage badge (no coverage measurement wired into CI yet) — both would render as empty/misleading or effectively fabricate a metric, which CLAUDE.md explicitly forbids. Add them once the underlying capability (semantic-release/tags, `pytest --cov` + a coverage badge job) actually exists, not just for visual parity.

## M2 — Ingestion architecture (planning, ahead of implementation)

### 2026-08-10 — Ingestion reads via the StudyLife REST API, not direct DB access **[owner: user]**
- **Decision:** The M2 ingestion worker will call the StudyLife REST API to pull notes/courses/calendar data, rather than reading the StudyLife database directly as originally specified in CLAUDE.md's architecture table ("Ingestion: Python-Worker, liest Notizen aus der StudyLife-DB").
- **Alternatives:** Direct DB access (original CLAUDE.md spec) — faster, no extra HTTP hop, but couples `studylife-ai` to StudyLife's internal schema and bypasses its access-control logic; breaks silently on StudyLife schema migrations.
- **Why:** Decoupling from the internal schema and reusing StudyLife's existing access logic/auth outweighs the minor performance cost of going through the API. This overrides the CLAUDE.md architecture table — flagged as a deliberate architecture change, not a silent one.
- **Status:** CLAUDE.md's architecture table updated same day to reflect this (no longer says "liest Notizen aus der StudyLife-DB"). API shape confirmed by reading the StudyLife source directly (`C:\Users\koerb\OneDrive\Documentos\Code\repos\studylife\studylife`):
  - `GET /api/notes` (`NotesController.cs`) returns all notes, no pagination.
  - Auth is a custom `X-Api-Key` header/middleware (not JWT/Identity) — built explicitly for non-interactive integrations (the code comments cite the existing Home Assistant integration as the reference use case); minted once via the StudyLife UI after a passkey login, then usable indefinitely as a static credential. No tenant-less service account exists — the key is bound to one StudyLife user.
  - No Swagger/OpenAPI in the StudyLife repo; `StudyLife.Shared/Dtos.cs` is the closest thing to a spec.

### 2026-08-11 — Incremental sync: client-side full-list diff (Option A), not a StudyLife API change **[owner: user]**
- **Decision:** The M2 ingestion worker polls `GET /api/notes` for the full list on every sync and diffs it client-side against the last known state, rather than having StudyLife's API do server-side filtering.
- **Mechanics:** Each Qdrant point carries `note_id` (+ a change fingerprint — `updated_at` or a content hash, still to be decided when the ingestion pipeline is actually built) as payload metadata; Qdrant itself is the "last known state" — no separate manifest/state store. Per sync: note ID present now but not before → new (chunk + embed + upsert); ID present both times with a changed fingerprint → changed (delete its existing chunks, re-chunk + upsert); ID known before but missing now → deleted (delete its chunks). Requires being able to filter-delete Qdrant points by `note_id` payload.
- **Alternatives:** Add an `updatedSince` query param to `NotesController.GetAll` (+ soft-delete/tombstone support for detecting deletions) so the ingestion worker only transfers what changed. More bandwidth-efficient, but requires changing and redeploying the separate StudyLife repo, including a data-model change (hard→soft delete) for deletion detection to work at all.
- **Why:** At this project's scale (one user's personal notes, likely low-thousands at most), the full-list transfer per sync is negligible (single-digit KB) — the bandwidth savings of the alternative don't justify touching a different, already-running production repo and its own release cycle. Ships entirely within `studylife-ai`, independent of StudyLife's deployment schedule.
- **Backlog — revisit later:** If note volume or sync frequency ever make the full-list transfer a real cost, revisit adding `updatedSince` to `NotesController.GetAll` on the StudyLife side to cut bandwidth. Not blocking for M2; noted here so it isn't forgotten.
- Same underlying question (no `UpdatedAt` on `StudySessionDto` at all) applies to `api/sessions` if session data ends up in ingestion scope — same client-side-diff approach would apply, just with content-hash as the only viable fingerprint since there's no timestamp to compare.

### 2026-08-11 — Chunking strategy: fixed-size with overlap, sentence-aligned **[owner: user]**
- **Decision:** Chunk note content into ~500-token windows with ~15% (≈75 token) overlap, boundary snapped to the nearest sentence break. Size measured via `tiktoken` (already a transitive dependency via LiteLLM/OpenAI) as an approximation, independent of the actual embedding model used.
- **Alternatives considered:** (1) whole-note-as-one-chunk — simplest, unambiguous source attribution, but embeddings dilute on long notes and can exceed embedding context limits; (3) paragraph-based split with size fallback — better semantic coherence, more implementation complexity/edge cases; (4) hybrid of 1+3 by a length threshold — best long-term fit, but the right threshold can't be picked without real usage data on how long/structured the user's actual notes are.
- **Context that ruled out a structure-aware split**: checked `Notes.razor:173` in the StudyLife source — note content is edited via a plain `<textarea>`, no Markdown/rich-text editor, no enforced structure, no length limit. So "Struktur-Awareness" (as named in CLAUDE.md) can't mean Markdown-header-based splitting; there's no reliable syntax to split on.
- **Why fixed-size over the others for the M2 starting point:** industry-standard default, works regardless of whether a given note happens to have paragraph structure or not, and the Qdrant payload schema (`note_id`, `chunk_index`, `title`, `course_id?`, `session_id?`, fingerprint) doesn't need to change if we later move to paragraph-based or hybrid chunking — so this isn't a dead end, just a simple starting point. Revisit once RAGAS eval (M3) shows whether chunk quality is actually a problem.
- Each Qdrant point's payload also needs `note_id` for the delete-on-change/delete-on-removal mechanics from the sync-strategy decision above.

### 2026-08-11 — Single-user scope for now; multi-user kept option-open in the data model **[owner: user]**
- **Decision:** `studylife-ai` targets a single StudyLife user for now (one `X-Api-Key` in config) — no per-request auth, no user routing in the `/chat` endpoint. Qdrant chunk payload still gets a `user_id` field (populated from config/the API key's owning account), even though nothing filters on it yet.
- **Why single-user now:** A StudyLife API key is bound to exactly one `AuthUserId` and only ever returns that user's data — so with one key configured, there's no cross-user data risk to design against yet. Building multi-user routing (which of possibly several StudyLife accounts is "the current chat user", managing multiple API keys, mandatory retrieval-time `user_id` filtering) pulls in scope that belongs with M4 (agent/API auth), not M2 ingestion.
- **Why keep `user_id` in the payload anyway:** StudyLife itself is multi-user (family members, fully separated data) — if `studylife-ai` ever needs to serve more than one StudyLife account, retrofitting `user_id` after the fact means re-ingesting every note. Adding the field now costs nothing (single constant value today) and avoids that migration later.
- **Backlog — revisit later:** Multi-user support (auth in the FastAPI service to know "who's asking", per-account API key management, mandatory `user_id` filter at retrieval time so one family member's queries can never surface another's notes/grades) is deferred, not designed yet. Flagging explicitly so it isn't silently forgotten, given StudyLife's own multi-user design makes this a plausible future need, not a hypothetical one.

### 2026-08-11 — Dedicated StudyLife API key for studylife-ai, not the shared Home Assistant key **[owner: user]**
- **Decision:** StudyLife got a second, independent API key slot (`AuthUserEntity.AiApiKeyHash`, endpoints under `api/settings/ai-api-key`) specifically for `studylife-ai`, instead of reusing the existing Home Assistant key (`ApiKeyHash`, `api/settings/ha-api-key`). Implemented as a full mirror of the HA key feature: entity field + EF migration (SQLite + Postgres), controller endpoints, a setup-page card, i18n in all 26 languages, tests — in the StudyLife repo, not this one.
- **Alternatives:** Reuse the Home Assistant key value for studylife-ai too — zero cross-repo work, but the two integrations would share one credential: leaking or rotating it breaks/endangers both, with no independent revocation and no separate audit trail.
- **Why:** StudyLife's `ha-api-key` mechanism is a single slot per user with no separate service-account concept — sharing it across two unrelated integrations is a real blast-radius/credential-hygiene issue, not just untidy. Mirroring the existing pattern (rather than generalizing into a named multi-key system) matches StudyLife's own CLAUDE.md stance against premature abstraction — there are exactly two integrations today.
- **Where the change lives:** entirely in the `studylife` repo (commit `f9b856a`), not `studylife-ai`. `studylife-ai`'s only change is that `.env`'s `STUDYLIFE_API_KEY` now comes from the "studylife-ai connection" setup card, not the Home Assistant one.
