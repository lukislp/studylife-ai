# Privacy - StudyAtrium AI

StudyAtrium AI is the optional assistant service of a [StudyAtrium](https://github.com/lukislp/studylife)
instance (chat over your notes and study data, an agent that can propose actions, and tag/summary
suggestions for captured notes). It runs next to the StudyAtrium server of whoever operates your
instance. There is no vendor server in between: the operator of the instance is responsible for
this service and for the AI provider they configured.

## What it reads

- From your StudyAtrium account, using the AI key you generated there: your notes, courses,
  study sessions (the last 1825 days by default) and course goals, so the assistant can answer
  questions about them.
- From StudyAtrium's backend, per request: your chat messages or agent instruction, and for notes
  saved with the capture extension the note title and text.
- Your user id, signed by StudyAtrium (a short-lived proxy token), so every request is tied to
  your own data and nobody else's.

## What it stores, where, and for how long

| Data | Where | How long |
|---|---|---|
| Text chunks of your notes, courses, sessions and goals plus their embedding vectors, tagged with your user id | The Qdrant vector database the operator configured (`QDRANT_URL`), one partition per user | Until you revoke your AI key, your account is deleted, or the item is deleted in StudyAtrium (the next sync, every 60 s by default, removes it) |
| Your AI key, encrypted at rest | SQLite file `registered_keys.db` (`REGISTERED_KEYS_DB_PATH`) | Until you revoke the key or delete your account |
| Agent conversation state (checkpoints of `/agent` runs, including the messages of that run) | SQLite file `agent_checkpoints.db` (`AGENT_CHECKPOINT_DB_PATH`) | 30 days after the run's last activity (`AGENT_CHECKPOINT_TTL_DAYS`), or until revoke/account deletion |
| Usage metrics: per call site, model and user id the number of calls, tokens, cost and latency | In memory, exposed at `/metrics` for the operator's monitoring (optionally protected by `METRICS_TOKEN`) | Counters live until the process restarts; the monitoring system keeps them as long as the operator configured |
| Log lines: model, call site, latency, token counts and cost per model call; warnings and errors without message or note content | Container log of the operator | As configured by the operator's log stack |

Chat messages themselves are not stored by this service outside agent checkpoints.

## Where your content is sent

To the AI provider the operator chose with `LLM_MODEL`, `EMBEDDING_MODEL`, `RERANK_MODEL` and
`DATE_PARSE_MODEL` (for example OpenAI, or a model the operator runs themselves). Which provider
that is, and whether it runs on infrastructure the operator controls, is reported by
`GET /internal/provider-info` and shown to you in StudyAtrium before you agree. Nothing is sent to
a model before you have agreed: StudyAtrium refuses AI requests without a consent that covers the
current provider, and this service verifies that consent itself before every chat, agent step and
capture enrichment (`AI_CONSENT_CHECK`, on by default). It sends no analytics or telemetry
anywhere else.

## Deleting your data

When you revoke your AI key, or delete your StudyAtrium account, StudyAtrium calls
`POST /internal/revoke-key` on this service. That removes, in one operation, your Qdrant
partition, all your agent checkpoints and your registered key. StudyAtrium queues the call and
retries it until this service confirms. If a key dies without that call (for example it was
regenerated), the sync loop purges the registration after 20 consecutive rejections. Backups of
the Qdrant volume or the SQLite files that the operator made are not touched by a purge.

## Source

This service is open source (AGPL-3.0): <https://github.com/lukislp/studylife-ai>.
