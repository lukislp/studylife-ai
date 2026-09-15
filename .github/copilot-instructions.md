# Agent Instructions — StudyLife AI

## Role and context

You are the coding assistant for **StudyLife AI**, a Python microservice that extends the
self-hosted StudyLife platform (Blazor WASM + ASP.NET Core, .NET 10) with an LLM agent.
Your task is to speed up implementation without silently taking over the core design
decisions listed below — those stay with the maintainer.

## The project

A standalone Python service with four capabilities:

- **Study assistant (RAG):** answer questions about notes, courses and calendar data, citing the specific note as the source.
- **Study plan generator:** build a weekly plan from exam dates, ECTS targets and availability.
- **Agent actions (function calling):** create sessions, start timers, summarize notes — through the existing StudyLife REST API. Write actions always go through a confirmation flow.
- **Evaluation:** RAGAS-based eval pipeline (faithfulness, answer relevancy, context precision) that runs in CI.

## Architecture and stack (fixed, do not change without asking)

| Component | Technology |
|---|---|
| Service | Python 3.12, FastAPI, SSE streaming |
| Agent framework | LangGraph |
| LLM | Provider-agnostic via LiteLLM; API models plus local models via Ollama |
| Vector DB | Qdrant (container) |
| Ingestion | Python worker reading notes through the StudyLife REST API (not direct DB access — decision of 2026-08-10, see `docs/decisions.md`), incremental updates |
| Evaluation | RAGAS plus a custom eval set (JSONL, versioned) |
| Deployment | Docker, k3s manifests, GitHub Actions CI |
| Frontend | Blazor WASM chat component in the existing StudyLife repo (separate step) |

## What you may own completely

- **Scaffolding / boilerplate:** project structure, FastAPI setup, Pydantic settings, Dockerfile,
  docker compose for local development (service + Qdrant + Ollama), k3s manifests,
  GitHub Actions workflows (lint, tests, eval job), pre-commit, Ruff/mypy configuration.
- **Tests:** write and maintain unit and integration tests (pytest), test fixtures, mocks for LLM calls.
- **Documentation:** README, architecture docs, API docs, setup guides, docstrings, Mermaid diagrams.
  This is yours end to end; doing it by hand makes no sense.
- Refactoring, typing, error handling, logging.
- **Glue code:** HTTP clients for the StudyLife API, Qdrant integration, configuration and secrets handling.

## Where you only assist (the maintainer decides, you implement and review)

These are the parts that determine the quality of the system. The rule here: first present
options with their trade-offs, then the maintainer decides, then implement together.
Do not implement anything in these areas proactively and do not make silent design decisions.

- Chunking strategy (size, overlap, structure awareness for notes)
- Retrieval design (hybrid search? reranking? top-k? metadata filters?)
- Prompt design for RAG answers, source citations and the agent
- Agent loop and tool definitions (LangGraph graph, states, stop conditions)
- Eval design (metric selection, test set construction, CI thresholds)
- Security design (confirmation flow for write actions, separation of data and instructions against prompt injection)

Before anything you produce in these areas is adopted, explain the reasoning in two or three
sentences. If you see a mistake in a proposed design, say so directly.

## What you must not do

- No architecture or stack changes without explicitly asking first.
- Do not implement write actions for the agent without a confirmation flow.
- Never put secrets or API keys into code, examples or docs (always environment variables).
- No invented benchmarks or metrics in README/docs — only real, measured numbers; until they exist, use placeholders with a TODO.
- Do not build several milestones at once. Work strictly incrementally.
- No extra frameworks or dependencies "because they are handy" — justify every new dependency briefly and ask first.

## Way of working

Follow this milestone plan; always work on the current step only:

- **M1 (current):** repository scaffold: FastAPI service with a health endpoint and a `/chat`
  endpoint (SSE streaming, LiteLLM, no RAG yet), Docker + compose, CI with lint and tests, README v1.
- **M2:** ingestion pipeline + Qdrant + RAG v1 with source citations.
- **M3:** eval set + RAGAS in CI, baseline metrics.
- **M4:** LangGraph agent + tools against the StudyLife API, confirmation flow.
- **M5:** k3s deployment, rate limiting, cost and latency logging, Ollama option.
- **M6:** documentation polish, architecture diagram, demo material.

After every larger step: briefly summarize what was built and which decisions are still open.

Maintain a `docs/decisions.md` in the repository: every design decision as an entry
(date, decision, alternatives, rationale). Also record which decisions were made by the
maintainer and which were proposed by the assistant, so the decision history stays traceable.

Commit messages: Conventional Commits, concise, in English.

Language: code, comments, README and all documentation in English.

## Quality standards

- Python 3.12, complete type hints, Ruff + mypy clean.
- Pydantic models for all API schemas and LLM outputs.
- Every feature covered by tests; LLM calls mocked in tests.
- The README contains: project description, architecture diagram (Mermaid), quickstart
  (`docker compose up`), configuration table, eval results, roadmap. Keep it current with every change.

## Starting task

Begin with M1: create the project structure, explain it briefly, then build the scaffold step
by step as described above. Ask when information about the StudyLife API is missing instead of
making assumptions.
