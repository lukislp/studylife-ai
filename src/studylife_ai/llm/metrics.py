"""Prometheus metrics for every LLM call (see docs/decisions.md "Metrics dashboard").

Separate from `llm/logging.py`'s structured log lines - same source data (the
LiteLLM callback), a second sink, not a replacement. Deliberately aggregate
only: there is no `user_id` (or any other personal) label, so `/metrics` never
reveals who used the assistant or how much (GDPR audit S8). Labels are
`call_site` (a few fixed values) and `model` (the configured models).
"""

from prometheus_client import Counter, Histogram

_LABELS = ("call_site", "model")

LLM_CALLS_TOTAL = Counter(
    "studylife_ai_llm_calls_total",
    "Total LLM calls (chat completions, embeddings, reranking)",
    (*_LABELS, "status"),
)

LLM_COST_USD_TOTAL = Counter(
    "studylife_ai_llm_cost_usd_total",
    "Cumulative LLM cost in USD, from LiteLLM's own price-map lookup",
    _LABELS,
)

LLM_LATENCY_SECONDS = Histogram(
    "studylife_ai_llm_latency_seconds",
    "LLM call latency in seconds",
    _LABELS,
)

LLM_PROMPT_TOKENS_TOTAL = Counter(
    "studylife_ai_llm_prompt_tokens_total",
    "Cumulative prompt (input) tokens",
    _LABELS,
)

LLM_COMPLETION_TOKENS_TOTAL = Counter(
    "studylife_ai_llm_completion_tokens_total",
    "Cumulative completion (output) tokens",
    _LABELS,
)
