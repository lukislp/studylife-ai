"""Cost/latency logging for every LiteLLM call (M5).

Hooked in once, globally, via LiteLLM's callback system - not threaded
through each call site individually. This means it also covers the
LangGraph agent's `ChatLiteLLM` model (agent/graph.py), which never goes
through `llm/client.py` at all.
"""

import logging
from datetime import datetime

import litellm
from litellm.integrations.custom_logger import CustomLogger

from studylife_ai.llm.metrics import (
    LLM_CALLS_TOTAL,
    LLM_COMPLETION_TOKENS_TOTAL,
    LLM_COST_USD_TOTAL,
    LLM_LATENCY_SECONDS,
    LLM_PROMPT_TOKENS_TOTAL,
)

logger = logging.getLogger("studylife_ai.llm.usage")


def _metadata(kwargs: dict[str, object]) -> dict[str, object]:
    litellm_params = kwargs.get("litellm_params")
    if not isinstance(litellm_params, dict):
        return {}
    metadata = litellm_params.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _call_site(kwargs: dict[str, object]) -> str:
    call_site = _metadata(kwargs).get("call_site")
    return call_site if isinstance(call_site, str) else "unknown"


def _user_id(kwargs: dict[str, object]) -> str:
    user_id = _metadata(kwargs).get("user_id")
    return user_id if isinstance(user_id, str) else "unknown"


class UsageLogger(CustomLogger):
    """Logs model, call site, latency, tokens and cost after each call, and
    records the same values as Prometheus metrics (`llm/metrics.py`) for the
    Grafana dashboard - see docs/decisions.md "Metrics dashboard".

    Cost is read from `response_cost`, the same value LiteLLM itself
    computed from its model price map - correct for known API models,
    `0.0` for local Ollama models (no separate `completion_cost()` call
    needed).
    """

    async def async_log_success_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        call_site = _call_site(kwargs)
        model = str(kwargs.get("model"))
        user_id = _user_id(kwargs)
        latency_seconds = (end_time - start_time).total_seconds()
        usage = getattr(response_obj, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        cost_usd = kwargs.get("response_cost")

        logger.info(
            "llm_call call_site=%s model=%s latency_ms=%.0f prompt_tokens=%s "
            "completion_tokens=%s cost_usd=%s",
            call_site,
            model,
            latency_seconds * 1000,
            prompt_tokens,
            completion_tokens,
            cost_usd,
        )

        labels = {"call_site": call_site, "model": model, "user_id": user_id}
        LLM_CALLS_TOTAL.labels(**labels, status="success").inc()
        LLM_LATENCY_SECONDS.labels(**labels).observe(latency_seconds)
        if isinstance(cost_usd, int | float):
            LLM_COST_USD_TOTAL.labels(**labels).inc(cost_usd)
        if isinstance(prompt_tokens, int):
            LLM_PROMPT_TOKENS_TOTAL.labels(**labels).inc(prompt_tokens)
        if isinstance(completion_tokens, int):
            LLM_COMPLETION_TOKENS_TOTAL.labels(**labels).inc(completion_tokens)

    async def async_log_failure_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        call_site = _call_site(kwargs)
        model = str(kwargs.get("model"))
        user_id = _user_id(kwargs)
        latency_seconds = (end_time - start_time).total_seconds()
        exception = kwargs.get("exception")

        logger.warning(
            "llm_call_failed call_site=%s model=%s latency_ms=%.0f error=%s",
            call_site,
            model,
            latency_seconds * 1000,
            exception,
        )

        labels = {"call_site": call_site, "model": model, "user_id": user_id}
        LLM_CALLS_TOTAL.labels(**labels, status="failure").inc()
        LLM_LATENCY_SECONDS.labels(**labels).observe(latency_seconds)


def configure_llm_usage_logging() -> None:
    """Registers `UsageLogger` with LiteLLM, once per process.

    Idempotent: `create_app()` and the eval entrypoint can both call this
    unconditionally without producing duplicate log lines.
    """
    if any(isinstance(callback, UsageLogger) for callback in litellm.callbacks):
        return
    litellm.callbacks.append(UsageLogger())
