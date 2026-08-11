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

logger = logging.getLogger("studylife_ai.llm.usage")


def _call_site(kwargs: dict[str, object]) -> str:
    litellm_params = kwargs.get("litellm_params")
    if not isinstance(litellm_params, dict):
        return "unknown"
    metadata = litellm_params.get("metadata")
    if not isinstance(metadata, dict):
        return "unknown"
    call_site = metadata.get("call_site")
    return call_site if isinstance(call_site, str) else "unknown"


class UsageLogger(CustomLogger):
    """Logs model, call site, latency, tokens and cost after each call.

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
        usage = getattr(response_obj, "usage", None)
        logger.info(
            "llm_call call_site=%s model=%s latency_ms=%.0f prompt_tokens=%s "
            "completion_tokens=%s cost_usd=%s",
            _call_site(kwargs),
            kwargs.get("model"),
            (end_time - start_time).total_seconds() * 1000,
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
            kwargs.get("response_cost"),
        )

    async def async_log_failure_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        exception = kwargs.get("exception")
        logger.warning(
            "llm_call_failed call_site=%s model=%s latency_ms=%.0f error=%s",
            _call_site(kwargs),
            kwargs.get("model"),
            (end_time - start_time).total_seconds() * 1000,
            exception,
        )


def configure_llm_usage_logging() -> None:
    """Registers `UsageLogger` with LiteLLM, once per process.

    Idempotent: `create_app()` and the eval entrypoint can both call this
    unconditionally without producing duplicate log lines.
    """
    if any(isinstance(callback, UsageLogger) for callback in litellm.callbacks):
        return
    litellm.callbacks.append(UsageLogger())
