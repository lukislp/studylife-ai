import logging
from datetime import datetime, timedelta
from types import SimpleNamespace

import litellm
import pytest
from pytest import MonkeyPatch

from studylife_ai.llm.logging import UsageLogger, configure_llm_usage_logging


def test_configure_llm_usage_logging_is_idempotent(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])

    configure_llm_usage_logging()
    configure_llm_usage_logging()

    assert [type(c) for c in litellm.callbacks] == [UsageLogger]


async def test_async_log_success_event_logs_call_site_model_latency_tokens_cost(
    caplog: pytest.LogCaptureFixture,
) -> None:
    kwargs = {
        "model": "openai/gpt-4o-mini",
        "response_cost": 0.0021,
        "litellm_params": {"metadata": {"call_site": "chat"}},
    }
    response_obj = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=120, completion_tokens=40))
    start = datetime(2026, 8, 11, 12, 0, 0)
    end = start + timedelta(milliseconds=850)

    with caplog.at_level(logging.INFO, logger="studylife_ai.llm.usage"):
        await UsageLogger().async_log_success_event(kwargs, response_obj, start, end)

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "call_site=chat" in message
    assert "model=openai/gpt-4o-mini" in message
    assert "latency_ms=850" in message
    assert "prompt_tokens=120" in message
    assert "completion_tokens=40" in message
    assert "cost_usd=0.0021" in message


async def test_async_log_success_event_defaults_call_site_to_unknown_without_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    kwargs: dict[str, object] = {"model": "ollama/llama3.2", "response_cost": 0.0}
    response_obj = SimpleNamespace(usage=None)
    start = datetime(2026, 8, 11, 12, 0, 0)

    with caplog.at_level(logging.INFO, logger="studylife_ai.llm.usage"):
        await UsageLogger().async_log_success_event(kwargs, response_obj, start, start)

    assert "call_site=unknown" in caplog.records[0].getMessage()


async def test_async_log_failure_event_logs_call_site_model_latency_and_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    kwargs = {
        "model": "openai/gpt-4o-mini",
        "exception": RuntimeError("boom"),
        "litellm_params": {"metadata": {"call_site": "rerank"}},
    }
    start = datetime(2026, 8, 11, 12, 0, 0)
    end = start + timedelta(milliseconds=200)

    with caplog.at_level(logging.WARNING, logger="studylife_ai.llm.usage"):
        await UsageLogger().async_log_failure_event(kwargs, None, start, end)

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "call_site=rerank" in message
    assert "latency_ms=200" in message
    assert "error=boom" in message
