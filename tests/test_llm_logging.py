import logging
from datetime import datetime, timedelta
from types import SimpleNamespace

import litellm
import pytest
from prometheus_client import generate_latest
from pytest import MonkeyPatch

from studylife_ai.llm.logging import UsageLogger, configure_llm_usage_logging


def _metrics_text() -> str:
    return generate_latest().decode("utf-8")


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


async def test_async_log_success_event_records_prometheus_metrics() -> None:
    # Unique call_site/user_id per test (not reused elsewhere in this file or test_chat.py etc.)
    # so the absolute counter values read back below can't be polluted by another test's
    # increments to the same label combination - Prometheus Counter/Histogram objects are
    # module-level global state, shared across the whole test run.
    kwargs = {
        "model": "openai/gpt-4o-mini",
        "response_cost": 0.5,
        "litellm_params": {
            "metadata": {"call_site": "metrics-test-success", "user_id": "metrics-test-user"}
        },
    }
    response_obj = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50))
    start = datetime(2026, 8, 11, 12, 0, 0)
    end = start + timedelta(milliseconds=500)

    await UsageLogger().async_log_success_event(kwargs, response_obj, start, end)

    text = _metrics_text()
    # Label order in the exposition format is alphabetical by label name (call_site, model,
    # status, user_id) - NOT declaration order (verified live via generate_latest(), since
    # prometheus_client doesn't document this explicitly).
    labels = (
        'call_site="metrics-test-success",model="openai/gpt-4o-mini",user_id="metrics-test-user"'
    )
    assert (
        'studylife_ai_llm_calls_total{call_site="metrics-test-success",'
        'model="openai/gpt-4o-mini",status="success",user_id="metrics-test-user"} 1.0'
    ) in text
    assert f"studylife_ai_llm_cost_usd_total{{{labels}}} 0.5" in text
    assert f"studylife_ai_llm_prompt_tokens_total{{{labels}}} 100.0" in text
    assert f"studylife_ai_llm_completion_tokens_total{{{labels}}} 50.0" in text
    assert f"studylife_ai_llm_latency_seconds_count{{{labels}}} 1.0" in text
    assert f"studylife_ai_llm_latency_seconds_sum{{{labels}}} 0.5" in text


async def test_async_log_failure_event_records_prometheus_call_and_latency_metrics() -> None:
    kwargs = {
        "model": "ollama/llama3.2",
        "exception": RuntimeError("boom"),
        "litellm_params": {
            "metadata": {"call_site": "metrics-test-failure", "user_id": "metrics-test-user"}
        },
    }
    start = datetime(2026, 8, 11, 12, 0, 0)
    end = start + timedelta(milliseconds=100)

    await UsageLogger().async_log_failure_event(kwargs, None, start, end)

    text = _metrics_text()
    labels = 'call_site="metrics-test-failure",model="ollama/llama3.2",user_id="metrics-test-user"'
    assert (
        'studylife_ai_llm_calls_total{call_site="metrics-test-failure",'
        'model="ollama/llama3.2",status="failure",user_id="metrics-test-user"} 1.0'
    ) in text
    assert f"studylife_ai_llm_latency_seconds_count{{{labels}}} 1.0" in text
    # No cost/token metrics for a failed call - nothing to attribute.
    assert f"studylife_ai_llm_cost_usd_total{{{labels}}}" not in text


async def test_async_log_success_event_defaults_user_id_to_unknown_without_metadata() -> None:
    kwargs = {
        "model": "openai/gpt-4o-mini",
        "response_cost": 0.1,
        "litellm_params": {"metadata": {"call_site": "metrics-test-no-user"}},
    }
    response_obj = SimpleNamespace(usage=None)
    start = datetime(2026, 8, 11, 12, 0, 0)

    await UsageLogger().async_log_success_event(kwargs, response_obj, start, start)

    text = _metrics_text()
    assert (
        'studylife_ai_llm_calls_total{call_site="metrics-test-no-user",'
        'model="openai/gpt-4o-mini",status="success",user_id="unknown"} 1.0'
    ) in text
