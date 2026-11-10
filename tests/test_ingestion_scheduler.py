import asyncio
import logging

import pytest
from pytest import MonkeyPatch

from studylife_ai.config import Settings
from studylife_ai.ingestion import scheduler as scheduler_module
from studylife_ai.ingestion.scheduler import run_periodic_sync
from studylife_ai.studylife.registered_keys import RegisteredKeyStore
from tests.conftest import TEST_AI_KEY_ENCRYPTION_KEY


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "studylife_api_base_url": "http://studylife.test",
        "ingestion_sync_interval_seconds": 60,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def _store() -> RegisteredKeyStore:
    return RegisteredKeyStore(":memory:", TEST_AI_KEY_ENCRYPTION_KEY)


def _end_loop_after(n_calls: int) -> object:
    """Fake asyncio.sleep that lets the loop body run `n_calls` times, then
    raises CancelledError - the same signal a real task cancellation would
    produce - so the loop under test ends deterministically without any real
    waiting or needing an external task to cancel it."""
    calls = 0

    async def _fake_sleep(_seconds: float) -> None:
        nonlocal calls
        calls += 1
        if calls >= n_calls:
            raise asyncio.CancelledError

    return _fake_sleep


async def test_run_periodic_sync_calls_sync_all_repeatedly(monkeypatch: MonkeyPatch) -> None:
    call_count = 0

    async def fake_sync_all(settings: Settings, **_kwargs: object) -> None:
        nonlocal call_count
        call_count += 1

    monkeypatch.setattr(scheduler_module, "sync_all", fake_sync_all)
    monkeypatch.setattr(scheduler_module.asyncio, "sleep", _end_loop_after(3))

    with pytest.raises(asyncio.CancelledError):
        await run_periodic_sync(_settings(), _store())

    assert call_count == 3


async def test_run_periodic_sync_logs_and_continues_past_a_runtime_error(
    monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def fake_sync_all(settings: Settings, **_kwargs: object) -> None:
        raise RuntimeError("No users registered - generate an AiApiKey first.")

    monkeypatch.setattr(scheduler_module, "sync_all", fake_sync_all)
    monkeypatch.setattr(scheduler_module.asyncio, "sleep", _end_loop_after(1))

    with (
        caplog.at_level(logging.INFO, logger="studylife_ai.ingestion.scheduler"),
        pytest.raises(asyncio.CancelledError),
    ):
        await run_periodic_sync(_settings(), _store())

    assert any(
        r.levelno == logging.INFO and "No users registered" in r.getMessage()
        for r in caplog.records
    )
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


async def test_run_periodic_sync_logs_and_continues_past_an_unexpected_exception(
    monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def fake_sync_all(settings: Settings, **_kwargs: object) -> None:
        raise ValueError("boom")

    monkeypatch.setattr(scheduler_module, "sync_all", fake_sync_all)
    monkeypatch.setattr(scheduler_module.asyncio, "sleep", _end_loop_after(1))

    with (
        caplog.at_level(logging.ERROR, logger="studylife_ai.ingestion.scheduler"),
        pytest.raises(asyncio.CancelledError),
    ):
        await run_periodic_sync(_settings(), _store())

    assert any("Periodic sync run failed" in r.getMessage() for r in caplog.records)


async def test_run_periodic_sync_sleeps_the_configured_interval(monkeypatch: MonkeyPatch) -> None:
    slept_for: list[float] = []

    async def fake_sync_all(settings: Settings, **_kwargs: object) -> None:
        return None

    async def fake_sleep(seconds: float) -> None:
        slept_for.append(seconds)
        raise asyncio.CancelledError

    monkeypatch.setattr(scheduler_module, "sync_all", fake_sync_all)
    monkeypatch.setattr(scheduler_module.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await run_periodic_sync(_settings(ingestion_sync_interval_seconds=45), _store())

    assert slept_for == [45]


async def test_run_periodic_sync_hands_the_shared_store_to_every_tick(
    monkeypatch: MonkeyPatch,
) -> None:
    """The loop must reuse the app's one store instead of letting sync_all() open (and
    re-migrate) a fresh one per tick."""
    seen: list[object] = []

    async def fake_sync_all(settings: Settings, *, registered_keys: object = None) -> None:
        seen.append(registered_keys)

    store = _store()
    monkeypatch.setattr(scheduler_module, "sync_all", fake_sync_all)
    monkeypatch.setattr(scheduler_module.asyncio, "sleep", _end_loop_after(3))

    with pytest.raises(asyncio.CancelledError):
        await run_periodic_sync(_settings(), store)

    assert seen == [store, store, store]
