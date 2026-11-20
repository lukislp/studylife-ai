"""GDPR audit S8 / 2026-11-20 S8b: the AI consent check is on everywhere; production also refuses
to start without it."""

import logging

import pytest
from pydantic import ValidationError

from studylife_ai.config import Settings


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {"unsafe_disable_ai_consent_check": False}
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def _production(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "environment": "production",
        "studylife_api_base_url": "https://studylife.test",
    }
    values.update(overrides)
    return _settings(**values)


def test_production_with_base_url_and_consent_check_starts() -> None:
    assert _production().unsafe_disable_ai_consent_check is False


def test_production_without_base_url_refuses_to_start() -> None:
    with pytest.raises(ValidationError, match="STUDYLIFE_API_BASE_URL"):
        _production(studylife_api_base_url=None)


def test_production_with_the_opt_out_refuses_to_start() -> None:
    with pytest.raises(ValidationError, match="UNSAFE_DISABLE_AI_CONSENT_CHECK"):
        _production(unsafe_disable_ai_consent_check=True)


@pytest.mark.parametrize("environment", ["local", "staging"])
def test_the_opt_out_is_off_by_default_in_every_environment(
    environment: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("UNSAFE_DISABLE_AI_CONSENT_CHECK", raising=False)
    settings = Settings(_env_file=None, environment=environment)  # type: ignore[arg-type]

    assert settings.unsafe_disable_ai_consent_check is False


@pytest.mark.parametrize("environment", ["local", "staging"])
def test_the_explicit_opt_out_works_outside_production(environment: str) -> None:
    settings = _settings(environment=environment, unsafe_disable_ai_consent_check=True)

    assert settings.unsafe_disable_ai_consent_check is True


def test_startup_warns_when_the_opt_out_is_on(caplog: pytest.LogCaptureFixture) -> None:
    import studylife_ai.main as main_module

    with caplog.at_level(logging.WARNING, logger="studylife_ai.main"):
        main_module.warn_if_consent_check_disabled(_settings(unsafe_disable_ai_consent_check=True))

    assert any("UNSAFE_DISABLE_AI_CONSENT_CHECK" in r.message for r in caplog.records)


def test_startup_is_quiet_when_the_check_is_on(caplog: pytest.LogCaptureFixture) -> None:
    import studylife_ai.main as main_module

    with caplog.at_level(logging.WARNING, logger="studylife_ai.main"):
        main_module.warn_if_consent_check_disabled(_settings())

    assert not caplog.records
