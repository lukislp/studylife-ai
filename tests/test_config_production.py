"""GDPR audit S8: production fails closed when the AI consent check cannot run."""

import pytest
from pydantic import ValidationError

from studylife_ai.config import Settings


def _production(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "environment": "production",
        "studylife_api_base_url": "https://studylife.test",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def test_production_with_base_url_and_consent_check_starts() -> None:
    settings = _production()

    assert settings.ai_consent_check is True


def test_production_without_base_url_refuses_to_start() -> None:
    with pytest.raises(ValidationError, match="STUDYLIFE_API_BASE_URL"):
        _production(studylife_api_base_url=None)


def test_production_with_consent_check_disabled_refuses_to_start() -> None:
    with pytest.raises(ValidationError, match="AI_CONSENT_CHECK"):
        _production(ai_consent_check=False)


@pytest.mark.parametrize("environment", ["local", "staging"])
def test_the_dev_escape_works_outside_production(environment: str) -> None:
    settings = Settings(
        _env_file=None,
        environment=environment,  # type: ignore[arg-type]
        studylife_api_base_url=None,
        ai_consent_check=False,
    )

    assert settings.ai_consent_check is False
    assert settings.studylife_api_base_url is None
