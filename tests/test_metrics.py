"""Audit O6-ai: optional bearer-token gate on GET /metrics (Settings.metrics_token)."""

from httpx import AsyncClient
from pytest import MonkeyPatch


async def test_metrics_is_open_by_default(client: AsyncClient) -> None:
    """METRICS_TOKEN unset (the default) - no behavior change from before this existed."""
    response = await client.get("/metrics")

    assert response.status_code == 200


async def test_metrics_rejects_requests_without_a_bearer_token_once_configured(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    from studylife_ai import config as config_module

    settings = config_module.get_settings()
    monkeypatch.setattr(
        "studylife_ai.main.get_settings",
        lambda: settings.model_copy(update={"metrics_token": "secret-token"}),
    )

    response = await client.get("/metrics")

    assert response.status_code == 401


async def test_metrics_rejects_a_wrong_bearer_token(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    from studylife_ai import config as config_module

    settings = config_module.get_settings()
    monkeypatch.setattr(
        "studylife_ai.main.get_settings",
        lambda: settings.model_copy(update={"metrics_token": "secret-token"}),
    )

    response = await client.get("/metrics", headers={"Authorization": "Bearer wrong-token"})

    assert response.status_code == 401


async def test_metrics_accepts_the_configured_bearer_token(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    from studylife_ai import config as config_module

    settings = config_module.get_settings()
    monkeypatch.setattr(
        "studylife_ai.main.get_settings",
        lambda: settings.model_copy(update={"metrics_token": "secret-token"}),
    )

    response = await client.get("/metrics", headers={"Authorization": "Bearer secret-token"})

    assert response.status_code == 200
