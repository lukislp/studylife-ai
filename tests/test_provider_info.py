import pytest
from httpx import AsyncClient
from pytest import MonkeyPatch

from studylife_ai.api.internal import SHARED_SECRET_HEADER
from studylife_ai.config import Settings
from studylife_ai.llm.provider_info import (
    describe_providers,
    describe_role,
    is_local_base_url,
    split_model,
)
from tests.conftest import TEST_SHARED_SECRET


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "llm_model": "ollama/llama3.2",
        "embedding_model": "ollama/nomic-embed-text",
        "rerank_model": None,
        "date_parse_model": None,
        "llm_api_base": None,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("openai/gpt-4o-mini", ("openai", "gpt-4o-mini")),
        ("OpenAI/gpt-4o", ("openai", "gpt-4o")),
        ("ollama/llama3.2:3b", ("ollama", "llama3.2:3b")),
        ("gpt-4o", ("", "gpt-4o")),
        ("bedrock/anthropic.claude-3/v1", ("bedrock", "anthropic.claude-3/v1")),
    ],
)
def test_split_model(model: str, expected: tuple[str, str]) -> None:
    assert split_model(model) == expected


@pytest.mark.parametrize(
    ("model", "label", "local"),
    [
        ("openai/gpt-4o-mini", "OpenAI", False),
        ("anthropic/claude-sonnet-4", "Anthropic", False),
        ("ollama/llama3.2", "Ollama (local)", True),
        ("ollama_chat/llama3.2", "Ollama (local)", True),
        ("lm_studio/qwen", "LM Studio (local)", True),
        ("somecloud/model-x", "somecloud", False),
        ("gpt-4o", "unknown", False),
    ],
)
def test_describe_role_labels_and_locality(model: str, label: str, local: bool) -> None:
    role = describe_role("chat", model)
    assert role.provider == label
    assert role.local is local
    assert role.role == "chat"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (None, False),
        ("", False),
        ("http://localhost:11434", True),
        ("http://127.0.0.1:8080/v1", True),
        ("http://[::1]:8080", True),
        ("http://192.168.1.20:8000", True),
        ("http://10.0.0.5", True),
        ("http://llm-gateway:8000", True),
        ("http://box.local:8000", True),
        ("https://api.openai.com/v1", False),
        ("https://8.8.8.8", False),
        ("not a url", False),
    ],
)
def test_is_local_base_url(url: str | None, expected: bool) -> None:
    assert is_local_base_url(url) is expected


def test_local_base_url_makes_an_openai_compatible_model_local() -> None:
    role = describe_role("chat", "openai/my-model", api_base="http://localhost:8000/v1")
    assert role.provider == "OpenAI"
    assert role.local is True


def test_public_base_url_does_not_make_a_model_local() -> None:
    assert describe_role("chat", "openai/gpt-4o", api_base="https://api.openai.com").local is False


def test_unknown_provider_is_never_local_even_with_a_local_base_url() -> None:
    assert describe_role("chat", "gpt-4o", api_base="http://localhost:8000").local is False


def test_all_local_defaults_are_not_external() -> None:
    info = describe_providers(_settings())
    assert [r.role for r in info.roles] == ["chat", "embeddings"]
    assert info.any_external is False


def test_one_external_role_makes_the_instance_external() -> None:
    info = describe_providers(
        _settings(llm_model="openai/gpt-4o-mini", embedding_model="openai/text-embedding-3-small")
    )
    assert info.any_external is True
    assert {r.provider for r in info.roles} == {"OpenAI"}


def test_optional_roles_appear_only_when_configured() -> None:
    info = describe_providers(
        _settings(rerank_model="openai/gpt-5-mini", date_parse_model="ollama/llama3.2")
    )
    by_role = {r.role: r for r in info.roles}
    assert set(by_role) == {"chat", "embeddings", "rerank", "date_parse"}
    assert by_role["rerank"].local is False
    assert by_role["date_parse"].local is True
    assert info.any_external is True


def test_embeddings_ignore_the_chat_base_url() -> None:
    info = describe_providers(
        _settings(
            embedding_model="openai/text-embedding-3-small", llm_api_base="http://localhost:1"
        )
    )
    embeddings = next(r for r in info.roles if r.role == "embeddings")
    assert embeddings.local is False


async def test_provider_info_endpoint_requires_the_shared_secret(
    internal_client: AsyncClient,
) -> None:
    assert (await internal_client.get("/internal/provider-info")).status_code == 401
    wrong = await internal_client.get(
        "/internal/provider-info", headers={SHARED_SECRET_HEADER: "wrong"}
    )
    assert wrong.status_code == 401


async def test_provider_info_endpoint_returns_the_description(
    internal_client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "studylife_ai.api.internal.get_settings",
        lambda: _settings(
            llm_model="openai/gpt-4o-mini",
            embedding_model="ollama/nomic-embed-text",
            studylife_shared_secret=TEST_SHARED_SECRET,
        ),
    )

    response = await internal_client.get(
        "/internal/provider-info", headers={SHARED_SECRET_HEADER: TEST_SHARED_SECRET}
    )

    assert response.status_code == 200
    assert response.json() == {
        "roles": [
            {"role": "chat", "provider": "OpenAI", "model": "gpt-4o-mini", "local": False},
            {
                "role": "embeddings",
                "provider": "Ollama (local)",
                "model": "nomic-embed-text",
                "local": True,
            },
        ],
        "any_external": True,
    }
