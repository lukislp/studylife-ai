import pytest
from httpx import AsyncClient
from pytest import MonkeyPatch

from studylife_ai.api.internal import SHARED_SECRET_HEADER
from studylife_ai.config import Settings
from studylife_ai.llm.provider_info import describe_providers, describe_role, split_model
from tests.conftest import TEST_SHARED_SECRET


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "llm_model": "ollama/llama3.2",
        "embedding_model": "ollama/nomic-embed-text",
        "rerank_model": None,
        "date_parse_model": None,
        "llm_api_base": None,
        "ai_models_run_locally": False,
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
    ("model", "label"),
    [
        ("openai/gpt-4o-mini", "OpenAI"),
        ("anthropic/claude-sonnet-4", "Anthropic"),
        ("ollama/llama3.2", "Ollama"),
        ("ollama_chat/llama3.2", "Ollama"),
        ("lm_studio/qwen", "LM Studio"),
        ("hosted_vllm/qwen", "vLLM"),
        ("somecloud/model-x", "somecloud"),
        ("gpt-4o", "unknown"),
    ],
)
def test_labels_name_the_provider_and_never_claim_locality(model: str, label: str) -> None:
    role = describe_role("chat", model)
    assert role.provider == label
    assert "local" not in role.provider.lower()
    assert role.role == "chat"


@pytest.mark.parametrize(
    "model",
    [
        "ollama/llama3.2",
        "ollama_chat/llama3.2",
        "lm_studio/qwen",
        "hosted_vllm/qwen",
        "openai/gpt-4o",
    ],
)
def test_a_prefix_alone_never_makes_a_model_local(model: str) -> None:
    assert describe_role("chat", model).local is False


@pytest.mark.parametrize(
    "api_base",
    [
        "http://localhost:11434",
        "http://192.168.1.20:8000",
        "http://litellm:4000",
        "http://box.local",
    ],
)
def test_a_private_base_url_alone_never_makes_a_model_local(api_base: str) -> None:
    # `openai/...` behind an in-cluster gateway still reaches OpenAI.
    info = describe_providers(_settings(llm_model="openai/gpt-4o", llm_api_base=api_base))
    chat = next(r for r in info.roles if r.role == "chat")
    assert chat.local is False
    assert info.any_external is True


def test_the_operator_declaration_makes_every_model_local() -> None:
    info = describe_providers(
        _settings(
            ai_models_run_locally=True,
            rerank_model="ollama/llama3.2",
            date_parse_model="ollama/llama3.2",
        )
    )
    assert [r.local for r in info.roles] == [True, True, True, True]
    assert info.any_external is False


def test_declared_local_is_ignored_for_a_model_without_a_provider_prefix() -> None:
    assert describe_role("chat", "gpt-4o", declared_local=True).local is False


def test_default_is_external_even_for_all_ollama_models() -> None:
    info = describe_providers(_settings())
    assert [r.role for r in info.roles] == ["chat", "embeddings"]
    assert all(not r.local for r in info.roles)
    assert info.any_external is True


def test_declaration_is_off_by_default() -> None:
    assert Settings().ai_models_run_locally is False  # type: ignore[call-arg]


def test_optional_roles_appear_only_when_configured() -> None:
    info = describe_providers(
        _settings(rerank_model="openai/gpt-5-mini", date_parse_model="ollama/llama3.2")
    )
    assert {r.role for r in info.roles} == {"chat", "embeddings", "rerank", "date_parse"}


def test_mixed_providers_are_each_reported() -> None:
    info = describe_providers(
        _settings(llm_model="openai/gpt-4o-mini", embedding_model="openai/text-embedding-3-small")
    )
    assert {r.provider for r in info.roles} == {"OpenAI"}
    assert info.any_external is True


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
                "provider": "Ollama",
                "model": "nomic-embed-text",
                "local": False,
            },
        ],
        "any_external": True,
    }
