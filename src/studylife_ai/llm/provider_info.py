"""Describes which AI providers and models this instance sends user data to.

Pure functions only: the description is derived in this one place from the configured LiteLLM
model strings (`llm_model`, `embedding_model`, `rerank_model`, `date_parse_model`) and
`llm_api_base`, and exposed by `GET /internal/provider-info` (see api/internal.py). StudyLife's
backend uses it to decide whether a user must give explicit consent before their notes and study
data are processed (docs/decisions.md "Provider description endpoint").
"""

import ipaddress
from urllib.parse import urlparse

from studylife_ai.config import Settings
from studylife_ai.schemas.internal import ProviderInfoResponse, ProviderRole

# LiteLLM provider prefixes mapped to a human-readable label. Anything not listed is reported
# under its raw prefix, so a new provider is never silently mislabelled.
_PROVIDER_LABELS: dict[str, str] = {
    "openai": "OpenAI",
    "azure": "Azure OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Google Gemini",
    "vertex_ai": "Google Vertex AI",
    "mistral": "Mistral AI",
    "cohere": "Cohere",
    "groq": "Groq",
    "deepseek": "DeepSeek",
    "xai": "xAI",
    "openrouter": "OpenRouter",
    "together_ai": "Together AI",
    "bedrock": "Amazon Bedrock",
    "huggingface": "Hugging Face",
    "perplexity": "Perplexity",
    "ollama": "Ollama (local)",
    "ollama_chat": "Ollama (local)",
    "lm_studio": "LM Studio (local)",
    "hosted_vllm": "vLLM (local)",
}

# Prefixes whose backend runs on a host the operator controls by definition.
_LOCAL_PREFIXES = frozenset({"ollama", "ollama_chat", "lm_studio", "hosted_vllm"})


def split_model(model: str) -> tuple[str, str]:
    """Split a LiteLLM model string into (provider prefix, model name). A string without a
    provider prefix has an empty prefix: the provider is then unknown."""
    prefix, sep, name = model.partition("/")
    if not sep:
        return "", model
    return prefix.strip().lower(), name.strip()


def is_local_base_url(api_base: str | None) -> bool:
    """True when the URL points at the machine itself or a private network address, i.e. the
    request cannot leave the operator's own network. Unparseable or public hosts are not local."""
    if not api_base:
        return False
    host = urlparse(api_base).hostname
    if not host:
        return False
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # A single-label name (a docker compose or cluster service name) is not publicly routable.
        return "." not in host
    return address.is_loopback or address.is_private or address.is_link_local


def describe_role(role: str, model: str, *, api_base: str | None = None) -> ProviderRole:
    """Describe one configured model. `api_base` is only passed for roles that are actually
    sent to `llm_api_base` (chat, rerank, date parsing); embeddings never use it."""
    prefix, name = split_model(model)
    label = _PROVIDER_LABELS.get(prefix, prefix) if prefix else "unknown"
    local = prefix in _LOCAL_PREFIXES or (bool(prefix) and is_local_base_url(api_base))
    return ProviderRole(role=role, provider=label, model=name, local=local)


def describe_providers(settings: Settings) -> ProviderInfoResponse:
    """Describe every configured role; optional roles that are unset are omitted."""
    base = settings.llm_api_base
    roles = [
        describe_role("chat", settings.llm_model, api_base=base),
        describe_role("embeddings", settings.embedding_model),
    ]
    if settings.rerank_model:
        roles.append(describe_role("rerank", settings.rerank_model, api_base=base))
    if settings.date_parse_model:
        roles.append(describe_role("date_parse", settings.date_parse_model, api_base=base))
    return ProviderInfoResponse(roles=roles, any_external=any(not r.local for r in roles))
