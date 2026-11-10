"""Describes which AI providers and models this instance sends user data to.

Pure functions only: the description is derived in this one place from the configured LiteLLM
model strings (`llm_model`, `embedding_model`, `rerank_model`, `date_parse_model`) and exposed by
`GET /internal/provider-info` (see api/internal.py). StudyLife's backend uses it to decide
whether a user must give explicit consent before their notes and study data are processed
(docs/decisions.md "Provider description endpoint").

Locality is never guessed. A label names who the model is *from* (the LiteLLM prefix), not where
it runs; `local` is true only when the operator declared it with `AI_MODELS_RUN_LOCALLY`.
"""

from studylife_ai.config import Settings
from studylife_ai.schemas.internal import ProviderInfoResponse, ProviderRole

# LiteLLM provider prefixes mapped to a human-readable label. Anything not listed is reported
# under its raw prefix, so a new provider is never silently mislabelled. No label says "local":
# a prefix such as `ollama` or `hosted_vllm` can just as well point at a remote host.
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
    "ollama": "Ollama",
    "ollama_chat": "Ollama",
    "lm_studio": "LM Studio",
    "hosted_vllm": "vLLM",
}


def split_model(model: str) -> tuple[str, str]:
    """Split a LiteLLM model string into (provider prefix, model name). A string without a
    provider prefix has an empty prefix: the provider is then unknown."""
    prefix, sep, name = model.partition("/")
    if not sep:
        return "", model
    return prefix.strip().lower(), name.strip()


def describe_role(role: str, model: str, *, declared_local: bool = False) -> ProviderRole:
    """Describe one configured model. `declared_local` is the operator's declaration
    (`Settings.ai_models_run_locally`); a model whose provider cannot even be named stays
    non-local whatever was declared."""
    prefix, name = split_model(model)
    label = _PROVIDER_LABELS.get(prefix, prefix) if prefix else "unknown"
    return ProviderRole(
        role=role, provider=label, model=name, local=declared_local and bool(prefix)
    )


def describe_providers(settings: Settings) -> ProviderInfoResponse:
    """Describe every configured role; optional roles that are unset are omitted."""
    declared = settings.ai_models_run_locally
    roles = [
        describe_role("chat", settings.llm_model, declared_local=declared),
        describe_role("embeddings", settings.embedding_model, declared_local=declared),
    ]
    if settings.rerank_model:
        roles.append(describe_role("rerank", settings.rerank_model, declared_local=declared))
    if settings.date_parse_model:
        roles.append(
            describe_role("date_parse", settings.date_parse_model, declared_local=declared)
        )
    return ProviderInfoResponse(roles=roles, any_external=any(not r.local for r in roles))
