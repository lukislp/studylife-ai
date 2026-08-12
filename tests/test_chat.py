import json
from types import SimpleNamespace

import pytest
from httpx import AsyncClient
from pytest import MonkeyPatch

from studylife_ai.api.identity import PROXY_TOKEN_HEADER
from studylife_ai.ingestion.qdrant_store import RetrievedChunk
from tests.conftest import TEST_SHARED_SECRET, TEST_USER_ID, make_proxy_token


def _make_fake_stream(texts: list[str]) -> object:
    async def fake_stream() -> object:
        for text in texts:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])

    return fake_stream()


def _parse_sse_events(body: str) -> list[dict[str, object]]:
    return [
        json.loads(chunk.removeprefix("data: "))
        for chunk in body.split("\n\n")
        if chunk and chunk != "data: [DONE]"
    ]


def _mock_no_retrieval(monkeypatch: MonkeyPatch) -> None:
    async def fake_retrieve_context(
        query: str, settings: object, store: object, user_id: str
    ) -> list[RetrievedChunk]:
        return []

    monkeypatch.setattr("studylife_ai.api.chat._retrieve_context", fake_retrieve_context)


async def test_chat_streams_llm_deltas_and_sources_as_sse(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    calls = []

    async def fake_acompletion(*_args: object, **kwargs: object) -> object:
        calls.append(kwargs)
        return _make_fake_stream(["Hello", ", world!"])

    monkeypatch.setattr("studylife_ai.llm.client.litellm.acompletion", fake_acompletion)
    _mock_no_retrieval(monkeypatch)

    response = await client.post("/chat", json={"messages": [{"role": "user", "content": "Hi"}]})

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert events == [{"delta": "Hello"}, {"delta": ", world!"}, {"sources": []}]
    assert response.text.strip().endswith("data: [DONE]")
    # M5: cost/latency logging (llm/logging.py) reads call_site/user_id from
    # this metadata - a regression here would silently break call-site/user
    # tagging without failing any logging test on its own.
    assert calls[0]["metadata"] == {"call_site": "chat", "user_id": TEST_USER_ID}


async def test_chat_passes_reasoning_effort_when_configured(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    """Regression test: reasoning models (e.g. gpt-5) spend real, billed tokens "thinking"
    before any visible output unless reasoning_effort is set - see docs/decisions.md."""
    from studylife_ai import config as config_module

    calls = []

    async def fake_acompletion(*_args: object, **kwargs: object) -> object:
        calls.append(kwargs)
        return _make_fake_stream(["Hello"])

    monkeypatch.setattr("studylife_ai.llm.client.litellm.acompletion", fake_acompletion)
    _mock_no_retrieval(monkeypatch)
    settings = config_module.get_settings()
    monkeypatch.setattr(
        "studylife_ai.api.chat.get_settings",
        lambda: settings.model_copy(update={"llm_reasoning_effort": "minimal"}),
    )

    response = await client.post("/chat", json={"messages": [{"role": "user", "content": "Hi"}]})

    assert response.status_code == 200
    assert calls[0]["reasoning_effort"] == "minimal"


async def test_chat_streams_error_event_on_llm_failure(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    async def fake_acompletion(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr("studylife_ai.llm.client.litellm.acompletion", fake_acompletion)
    _mock_no_retrieval(monkeypatch)

    response = await client.post("/chat", json={"messages": [{"role": "user", "content": "Hi"}]})

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert events == [{"error": "LLM request failed"}, {"sources": []}]
    assert response.text.strip().endswith("data: [DONE]")


@pytest.mark.parametrize("payload", [{"messages": []}, {}])
async def test_chat_rejects_invalid_request(
    client: AsyncClient, payload: dict[str, object]
) -> None:
    response = await client.post("/chat", json=payload)

    assert response.status_code == 422


async def test_chat_augments_llm_messages_with_retrieved_context(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    chunk = RetrievedChunk(
        content_type="note",
        entity_id=1,
        chunk_index=0,
        content="det(A - λI) = 0",
        title="Eigenwerte",
        course_id=3,
        session_id=None,
        score=0.9,
        session_start=None,
    )

    async def fake_retrieve_context(
        query: str, settings: object, store: object, user_id: str
    ) -> list[RetrievedChunk]:
        assert query == "Was sind Eigenwerte?"
        return [chunk]

    monkeypatch.setattr("studylife_ai.api.chat._retrieve_context", fake_retrieve_context)

    captured_messages: list[dict[str, str]] = []

    async def fake_acompletion(
        *_args: object, messages: list[dict[str, str]], **_kwargs: object
    ) -> object:
        captured_messages.extend(messages)
        return _make_fake_stream(["Eigenwerte..."])

    monkeypatch.setattr("studylife_ai.llm.client.litellm.acompletion", fake_acompletion)

    response = await client.post(
        "/chat", json={"messages": [{"role": "user", "content": "Was sind Eigenwerte?"}]}
    )

    assert response.status_code == 200
    assert captured_messages[0]["role"] == "system"
    assert "[1] Note: Eigenwerte\ndet(A - λI) = 0" in captured_messages[0]["content"]
    # A second system message injects the current date (see api/chat.py - matches api/agent.py's
    # identical fix) so the model can answer relative-date questions correctly.
    assert captured_messages[1]["role"] == "system"
    assert "current date" in captured_messages[1]["content"]
    assert captured_messages[2] == {"role": "user", "content": "Was sind Eigenwerte?"}

    events = _parse_sse_events(response.text)
    assert events[-1] == {
        "sources": [{"content_type": "note", "entity_id": 1, "title": "Eigenwerte", "course_id": 3}]
    }


async def test_chat_returns_401_when_proxy_token_header_is_missing(
    client: AsyncClient,
) -> None:
    response = await client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}]},
        headers={PROXY_TOKEN_HEADER: ""},
    )

    assert response.status_code == 401


async def test_chat_returns_401_for_a_token_signed_with_the_wrong_secret(
    client: AsyncClient,
) -> None:
    bad_token = make_proxy_token("test-user", secret="wrong-secret")

    response = await client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}]},
        headers={PROXY_TOKEN_HEADER: bad_token},
    )

    assert response.status_code == 401


async def test_chat_returns_401_for_an_expired_token(client: AsyncClient) -> None:
    expired_token = make_proxy_token("test-user", secret=TEST_SHARED_SECRET, expires_in=-60)

    response = await client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hi"}]},
        headers={PROXY_TOKEN_HEADER: expired_token},
    )

    assert response.status_code == 401


async def test_chat_falls_back_to_no_context_when_retrieval_fails(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    async def failing_retrieve_context(
        query: str, settings: object, store: object, user_id: str
    ) -> list[RetrievedChunk]:
        raise RuntimeError("qdrant unreachable")

    monkeypatch.setattr("studylife_ai.api.chat._retrieve_context", failing_retrieve_context)

    async def fake_acompletion(*_args: object, **_kwargs: object) -> object:
        return _make_fake_stream(["Hello"])

    monkeypatch.setattr("studylife_ai.llm.client.litellm.acompletion", fake_acompletion)

    response = await client.post("/chat", json={"messages": [{"role": "user", "content": "Hi"}]})

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert events == [{"delta": "Hello"}, {"sources": []}]
