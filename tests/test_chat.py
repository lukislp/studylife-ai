import json
from types import SimpleNamespace

import pytest
from httpx import AsyncClient
from pytest import MonkeyPatch


def _make_fake_stream(texts: list[str]) -> object:
    async def fake_stream() -> object:
        for text in texts:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])

    return fake_stream()


def _parse_sse_events(body: str) -> list[dict[str, str]]:
    return [
        json.loads(chunk.removeprefix("data: "))
        for chunk in body.split("\n\n")
        if chunk and chunk != "data: [DONE]"
    ]


async def test_chat_streams_llm_deltas_as_sse(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    async def fake_acompletion(*_args: object, **_kwargs: object) -> object:
        return _make_fake_stream(["Hello", ", world!"])

    monkeypatch.setattr("studylife_ai.llm.client.litellm.acompletion", fake_acompletion)

    response = await client.post("/chat", json={"messages": [{"role": "user", "content": "Hi"}]})

    assert response.status_code == 200
    assert _parse_sse_events(response.text) == [{"delta": "Hello"}, {"delta": ", world!"}]
    assert response.text.strip().endswith("data: [DONE]")


async def test_chat_streams_error_event_on_llm_failure(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    async def fake_acompletion(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr("studylife_ai.llm.client.litellm.acompletion", fake_acompletion)

    response = await client.post("/chat", json={"messages": [{"role": "user", "content": "Hi"}]})

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert events == [{"error": "LLM request failed"}]
    assert response.text.strip().endswith("data: [DONE]")


@pytest.mark.parametrize("payload", [{"messages": []}, {}])
async def test_chat_rejects_invalid_request(
    client: AsyncClient, payload: dict[str, object]
) -> None:
    response = await client.post("/chat", json=payload)

    assert response.status_code == 422
