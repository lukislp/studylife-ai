from datetime import date

from pytest import MonkeyPatch

from studylife_ai.rag import date_parse as date_parse_module
from studylife_ai.rag.date_parse import DateRange, _build_prompt, _parse_response, parse_date_range


def test_build_prompt_includes_query_and_todays_date() -> None:
    prompt = _build_prompt("welche Sessions hatten wir letzte Woche?", today=date(2026, 8, 12))

    assert "welche Sessions hatten wir letzte Woche?" in prompt
    assert "2026-08-12, Wednesday" in prompt


def test_build_prompt_covers_named_phrase_types_and_none_instruction() -> None:
    prompt = _build_prompt("query", today=date(2026, 8, 12))

    assert "letzte Woche" in prompt and "last week" in prompt
    assert "diese Woche" in prompt and "this week" in prompt
    assert "im Mai" in prompt and "in May" in prompt
    assert "seit Anfang des Monats" in prompt
    assert "NONE" in prompt
    assert "START|END" in prompt


def test_parse_response_well_formed_range() -> None:
    assert _parse_response("2026-08-03|2026-08-09") == DateRange(date(2026, 8, 3), date(2026, 8, 9))


def test_parse_response_single_bare_date() -> None:
    assert _parse_response("2026-08-10") == DateRange(date(2026, 8, 10), date(2026, 8, 10))


def test_parse_response_swaps_reversed_range() -> None:
    assert _parse_response("2026-08-09|2026-08-03") == DateRange(date(2026, 8, 3), date(2026, 8, 9))


def test_parse_response_tolerates_surrounding_prose_and_whitespace() -> None:
    assert _parse_response("  2026-08-03 | 2026-08-09  \n") == DateRange(
        date(2026, 8, 3), date(2026, 8, 9)
    )


def test_parse_response_none_reply_returns_none() -> None:
    assert _parse_response("NONE") is None


def test_parse_response_empty_or_garbage_returns_none() -> None:
    assert _parse_response("") is None
    assert _parse_response("I'm not sure what you mean") is None


def test_parse_response_syntactically_matching_but_invalid_date_returns_none() -> None:
    assert _parse_response("2026-13-99|2026-13-99") is None


async def test_parse_date_range_returns_range_on_well_formed_response(
    monkeypatch: MonkeyPatch,
) -> None:
    async def fake_complete_chat(messages: list[object], **kwargs: object) -> str:
        return "2026-08-03|2026-08-09"

    monkeypatch.setattr(date_parse_module, "complete_chat", fake_complete_chat)

    result = await parse_date_range(
        "letzte Woche",
        model="openai/gpt-4o",
        api_base=None,
        timeout=30.0,
        today=date(2026, 8, 12),
    )

    assert result == DateRange(date(2026, 8, 3), date(2026, 8, 9))


async def test_parse_date_range_returns_none_for_none_reply(monkeypatch: MonkeyPatch) -> None:
    async def fake_complete_chat(messages: list[object], **kwargs: object) -> str:
        return "NONE"

    monkeypatch.setattr(date_parse_module, "complete_chat", fake_complete_chat)

    result = await parse_date_range(
        "was haben wir in Analysis behandelt?",
        model="openai/gpt-4o",
        api_base=None,
        timeout=30.0,
        today=date(2026, 8, 12),
    )

    assert result is None


async def test_parse_date_range_returns_none_and_does_not_raise_on_call_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    async def failing_complete_chat(*args: object, **kwargs: object) -> str:
        raise RuntimeError("network error")

    monkeypatch.setattr(date_parse_module, "complete_chat", failing_complete_chat)

    result = await parse_date_range(
        "letzte Woche", model="openai/gpt-4o", api_base=None, timeout=30.0, today=date(2026, 8, 12)
    )

    assert result is None


async def test_parse_date_range_pins_temperature_to_zero(monkeypatch: MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_complete_chat(messages: list[object], **kwargs: object) -> str:
        captured.update(kwargs)
        return "NONE"

    monkeypatch.setattr(date_parse_module, "complete_chat", fake_complete_chat)

    await parse_date_range(
        "query", model="openai/gpt-4o", api_base=None, timeout=30.0, today=date(2026, 8, 12)
    )

    assert captured["temperature"] == 0.0
