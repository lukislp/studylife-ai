from datetime import date

from pytest import MonkeyPatch

from studylife_ai.rag import date_parse as date_parse_module
from studylife_ai.rag.date_parse import DateRange, _build_prompt, _parse_response, parse_date_range


def test_build_prompt_includes_query_and_todays_date() -> None:
    prompt = _build_prompt("welche Sessions hatten wir letzte Woche?", today=date(2026, 8, 12))

    assert "welche Sessions hatten wir letzte Woche?" in prompt
    assert "2026-08-12, Wednesday" in prompt


def test_build_prompt_covers_named_categories_and_forbids_llm_arithmetic() -> None:
    prompt = _build_prompt("query", today=date(2026, 8, 12))

    assert "THIS_WEEK" in prompt and "diese Woche" in prompt
    assert "LAST_WEEK" in prompt and "letzte Woche" in prompt
    assert "THIS_MONTH" in prompt
    assert "MONTH:<1-12>" in prompt and "im Mai" in prompt
    assert "SINCE_MONTH_START" in prompt and "seit Anfang des Monats" in prompt
    assert "NONE" in prompt
    assert "DATE:<YYYY-MM-DD>" in prompt
    # Regression test for the 2026-08-12 "letzte Woche"/"diese Woche" boundary bug: the LLM must
    # never be asked to compute week/month boundaries itself (see docs/decisions.md "NL
    # date-range resolution: LLM classifies, Python computes").
    assert "Do NOT calculate any week/month boundary dates yourself" in prompt


def test_parse_response_exact_date() -> None:
    assert _parse_response("DATE:2026-08-10", today=date(2026, 8, 12)) == DateRange(
        date(2026, 8, 10), date(2026, 8, 10)
    )


def test_parse_response_bare_date_without_prefix_still_recovers() -> None:
    assert _parse_response("2026-08-10", today=date(2026, 8, 12)) == DateRange(
        date(2026, 8, 10), date(2026, 8, 10)
    )


def test_parse_response_none_reply_returns_none() -> None:
    assert _parse_response("NONE", today=date(2026, 8, 12)) is None


def test_parse_response_empty_or_garbage_returns_none() -> None:
    assert _parse_response("", today=date(2026, 8, 12)) is None
    assert _parse_response("I'm not sure what you mean", today=date(2026, 8, 12)) is None


def test_parse_response_invalid_date_returns_none() -> None:
    assert _parse_response("DATE:2026-13-99", today=date(2026, 8, 12)) is None


def test_parse_response_this_week_is_monday_to_sunday_containing_today() -> None:
    # 2026-08-12 is a Wednesday; this week is Mon 2026-08-10 - Sun 2026-08-16.
    assert _parse_response("THIS_WEEK", today=date(2026, 8, 12)) == DateRange(
        date(2026, 8, 10), date(2026, 8, 16)
    )


def test_parse_response_last_week_is_the_previous_monday_to_sunday() -> None:
    # Last week: Mon 2026-08-03 - Sun 2026-08-09.
    assert _parse_response("LAST_WEEK", today=date(2026, 8, 12)) == DateRange(
        date(2026, 8, 3), date(2026, 8, 9)
    )


def test_parse_response_this_week_and_last_week_from_a_monday() -> None:
    # Boundary case: today itself is a Monday - "this week" must still start today, not the
    # previous week, and "last week" must be the fully completed week just before it.
    monday = date(2026, 8, 10)
    assert _parse_response("THIS_WEEK", today=monday) == DateRange(
        date(2026, 8, 10), date(2026, 8, 16)
    )
    assert _parse_response("LAST_WEEK", today=monday) == DateRange(
        date(2026, 8, 3), date(2026, 8, 9)
    )


def test_parse_response_this_month() -> None:
    assert _parse_response("THIS_MONTH", today=date(2026, 8, 12)) == DateRange(
        date(2026, 8, 1), date(2026, 8, 31)
    )


def test_parse_response_since_month_start() -> None:
    assert _parse_response("SINCE_MONTH_START", today=date(2026, 8, 12)) == DateRange(
        date(2026, 8, 1), date(2026, 8, 12)
    )


def test_parse_response_named_month_in_the_past_this_year() -> None:
    # "im Mai" asked in August 2026 - May already happened this year, so this year's May.
    assert _parse_response("MONTH:5", today=date(2026, 8, 12)) == DateRange(
        date(2026, 5, 1), date(2026, 5, 31)
    )


def test_parse_response_named_month_not_yet_happened_this_year_resolves_to_last_year() -> None:
    # "im Dezember" asked in August 2026 - December hasn't happened yet this year, so last
    # December.
    assert _parse_response("MONTH:12", today=date(2026, 8, 12)) == DateRange(
        date(2025, 12, 1), date(2025, 12, 31)
    )


def test_parse_response_named_month_including_the_current_month_resolves_to_this_year() -> None:
    assert _parse_response("MONTH:8", today=date(2026, 8, 12)) == DateRange(
        date(2026, 8, 1), date(2026, 8, 31)
    )


def test_parse_response_out_of_range_month_returns_none() -> None:
    assert _parse_response("MONTH:13", today=date(2026, 8, 12)) is None


async def test_parse_date_range_returns_range_on_well_formed_response(
    monkeypatch: MonkeyPatch,
) -> None:
    async def fake_complete_chat(messages: list[object], **kwargs: object) -> str:
        return "LAST_WEEK"

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
