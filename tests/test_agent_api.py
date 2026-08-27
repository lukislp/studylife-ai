from unittest.mock import AsyncMock

from httpx import AsyncClient
from langchain_core.messages import AIMessage, ToolCall
from litellm.exceptions import Timeout as LiteLLMTimeout
from pytest import MonkeyPatch

from studylife_ai.api import agent as agent_module
from studylife_ai.api.identity import PROXY_TOKEN_HEADER
from studylife_ai.config import Settings
from studylife_ai.main import app
from studylife_ai.studylife.models import CourseDto, StudySessionDto
from tests.conftest import TEST_SHARED_SECRET, TEST_USER_ID, make_proxy_token
from tests.fakes import FakeToolCallingModel


def test_agent_system_prompt_frames_search_notes_output_as_untrusted_data() -> None:
    """Regression test: the agent's system prompt must state that tool-returned note content
    (search_notes) is untrusted data, never instructions - see docs/decisions.md and
    agent/tools.py's own per-call framing on the tool output itself (belt-and-braces)."""
    assert "untrusted" in agent_module._AGENT_SYSTEM_PROMPT.lower()
    assert "search_notes" in agent_module._AGENT_SYSTEM_PROMPT
    assert "never" in agent_module._AGENT_SYSTEM_PROMPT.lower()


_CREATE_SESSION_CALL = ToolCall(
    name="create_study_session",
    args={
        "course_id": 6,
        "course_name": "Lineare Algebra",
        "start_time": "2026-08-12T10:00:00",
        "end_time": "2026-08-12T11:00:00",
    },
    id="call_1",
    type="tool_call",
)


def _agent_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "studylife_api_base_url": "http://studylife.test",
        "studylife_shared_secret": TEST_SHARED_SECRET,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


async def _install_fake_agent(
    monkeypatch: MonkeyPatch,
    responses: list[AIMessage],
    *,
    fail_on_call_index: int | None = None,
    fail_exception: Exception | None = None,
) -> AsyncMock:
    """Stands in for the two things api/agent.py._build_agent()
    constructs per request: the StudyLifeClient (a fake, so tool calls never
    hit real HTTP) and the model bound inside build_agent() (ChatLiteLLM,
    swapped for a scripted fake). Also registers a fake AiApiKey for
    TEST_USER_ID in the test-isolated in-memory RegisteredKeyStore (see
    conftest.py) - without one, _studylife_client_for() 404s before ever
    reaching StudyLifeClient. The checkpointer itself is NOT faked here -
    api/agent.py reads the real one the app's lifespan already builds
    (see main.py), which is fine across tests since each uses a fresh random
    thread_id.

    `fail_on_call_index`/`fail_exception` (see FakeToolCallingModel) simulate a transient
    provider failure at a specific model call within the run - the same fake model instance
    is reused across /agent and /agent/confirm (both call _build_agent, which is monkeypatched
    to keep returning this one object), so its call-count state persists across that boundary
    exactly like the real per-thread checkpointed run would."""
    fake_studylife = AsyncMock()
    # api/agent.py uses `async with studylife_client:` on an already-built
    # instance - AsyncMock's default __aenter__ return value is a DIFFERENT
    # auto-mock, not this instance, so assertions like
    # create_session.assert_awaited_once() below would silently check the
    # wrong object without this.
    fake_studylife.__aenter__.return_value = fake_studylife
    fake_studylife.create_session.return_value = StudySessionDto(
        id=99,
        course_id=6,
        course_name="Lineare Algebra",
        start_time="2026-08-12T10:00:00",  # type: ignore[arg-type]
        end_time="2026-08-12T11:00:00",  # type: ignore[arg-type]
    )
    fake_model = FakeToolCallingModel(
        responses=responses, fail_on_call_index=fail_on_call_index, fail_exception=fail_exception
    )
    # build_agent has no model= override - ChatLiteLLM is constructed inside
    # it, so swap that constructor for the duration of this build() call.
    monkeypatch.setattr("studylife_ai.agent.graph.ChatLiteLLM", lambda **kwargs: fake_model)
    monkeypatch.setattr("studylife_ai.api.agent.StudyLifeClient", lambda **kwargs: fake_studylife)
    monkeypatch.setattr("studylife_ai.api.agent.get_settings", _agent_settings)
    await app.state.registered_key_store.set(TEST_USER_ID, "fake-registered-key")
    return fake_studylife


async def test_agent_answers_directly_when_no_tool_needed(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    await _install_fake_agent(monkeypatch, [AIMessage(content="Du hast 58 Kurse.")])

    response = await client.post("/agent", json={"message": "Wie viele Kurse habe ich?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Du hast 58 Kurse."
    assert body["pending_actions"] == []


async def test_agent_returns_pending_action_for_write_tool(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    propose = AIMessage(content="", tool_calls=[_CREATE_SESSION_CALL])
    await _install_fake_agent(monkeypatch, [propose, AIMessage(content="Done.")])

    response = await client.post(
        "/agent", json={"message": "leg mir eine Session für Lineare Algebra an"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] is None
    assert len(body["pending_actions"]) == 1
    assert body["pending_actions"][0]["tool"] == "create_study_session"
    assert isinstance(body["pending_actions"][0]["thread_id"], str)


async def test_agent_confirm_approve_executes_the_tool(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    propose = AIMessage(content="", tool_calls=[_CREATE_SESSION_CALL])
    fake_studylife = await _install_fake_agent(
        monkeypatch, [propose, AIMessage(content="Session erstellt.")]
    )

    propose_response = await client.post(
        "/agent", json={"message": "leg mir eine Session für Lineare Algebra an"}
    )
    thread_id = propose_response.json()["pending_actions"][0]["thread_id"]

    confirm_response = await client.post(
        "/agent/confirm", json={"thread_id": thread_id, "decision": "approve"}
    )

    assert confirm_response.status_code == 200
    assert confirm_response.json()["answer"] == "Session erstellt."
    fake_studylife.create_session.assert_awaited_once()


async def test_agent_confirm_reject_does_not_execute_the_tool(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    propose = AIMessage(content="", tool_calls=[_CREATE_SESSION_CALL])
    fake_studylife = await _install_fake_agent(
        monkeypatch, [propose, AIMessage(content="Ok, abgebrochen.")]
    )

    propose_response = await client.post(
        "/agent", json={"message": "leg mir eine Session für Lineare Algebra an"}
    )
    thread_id = propose_response.json()["pending_actions"][0]["thread_id"]

    confirm_response = await client.post(
        "/agent/confirm", json={"thread_id": thread_id, "decision": "reject"}
    )

    assert confirm_response.status_code == 200
    fake_studylife.create_session.assert_not_awaited()


async def test_agent_confirm_with_unknown_thread_id_returns_404(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    await _install_fake_agent(monkeypatch, [AIMessage(content="irrelevant")])

    response = await client.post(
        "/agent/confirm",
        json={"thread_id": f"{TEST_USER_ID}:does-not-exist", "decision": "approve"},
    )

    assert response.status_code == 404


async def test_agent_confirm_rejects_a_thread_belonging_to_another_user(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    """The thread_id ownership check (see docs/decisions.md "M4.5 Multi-user
    support") must reject before the checkpointer is ever touched - a
    mismatched user_id prefix is enough, regardless of whether the thread
    even exists."""
    await _install_fake_agent(monkeypatch, [AIMessage(content="irrelevant")])

    response = await client.post(
        "/agent/confirm",
        json={"thread_id": "someone-else:whatever", "decision": "approve"},
    )

    assert response.status_code == 403


async def test_agent_returns_503_when_not_configured(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "studylife_ai.api.agent.get_settings",
        lambda: _agent_settings(studylife_api_base_url=None),
    )

    response = await client.post("/agent", json={"message": "hi"})

    assert response.status_code == 503


async def test_agent_returns_404_when_no_key_registered_for_user(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    """A user who never generated an AiApiKey (or had it revoked) has no
    entry in RegisteredKeyStore - the agent needs a real, usable credential
    for StudyLife API calls, which the proxy token alone doesn't provide."""
    monkeypatch.setattr("studylife_ai.api.agent.get_settings", _agent_settings)
    # Deliberately do NOT register a key for TEST_USER_ID.

    response = await client.post("/agent", json={"message": "hi"})

    assert response.status_code == 404


async def test_agent_returns_401_when_proxy_token_header_is_missing(
    client: AsyncClient,
) -> None:
    response = await client.post("/agent", json={"message": "hi"}, headers={PROXY_TOKEN_HEADER: ""})

    assert response.status_code == 401


async def test_agent_returns_401_for_a_token_signed_with_the_wrong_secret(
    client: AsyncClient,
) -> None:
    bad_token = make_proxy_token(TEST_USER_ID, secret="wrong-secret")

    response = await client.post(
        "/agent", json={"message": "hi"}, headers={PROXY_TOKEN_HEADER: bad_token}
    )

    assert response.status_code == 401


async def test_agent_confirm_surfaces_all_pending_actions_in_one_turn(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    save_note_call = ToolCall(
        name="save_note",
        args={"title": "Zusammenfassung", "content": "..."},
        id="call_2",
        type="tool_call",
    )
    propose = AIMessage(content="", tool_calls=[_CREATE_SESSION_CALL, save_note_call])
    fake_studylife = await _install_fake_agent(
        monkeypatch, [propose, AIMessage(content="Beides erledigt.")]
    )

    response = await client.post(
        "/agent", json={"message": "leg mir eine Session an und speichere eine Notiz"}
    )

    assert response.status_code == 200
    body = response.json()
    tools_proposed = {a["tool"] for a in body["pending_actions"]}
    assert tools_proposed == {"create_study_session", "save_note"}

    thread_id = body["pending_actions"][0]["thread_id"]
    confirm_response = await client.post(
        "/agent/confirm", json={"thread_id": thread_id, "decision": "approve"}
    )

    assert confirm_response.status_code == 200
    fake_studylife.create_session.assert_awaited_once()
    fake_studylife.create_note.assert_awaited_once()


async def test_agent_confirm_tool_failure_returns_502_and_invalidates_thread(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    propose = AIMessage(content="", tool_calls=[_CREATE_SESSION_CALL])
    fake_studylife = await _install_fake_agent(
        monkeypatch, [propose, AIMessage(content="unused - tool raises before this")]
    )
    fake_studylife.create_session.side_effect = RuntimeError("StudyLife API rejected the request")

    propose_response = await client.post(
        "/agent", json={"message": "leg mir eine Session für Lineare Algebra an"}
    )
    thread_id = propose_response.json()["pending_actions"][0]["thread_id"]

    confirm_response = await client.post(
        "/agent/confirm", json={"thread_id": thread_id, "decision": "approve"}
    )
    assert confirm_response.status_code == 502

    # The thread must not be retryable afterwards - it should now report as
    # having no pending action, not silently retry the same failing call.
    retry_response = await client.post(
        "/agent/confirm", json={"thread_id": thread_id, "decision": "approve"}
    )
    assert retry_response.status_code == 404


async def test_agent_confirm_transient_failure_preserves_checkpoint_for_retry(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    """Regression test (docs/decisions.md "Preserve the checkpoint on a transient agent
    failure"): unlike a permanent failure (see the 502 test above), a one-off provider hiccup
    (timeout/429/5xx) during POST /agent/confirm must NOT discard the pending action - the
    same thread_id must still be resumable afterwards, so the client can just retry the
    confirm call instead of losing the whole propose-confirm flow.

    The failure is placed in the model's SECOND call (index 1) - the follow-up "here's what I
    did" call that runs AFTER create_study_session has already executed successfully, not in
    the tool call itself - so this also verifies the tool isn't silently re-invoked by the
    retry (`create_session.assert_awaited_once()` below): only the model step that actually
    failed needs to be retried, since LangGraph only re-runs a step that never completed."""
    propose = AIMessage(content="", tool_calls=[_CREATE_SESSION_CALL])
    fake_studylife = await _install_fake_agent(
        monkeypatch,
        [propose, AIMessage(content="Session erstellt.")],
        fail_on_call_index=1,
        fail_exception=LiteLLMTimeout(
            message="upstream timed out", model="test-model", llm_provider="test-provider"
        ),
    )

    propose_response = await client.post(
        "/agent", json={"message": "leg mir eine Session für Lineare Algebra an"}
    )
    thread_id = propose_response.json()["pending_actions"][0]["thread_id"]

    confirm_response = await client.post(
        "/agent/confirm", json={"thread_id": thread_id, "decision": "approve"}
    )
    assert confirm_response.status_code == 503
    # The tool itself already succeeded on this first attempt - only the follow-up model call
    # failed.
    fake_studylife.create_session.assert_awaited_once()

    # Retry the exact same confirm call against the exact same thread_id - it must still be
    # pending, not 404 (contrast with the permanent-failure test above, which expects exactly
    # a 404 here).
    retry_response = await client.post(
        "/agent/confirm", json={"thread_id": thread_id, "decision": "approve"}
    )

    assert retry_response.status_code == 200
    assert retry_response.json()["answer"] == "Session erstellt."
    # Still exactly once - the retry did not re-run the already-completed tool call.
    fake_studylife.create_session.assert_awaited_once()


async def test_agent_pending_action_shows_real_catalog_course_name(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    """Regression test: the human-confirmation display previously showed whatever
    `course_name` the model typed into create_study_session's args verbatim - even though
    StudyLife's own API ignores that field and derives the real name from `course_id`
    server-side (see docs/decisions.md "F15/O6-ai phase B"). The model can get the display
    name wrong (e.g. a stale/mistranslated name); the confirmation should show the real
    catalog name, resolved from the already-fetched list_courses result, not the model's own
    text - so the user confirms against truth."""
    list_courses_call = ToolCall(name="list_courses", args={}, id="call_0", type="tool_call")
    list_then_create = AIMessage(content="", tool_calls=[list_courses_call])
    propose = AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                name="create_study_session",
                args={
                    "course_id": 6,
                    "course_name": "Linear Algebra (wrong/stale name)",
                    "start_time": "2026-08-12T10:00:00",
                    "end_time": "2026-08-12T11:00:00",
                },
                id="call_1",
                type="tool_call",
            )
        ],
    )
    fake_studylife = await _install_fake_agent(
        monkeypatch, [list_then_create, propose, AIMessage(content="Done.")]
    )
    fake_studylife.get_courses.return_value = [
        CourseDto(id=6, semester=3, name="Lineare Algebra", code="MATH101", ects=5)
    ]

    response = await client.post(
        "/agent", json={"message": "leg mir eine Session für Lineare Algebra an"}
    )

    assert response.status_code == 200
    args = response.json()["pending_actions"][0]["args"]
    assert args["course_name"] == "Lineare Algebra"
