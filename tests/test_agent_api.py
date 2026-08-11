from unittest.mock import AsyncMock

from httpx import AsyncClient
from langchain_core.messages import AIMessage, ToolCall
from langgraph.checkpoint.memory import InMemorySaver
from pytest import MonkeyPatch

from studylife_ai.agent.graph import build_agent
from studylife_ai.agent.tools import build_tools
from studylife_ai.config import get_settings
from studylife_ai.main import app
from studylife_ai.studylife.models import StudySessionDto
from tests.fakes import FakeToolCallingModel

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


def _install_fake_agent(monkeypatch: MonkeyPatch, responses: list[AIMessage]) -> AsyncMock:
    fake_studylife = AsyncMock()
    fake_studylife.create_session.return_value = StudySessionDto(
        id=99,
        course_id=6,
        course_name="Lineare Algebra",
        start_time="2026-08-12T10:00:00",  # type: ignore[arg-type]
        end_time="2026-08-12T11:00:00",  # type: ignore[arg-type]
    )
    tools = build_tools(studylife=fake_studylife, store=AsyncMock(), settings=get_settings())
    fake_model = FakeToolCallingModel(responses=responses)
    # build_agent has no model= override - ChatLiteLLM is constructed inside
    # it, so swap that constructor for the duration of this build() call.
    monkeypatch.setattr("studylife_ai.agent.graph.ChatLiteLLM", lambda **kwargs: fake_model)
    checkpointer = InMemorySaver()
    compiled = build_agent(tools=tools, checkpointer=checkpointer, settings=get_settings())
    monkeypatch.setattr(app.state, "agent", compiled, raising=False)
    monkeypatch.setattr(app.state, "agent_checkpointer", checkpointer, raising=False)
    return fake_studylife


async def test_agent_answers_directly_when_no_tool_needed(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    _install_fake_agent(monkeypatch, [AIMessage(content="Du hast 58 Kurse.")])

    response = await client.post("/agent", json={"message": "Wie viele Kurse habe ich?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Du hast 58 Kurse."
    assert body["pending_actions"] == []


async def test_agent_returns_pending_action_for_write_tool(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    propose = AIMessage(content="", tool_calls=[_CREATE_SESSION_CALL])
    _install_fake_agent(monkeypatch, [propose, AIMessage(content="Done.")])

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
    fake_studylife = _install_fake_agent(
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
    fake_studylife = _install_fake_agent(
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
    _install_fake_agent(monkeypatch, [AIMessage(content="irrelevant")])

    response = await client.post(
        "/agent/confirm", json={"thread_id": "does-not-exist", "decision": "approve"}
    )

    assert response.status_code == 404


async def test_agent_returns_503_when_not_configured(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(app.state, "agent", None, raising=False)

    response = await client.post("/agent", json={"message": "hi"})

    assert response.status_code == 503


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
    fake_studylife = _install_fake_agent(
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
    fake_studylife = _install_fake_agent(
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
