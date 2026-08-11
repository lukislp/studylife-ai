import uuid
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, HumanMessage, ToolCall
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pytest import MonkeyPatch

from studylife_ai.agent import graph as graph_module
from studylife_ai.agent.graph import build_agent
from studylife_ai.agent.tools import build_tools
from studylife_ai.config import Settings
from studylife_ai.studylife.models import StudySessionDto
from tests.fakes import FakeToolCallingModel


def _settings() -> Settings:
    return Settings(
        embedding_model="ollama/nomic-embed-text",
        llm_model="ollama/llama3.2",
        llm_api_base="http://localhost:11434",
        retrieval_top_k=5,
    )


def _build_agent_with_fake_model(monkeypatch: MonkeyPatch, responses: list[AIMessage]) -> Any:
    model = FakeToolCallingModel(responses=responses)
    monkeypatch.setattr(graph_module, "ChatLiteLLM", lambda **kwargs: model)
    fake_studylife = AsyncMock()
    fake_studylife.create_session.return_value = StudySessionDto(
        id=99,
        course_id=6,
        course_name="Lineare Algebra",
        start_time=datetime(2026, 8, 12, 10, 0),
        end_time=datetime(2026, 8, 12, 11, 0),
    )
    tools = build_tools(
        studylife=fake_studylife, store=AsyncMock(), settings=_settings(), user_id="test-user"
    )
    agent = build_agent(tools=tools, checkpointer=InMemorySaver(), settings=_settings())
    return agent, fake_studylife


async def test_write_tool_pauses_and_does_not_execute_until_approved(
    monkeypatch: MonkeyPatch,
) -> None:
    propose_msg = AIMessage(
        content="",
        tool_calls=[
            ToolCall(
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
        ],
    )
    final_msg = AIMessage(content="Done, session created.")
    agent, fake_studylife = _build_agent_with_fake_model(monkeypatch, [propose_msg, final_msg])

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    result = await agent.ainvoke(
        {"messages": [HumanMessage("leg mir eine Session für Lineare Algebra an")]},
        config=config,
    )

    assert result.get("__interrupt__")
    interrupt = result["__interrupt__"][0]
    assert interrupt.value["action_requests"][0]["name"] == "create_study_session"
    fake_studylife.create_session.assert_not_awaited()

    resumed = await agent.ainvoke(
        Command(resume={"decisions": [{"type": "approve"}]}), config=config
    )

    assert not resumed.get("__interrupt__")
    fake_studylife.create_session.assert_awaited_once()


async def test_write_tool_rejection_does_not_execute(monkeypatch: MonkeyPatch) -> None:
    propose_msg = AIMessage(
        content="",
        tool_calls=[
            ToolCall(
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
        ],
    )
    final_msg = AIMessage(content="Ok, nicht angelegt.")
    agent, fake_studylife = _build_agent_with_fake_model(monkeypatch, [propose_msg, final_msg])

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    await agent.ainvoke(
        {"messages": [HumanMessage("leg mir eine Session für Lineare Algebra an")]},
        config=config,
    )

    resumed = await agent.ainvoke(
        Command(resume={"decisions": [{"type": "reject", "message": "Nutzer hat abgelehnt"}]}),
        config=config,
    )

    assert not resumed.get("__interrupt__")
    fake_studylife.create_session.assert_not_awaited()


async def test_read_only_answer_does_not_pause(monkeypatch: MonkeyPatch) -> None:
    final_msg = AIMessage(content="Du hast 58 Kurse.")
    agent, fake_studylife = _build_agent_with_fake_model(monkeypatch, [final_msg])

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = await agent.ainvoke(
        {"messages": [HumanMessage("Wie viele Kurse habe ich?")]}, config=config
    )

    assert not result.get("__interrupt__")
    fake_studylife.create_session.assert_not_awaited()
