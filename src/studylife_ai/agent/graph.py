"""Builds the LangGraph agent: a tool-calling loop that pauses before either
write tool executes (see docs/decisions.md "M4 agent stack" for why
`create_agent`/`HumanInTheLoopMiddleware`, not the deprecated
`create_react_agent`, and why the resume payload has the exact shape used
in `api/agent.py`).
"""

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import BaseTool
from langchain_litellm.chat_models.litellm import ChatLiteLLM
from langgraph.checkpoint.base import BaseCheckpointSaver

from studylife_ai.config import Settings

# Only the two write tools pause for confirmation - read tools (list_courses,
# search_notes) execute immediately, they have no side effects to confirm.
_WRITE_TOOLS = {"create_study_session", "save_note"}


def build_agent(
    *, tools: list[BaseTool], checkpointer: BaseCheckpointSaver[Any], settings: Settings
) -> Any:
    model = ChatLiteLLM(model=settings.llm_model, api_base=settings.llm_api_base)
    return create_agent(
        model=model,
        tools=tools,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={name: True for name in _WRITE_TOOLS},
                description_prefix="This action needs your confirmation before it runs",
            )
        ],
        checkpointer=checkpointer,
    )
