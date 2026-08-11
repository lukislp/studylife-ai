"""Agent endpoint: tool-calling with a confirm step before any write (see
docs/decisions.md "M4 agent stack"). Two endpoints: `POST /agent` answers
directly or proposes one or more write actions; `POST /agent/confirm`
executes (or rejects) all of a turn's previously proposed actions, resuming
the paused LangGraph run by its `thread_id`.

Multi-user (see docs/decisions.md "M4.5 Multi-user support"): the agent
graph is rebuilt fresh per request from the calling user's own
`StudyLifeClient`, not shared across users via a single startup-built graph.
`thread_id` embeds the owning `user_id` (`f"{user_id}:{uuid4()}"`), and
`POST /agent/confirm` rejects with 403 if the caller's own resolved
`user_id` doesn't match the thread_id's prefix - before the checkpointer is
ever touched, so no other user's pending-action state is ever read.
"""

import logging
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command

from studylife_ai.agent.graph import build_agent
from studylife_ai.agent.tools import build_tools
from studylife_ai.api.identity import ResolvedIdentity, resolve_identity, verify_identity
from studylife_ai.config import get_settings
from studylife_ai.schemas.agent import AgentRequest, AgentResponse, ConfirmRequest, PendingAction
from studylife_ai.studylife.client import StudyLifeClient

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agent"])

_NOT_CONFIGURED_DETAIL = "Agent not available - STUDYLIFE_API_BASE_URL must be set."


def _build_agent(
    http_request: Request, identity: ResolvedIdentity, studylife_client: StudyLifeClient
) -> Any:
    """Builds a fresh agent graph scoped to `studylife_client` (see
    docs/decisions.md "M4.5 Multi-user support" - "Agent graph: rebuilt per
    request"). Caller owns `studylife_client`'s lifecycle - build it with
    `async with StudyLifeClient(...)` so it's closed even if this raises."""
    settings = get_settings()
    tools = build_tools(
        studylife=studylife_client,
        store=http_request.app.state.qdrant_store,
        settings=settings,
        user_id=identity.user_id,
    )
    return build_agent(
        tools=tools, checkpointer=http_request.app.state.agent_checkpointer, settings=settings
    )


def _response_from_result(result: dict[str, Any], thread_id: str) -> AgentResponse:
    interrupts = result.get("__interrupt__")
    if interrupts:
        # Usually one action, but the model can propose several writes in a
        # single turn (e.g. "create a session and save a note") - surface
        # all of them, since /agent/confirm applies one decision to all.
        return AgentResponse(
            pending_actions=[
                PendingAction(
                    tool=action_request["name"],
                    args=action_request["args"],
                    description=action_request.get("description", ""),
                    thread_id=thread_id,
                )
                for action_request in interrupts[0].value["action_requests"]
            ]
        )
    last_ai_message = next(
        (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None
    )
    return AgentResponse(answer=str(last_ai_message.content) if last_ai_message else "")


async def _invoke_and_handle_failure(
    compiled_agent: Any, checkpointer: Any, input_or_command: Any, thread_id: str
) -> dict[str, Any]:
    """Runs the agent and translates a tool/model failure into a clean HTTP
    error instead of an unhandled 500 - and, critically, deletes the
    thread's checkpoint on failure. Without this, a failed write tool (e.g.
    StudyLife rejecting an invalid course_id) leaves the paused state
    exactly where it was: retrying POST /agent/confirm with the same
    thread_id would just retry - and fail - the identical tool call forever,
    since the thread still looks "pending" to the 404 check.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result: dict[str, Any] = await compiled_agent.ainvoke(input_or_command, config=config)
    except Exception:
        logger.exception("Agent run failed for thread_id=%s", thread_id)
        await checkpointer.adelete_thread(thread_id)
        raise HTTPException(
            status_code=502, detail="The action failed while running. Please try again."
        ) from None
    return result


@router.post("/agent")
async def run_agent(
    request: AgentRequest,
    http_request: Request,
    identity: ResolvedIdentity = Depends(resolve_identity),
) -> AgentResponse:
    settings = get_settings()
    if not settings.studylife_api_base_url:
        raise HTTPException(status_code=503, detail=_NOT_CONFIGURED_DETAIL)

    thread_id = f"{identity.user_id}:{uuid.uuid4()}"
    async with StudyLifeClient(
        base_url=settings.studylife_api_base_url, api_key=identity.ai_api_key
    ) as studylife_client:
        await verify_identity(studylife_client)
        compiled_agent = _build_agent(http_request, identity, studylife_client)
        # The model has no other way to know "today" - without this, relative
        # dates ("morgen", "nächste Woche") get resolved against training data
        # instead of the real current date (confirmed live: "morgen" became a
        # date in 2023). Local time, not UTC - matches StudyLife's own sessions,
        # which store naive local timestamps (see docs/decisions.md).
        now = datetime.now().strftime("%Y-%m-%d %H:%M, %A")
        result = await _invoke_and_handle_failure(
            compiled_agent,
            http_request.app.state.agent_checkpointer,
            {
                "messages": [
                    SystemMessage(f"The current date and time is {now}."),
                    HumanMessage(request.message),
                ]
            },
            thread_id,
        )
    return _response_from_result(result, thread_id)


@router.post("/agent/confirm")
async def confirm_agent_action(
    request: ConfirmRequest,
    http_request: Request,
    identity: ResolvedIdentity = Depends(resolve_identity),
) -> AgentResponse:
    if not request.thread_id.startswith(f"{identity.user_id}:"):
        # Rejected before the checkpointer - or even a StudyLifeClient - is
        # ever touched: no other user's pending-action state is read, let
        # alone resumed, on a mismatch.
        raise HTTPException(status_code=403, detail="This action does not belong to you.")

    settings = get_settings()
    if not settings.studylife_api_base_url:
        raise HTTPException(status_code=503, detail=_NOT_CONFIGURED_DETAIL)

    async with StudyLifeClient(
        base_url=settings.studylife_api_base_url, api_key=identity.ai_api_key
    ) as studylife_client:
        await verify_identity(studylife_client)
        compiled_agent = _build_agent(http_request, identity, studylife_client)
        decision: dict[str, Any] = {"type": request.decision}
        if request.decision == "reject" and request.message:
            decision["message"] = request.message

        state = await compiled_agent.aget_state({"configurable": {"thread_id": request.thread_id}})
        if not state.next:
            # No paused run under this thread_id - never proposed, already
            # confirmed/rejected once, invalidated after a prior failure (see
            # _invoke_and_handle_failure), or the id is wrong. Resuming anyway
            # would silently start a fresh, empty run instead of erroring.
            raise HTTPException(status_code=404, detail="No pending action for this thread_id.")

        # Same decision applies to every pending action in this turn (see
        # schemas/agent.py) - the resume payload needs exactly as many decisions
        # as there were action_requests, or HumanInTheLoopMiddleware raises.
        pending_count = len(state.tasks[0].interrupts[0].value["action_requests"])
        result = await _invoke_and_handle_failure(
            compiled_agent,
            http_request.app.state.agent_checkpointer,
            Command(resume={"decisions": [decision] * pending_count}),
            request.thread_id,
        )
    return _response_from_result(result, request.thread_id)
