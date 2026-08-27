"""Shared test doubles."""

from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.outputs import ChatResult


class FakeToolCallingModel(FakeMessagesListChatModel):
    """Cycles through pre-set responses, ignoring the tools it's bound to -
    real tool-call decisions come from the `responses` list, not the model."""

    fail_on_call_index: int | None = None
    """0-based index (across this model's `_generate` calls) at which to raise
    `fail_exception` once, instead of returning the next canned response - simulates a
    transient LLM-provider failure partway through an agent run (see
    tests/test_agent_api.py's confirm-preserves-checkpoint-on-transient test). The failed
    call does NOT consume a response from `responses` - the next successful call still gets
    the same one it would have gotten had this call not failed."""
    fail_exception: Exception | None = None
    call_count: int = 0

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        index = self.call_count
        self.call_count += 1
        if self.fail_on_call_index is not None and index == self.fail_on_call_index:
            assert self.fail_exception is not None
            raise self.fail_exception
        return super()._generate(*args, **kwargs)
