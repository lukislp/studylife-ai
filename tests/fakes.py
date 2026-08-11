"""Shared test doubles."""

from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel


class FakeToolCallingModel(FakeMessagesListChatModel):
    """Cycles through pre-set responses, ignoring the tools it's bound to -
    real tool-call decisions come from the `responses` list, not the model."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self
