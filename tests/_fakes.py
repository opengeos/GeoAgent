"""Test helpers shared across the chat / confirm-callback test modules.

Centralises a tool-binding-capable fake chat model. LangChain's stock
:class:`~langchain_core.language_models.fake_chat_models.GenericFakeChatModel`
does not implement ``bind_tools``, which deepagents' middleware calls
when wiring tools into the agent graph. We need a model that:

1. Yields a pre-baked sequence of :class:`~langchain_core.messages.AIMessage`
   responses, optionally with ``tool_calls`` so the agent will execute
   tools.
2. Returns ``self`` from ``bind_tools`` so the deepagents middleware
   stack can compose tool schemas onto it without raising.

This file lives under ``tests/`` (not the package) because the fake is
a test-only construct.
"""

from __future__ import annotations

from typing import Any, Iterable

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel


class ToolBindingFakeChatModel(GenericFakeChatModel):
    """A :class:`GenericFakeChatModel` that no-ops ``bind_tools``.

    Pass an iterable of ``AIMessage`` (or convertible) instances; each
    invocation pops the next message in order, just like the parent
    class.
    """

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ToolBindingFakeChatModel":
        """No-op tool binding — return self so middleware can chain."""
        return self


def make_fake(messages: Iterable[Any]) -> ToolBindingFakeChatModel:
    """Build a :class:`ToolBindingFakeChatModel` from a list of messages."""
    return ToolBindingFakeChatModel(messages=iter(list(messages)))


__all__ = ["ToolBindingFakeChatModel", "make_fake"]
