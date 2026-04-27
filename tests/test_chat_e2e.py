"""End-to-end tests for :class:`geoagent.GeoAgent.chat`.

These tests use a fake chat model so they run in CI without an LLM
provider. They exercise the deepagents graph end-to-end: the agent
receives a query, the fake model produces a final message, and the
facade reconstructs a :class:`GeoAgentResponse`.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from geoagent import GeoAgent
from tests._fakes import make_fake


def test_chat_returns_response_with_answer_text() -> None:
    """A fake LLM that emits a final AIMessage produces a successful response."""
    agent = GeoAgent(llm=make_fake([AIMessage(content="Hello, world.")]))
    resp = agent.chat("hi")
    assert resp.success is True
    assert resp.error_message is None
    assert resp.answer_text == "Hello, world."
    assert resp.executed_tools == []
    assert resp.cancelled_tools == []
    assert resp.execution_time >= 0


def test_chat_propagates_exceptions_into_failure_response() -> None:
    """Errors raised during graph invocation surface as ``success=False``.

    Build a fake LLM whose message iterator is empty: the first attempt
    to generate a response raises, which the chat() try/except converts
    into a structured failure response.
    """
    agent = GeoAgent(llm=make_fake([]))
    resp = agent.chat("hi")
    assert resp.success is False
    assert resp.error_message  # populated, exact text varies per LangChain release


def test_chat_search_alias_routes_through_chat() -> None:
    """``search`` / ``analyze`` / ``visualize`` are thin wrappers over chat."""
    agent = GeoAgent(llm=make_fake([AIMessage(content="search done")]))
    resp = agent.search("any data")
    assert resp.success and resp.answer_text == "search done"


def test_chat_runs_independent_threads_per_instance() -> None:
    """Two ``GeoAgent`` instances must use distinct LangGraph threads."""
    a = GeoAgent(llm=make_fake([AIMessage(content="first")]))
    b = GeoAgent(llm=make_fake([AIMessage(content="second")]))
    assert a._thread_id != b._thread_id
    assert a.chat("q1").answer_text == "first"
    assert b.chat("q2").answer_text == "second"
