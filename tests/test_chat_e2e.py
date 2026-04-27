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


def test_target_map_rebuilds_graph_and_adds_mapping_subagent() -> None:
    """Passing ``target_map`` rebuilds the graph so the mapping subagent appears.

    When the agent is constructed without a map, ``default_subagents``
    omits the mapping subagent. Supplying a ``target_map`` to chat()
    must update the context and rebuild the graph; the new subagent
    list must include "mapping".
    """
    from geoagent.agents.coordinator import default_subagents
    from geoagent.testing import MockLeafmap

    agent = GeoAgent(llm=make_fake([AIMessage(content="ok")]))
    initial_thread = agent._thread_id
    assert "mapping" not in {s["name"] for s in default_subagents(agent._context)}

    fake_map = MockLeafmap()
    # Re-prime the underlying fake model with a fresh response sequence,
    # since the previous chat() consumed the original iterator.
    agent._construction["llm"] = make_fake([AIMessage(content="after")])
    resp = agent.chat("anything", target_map=fake_map)

    assert resp.success
    assert agent._context.map_obj is fake_map
    assert (
        agent._thread_id != initial_thread
    ), "graph rebuild should also rotate the thread id"
    assert "mapping" in {s["name"] for s in default_subagents(agent._context)}


def test_target_map_same_object_does_not_rebuild() -> None:
    """Re-passing the *same* map object should not rebuild the graph."""
    from geoagent.testing import MockLeafmap

    fake_map = MockLeafmap()
    from geoagent.core.context import GeoAgentContext

    agent = GeoAgent(
        llm=make_fake([AIMessage(content="x"), AIMessage(content="y")]),
        context=GeoAgentContext(map_obj=fake_map),
    )
    initial_thread = agent._thread_id
    initial_graph = agent._graph
    agent.chat("first", target_map=fake_map)
    assert agent._thread_id == initial_thread
    assert agent._graph is initial_graph
