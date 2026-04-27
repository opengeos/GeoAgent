"""Tests for create_geo_agent + for_leafmap / for_anymap / for_qgis.

These tests construct the deepagents-compiled agents with mocks. They do
not exercise an LLM — they verify that the factory wires up the right
tools and ``interrupt_on`` configuration.
"""

from __future__ import annotations

import pytest

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from geoagent.core.context import GeoAgentContext
from geoagent.core.factory import (
    create_geo_agent,
    for_anymap,
    for_leafmap,
    for_qgis,
)
from geoagent.testing import MockAnymap, MockLeafmap, MockQGISIface, MockQGISProject

deepagents = pytest.importorskip("deepagents")


def _fake_llm() -> FakeListChatModel:
    """Return a minimal LangChain BaseChatModel for factory tests.

    deepagents requires the model arg to be either a provider:model string or
    a real ``BaseChatModel`` instance. We don't actually invoke the agent
    here — we only verify that the factory can construct it.
    """
    return FakeListChatModel(responses=["ok"])


def test_for_leafmap_builds_agent() -> None:
    m = MockLeafmap()
    agent = for_leafmap(m, llm=_fake_llm(), include_stac=False)
    assert agent is not None
    # Compiled deepagents graphs expose `.invoke` and `.stream`.
    assert hasattr(agent, "invoke") or hasattr(agent, "ainvoke")


def test_for_anymap_builds_agent() -> None:
    m = MockAnymap()
    agent = for_anymap(m, llm=_fake_llm(), include_stac=False)
    assert agent is not None


def test_for_qgis_builds_agent_with_iface() -> None:
    iface = MockQGISIface()
    project = MockQGISProject()
    agent = for_qgis(iface, project, llm=_fake_llm(), include_stac=False)
    assert agent is not None


def test_for_qgis_builds_agent_with_no_iface() -> None:
    # Should not raise; tool list will simply lack QGIS-specific tools.
    agent = for_qgis(None, llm=_fake_llm(), include_stac=False)
    assert agent is not None


def test_create_geo_agent_accepts_explicit_tools_and_context() -> None:
    from geoagent.tools.leafmap import leafmap_tools

    m = MockLeafmap()
    tools = leafmap_tools(m)
    context = GeoAgentContext(map_obj=m, current_layer="NDVI")
    agent = create_geo_agent(tools=tools, context=context, llm=_fake_llm())
    assert agent is not None


def test_factory_assembles_interrupt_on_for_confirm_tools(monkeypatch) -> None:
    """The factory must populate `interrupt_on` from tool metadata."""
    captured: dict[str, object] = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return object()

    import geoagent.core.factory as factory_mod

    monkeypatch.setattr(
        factory_mod, "_require_deepagents", lambda: fake_create_deep_agent
    )

    from geoagent.tools.leafmap import leafmap_tools

    m = MockLeafmap()
    create_geo_agent(tools=leafmap_tools(m), llm=_fake_llm())

    assert "interrupt_on" in captured
    interrupt_on = captured["interrupt_on"]
    assert interrupt_on is not None
    assert "remove_layer" in interrupt_on
    assert "save_map" in interrupt_on
    # safe tools must NOT be in interrupt_on
    assert "list_layers" not in interrupt_on
    assert "zoom_in" not in interrupt_on


def test_factory_passes_context_schema(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return object()

    import geoagent.core.factory as factory_mod

    monkeypatch.setattr(
        factory_mod, "_require_deepagents", lambda: fake_create_deep_agent
    )

    create_geo_agent(
        tools=[],
        context=GeoAgentContext(current_layer="NDVI"),
        llm=_fake_llm(),
    )
    assert captured["context_schema"] is GeoAgentContext


def test_for_leafmap_omits_checkpointer_by_default(monkeypatch) -> None:
    """``for_leafmap`` returns a graph that accepts a bare ``.invoke()``.

    deepagents' default checkpointer (``MemorySaver``) requires every
    ``.invoke()`` call to pass ``config={"configurable": {"thread_id": ...}}``.
    Direct-invoke users (``graph.invoke({"messages": [...]})``) shouldn't
    have to know that, so the convenience factories opt out by default.
    Pinning this captures the contract: no checkpointer reaches
    ``create_deep_agent`` unless the caller asks for one.
    """
    captured: dict[str, object] = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return object()

    import geoagent.core.factory as factory_mod

    monkeypatch.setattr(
        factory_mod, "_require_deepagents", lambda: fake_create_deep_agent
    )

    for_leafmap(MockLeafmap(), llm=_fake_llm(), include_stac=False)
    assert captured.get("checkpointer") is None


def test_for_qgis_omits_checkpointer_by_default(monkeypatch) -> None:
    """``for_qgis`` mirrors ``for_leafmap``'s no-checkpointer default."""
    captured: dict[str, object] = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return object()

    import geoagent.core.factory as factory_mod

    monkeypatch.setattr(
        factory_mod, "_require_deepagents", lambda: fake_create_deep_agent
    )

    for_qgis(MockQGISIface(), llm=_fake_llm(), include_stac=False)
    assert captured.get("checkpointer") is None


def test_for_leafmap_honors_explicit_checkpointer(monkeypatch) -> None:
    """A caller-supplied checkpointer is not stomped by the default."""
    from langgraph.checkpoint.memory import MemorySaver

    captured: dict[str, object] = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return object()

    import geoagent.core.factory as factory_mod

    monkeypatch.setattr(
        factory_mod, "_require_deepagents", lambda: fake_create_deep_agent
    )

    saver = MemorySaver()
    for_leafmap(
        MockLeafmap(),
        llm=_fake_llm(),
        include_stac=False,
        checkpointer=saver,
    )
    assert captured.get("checkpointer") is saver


def test_for_qgis_wraps_graph_to_force_inline_tool_execution(monkeypatch) -> None:
    """``for_qgis`` returns a wrapper that activates inline tool execution.

    LangGraph's ``ToolNode`` always offloads tool calls to a worker
    thread pool. Under QGIS that corrupts iface and crashes the
    process; the QGIS path must run tools inline on the calling
    thread. The wrapper applied by ``for_qgis`` enters the
    :func:`inline_tool_execution` context manager around every
    ``invoke`` / ``stream`` / ``ainvoke`` / ``astream`` /
    ``astream_events`` call. The context manager flips a
    :class:`contextvars.ContextVar` that the stable wrapper around
    LangGraph's executor factory consults — so inside the call,
    ``get_executor_for_config`` returns the inline executor; outside,
    the gating flag is reset and the wrapper delegates to the
    original factory.
    """
    import langgraph.prebuilt.tool_node as _tool_node_mod

    from geoagent.tools import _inline_executor as _inline_mod
    from geoagent.tools._inline_executor import _InlineExecutor

    captured: dict[str, object] = {}

    class _FakeInner:
        def invoke(self, *args, **kwargs):
            # Snapshot the executor factory at the moment ``invoke`` runs.
            captured["executor"] = _tool_node_mod.get_executor_for_config({})
            captured["flag_inside"] = _inline_mod._inline_active.get()
            return "ok"

    import geoagent.core.factory as factory_mod

    monkeypatch.setattr(
        factory_mod, "_require_deepagents", lambda: lambda **_: _FakeInner()
    )

    agent = for_qgis(MockQGISIface(), llm=_fake_llm(), include_stac=False)

    result = agent.invoke({"messages": []})

    assert result == "ok"
    assert isinstance(captured["executor"], _InlineExecutor)
    assert captured["flag_inside"] is True
    # The gating ContextVar is reset once invoke() returns. The wrapper
    # itself stays installed (it's permanent and concurrency-safe).
    assert _inline_mod._inline_active.get() is False


def test_for_qgis_wrapper_forwards_other_attributes(monkeypatch) -> None:
    """Non-overridden methods / attributes pass through to the inner graph.

    The wrapper only wraps invoke / stream / ainvoke / astream;
    everything else (``get_state``, custom attrs, etc.) must be
    visible via ``__getattr__``.
    """

    class _FakeInner:
        custom_attr = 42

        def get_state(self):
            return "inner-state"

    import geoagent.core.factory as factory_mod

    monkeypatch.setattr(
        factory_mod, "_require_deepagents", lambda: lambda **_: _FakeInner()
    )

    agent = for_qgis(MockQGISIface(), llm=_fake_llm(), include_stac=False)
    assert agent.custom_attr == 42
    assert agent.get_state() == "inner-state"
