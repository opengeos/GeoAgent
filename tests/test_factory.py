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
