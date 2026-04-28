"""Tests for factory helpers."""

from __future__ import annotations

from geoagent import GeoAgentContext, create_agent, for_leafmap
from geoagent.testing import MockLeafmap


def test_create_agent_empty_tools() -> None:
    a = create_agent(tools=[], context=GeoAgentContext())
    assert a.strands_agent.tool_names == []


def test_for_leafmap_registers_tools() -> None:
    m = MockLeafmap()
    a = for_leafmap(m)
    names = set(a.strands_agent.tool_names)
    assert "list_layers" in names


def test_for_leafmap_accepts_provider_and_model_id() -> None:
    m = MockLeafmap()
    a = for_leafmap(m, provider="anthropic", model_id="claude-sonnet-4-6")
    assert a.config.provider == "anthropic"
    assert a.config.model == "claude-sonnet-4-6"
