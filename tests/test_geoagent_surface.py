"""GeoAgent surface compatibility with Strands-like attributes."""

from __future__ import annotations

from geoagent import for_leafmap
from geoagent.testing import MockLeafmap


class _MockModel:
    """Provide a test double for MockModel."""

    stateful = False


def test_geoagent_exposes_tool_names_and_registry() -> None:
    """Verify that geoagent exposes tool names and registry."""
    agent = for_leafmap(MockLeafmap(), model=_MockModel())
    names = agent.tool_names
    assert "get_map_state" in names
    configs = agent.tool_registry.get_all_tools_config()
    assert any(item["name"] == "get_map_state" for item in configs)
