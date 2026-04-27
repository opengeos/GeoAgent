"""Tests for the ConfirmCallback bridge in :class:`geoagent.GeoAgent.chat`.

When a confirmation-required tool fires, deepagents pauses the graph
and emits an ``__interrupt__`` payload. ``GeoAgent.chat`` is expected
to:

1. Build a :class:`ConfirmRequest` for each pending action.
2. Invoke the user-supplied ``confirm`` callback.
3. Resume with ``approve`` or ``reject`` decisions accordingly.
4. Track names in :attr:`GeoAgentResponse.executed_tools` /
   :attr:`GeoAgentResponse.cancelled_tools`.

These tests pin that contract using a fake chat model that emits a
single tool call followed by a final AI message.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from geoagent import GeoAgent, geo_tool
from geoagent.core.safety import ConfirmRequest
from tests._fakes import make_fake


@geo_tool(category="map", requires_confirmation=True)
def remove_layer(name: str) -> str:
    """Remove a named layer from the active map (test fixture)."""
    return f"Removed {name}"


@tool
def safe_inspect(query: str) -> str:
    """A safe inspection tool (no confirmation required)."""
    return f"inspected {query}"


def _make_agent_with_remove_call(*, confirm) -> GeoAgent:
    """Construct a GeoAgent whose first LLM step issues a remove_layer call."""
    return GeoAgent(
        llm=make_fake(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tc1",
                            "name": "remove_layer",
                            "args": {"name": "NDVI"},
                        }
                    ],
                ),
                AIMessage(content="Done."),
            ]
        ),
        tools=[remove_layer],
        confirm=confirm,
    )


def test_confirm_callback_receives_request_and_approves() -> None:
    seen: list[ConfirmRequest] = []

    def confirm(request: ConfirmRequest) -> bool:
        seen.append(request)
        return True

    agent = _make_agent_with_remove_call(confirm=confirm)
    resp = agent.chat("Remove the NDVI layer")

    assert resp.success is True
    assert len(seen) == 1
    request = seen[0]
    assert request.tool_name == "remove_layer"
    assert request.args == {"name": "NDVI"}
    # The cached @geo_tool description should flow through to the request.
    assert "Remove a named layer" in request.description
    assert "remove_layer" in resp.executed_tools
    assert "remove_layer" not in resp.cancelled_tools


def test_confirm_callback_can_reject() -> None:
    def confirm(_request: ConfirmRequest) -> bool:
        return False

    agent = _make_agent_with_remove_call(confirm=confirm)
    resp = agent.chat("Remove the NDVI layer")

    assert resp.success is True
    assert "remove_layer" in resp.cancelled_tools
    assert "remove_layer" not in resp.executed_tools
    # The tool body should not have run, so the rejection ToolMessage
    # is what the LLM saw — there is no "Removed NDVI" message.
    contents = [getattr(m, "content", "") for m in (resp.messages or [])]
    assert not any("Removed NDVI" == c for c in contents)


def test_default_confirm_rejects_destructive_tools_silently() -> None:
    """When no callback is wired, auto_approve_safe_only rejects everything."""
    agent = _make_agent_with_remove_call(confirm=None)
    resp = agent.chat("Remove the NDVI layer")
    assert resp.success is True
    assert resp.cancelled_tools == ["remove_layer"]
    assert resp.executed_tools == []


def test_safe_tool_does_not_trigger_confirm() -> None:
    """A tool without ``requires_confirmation`` runs without prompting."""
    seen: list[ConfirmRequest] = []

    def confirm(request: ConfirmRequest) -> bool:
        seen.append(request)
        return True

    agent = GeoAgent(
        llm=make_fake(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tc1",
                            "name": "safe_inspect",
                            "args": {"query": "all"},
                        }
                    ],
                ),
                AIMessage(content="Done."),
            ]
        ),
        tools=[safe_inspect],
        confirm=confirm,
    )
    resp = agent.chat("Inspect everything")
    assert resp.success is True
    assert seen == []  # callback never fired
    assert resp.cancelled_tools == []
