"""Tests for factory helpers."""

from __future__ import annotations

from geoagent import (
    GeoAgentContext,
    create_agent,
    for_browser_maplibre,
    for_geoai,
    for_leafmap,
)
from geoagent.testing import MockLeafmap, MockQGISIface, MockQGISProject


class _MockModel:
    """Provide a test double for MockModel."""

    stateful = False


def test_create_agent_empty_tools() -> None:
    """Verify that create agent empty tools."""
    a = create_agent(tools=[], context=GeoAgentContext(), model=_MockModel())
    assert a.strands_agent.tool_names == []


def test_for_leafmap_registers_tools() -> None:
    """Verify that for leafmap registers tools."""
    m = MockLeafmap()
    a = for_leafmap(m, model=_MockModel())
    names = set(a.strands_agent.tool_names)
    assert "list_layers" in names


def test_for_leafmap_accepts_provider_and_model_id() -> None:
    """Verify that for leafmap accepts provider and model id."""
    m = MockLeafmap()
    a = for_leafmap(
        m,
        provider="anthropic",
        model_id="claude-sonnet-4-6",
        model=_MockModel(),
    )
    assert a.config.provider == "anthropic"
    assert a.config.model == "claude-sonnet-4-6"


def test_for_browser_maplibre_registers_tools() -> None:
    """Verify the browser MapLibre factory registers browser tools."""
    session = object()
    a = for_browser_maplibre(session, model=_MockModel())
    names = set(a.strands_agent.tool_names)
    assert "get_map_state" in names
    assert "add_marker" in names
    assert a.context.metadata["integration"] == "browser_maplibre"
    assert "live\nMapLibre map" in a.context.metadata["system_prompt"]


def test_for_browser_maplibre_code_tool_is_opt_in() -> None:
    """Verify browser JavaScript execution is available only when enabled."""
    session = object()
    default_agent = for_browser_maplibre(session, model=_MockModel())
    code_agent = for_browser_maplibre(
        session,
        model=_MockModel(),
        allow_browser_code=True,
    )

    assert "run_maplibre_script" not in default_agent.strands_agent.tool_names
    assert "run_maplibre_script" in code_agent.strands_agent.tool_names
    assert "Browser JavaScript code execution is enabled" in (
        code_agent.context.metadata["system_prompt"]
    )


def test_factory_accepts_gemini_provider() -> None:
    """Verify that factory accepts gemini provider."""
    m = MockLeafmap()
    a = for_leafmap(
        m,
        provider="gemini",
        model_id="gemini-3.1-pro-preview",
        model=_MockModel(),
    )
    assert a.config.provider == "gemini"
    assert a.config.model == "gemini-3.1-pro-preview"


def test_factory_accepts_litellm_provider() -> None:
    """Verify that factory accepts LiteLLM provider."""
    m = MockLeafmap()
    a = for_leafmap(
        m,
        provider="litellm",
        model_id="openai/gpt-5.5",
        model=_MockModel(),
    )
    assert a.config.provider == "litellm"
    assert a.config.model == "openai/gpt-5.5"


def test_for_geoai_registers_qgis_and_geoai_tools() -> None:
    """Verify the GeoAI factory combines QGIS and GeoAI tool surfaces."""
    iface = MockQGISIface()
    project = MockQGISProject()

    a = for_geoai(iface, project=project, model=_MockModel())
    names = set(a.strands_agent.tool_names)

    assert "list_project_layers" in names
    assert "segment_image_with_text_prompt" in names
    assert "regularize_segmentation_mask_to_vector" in names
    assert "smooth_segmentation_mask_to_vector" in names
    assert a.context.metadata["integration"] == "geoai"
    assert "SamGeo text-prompt segmentation" in a.context.metadata["system_prompt"]


def test_for_geoai_permission_profiles_filter_segmentation_tool() -> None:
    """Verify Inspect-only hides long-running GeoAI segmentation."""
    iface = MockQGISIface()
    project = MockQGISProject()

    inspect = for_geoai(
        iface,
        project=project,
        model=_MockModel(),
        permission_profile="Inspect only",
    )
    processing = for_geoai(
        iface,
        project=project,
        model=_MockModel(),
        permission_profile="Run processing",
    )

    assert "segment_image_with_text_prompt" not in inspect.strands_agent.tool_names
    assert "segment_image_with_text_prompt" in processing.strands_agent.tool_names
