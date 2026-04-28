"""Tests for the GEE Data Catalogs GeoAgent tool factory."""

from __future__ import annotations

import sys

import pytest

from geoagent import for_gee_data_catalogs
from geoagent.testing import MockQGISIface, MockQGISProject
from geoagent.tools.gee_data_catalogs import gee_data_catalogs_tools


class _MockModel:
    """Provide a test double for MockModel."""

    stateful = False


def test_gee_data_catalogs_module_imports_without_qgis() -> None:
    """Verify GEE Data Catalogs tools are import-safe outside QGIS."""
    if "qgis" in sys.modules:
        pytest.skip("qgis is already imported in this environment.")
    assert "geoagent.tools.gee_data_catalogs" in sys.modules
    assert "qgis" not in sys.modules
    assert "gee_data_catalogs" not in sys.modules


def test_gee_data_catalogs_tools_returns_empty_for_none_iface() -> None:
    """Verify the GEE Data Catalogs factory returns no tools without iface."""
    assert gee_data_catalogs_tools(None) == []


def test_gee_data_catalogs_tools_expose_catalog_surface() -> None:
    """Verify the plugin-specific tool surface is registered."""
    tools = {t.tool_name: t for t in gee_data_catalogs_tools(object())}

    assert "search_gee_datasets" in tools
    assert "get_gee_dataset_info" in tools
    assert "summarize_gee_catalog" in tools
    assert "initialize_earth_engine" in tools
    assert "load_gee_dataset" in tools
    assert "open_gee_catalog_panel" in tools
    assert "configure_gee_dataset_load" in tools


def test_for_gee_data_catalogs_registers_catalog_and_qgis_tools() -> None:
    """Verify the GEE factory combines catalog and QGIS tool surfaces."""
    agent = for_gee_data_catalogs(
        MockQGISIface(),
        MockQGISProject(),
        model=_MockModel(),
    )
    names = set(agent.strands_agent.tool_names)

    assert "open_gee_catalog_panel" in names
    assert "configure_gee_dataset_load" in names
    assert "list_project_layers" in names
    assert agent.context.metadata["integration"] == "gee_data_catalogs"
