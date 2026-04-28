"""Tests for the WhiteboxTools GeoAgent adapter."""

from __future__ import annotations

import json
import sys
from types import ModuleType

import pytest

from geoagent import for_whitebox
from geoagent.testing import MockQGISIface, MockQGISLayer, MockQGISProject
from geoagent.tools.whitebox import whitebox_tools


class _MockModel:
    """Provide a test double for MockModel."""

    stateful = False


def test_whitebox_module_imports_without_qgis_or_whitebox() -> None:
    """Verify the adapter module is import-safe outside QGIS and Whitebox."""
    if "qgis" in sys.modules:
        pytest.skip("qgis is already imported in this environment.")
    assert "geoagent.tools.whitebox" in sys.modules
    assert "qgis" not in sys.modules
    assert "whitebox" not in sys.modules


def test_whitebox_tools_returns_empty_for_none_iface() -> None:
    """Verify the Whitebox factory returns no tools without iface."""
    assert whitebox_tools(None) == []


def test_whitebox_tools_expose_routed_surface() -> None:
    """Verify the adapter exposes a small broker surface."""
    tools = {tool.tool_name: tool for tool in whitebox_tools(object())}

    assert set(tools) == {
        "summarize_whitebox_tools",
        "search_whitebox_tools",
        "get_whitebox_tool_info",
        "run_whitebox_tool",
    }


def test_search_whitebox_tools_returns_compact_results(monkeypatch) -> None:
    """Verify search wraps WhiteboxTools list_tools output."""

    class _FakeWBT:
        verbose = True

        def list_tools(self, keywords=None):
            assert keywords == ["slope"]
            return {
                "slope": "Calculates slope from a DEM.",
                "average_normal_vector_angular_deviation": "Terrain metric.",
            }

        def toolbox(self, tool_name):
            return "Terrain Analysis" if tool_name == "slope" else "Math"

    whitebox_module = ModuleType("whitebox")
    whitebox_module.WhiteboxTools = _FakeWBT
    monkeypatch.setitem(sys.modules, "whitebox", whitebox_module)

    tools = {tool.tool_name: tool for tool in whitebox_tools(object())}
    result = tools["search_whitebox_tools"].__wrapped__(
        query="slope",
        category="terrain",
        max_results=5,
    )

    assert result["count"] == 1
    assert result["tools"] == [
        {
            "name": "slope",
            "description": "Calculates slope from a DEM.",
            "category": "Terrain Analysis",
        }
    ]


def test_get_whitebox_tool_info_parses_parameter_json(monkeypatch) -> None:
    """Verify tool info exposes flags, keys, optionality, and help."""

    class _FakeWBT:
        verbose = True

        def tool_parameters(self, tool_name):
            assert tool_name == "slope"
            return json.dumps(
                {
                    "parameters": [
                        {
                            "name": "Input DEM File",
                            "flags": ["-i", "--dem"],
                            "description": "Input raster DEM file.",
                            "parameter_type": {"ExistingFile": "Raster"},
                            "default_value": None,
                            "optional": False,
                        }
                    ]
                }
            )

        def toolbox(self, tool_name):
            return "Terrain Analysis"

        def tool_help(self, tool_name):
            return "Slope help text"

    whitebox_module = ModuleType("whitebox")
    whitebox_module.WhiteboxTools = _FakeWBT
    monkeypatch.setitem(sys.modules, "whitebox", whitebox_module)

    tools = {tool.tool_name: tool for tool in whitebox_tools(object())}
    result = tools["get_whitebox_tool_info"].__wrapped__("slope")

    assert result["tool_name"] == "slope"
    assert result["category"] == "Terrain Analysis"
    assert result["parameters"][0]["flags"] == ["-i", "--dem"]
    assert "dem" in result["parameters"][0]["keys"]
    assert result["parameters"][0]["optional"] is False
    assert result["help"] == "Slope help text"


def test_run_whitebox_tool_builds_args_and_loads_output(monkeypatch, tmp_path) -> None:
    """Verify run_tool args, callback capture, and QGIS output loading."""
    input_path = tmp_path / "dem.tif"
    output_path = tmp_path / "slope.tif"
    input_path.write_text("dem", encoding="utf-8")

    captured = {}

    class _FakeWBT:
        verbose = True

        def __init__(self):
            self.verbose = True

        def set_working_dir(self, path):
            captured["working_dir"] = path

        def set_compress_rasters(self, value):
            captured["compress_rasters"] = value

        def set_max_procs(self, value):
            captured["max_procs"] = value

        def tool_parameters(self, tool_name):
            assert tool_name == "slope"
            return json.dumps(
                {
                    "parameters": [
                        {
                            "name": "Input DEM File",
                            "flags": ["-i", "--dem"],
                            "parameter_type": {"ExistingFile": "Raster"},
                            "optional": False,
                        },
                        {
                            "name": "Output File",
                            "flags": ["-o", "--output"],
                            "parameter_type": {"NewFile": "Raster"},
                            "optional": False,
                        },
                        {
                            "name": "Z Conversion Factor",
                            "flags": ["--zfactor"],
                            "parameter_type": "Float",
                            "optional": True,
                        },
                    ]
                }
            )

        def run_tool(self, tool_name, args, callback=None):
            captured["tool_name"] = tool_name
            captured["args"] = args
            output_path.write_text("slope", encoding="utf-8")
            if callback:
                callback("Progress: 100%")
            return 0

    whitebox_module = ModuleType("whitebox")
    whitebox_module.WhiteboxTools = _FakeWBT
    monkeypatch.setitem(sys.modules, "whitebox", whitebox_module)

    project = MockQGISProject()
    project.addMapLayer(MockQGISLayer("DEM layer", str(input_path), "raster"))
    iface = MockQGISIface(project)
    tools = {tool.tool_name: tool for tool in whitebox_tools(iface, project)}

    result = tools["run_whitebox_tool"].__wrapped__(
        "slope",
        {"dem": "DEM layer", "output": str(output_path), "zfactor": 1.2},
        working_dir=str(tmp_path),
        output_layer_name="Slope",
        compress_rasters=True,
        max_procs=2,
    )

    assert captured["tool_name"] == "slope"
    assert captured["args"] == [
        f"--dem={input_path}",
        f"--output={output_path}",
        "--zfactor=1.2",
    ]
    assert captured["working_dir"] == str(tmp_path)
    assert captured["compress_rasters"] is True
    assert captured["max_procs"] == 2
    assert result["success"] is True
    assert result["messages"] == ["Progress: 100%"]
    assert result["layers_added"] == [
        {"path": str(output_path), "added": True, "layer_name": "Slope"}
    ]
    assert "Slope" in project.mapLayers()


def test_for_whitebox_registers_tools_and_respects_fast_mode(monkeypatch) -> None:
    """Verify the factory combines Whitebox and QGIS tools."""
    whitebox_module = ModuleType("whitebox")
    whitebox_module.WhiteboxTools = object
    monkeypatch.setitem(sys.modules, "whitebox", whitebox_module)

    agent = for_whitebox(MockQGISIface(), MockQGISProject(), model=_MockModel())
    names = set(agent.strands_agent.tool_names)

    assert "search_whitebox_tools" in names
    assert "run_whitebox_tool" in names
    assert "list_project_layers" in names
    assert agent.context.metadata["integration"] == "whitebox"

    fast_agent = for_whitebox(
        MockQGISIface(),
        MockQGISProject(),
        model=_MockModel(),
        fast=True,
    )
    fast_names = set(fast_agent.strands_agent.tool_names)

    assert "search_whitebox_tools" in fast_names
    assert "get_whitebox_tool_info" in fast_names
    assert "run_whitebox_tool" not in fast_names
