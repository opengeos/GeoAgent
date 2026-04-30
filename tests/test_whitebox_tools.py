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
        "run_whitebox_flow_accumulation",
        "run_whitebox_fill_sinks",
        "run_whitebox_color_shaded_relief",
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


def test_get_whitebox_tool_info_reports_invalid_parameter_json(monkeypatch) -> None:
    """Verify invalid Whitebox metadata JSON produces an actionable error."""

    class _FakeWBT:
        verbose = True

        def tool_parameters(self, tool_name):
            assert tool_name == "slope"
            return ""

    whitebox_module = ModuleType("whitebox")
    whitebox_module.WhiteboxTools = _FakeWBT
    monkeypatch.setitem(sys.modules, "whitebox", whitebox_module)

    tools = {tool.tool_name: tool for tool in whitebox_tools(object())}
    with pytest.raises(ValueError, match="invalid parameter metadata JSON"):
        tools["get_whitebox_tool_info"].__wrapped__("slope")


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


def test_run_whitebox_flow_accumulation_uses_active_layer(
    monkeypatch, tmp_path
) -> None:
    """Verify the flow accumulation convenience tool uses the active DEM."""
    input_path = tmp_path / "dem.tif"
    output_path = tmp_path / "flow_accumulation.tif"
    input_path.write_text("dem", encoding="utf-8")

    captured = {}

    class _FakeWBT:
        verbose = True

        def __init__(self):
            self.verbose = True

        def tool_parameters(self, tool_name):
            assert tool_name == "d8_flow_accumulation"
            return json.dumps(
                {
                    "parameters": [
                        {
                            "name": "Input Raster File",
                            "flags": ["-i", "--input"],
                            "parameter_type": {"ExistingFile": "Raster"},
                            "optional": False,
                        },
                        {
                            "name": "Output File",
                            "flags": ["-o", "--output"],
                            "parameter_type": {"NewFile": "Raster"},
                            "optional": False,
                        },
                    ]
                }
            )

        def run_tool(self, tool_name, args, callback=None):
            captured["tool_name"] = tool_name
            captured["args"] = args
            output_path.write_text("flow", encoding="utf-8")
            return 0

    whitebox_module = ModuleType("whitebox")
    whitebox_module.WhiteboxTools = _FakeWBT
    monkeypatch.setitem(sys.modules, "whitebox", whitebox_module)

    project = MockQGISProject()
    layer = MockQGISLayer("DEM layer", str(input_path), "raster")
    project.addMapLayer(layer)
    iface = MockQGISIface(project)
    iface.setActiveLayer(layer)
    tools = {tool.tool_name: tool for tool in whitebox_tools(iface, project)}

    result = tools["run_whitebox_flow_accumulation"].__wrapped__(
        output_path=str(output_path),
        output_layer_name="Flow Accumulation",
    )

    assert captured["tool_name"] == "d8_flow_accumulation"
    assert captured["args"] == [
        f"--input={input_path}",
        f"--output={output_path}",
    ]
    assert result["success"] is True
    assert result["layers_added"] == [
        {
            "path": str(output_path),
            "added": True,
            "layer_name": "Flow Accumulation",
        }
    ]
    assert "Flow Accumulation" in project.mapLayers()


def test_run_whitebox_fill_sinks_uses_active_layer(monkeypatch, tmp_path) -> None:
    """Verify sink-filling defaults to fill_depressions on the active DEM."""
    input_path = tmp_path / "dem.tif"
    output_path = tmp_path / "dem_filled.tif"
    input_path.write_text("dem", encoding="utf-8")

    captured = {}

    class _FakeWBT:
        verbose = True

        def __init__(self):
            self.verbose = True

        def tool_parameters(self, tool_name):
            assert tool_name == "fill_depressions"
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
                            "name": "Maximum Depression Depth (z units)",
                            "flags": ["--max_depth"],
                            "parameter_type": "Float",
                            "optional": True,
                        },
                        {
                            "name": "Fix flats?",
                            "flags": ["--fix_flats"],
                            "parameter_type": "Boolean",
                            "optional": True,
                        },
                    ]
                }
            )

        def run_tool(self, tool_name, args, callback=None):
            captured["tool_name"] = tool_name
            captured["args"] = args
            output_path.write_text("filled", encoding="utf-8")
            return 0

    whitebox_module = ModuleType("whitebox")
    whitebox_module.WhiteboxTools = _FakeWBT
    monkeypatch.setitem(sys.modules, "whitebox", whitebox_module)

    project = MockQGISProject()
    layer = MockQGISLayer("DEM layer", str(input_path), "raster")
    project.addMapLayer(layer)
    iface = MockQGISIface(project)
    iface.setActiveLayer(layer)
    tools = {tool.tool_name: tool for tool in whitebox_tools(iface, project)}

    result = tools["run_whitebox_fill_sinks"].__wrapped__(
        output_path=str(output_path),
        output_layer_name="DEM Filled",
        max_depth=10,
        fix_flats=True,
    )

    assert captured["tool_name"] == "fill_depressions"
    assert captured["args"] == [
        f"--dem={input_path}",
        f"--output={output_path}",
        "--max_depth=10.0",
        "--fix_flats",
    ]
    assert result["success"] is True
    assert result["layers_added"] == [
        {
            "path": str(output_path),
            "added": True,
            "layer_name": "DEM Filled",
        }
    ]
    assert "DEM Filled" in project.mapLayers()


def test_run_whitebox_fill_sinks_can_explicitly_breach(monkeypatch, tmp_path) -> None:
    """Verify breaching remains available only when explicitly requested."""
    input_path = tmp_path / "dem.tif"
    output_path = tmp_path / "dem_breached.tif"
    input_path.write_text("dem", encoding="utf-8")

    captured = {}

    class _FakeWBT:
        verbose = True

        def __init__(self):
            self.verbose = True

        def tool_parameters(self, tool_name):
            assert tool_name == "breach_depressions"
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
                            "name": "Maximum Breach Channel Length",
                            "flags": ["--max_length"],
                            "parameter_type": "Float",
                            "optional": True,
                        },
                        {
                            "name": "Fill single-cell pits?",
                            "flags": ["--fill_pits"],
                            "parameter_type": "Boolean",
                            "optional": True,
                        },
                    ]
                }
            )

        def run_tool(self, tool_name, args, callback=None):
            captured["tool_name"] = tool_name
            captured["args"] = args
            output_path.write_text("breached", encoding="utf-8")
            return 0

    whitebox_module = ModuleType("whitebox")
    whitebox_module.WhiteboxTools = _FakeWBT
    monkeypatch.setitem(sys.modules, "whitebox", whitebox_module)

    project = MockQGISProject()
    layer = MockQGISLayer("DEM layer", str(input_path), "raster")
    project.addMapLayer(layer)
    iface = MockQGISIface(project)
    iface.setActiveLayer(layer)
    tools = {tool.tool_name: tool for tool in whitebox_tools(iface, project)}

    result = tools["run_whitebox_fill_sinks"].__wrapped__(
        method="breach_depressions",
        output_path=str(output_path),
        output_layer_name="DEM Breached",
        max_length=50,
        fill_pits=True,
    )

    assert captured["tool_name"] == "breach_depressions"
    assert captured["args"] == [
        f"--dem={input_path}",
        f"--output={output_path}",
        "--max_length=50.0",
        "--fill_pits",
    ]
    assert result["success"] is True
    assert "DEM Breached" in project.mapLayers()


def test_run_whitebox_fill_sinks_ignores_model_supplied_noop_defaults(
    monkeypatch, tmp_path
) -> None:
    """Verify empty strings, false flags, and zero limits are not passed."""
    input_path = tmp_path / "dem.tif"
    output_path = tmp_path / "dem_filled.tif"
    input_path.write_text("dem", encoding="utf-8")

    captured = {}

    class _FakeWBT:
        verbose = True

        def __init__(self):
            self.verbose = True

        def tool_parameters(self, tool_name):
            assert tool_name == "fill_depressions_wang_and_liu"
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
                            "name": "Fix flats?",
                            "flags": ["--fix_flats"],
                            "parameter_type": "Boolean",
                            "optional": True,
                        },
                    ]
                }
            )

        def run_tool(self, tool_name, args, callback=None):
            captured["tool_name"] = tool_name
            captured["args"] = args
            output_path.write_text("filled", encoding="utf-8")
            return 0

    whitebox_module = ModuleType("whitebox")
    whitebox_module.WhiteboxTools = _FakeWBT
    monkeypatch.setitem(sys.modules, "whitebox", whitebox_module)

    project = MockQGISProject()
    layer = MockQGISLayer("DEM layer", str(input_path), "raster")
    project.addMapLayer(layer)
    iface = MockQGISIface(project)
    iface.setActiveLayer(layer)
    tools = {tool.tool_name: tool for tool in whitebox_tools(iface, project)}

    result = tools["run_whitebox_fill_sinks"].__wrapped__(
        layer_name="",
        output_path=str(output_path),
        output_layer_name="",
        method="fill_depressions_wang_and_liu",
        max_depth=0,
        max_length=0,
        flat_increment=0,
        fill_pits=False,
        fix_flats=False,
    )

    assert captured["tool_name"] == "fill_depressions_wang_and_liu"
    assert captured["args"] == [
        f"--dem={input_path}",
        f"--output={output_path}",
    ]
    assert result["success"] is True


def test_run_whitebox_color_shaded_relief_uses_active_layer(
    monkeypatch, tmp_path
) -> None:
    """Verify the color shaded relief convenience tool uses the active DEM."""
    input_path = tmp_path / "dem.tif"
    output_path = tmp_path / "relief.tif"
    input_path.write_text("dem", encoding="utf-8")

    captured = {}

    class _FakeWBT:
        verbose = True

        def __init__(self):
            self.verbose = True

        def tool_parameters(self, tool_name):
            assert tool_name == "hypsometrically_tinted_hillshade"
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
                            "name": "Illumination Source Altitude (degrees)",
                            "flags": ["--altitude"],
                            "parameter_type": "Float",
                            "optional": True,
                        },
                        {
                            "name": "Hillshade Weight",
                            "flags": ["--hs_weight"],
                            "parameter_type": "Float",
                            "optional": True,
                        },
                        {
                            "name": "Brightness",
                            "flags": ["--brightness"],
                            "parameter_type": "Float",
                            "optional": True,
                        },
                        {
                            "name": "Atmospheric Effects",
                            "flags": ["--atmospheric"],
                            "parameter_type": "Float",
                            "optional": True,
                        },
                        {
                            "name": "Palette",
                            "flags": ["--palette"],
                            "parameter_type": {"OptionList": ["atlas", "viridis"]},
                            "optional": True,
                        },
                        {
                            "name": "Reverse palette?",
                            "flags": ["--reverse"],
                            "parameter_type": "Boolean",
                            "optional": True,
                        },
                        {
                            "name": "Full 360-degree hillshade mode?",
                            "flags": ["--full_mode"],
                            "parameter_type": "Boolean",
                            "optional": True,
                        },
                    ]
                }
            )

        def run_tool(self, tool_name, args, callback=None):
            captured["tool_name"] = tool_name
            captured["args"] = args
            output_path.write_text("relief", encoding="utf-8")
            return 0

    whitebox_module = ModuleType("whitebox")
    whitebox_module.WhiteboxTools = _FakeWBT
    monkeypatch.setitem(sys.modules, "whitebox", whitebox_module)

    project = MockQGISProject()
    layer = MockQGISLayer("DEM layer", str(input_path), "raster")
    project.addMapLayer(layer)
    iface = MockQGISIface(project)
    iface.setActiveLayer(layer)
    tools = {tool.tool_name: tool for tool in whitebox_tools(iface, project)}

    result = tools["run_whitebox_color_shaded_relief"].__wrapped__(
        output_path=str(output_path),
        output_layer_name="Color Shaded Relief",
        palette="viridis",
        hillshade_weight=0.6,
        full_mode=True,
    )

    assert captured["tool_name"] == "hypsometrically_tinted_hillshade"
    assert captured["args"] == [
        f"--dem={input_path}",
        f"--output={output_path}",
        "--altitude=45.0",
        "--hs_weight=0.6",
        "--brightness=0.5",
        "--atmospheric=0.0",
        "--palette=viridis",
        "--reverse=false",
        "--full_mode",
    ]
    assert result["success"] is True
    assert result["layers_added"] == [
        {
            "path": str(output_path),
            "added": True,
            "layer_name": "Color Shaded Relief",
        }
    ]
    assert "Color Shaded Relief" in project.mapLayers()


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


def test_whitebox_prompt_instructs_pyqgis_fallback() -> None:
    """Verify QGIS API fallback execution stays in the Whitebox prompt."""
    from geoagent.core.factory import WHITEBOX_SYSTEM_PROMPT

    assert "run_pyqgis_script" in WHITEBOX_SYSTEM_PROMPT
    assert "Do not merely provide a script" in WHITEBOX_SYSTEM_PROMPT


def test_qgis_default_prompt_instructs_pyqgis_fallback() -> None:
    """Verify the default for_qgis prompt also points the agent at run_pyqgis_script."""
    from geoagent.core.factory import QGIS_SYSTEM_PROMPT

    assert "run_pyqgis_script" in QGIS_SYSTEM_PROMPT
    assert "Do not merely provide a script" in QGIS_SYSTEM_PROMPT
