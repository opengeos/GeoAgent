"""Tests for the GeoAI QGIS plugin tool adapter."""

from __future__ import annotations

import sys
import types
from typing import Any

from geoagent.core.decorators import needs_confirmation
from geoagent.testing import MockQGISIface, MockQGISLayer, MockQGISProject
from geoagent.tools.geoai import geoai_tools


def _install_fake_geoai_modules(monkeypatch, samgeo_client, task_calls):
    """Install fake GeoAI plugin modules used by the adapter."""
    geoai_pkg = types.ModuleType("geoai")
    geoai_pkg.__path__ = []
    core_pkg = types.ModuleType("geoai.core")
    core_pkg.__path__ = []
    samgeo_module = types.ModuleType("geoai.core.samgeo_subprocess")
    samgeo_module.SamGeoSubprocessClient = samgeo_client
    task_module = types.ModuleType("geoai.core.geoai_task_subprocess")

    def run_geoai_task(action, params):
        task_calls.append((action, params))
        return {"output_path": params["output_path"], "action": action}

    task_module.run_geoai_task = run_geoai_task
    monkeypatch.setitem(sys.modules, "geoai", geoai_pkg)
    monkeypatch.setitem(sys.modules, "geoai.core", core_pkg)
    monkeypatch.setitem(sys.modules, "geoai.core.samgeo_subprocess", samgeo_module)
    monkeypatch.setitem(sys.modules, "geoai.core.geoai_task_subprocess", task_module)


class _Plugin:
    __module__ = "geoai.geoai_plugin"

    def __init__(self, plugin_dir: str, calls: list[tuple[str, Any]]):
        self.plugin_dir = plugin_dir
        self.calls = calls

    def _ensure_dependencies(self, action_name):
        self.calls.append(("ensure_dependencies", action_name))
        return True


def test_geoai_tools_module_imports_without_qgis() -> None:
    """Verify the GeoAI tools module is import-safe outside QGIS."""
    assert "geoagent.tools.geoai" in sys.modules


def test_geoai_tools_returns_empty_for_none_iface() -> None:
    """Verify the GeoAI factory returns no tools without a QGIS iface."""
    assert geoai_tools(None) == []


def test_geoai_tool_reports_missing_plugin() -> None:
    """Verify missing GeoAI plugin is reported as a structured result."""
    iface = MockQGISIface()
    tool = {t.tool_name: t for t in geoai_tools(iface)}[
        "segment_image_with_text_prompt"
    ]

    result = tool(prompt="building")

    assert result["success"] is False
    assert "GeoAI QGIS plugin is not loaded" in result["error"]


def test_geoai_tool_metadata_requires_confirmation() -> None:
    """Verify segmentation is confirmation-gated and long-running."""
    iface = MockQGISIface()
    tool = {t.tool_name: t for t in geoai_tools(iface)}[
        "segment_image_with_text_prompt"
    ]

    meta = getattr(tool, "_geoagent_meta")
    assert meta.category == "geoai"
    assert meta.long_running is True
    assert needs_confirmation(tool) is True


def test_segment_image_with_text_prompt_uses_geoai_samgeo_client(
    monkeypatch,
    tmp_path,
) -> None:
    """Verify text-prompt segmentation calls GeoAI's SamGeo subprocess surface."""
    calls: list[tuple[str, Any]] = []

    class _FakeSamGeoClient:
        def __init__(self, **kwargs):
            calls.append(("init_args", kwargs))
            self.masks = []

        def initialize(self):
            calls.append(("initialize", None))

        def set_image(self, source, bands=None):
            calls.append(("set_image", {"source": source, "bands": bands}))

        def generate_masks(self, prompt, min_size=None, max_size=None):
            calls.append(
                (
                    "generate_masks",
                    {"prompt": prompt, "min_size": min_size, "max_size": max_size},
                )
            )
            self.masks = [object(), object()]

        def save_masks(self, output, unique=False):
            calls.append(("save_masks", {"output": output, "unique": unique}))

    task_calls: list[tuple[str, Any]] = []
    _install_fake_geoai_modules(monkeypatch, _FakeSamGeoClient, task_calls)

    project = MockQGISProject()
    iface = MockQGISIface(project=project)
    image_layer = MockQGISLayer("Imagery", "/tmp/image.tif", "raster")
    project.addMapLayer(image_layer)
    iface.setActiveLayer(image_layer)
    output_path = str(tmp_path / "mask.tif")

    tool = {
        item.tool_name: item
        for item in geoai_tools(
            iface, project, plugin=_Plugin(str(tmp_path / "geoai"), calls)
        )
    }["segment_image_with_text_prompt"]
    result = tool(
        prompt="buildings",
        output_path=output_path,
        bands="3,2,1",
        min_size=10,
        max_size=1000,
        unique=True,
        output_layer_name="Building mask",
    )

    assert result["success"] is True
    assert result["prompt"] == "buildings"
    assert result["source"] == "/tmp/image.tif"
    assert result["source_type"] == "active_qgis_layer"
    assert result["mask_count"] == 2
    assert result["output_path"] == output_path
    assert result["added_to_qgis"] is True
    assert result["output_layer_name"] == "Building mask"
    output_layer = project.mapLayersByName("Building mask")[0]
    assert output_layer.source() == output_path
    assert output_layer.symbology == {}
    assert ("ensure_dependencies", "samgeo") in calls
    assert ("initialize", None) in calls
    assert (
        "set_image",
        {"source": "/tmp/image.tif", "bands": [3, 2, 1]},
    ) in calls
    assert (
        "generate_masks",
        {"prompt": "buildings", "min_size": 10, "max_size": 1000},
    ) in calls
    assert ("save_masks", {"output": output_path, "unique": True}) in calls


def test_segment_image_with_text_prompt_skips_duplicate_call(monkeypatch, tmp_path):
    """Verify same-turn duplicate segmentation calls do not rerun SamGeo."""
    calls: list[tuple[str, Any]] = []
    task_calls: list[tuple[str, Any]] = []

    class _FakeSamGeoClient:
        def __init__(self, **_kwargs):
            self.masks = []

        def initialize(self):
            calls.append(("initialize", None))

        def set_image(self, source, bands=None):
            calls.append(("set_image", {"source": source, "bands": bands}))

        def generate_masks(self, prompt, min_size=None, max_size=None):
            calls.append(("generate_masks", {"min_size": min_size}))
            self.masks = [object()]

        def save_masks(self, output, unique=False):
            calls.append(("save_masks", output))

    _install_fake_geoai_modules(monkeypatch, _FakeSamGeoClient, task_calls)
    project = MockQGISProject()
    iface = MockQGISIface(project=project)
    image_layer = MockQGISLayer("Imagery", "/tmp/image.tif", "raster")
    project.addMapLayer(image_layer)
    iface.setActiveLayer(image_layer)
    tools = {
        item.tool_name: item
        for item in geoai_tools(
            iface, project, plugin=_Plugin(str(tmp_path / "geoai"), calls)
        )
    }

    first = tools["segment_image_with_text_prompt"](
        prompt="buildings",
        output_layer_name="Buildings SamGeo Mask",
        min_size=0,
    )
    second = tools["segment_image_with_text_prompt"](
        prompt="buildings",
        output_layer_name="Buildings SamGeo Mask",
        min_size=1,
    )

    assert first["success"] is True
    assert second["skipped_duplicate"] is True
    assert second["output_path"] == first["output_path"]
    assert ("generate_masks", {"min_size": 0}) in calls
    assert [name for name, _ in calls].count("generate_masks") == 1
    assert [name for name, _ in calls].count("save_masks") == 1


def test_segment_image_with_text_prompt_saves_vector_output(monkeypatch, tmp_path):
    """Verify segmentation can save a regularized vector output."""
    calls: list[tuple[str, Any]] = []
    task_calls: list[tuple[str, Any]] = []

    class _FakeSamGeoClient:
        def __init__(self, **_kwargs):
            self.masks = []

        def initialize(self):
            calls.append(("initialize", None))

        def set_image(self, source, bands=None):
            calls.append(("set_image", {"source": source, "bands": bands}))

        def generate_masks(self, prompt, min_size=None, max_size=None):
            calls.append(("generate_masks", prompt))
            self.masks = [object()]

        def save_masks(self, output, unique=False):
            calls.append(("save_masks", {"output": output, "unique": unique}))

    _install_fake_geoai_modules(monkeypatch, _FakeSamGeoClient, task_calls)
    project = MockQGISProject()
    iface = MockQGISIface(project=project)
    image_layer = MockQGISLayer("Imagery", "/tmp/image.tif", "raster")
    project.addMapLayer(image_layer)
    iface.setActiveLayer(image_layer)
    output_path = str(tmp_path / "buildings.geojson")

    tools = {
        item.tool_name: item
        for item in geoai_tools(
            iface, project, plugin=_Plugin(str(tmp_path / "geoai"), calls)
        )
    }
    result = tools["segment_image_with_text_prompt"](
        prompt="buildings",
        output_path=output_path,
        output_format="geojson",
        vector_mode="regularize",
        epsilon=3.5,
        min_area=25,
        output_layer_name="Buildings vector",
    )

    assert result["success"] is True
    assert result["output_kind"] == "vector"
    assert result["output_path"] == output_path
    assert result["added_to_qgis"] is True
    assert project.mapLayersByName("Buildings vector")[0].type() == "vector"
    assert len(task_calls) == 1
    action, params = task_calls[0]
    assert action == "vectorize_mask"
    assert params["output_path"] == output_path
    assert params["epsilon"] == 3.5
    assert params["min_area"] == 25
    assert params["mask_path"].endswith(".tif")


def test_generic_vector_output_defaults_to_geopackage(monkeypatch, tmp_path):
    """Verify generic vector output writes GeoPackage by default."""
    calls: list[tuple[str, Any]] = []
    task_calls: list[tuple[str, Any]] = []

    class _FakeSamGeoClient:
        def __init__(self, **_kwargs):
            self.masks = []

        def initialize(self):
            pass

        def set_image(self, source, bands=None):
            pass

        def generate_masks(self, prompt, min_size=None, max_size=None):
            self.masks = [object()]

        def save_masks(self, output, unique=False):
            pass

    _install_fake_geoai_modules(monkeypatch, _FakeSamGeoClient, task_calls)
    project = MockQGISProject()
    iface = MockQGISIface(project=project)
    image_layer = MockQGISLayer("Imagery", "/tmp/image.tif", "raster")
    project.addMapLayer(image_layer)
    iface.setActiveLayer(image_layer)
    tools = {
        item.tool_name: item
        for item in geoai_tools(
            iface, project, plugin=_Plugin(str(tmp_path / "geoai"), calls)
        )
    }

    result = tools["segment_image_with_text_prompt"](
        prompt="buildings",
        output_format="vector",
    )

    assert result["success"] is True
    assert result["output_format"] == "vector"
    assert result["output_path"].endswith(".gpkg")
    assert task_calls[0][0] == "raster_to_vector"
    assert task_calls[0][1]["output_format"] == "gpkg"


def test_regularize_and_smooth_vector_tools_use_geoai_tasks(monkeypatch, tmp_path):
    """Verify standalone vector post-processing tools route to GeoAI tasks."""
    calls: list[tuple[str, Any]] = []
    task_calls: list[tuple[str, Any]] = []

    class _UnusedSamGeoClient:
        pass

    _install_fake_geoai_modules(monkeypatch, _UnusedSamGeoClient, task_calls)
    project = MockQGISProject()
    iface = MockQGISIface(project=project)
    plugin = _Plugin(str(tmp_path / "geoai"), calls)
    tools = {
        item.tool_name: item for item in geoai_tools(iface, project, plugin=plugin)
    }

    regularized = tools["regularize_segmentation_mask_to_vector"](
        mask_path="/tmp/mask.tif",
        output_path=str(tmp_path / "regularized.gpkg"),
        epsilon=1.5,
        min_area=10,
        output_layer_name="Regularized",
    )
    smoothed = tools["smooth_segmentation_mask_to_vector"](
        mask_path="/tmp/mask.tif",
        output_path=str(tmp_path / "smooth.geojson"),
        smooth_iterations=5,
        simplify_tolerance=0.25,
        output_layer_name="Smoothed",
    )

    assert regularized["success"] is True
    assert smoothed["success"] is True
    assert project.mapLayersByName("Regularized")
    assert project.mapLayersByName("Smoothed")
    assert task_calls[0] == (
        "vectorize_mask",
        {
            "mask_path": "/tmp/mask.tif",
            "output_path": str(tmp_path / "regularized.gpkg"),
            "output_format": "gpkg",
            "epsilon": 1.5,
            "min_area": 10.0,
        },
    )
    assert task_calls[1] == (
        "smooth_vector",
        {
            "mask_path": "/tmp/mask.tif",
            "output_path": str(tmp_path / "smooth.geojson"),
            "output_format": "gpkg",
            "smooth_iterations": 5,
            "min_area": None,
            "simplify_tolerance": 0.25,
        },
    )
