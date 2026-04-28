"""Tests for the GEE Data Catalogs GeoAgent tool factory."""

from __future__ import annotations

import sys
from types import ModuleType

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
    assert "calculate_gee_normalized_difference" in tools
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


def test_load_gee_dataset_clips_raster_to_feature_collection(monkeypatch) -> None:
    """Verify raster clipping uses ee.Image.clipToCollection."""

    class _Canvas:
        def __init__(self) -> None:
            self.refreshed = False

        def refresh(self) -> None:
            self.refreshed = True

    class _Iface:
        def __init__(self) -> None:
            self.canvas = _Canvas()

        def mapCanvas(self) -> _Canvas:
            return self.canvas

    class _FakeFeatureCollection:
        def __init__(self, asset_id: str) -> None:
            self.asset_id = asset_id
            self.filters = []

        def filter(self, filter_obj):
            self.filters.append(filter_obj)
            return self

    class _FakeImage:
        def __init__(self, asset_id: str) -> None:
            self.asset_id = asset_id
            self.clipped_to = None

        def clipToCollection(self, feature_collection):
            self.clipped_to = feature_collection
            return self

    class _FakeFilter:
        @staticmethod
        def eq(prop: str, value: object):
            return ("eq", prop, value)

    ee_module = ModuleType("ee")
    ee_module.Filter = _FakeFilter
    ee_module.FeatureCollection = _FakeFeatureCollection
    ee_module.Image = lambda value: (
        value if isinstance(value, _FakeImage) else _FakeImage(value)
    )

    captured = {}

    ee_utils = ModuleType("gee_data_catalogs.core.ee_utils")
    ee_utils.add_ee_layer = lambda obj, vis, name: captured.update(
        {"object": obj, "vis": vis, "name": name}
    )
    ee_utils.detect_asset_type = lambda asset_id: "Image"
    ee_utils.filter_image_collection = lambda collection, **kwargs: collection
    ee_utils.initialize_ee = lambda project=None: None
    ee_utils.is_ee_initialized = lambda: True

    monkeypatch.setitem(sys.modules, "ee", ee_module)
    monkeypatch.setitem(
        sys.modules, "gee_data_catalogs", ModuleType("gee_data_catalogs")
    )
    monkeypatch.setitem(
        sys.modules, "gee_data_catalogs.core", ModuleType("gee_data_catalogs.core")
    )
    monkeypatch.setitem(sys.modules, "gee_data_catalogs.core.ee_utils", ee_utils)

    iface = _Iface()
    tools = {t.tool_name: t for t in gee_data_catalogs_tools(iface)}
    result = tools["load_gee_dataset"].__wrapped__(
        "JRC/GSW1_4/GlobalSurfaceWater",
        bands="occurrence",
        min_value=0,
        max_value=100,
        palette="white,blue",
        clip_collection_asset_id="TIGER/2018/States",
        clip_filter_property="NAME",
        clip_filter_value="Tennessee",
    )

    loaded = captured["object"]
    assert result["success"] is True
    assert result["clip"]["method"] == "ee.Image.clipToCollection"
    assert result["clip"]["collection_asset_id"] == "TIGER/2018/States"
    assert loaded.clipped_to.asset_id == "TIGER/2018/States"
    assert loaded.clipped_to.filters == [("eq", "NAME", "Tennessee")]
    assert captured["vis"]["bands"] == ["occurrence"]
    assert iface.canvas.refreshed is True


def test_calculate_gee_normalized_difference_loads_index(monkeypatch) -> None:
    """Verify normalized difference indexes use Earth Engine band math."""

    class _Canvas:
        def __init__(self) -> None:
            self.refreshed = False

        def refresh(self) -> None:
            self.refreshed = True

    class _Iface:
        def __init__(self) -> None:
            self.canvas = _Canvas()

        def mapCanvas(self) -> _Canvas:
            return self.canvas

    class _FakeFeatureCollection:
        def __init__(self, asset_id: str) -> None:
            self.asset_id = asset_id
            self.filters = []

        def filter(self, filter_obj):
            self.filters.append(filter_obj)
            return self

    class _FakeImage:
        def __init__(self, asset_id: str) -> None:
            self.asset_id = asset_id
            self.normalized_bands = None
            self.name = None
            self.clipped_to = None

        def normalizedDifference(self, bands):
            self.normalized_bands = list(bands)
            return self

        def rename(self, name: str):
            self.name = name
            return self

        def clipToCollection(self, feature_collection):
            self.clipped_to = feature_collection
            return self

    class _FakeFilter:
        @staticmethod
        def eq(prop: str, value: object):
            return ("eq", prop, value)

    ee_module = ModuleType("ee")
    ee_module.Filter = _FakeFilter
    ee_module.FeatureCollection = _FakeFeatureCollection
    ee_module.Image = lambda value: (
        value if isinstance(value, _FakeImage) else _FakeImage(value)
    )

    captured = {}

    ee_utils = ModuleType("gee_data_catalogs.core.ee_utils")
    ee_utils.add_ee_layer = lambda obj, vis, name: captured.update(
        {"object": obj, "vis": vis, "name": name}
    )
    ee_utils.detect_asset_type = lambda asset_id: "Image"
    ee_utils.filter_image_collection = lambda collection, **kwargs: collection
    ee_utils.initialize_ee = lambda project=None: None
    ee_utils.is_ee_initialized = lambda: True

    monkeypatch.setitem(sys.modules, "ee", ee_module)
    monkeypatch.setitem(
        sys.modules, "gee_data_catalogs", ModuleType("gee_data_catalogs")
    )
    monkeypatch.setitem(
        sys.modules, "gee_data_catalogs.core", ModuleType("gee_data_catalogs.core")
    )
    monkeypatch.setitem(sys.modules, "gee_data_catalogs.core.ee_utils", ee_utils)

    iface = _Iface()
    tools = {t.tool_name: t for t in gee_data_catalogs_tools(iface)}
    result = tools["calculate_gee_normalized_difference"].__wrapped__(
        "NASA/HLS/HLSS30/v002",
        positive_band="B8",
        negative_band="B4",
        index_name="NDVI",
        layer_name="HLS S2 NDVI",
        clip_collection_asset_id="TIGER/2018/States",
        clip_filter_property="NAME",
        clip_filter_value="Tennessee",
    )

    loaded = captured["object"]
    assert result["success"] is True
    assert result["formula"] == "(B8 - B4) / (B8 + B4)"
    assert result["bands"] == ["B8", "B4"]
    assert loaded.normalized_bands == ["B8", "B4"]
    assert loaded.name == "NDVI"
    assert loaded.clipped_to.asset_id == "TIGER/2018/States"
    assert captured["name"] == "HLS S2 NDVI"
    assert captured["vis"]["bands"] == ["NDVI"]
    assert captured["vis"]["min"] == -1.0
    assert captured["vis"]["max"] == 1.0
    assert iface.canvas.refreshed is True
