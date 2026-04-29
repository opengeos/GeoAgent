"""Tests for the GEE Data Catalogs GeoAgent tool factory."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from geoagent import for_gee_data_catalogs
from geoagent.testing import MockQGISIface, MockQGISProject
from geoagent.tools.gee_data_catalogs import (
    _xyz_uri_from_tile_url,
    gee_data_catalogs_tools,
)


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


def test_ee_xyz_uri_encodes_nested_query_parameters() -> None:
    """Verify QGIS parses EE token query params as part of the tile URL."""
    uri = _xyz_uri_from_tile_url(
        "https://example.com/{z}/{x}/{y}?token=abc&expires=123"
    )

    expected = (
        "type=xyz&url=https://example.com/{z}/{x}/{y}"
        "%3Ftoken%3Dabc%26expires%3D123&zmax=24&zmin=0"
    )
    assert uri == expected


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

    class _FakeLayer:
        def isValid(self):
            return True

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

    def _add_ee_layer(obj, vis, name):
        captured.update({"object": obj, "vis": vis, "name": name})
        return _FakeLayer()

    ee_utils.add_ee_layer = _add_ee_layer
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
    assert result["composite_method"] is None
    assert result["clip"]["method"] == "ee.Image.clipToCollection"
    assert result["clip"]["collection_asset_id"] == "TIGER/2018/States"
    assert loaded.clipped_to.asset_id == "TIGER/2018/States"
    assert loaded.clipped_to.filters == [("eq", "NAME", "Tennessee")]
    assert captured["vis"]["bands"] == ["occurrence"]
    assert iface.canvas.refreshed is True


def test_load_gee_dataset_uses_mosaic_as_composite_method(monkeypatch) -> None:
    """Verify mosaic is handled as an ImageCollection method, not an ee.Reducer."""

    class _Canvas:
        def __init__(self) -> None:
            self.extent = None

        def setExtent(self, extent) -> None:
            self.extent = extent

        def refresh(self) -> None:
            pass

    class _Iface:
        def __init__(self) -> None:
            self.canvas = _Canvas()

        def mapCanvas(self) -> _Canvas:
            return self.canvas

    class _FakeImage:
        def __init__(self, method: str) -> None:
            self.method = method

    class _FakeImageCollection:
        def __init__(self, asset_id: str) -> None:
            self.asset_id = asset_id

        def mosaic(self):
            return _FakeImage("mosaic")

        def median(self):
            return _FakeImage("median")

        def mean(self):
            return _FakeImage("mean")

        def min(self):
            return _FakeImage("min")

        def max(self):
            return _FakeImage("max")

        def first(self):
            return _FakeImage("first")

    class _FakeLayer:
        def isValid(self):
            return True

    ee_module = ModuleType("ee")
    ee_module.ImageCollection = _FakeImageCollection

    captured = {}

    ee_utils = ModuleType("gee_data_catalogs.core.ee_utils")

    def _add_ee_layer(obj, vis, name):
        captured.update({"object": obj, "vis": vis, "name": name})
        return _FakeLayer()

    ee_utils.add_ee_layer = _add_ee_layer
    ee_utils.detect_asset_type = lambda asset_id: "ImageCollection"

    def _filter_image_collection(collection, **kwargs):
        captured["filter_kwargs"] = kwargs
        return collection

    ee_utils.filter_image_collection = _filter_image_collection
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
        "TEST/IMAGE_COLLECTION",
        reducer="mosaic",
        bands="WTR",
        min_value=0,
        max_value=2,
        palette="white,blue",
    )

    assert result["success"] is True
    assert result["composite_method"] == "mosaic"
    assert result["requested_reducer"] == "mosaic"
    assert captured["object"].method == "mosaic"
    assert captured["vis"]["bands"] == ["WTR"]

    false_color = tools["load_gee_dataset"].__wrapped__(
        "COPERNICUS/S2_SR_HARMONIZED",
        layer_name="Sentinel-2 false color composite - San Francisco",
        reducer="first",
        bands="B8,B4,B3",
        min_value=0,
        max_value=3000,
        bbox="-122.55,37.65,-122.3,37.85",
        start_date="2025-06-01",
        end_date="2025-08-31",
        cloud_cover=10,
    )

    snippet = false_color["earth_engine_python_snippet"]
    assert false_color["success"] is True
    assert false_color["composite_method"] == "first"
    assert "ee.Initialize()" not in snippet
    assert "import geemap" in snippet
    assert "m = geemap.Map()" in snippet
    assert "image = collection.first()" in snippet
    assert (
        "vis_params = {'bands': ['B8', 'B4', 'B3'], 'min': 0.0, 'max': 3000.0}"
        in snippet
    )
    assert (
        "m.add_layer(image, vis_params, "
        "'Sentinel-2 false color composite - San Francisco')" in snippet
    )


def test_load_gee_dataset_filters_explicit_image_collection_bbox(
    monkeypatch,
) -> None:
    """Verify explicit ImageCollection bbox filters are passed through."""

    class _Canvas:
        def __init__(self) -> None:
            self.extent = None

        def setExtent(self, extent) -> None:
            self.extent = extent

        def refresh(self) -> None:
            pass

    class _Iface:
        def __init__(self) -> None:
            self.canvas = _Canvas()

        def mapCanvas(self) -> _Canvas:
            return self.canvas

    class _FakeImage:
        def __init__(self, method: str) -> None:
            self.method = method

    class _FakeImageCollection:
        def __init__(self, asset_id: str) -> None:
            self.asset_id = asset_id

        def mosaic(self):
            return _FakeImage("mosaic")

    class _FakeLayer:
        def isValid(self):
            return True

    ee_module = ModuleType("ee")
    ee_module.ImageCollection = _FakeImageCollection

    captured = {}

    ee_utils = ModuleType("gee_data_catalogs.core.ee_utils")
    ee_utils.add_ee_layer = lambda obj, vis, name: _FakeLayer()
    ee_utils.detect_asset_type = lambda asset_id: "ImageCollection"

    def _filter_image_collection(collection, **kwargs):
        captured["filter_kwargs"] = kwargs
        return collection

    ee_utils.filter_image_collection = _filter_image_collection
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
        "TEST/IMAGE_COLLECTION",
        reducer="mosaic",
        bbox="-84,35,-83,36",
    )

    assert result["success"] is True
    assert captured["filter_kwargs"]["bbox"] == [-84.0, 35.0, -83.0, 36.0]
    assert result["bbox"] == [-84.0, 35.0, -83.0, 36.0]
    assert result["zoom"] == {
        "success": True,
        "target": "bbox",
        "bbox": [-84.0, 35.0, -83.0, 36.0],
    }
    assert "import geemap" in result["earth_engine_python_snippet"]
    assert "ee.Initialize()" not in result["earth_engine_python_snippet"]
    assert "collection.filterBounds" in result["earth_engine_python_snippet"]
    assert "image = collection.mosaic()" in result["earth_engine_python_snippet"]
    assert "m.add_layer(image, vis_params" in result["earth_engine_python_snippet"]
    assert iface.canvas.extent == (-84.0, 35.0, -83.0, 36.0)
    assert result["diagnostics"]["bbox"] == [-84.0, 35.0, -83.0, 36.0]


def test_load_gee_dataset_renders_opera_dswx_hls_with_valid_band_and_mode(
    monkeypatch,
) -> None:
    """Verify OPERA DSWx-HLS uses documented bands and remapped mode rendering."""

    class _Canvas:
        def refresh(self) -> None:
            pass

    class _Iface:
        def mapCanvas(self) -> _Canvas:
            return _Canvas()

    class _FakeImage:
        def __init__(self, band: str | None = None) -> None:
            self.band = band
            self.mask = None

        def select(self, band: str):
            return _FakeImage(band)

        def lt(self, value: int):
            return ("lt", self.band, value)

        def updateMask(self, mask):
            self.mask = mask
            return self

    class _FakeModeImage:
        def __init__(self) -> None:
            self.name = None
            self.selected = None
            self.mask = None

        def rename(self, name: str):
            self.name = name
            return self

        def select(self, band: str):
            self.selected = band
            return self

        def remap(self, source_values, target_values):
            self.source_values = source_values
            self.target_values = target_values
            return self

        def neq(self, value):
            return ("neq", value)

        def updateMask(self, mask):
            self.mask = mask
            return {
                "source_band": self.selected,
                "source_values": self.source_values,
                "target_values": self.target_values,
                "mask": mask,
            }

    class _FakeImageCollection:
        def __init__(self, asset_id: str) -> None:
            self.asset_id = asset_id
            self.mapped_image = None
            self.reducer = None
            self.date_filter = None
            self.bounds_filter = None

        def filterDate(self, start, end):
            self.date_filter = (start, end)
            return self

        def filterBounds(self, geometry):
            self.bounds_filter = geometry
            return self

        def map(self, fn):
            self.mapped_image = fn(_FakeImage())
            return self

        def reduce(self, reducer):
            self.reducer = reducer
            return _FakeModeImage()

    class _FakeReducer:
        @staticmethod
        def mode():
            return "mode"

        @staticmethod
        def max():
            return "max"

    class _FakeLayer:
        def isValid(self):
            return True

    ee_module = ModuleType("ee")
    ee_module.ImageCollection = _FakeImageCollection
    ee_module.Reducer = _FakeReducer
    ee_module.Geometry = type(
        "Geometry",
        (),
        {"Rectangle": staticmethod(lambda bbox: ("rectangle", bbox))},
    )

    captured = {}

    ee_utils = ModuleType("gee_data_catalogs.core.ee_utils")

    def _add_ee_layer(obj, vis, name):
        captured.update({"object": obj, "vis": vis, "name": name})
        return _FakeLayer()

    ee_utils.add_ee_layer = _add_ee_layer
    ee_utils.detect_asset_type = lambda asset_id: "ImageCollection"

    def _filter_image_collection(collection, **kwargs):
        raise AssertionError("DSWx should not use generic collection filtering")

    ee_utils.filter_image_collection = _filter_image_collection
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

    tools = {t.tool_name: t for t in gee_data_catalogs_tools(_Iface())}
    result = tools["load_gee_dataset"].__wrapped__(
        "OPERA/DSWX/L3_V1/HLS",
        reducer="mosaic",
        bands="B01_WTR",
        start_date="2025-01-01",
        end_date="2025-12-31",
        bbox="-84.05,35.85,-83.75,36.10",
        cloud_cover=20,
    )

    assert result["success"] is True
    assert result["composite_method"] == "mode"
    assert result["requested_reducer"] == "mosaic"
    assert result["rendered_band"] == "WTR_Water_classification"
    assert result["diagnostics"]["filtering"] == "direct_dswx_filterDate_filterBounds"
    assert result["diagnostics"]["ignored_cloud_cover"] == 20
    assert result["bbox"] == [-84.05, 35.85, -83.75, 36.1]
    assert result["diagnostics"]["bbox"] == [-84.05, 35.85, -83.75, 36.1]
    assert "ee.Reducer.mode()" in result["earth_engine_python_snippet"]
    assert "image.updateMask(image.neq(0))" in result["earth_engine_python_snippet"]
    assert "m = geemap.Map()" in result["earth_engine_python_snippet"]
    assert captured["object"]["source_band"] == "WTR_Water_classification"
    assert captured["object"]["source_values"] == [0, 1, 2, 252, 253, 254]
    assert captured["object"]["target_values"] == [0, 1, 2, 3, 4, 5]
    assert captured["object"]["mask"] == ("neq", 0)
    assert captured["vis"]["min"] == 0.0
    assert captured["vis"]["max"] == 5.0
    assert captured["vis"]["palette"] == [
        "ffffff",
        "0000ff",
        "0088ff",
        "f2f2f2",
        "dfdfdf",
        "da00ff",
    ]


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

    class _FakeLayer:
        def isValid(self):
            return True

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

    def _add_ee_layer(obj, vis, name):
        captured.update({"object": obj, "vis": vis, "name": name})
        return _FakeLayer()

    ee_utils.add_ee_layer = _add_ee_layer
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
    assert "normalizedDifference(['B8', 'B4'])" in result["earth_engine_python_snippet"]
    assert "ee.Initialize()" not in result["earth_engine_python_snippet"]
    assert (
        "m.add_layer(image, vis_params, 'HLS S2 NDVI')"
        in result["earth_engine_python_snippet"]
    )
    assert captured["vis"]["bands"] == ["NDVI"]
    assert captured["vis"]["min"] == -1.0
    assert captured["vis"]["max"] == 1.0
    assert iface.canvas.refreshed is True
