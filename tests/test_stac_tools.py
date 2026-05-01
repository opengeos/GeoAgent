"""Tests for STAC tool helpers and factory wiring."""

from __future__ import annotations

import sys
import types

from geoagent import for_stac
from geoagent.testing import MockQGISIface, MockQGISLayer, MockQGISProject
from geoagent.tools.stac import (
    PLANETARY_COMPUTER_STAC_URL,
    _qgis_raster_uri,
    _zoom_to_qgis_layer,
    stac_tools,
)


class _MockModel:
    """Tiny model stand-in for GeoAgent factory tests."""

    stateful = False


class _FakeItem:
    def __init__(
        self,
        item_id="item-1",
        cloud_cover=25.0,
        bbox=None,
        collection="sentinel",
        assets=None,
    ):
        self.item_id = item_id
        self.cloud_cover = cloud_cover
        self.bbox = bbox or [-84, 35, -83, 36]
        self.collection = collection
        self.assets = assets or {
            "visual": {
                "href": "https://example.com/visual.tif",
                "type": "image/tiff; application=geotiff",
                "roles": ["visual"],
            }
        }

    def to_dict(self):
        west, south, east, north = self.bbox
        return {
            "id": self.item_id,
            "collection": self.collection,
            "bbox": self.bbox,
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [west, south],
                        [east, south],
                        [east, north],
                        [west, north],
                        [west, south],
                    ]
                ],
            },
            "properties": {
                "datetime": "2024-01-01T00:00:00Z",
                "eo:cloud_cover": self.cloud_cover,
                "s2:mgrs_tile": "17SKV",
            },
            "assets": self.assets,
        }


class _FakeSearch:
    def __init__(self, items=None):
        self._items = list(items or [_FakeItem()])

    def items(self):
        return list(self._items)


class _FakeCollection:
    id = "sentinel"

    def to_dict(self):
        return {"id": "sentinel", "title": "Sentinel", "description": "Imagery"}


class _FakeClient:
    last_url = None
    last_timeout = None
    last_search_kwargs = None
    items = None

    @classmethod
    def open(cls, catalog_url, **kwargs):
        cls.last_url = catalog_url
        cls.last_timeout = kwargs.get("timeout")
        return cls()

    def get_collections(self):
        return [_FakeCollection()]

    def search(self, **kwargs):
        self.__class__.last_search_kwargs = dict(kwargs)
        return _FakeSearch(self.__class__.items)

    def get_item(self, _item_id, recursive=True):  # noqa: ARG002
        return _FakeItem()


def _install_fake_stac_modules(monkeypatch):
    pystac_client = types.ModuleType("pystac_client")
    _FakeClient.items = [
        _FakeItem("cloudy", 72.0),
        _FakeItem("clear", 2.5),
        _FakeItem("partly-cloudy", 18.0),
    ]
    pystac_client.Client = _FakeClient
    planetary_computer = types.ModuleType("planetary_computer")
    planetary_computer.sign = lambda href: f"{href}?signed=true"
    monkeypatch.setitem(sys.modules, "pystac_client", pystac_client)
    monkeypatch.setitem(sys.modules, "planetary_computer", planetary_computer)


def test_stac_search_lists_items(monkeypatch) -> None:
    """STAC search should return compact item summaries."""
    _install_fake_stac_modules(monkeypatch)
    tools = {tool.tool_name: tool for tool in stac_tools()}

    result = tools["search_stac_items"].__wrapped__(
        catalog_url="https://example.com/stac",
        collection="sentinel-2-l2a",
        bbox="-84,35,-83,36",
    )

    assert result["success"] is True
    assert result["count"] == 3
    assert _FakeClient.last_timeout == (3.05, 15.0)
    assert result["items"][0]["id"] == "clear"
    assert result["items"][0]["cloud_cover"] == 2.5
    assert result["items"][0]["asset_keys"] == ["visual"]
    assert result["items"][0]["preferred_assets"][0]["key"] == "visual"
    assert result["items"][0]["contains_query_center"] is True


def test_stac_search_defaults_to_planetary_computer(monkeypatch) -> None:
    """STAC search should use Planetary Computer when no catalog is provided."""
    _install_fake_stac_modules(monkeypatch)
    tools = {tool.tool_name: tool for tool in stac_tools()}

    result = tools["search_stac_items"].__wrapped__(collection="sentinel")

    assert result["success"] is True
    assert result["catalog_url"] == PLANETARY_COMPUTER_STAC_URL
    assert _FakeClient.last_url == PLANETARY_COMPUTER_STAC_URL


def test_stac_search_filters_cloud_free_items(monkeypatch) -> None:
    """Cloud-free requests should filter and rank by eo:cloud_cover."""
    _install_fake_stac_modules(monkeypatch)
    tools = {tool.tool_name: tool for tool in stac_tools()}

    result = tools["search_stac_items"].__wrapped__(
        collection="sentinel-2-l2a",
        query_text="cloud-free true color",
    )

    assert result["max_cloud_cover"] == 10.0
    assert result["sorted_by"] == "cloud_cover"
    assert result["items"][0]["id"] == "clear"
    assert [item["id"] for item in result["items"]] == ["clear"]
    assert _FakeClient.last_search_kwargs["query"] == {"eo:cloud_cover": {"lte": 10.0}}


def test_stac_search_ignores_cloud_filter_for_dem_collection(monkeypatch) -> None:
    """Cloud filters should not hide DEM collections without cloud metadata."""
    _install_fake_stac_modules(monkeypatch)
    _FakeClient.items = [
        _FakeItem(
            "dem",
            cloud_cover=None,
            collection="cop-dem-glo-30",
            assets={
                "rendered_preview": {
                    "href": "https://example.com/preview.png",
                    "type": "image/png",
                    "roles": ["overview"],
                },
                "data": {
                    "href": "https://example.com/dem.tif",
                    "type": "image/tiff; application=geotiff",
                    "roles": ["data"],
                },
            },
        )
    ]
    tools = {tool.tool_name: tool for tool in stac_tools()}

    result = tools["search_stac_items"].__wrapped__(
        collection="cop-dem-glo-30",
        bbox="-122.459,47.481,-122.224,47.734",
        max_cloud_cover=0,
    )

    assert result["count"] == 1
    assert result["max_cloud_cover"] is None
    assert result["ignored_max_cloud_cover"] == 0.0
    assert "query" not in _FakeClient.last_search_kwargs
    assert result["items"][0]["preferred_assets"][0]["key"] == "data"


def test_stac_search_prefers_spatial_fit_before_cloud_cover(monkeypatch) -> None:
    """Cloud ranking should not pick a tile that misses the requested center."""
    _install_fake_stac_modules(monkeypatch)
    _FakeClient.items = [
        _FakeItem("edge-clear", 0.1, bbox=[-85.0, 35.0, -84.5, 36.0]),
        _FakeItem("center-low-cloud", 4.0, bbox=[-84.2, 35.8, -83.7, 36.2]),
    ]
    tools = {tool.tool_name: tool for tool in stac_tools()}

    result = tools["search_stac_items"].__wrapped__(
        collection="sentinel-2-l2a",
        bbox="-84.1,35.85,-83.75,36.1",
        max_cloud_cover=10,
    )

    assert result["sorted_by"] == "spatial_fit,cloud_cover"
    assert result["items"][0]["id"] == "center-low-cloud"
    assert result["items"][0]["contains_query_center"] is True


def test_stac_assets_are_signed_when_available(monkeypatch) -> None:
    """Asset inspection should use planetary-computer signing when present."""
    _install_fake_stac_modules(monkeypatch)
    tools = {tool.tool_name: tool for tool in stac_tools()}

    result = tools["get_stac_item_assets"].__wrapped__(
        "https://example.com/stac",
        "sentinel",
        "item-1",
    )

    assert result["success"] is True
    assert result["assets"][0]["href"].endswith("?signed=true")


def test_add_stac_asset_to_qgis_loads_raster(monkeypatch) -> None:
    """The QGIS STAC loader should add concrete asset URLs as raster layers."""
    _install_fake_stac_modules(monkeypatch)
    project = MockQGISProject()
    iface = MockQGISIface(project)
    tools = {tool.tool_name: tool for tool in stac_tools(iface, project)}

    result = tools["add_stac_asset_to_qgis"].__wrapped__(
        "/tmp/visual.tif",
        "Visual",
    )

    assert result["loaded"] is True
    assert project.mapLayers()


def test_add_stac_asset_to_qgis_loads_remote_raster(monkeypatch) -> None:
    """Remote STAC COGs should be passed through to QGIS for loading."""
    _install_fake_stac_modules(monkeypatch)
    project = MockQGISProject()
    iface = MockQGISIface(project)
    tools = {tool.tool_name: tool for tool in stac_tools(iface, project)}

    result = tools["add_stac_asset_to_qgis"].__wrapped__(
        "https://example.com/visual.tif",
        "Visual",
    )

    assert result["success"] is True
    assert result["loaded"] is True
    assert result["zoomed"] is True
    assert result["asset_href"].startswith("https://example.com/visual.tif")
    assert result["qgis_uri"].startswith("/vsicurl/https://example.com/visual.tif")
    assert "Visual" in project.mapLayers()
    assert project.mapLayers()["Visual"].source().startswith("/vsicurl/https://")
    assert iface.activeLayer() is project.mapLayers()["Visual"]
    assert iface.mapCanvas().refresh_count > 0


def test_qgis_raster_uri_uses_vsicurl_for_http_assets() -> None:
    """Remote STAC COGs should use the qgis-stac-plugin /vsicurl pattern."""
    assert _qgis_raster_uri("https://example.com/a.tif?sig=abc") == (
        "/vsicurl/https://example.com/a.tif?sig=abc"
    )
    assert _qgis_raster_uri("/vsicurl/https://example.com/a.tif") == (
        "/vsicurl/https://example.com/a.tif"
    )
    assert _qgis_raster_uri("/tmp/a.tif") == "/tmp/a.tif"


def test_stac_zoom_prefers_layer_extent_over_iface_zoom() -> None:
    """Loaded STAC rasters should drive the canvas extent directly."""
    iface = MockQGISIface()
    layer = MockQGISLayer(
        "Visual",
        source="/tmp/visual.tif",
        layer_type="raster",
        extent=(-84.1, 35.85, -83.75, 36.1),
    )

    zoomed = _zoom_to_qgis_layer(iface, layer)

    assert zoomed is True
    assert iface.activeLayer() is layer
    assert iface.mapCanvas().extent() == (-84.1, 35.85, -83.75, 36.1)
    assert iface.mapCanvas().refresh_count > 0


def test_current_stac_search_extent_uses_qgis_canvas(monkeypatch) -> None:
    """STAC tools should expose current QGIS extent without PyQGIS scripts."""
    _install_fake_stac_modules(monkeypatch)
    iface = MockQGISIface()
    iface.mapCanvas().setExtent((-84, 35, -83, 36))
    tools = {tool.tool_name: tool for tool in stac_tools(iface)}

    result = tools["get_current_stac_search_extent"].__wrapped__()

    assert result["success"] is True
    assert result["bbox"] == [-84.0, 35.0, -83.0, 36.0]


def test_for_stac_registers_stac_tools(monkeypatch) -> None:
    """The STAC factory should expose STAC tools with QGIS tools."""
    _install_fake_stac_modules(monkeypatch)
    agent = for_stac(MockQGISIface(), MockQGISProject(), model=_MockModel())

    names = set(agent.strands_agent.tool_names)

    assert "get_current_stac_search_extent" in names
    assert "search_stac_items" in names
    assert "add_stac_asset_to_qgis" in names
    assert "run_pyqgis_script" not in names
    assert agent.context.metadata["integration"] == "stac"
