"""Tests for the HyperCoast GeoAgent tool factory."""

from __future__ import annotations

import json
import os
import sys
import types
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from geoagent import for_hypercoast
from geoagent.testing import MockQGISIface, MockQGISLayer, MockQGISProject
from geoagent.tools import hypercoast_tools
import geoagent.tools.hypercoast as hypercoast


class _MockModel:
    """Tiny model stand-in for GeoAgent factory tests."""

    stateful = False


class _FakeDataVars:
    """Small data_vars container for dataset metadata tests."""

    def __iter__(self):
        """Return fake variable names."""
        return iter(["reflectance"])


class _FakeXarrayDataset:
    """Small xarray-like dataset for metadata tests."""

    data_vars = _FakeDataVars()
    dims = {"wavelength": 3, "y": 2, "x": 2}


class _FakeDataset:
    """Small HyperCoast dataset stand-in."""

    created_data_types: list[str] = []

    def __init__(self, filepath: str, data_type: str = "auto") -> None:
        """Initialize the fake dataset.

        Args:
            filepath: Source dataset path.
            data_type: Requested data type.
        """
        self.filepath = filepath
        self.created_data_types.append(data_type)
        self.data_type = "EMIT" if data_type == "auto" else data_type
        self.dataset = _FakeXarrayDataset()
        self.wavelengths = [450.0, 550.0, 650.0]
        self.bounds = (-85.0, 34.0, -83.0, 36.0)
        self.crs = "EPSG:4326"
        self.selected_variable = None
        self.last_error = None
        self.export_calls: list[dict[str, Any]] = []

    def set_selected_variable(self, variable_name: str | None) -> None:
        """Record the selected variable.

        Args:
            variable_name: Selected data variable.
        """
        self.selected_variable = variable_name

    def load(self) -> bool:
        """Report successful dataset loading."""
        return True

    def load_and_export(self, output_path: str, wavelengths: list[float]) -> str:
        """Record a combined load/export request.

        Args:
            output_path: Output GeoTIFF path.
            wavelengths: RGB wavelengths.

        Returns:
            Output path.
        """
        self.export_calls.append(
            {"output_path": output_path, "wavelengths": wavelengths}
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("fake", encoding="utf-8")
        return output_path


class _FakeCommon:
    """Small HyperCoast common module stand-in."""

    def __init__(self) -> None:
        """Initialize call storage."""
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def search_emit(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Record an EMIT search request.

        Args:
            **kwargs: Search options.

        Returns:
            Fake granule list.
        """
        self.calls.append(("search_emit", kwargs))
        return [
            {
                "id": "emit-1",
                "umm": {
                    "GranuleUR": "EMIT_L2A_RFL_001",
                    "TemporalExtent": {"RangeDateTime": {}},
                },
            }
        ]

    def search_pace(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Record a PACE search request.

        Args:
            **kwargs: Search options.

        Returns:
            Fake granule list.
        """
        self.calls.append(("search_pace", kwargs))
        return [{"id": "pace-1", "umm": {"GranuleUR": "PACE_OCI_001"}}]

    def download_emit(
        self,
        granules: list[Any],
        out_dir: str | None = None,
        threads: int = 8,
    ) -> list[str]:
        """Record an EMIT download request.

        Args:
            granules: Granules to download.
            out_dir: Output directory.
            threads: Download thread count.

        Returns:
            Fake downloaded paths.
        """
        self.calls.append(
            (
                "download_emit",
                {"granules": granules, "out_dir": out_dir, "threads": threads},
            )
        )
        return [os.path.join(str(out_dir), "emit.nc")]

    def download_pace(
        self,
        granules: list[Any],
        out_dir: str | None = None,
        provider: str | None = None,
        threads: int = 8,
    ) -> list[str]:
        """Record a PACE download request.

        Args:
            granules: Granules to download.
            out_dir: Output directory.
            provider: Optional provider.
            threads: Download thread count.

        Returns:
            Fake downloaded paths.
        """
        self.calls.append(
            (
                "download_pace",
                {
                    "granules": granules,
                    "out_dir": out_dir,
                    "provider": provider,
                    "threads": threads,
                },
            )
        )
        return [os.path.join(str(out_dir), "pace.nc")]


class _FakeDownloadResponse:
    """Small context-manager response for direct download tests."""

    def __init__(self, payload: bytes) -> None:
        """Initialize fake response bytes.

        Args:
            payload: Bytes returned by ``read``.
        """
        self.payload = BytesIO(payload)

    def __enter__(self):
        """Return this response."""
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Close context manager without special handling."""
        return None

    def read(self, size: int = -1) -> bytes:
        """Return payload bytes."""
        return self.payload.read(size)


class _FakeOpener:
    """Small urllib opener stand-in for direct download tests."""

    def __init__(self, payload: bytes) -> None:
        """Initialize fake opener.

        Args:
            payload: Bytes returned for every request.
        """
        self.payload = payload

    def open(self, request, timeout: int = 120):
        """Return a fake response."""
        _ = request, timeout
        return _FakeDownloadResponse(self.payload)


class _FakeLayer:
    """Small QGIS layer stand-in with custom properties."""

    def __init__(self, name: str, source: str) -> None:
        """Initialize the fake layer.

        Args:
            name: Layer name.
            source: Layer source path.
        """
        self._name = name
        self._source = source
        self.properties: dict[str, Any] = {}

    def id(self) -> str:
        """Return the layer identifier."""
        return self._name

    def name(self) -> str:
        """Return the layer name."""
        return self._name

    def source(self) -> str:
        """Return the layer source."""
        return self._source

    def isValid(self) -> bool:
        """Return whether the layer is valid."""
        return True

    def setCustomProperty(self, key: str, value: Any) -> None:
        """Store a custom property.

        Args:
            key: Property key.
            value: Property value.
        """
        self.properties[key] = value


class _FakePlugin:
    """Small HyperCoast plugin stand-in."""

    plugin_dir = "/tmp/hypercoast_qgis"

    def __init__(self) -> None:
        """Initialize registration and panel call storage."""
        self.registered: dict[str, Any] = {}
        self.opened: list[str] = []

    def register_hyperspectral_layer(
        self, layer_id: str, data_info: dict[str, Any]
    ) -> None:
        """Record a registered HyperCoast layer.

        Args:
            layer_id: Layer id.
            data_info: Layer metadata.
        """
        self.registered[layer_id] = data_info

    def show_load_dialog(self) -> None:
        """Record opening the load dialog."""
        self.opened.append("load")


def test_hypercoast_module_imports_without_qgis_or_plugin() -> None:
    """Verify HyperCoast tools are import-safe outside QGIS and HyperCoast."""
    assert "geoagent.tools.hypercoast" in sys.modules
    if "qgis" in sys.modules:
        pytest.skip("qgis is already imported in this environment.")
    assert "qgis" not in sys.modules
    assert "hypercoast.common" not in sys.modules
    assert "hypercoast_qgis" not in sys.modules


def test_hypercoast_tools_returns_empty_for_none_iface() -> None:
    """Verify the HyperCoast factory returns no tools without iface."""
    assert hypercoast_tools(None) == []


def test_hypercoast_tools_expose_expected_surface() -> None:
    """Verify HyperCoast tool names and confirmation metadata."""
    tools = {tool.tool_name: tool for tool in hypercoast_tools(object())}

    assert set(tools) == {
        "list_hypercoast_data_types",
        "search_hypercoast_data",
        "display_hypercoast_footprints",
        "get_selected_hypercoast_footprints",
        "download_hypercoast_data",
        "preview_hypercoast_dataset",
        "load_hypercoast_rgb",
        "load_hypercoast_variable",
        "open_hypercoast_panel",
    }
    assert tools["display_hypercoast_footprints"]._geoagent_meta.requires_confirmation
    assert tools["download_hypercoast_data"]._geoagent_meta.requires_confirmation
    assert tools["download_hypercoast_data"]._geoagent_meta.long_running
    assert tools["load_hypercoast_rgb"]._geoagent_meta.requires_confirmation
    assert tools["load_hypercoast_rgb"]._geoagent_meta.long_running
    assert tools["load_hypercoast_variable"]._geoagent_meta.requires_confirmation
    assert tools["load_hypercoast_variable"]._geoagent_meta.long_running
    assert tools["open_hypercoast_panel"]._geoagent_meta.requires_confirmation


def test_list_hypercoast_data_types_includes_presets() -> None:
    """Verify supported data types and RGB presets are advertised."""
    tools = {tool.tool_name: tool for tool in hypercoast_tools(object())}

    result = tools["list_hypercoast_data_types"].__wrapped__()

    assert result["success"] is True
    assert "EMIT" in result["data_types"]
    assert "PACE" in result["data_types"]
    assert result["wavelength_presets"]["True Color (RGB)"] == [650.0, 550.0, 450.0]


def test_search_hypercoast_data_dispatches_to_emit(monkeypatch) -> None:
    """Verify EMIT search uses HyperCoast common lazily."""
    common = _FakeCommon()
    monkeypatch.setattr(hypercoast, "_load_hypercoast_common", lambda: common)
    tools = {tool.tool_name: tool for tool in hypercoast_tools(object())}

    result = tools["search_hypercoast_data"].__wrapped__(
        source="emit",
        bbox="-85,34,-83,36",
        temporal="2024-01-01,2024-01-31",
        count=2,
    )

    assert result["success"] is True
    assert result["count"] == 1
    assert result["source"] == "emit"
    assert result["bbox"] == [-85.0, 34.0, -83.0, 36.0]
    assert result["granules"][0]["granule_ur"] == "EMIT_L2A_RFL_001"
    assert common.calls == [
        (
            "search_emit",
            {
                "bbox": [-85.0, 34.0, -83.0, 36.0],
                "temporal": "2024-01-01,2024-01-31",
                "count": 2,
            },
        )
    ]


def test_search_hypercoast_data_can_use_current_extent(monkeypatch) -> None:
    """Verify current QGIS extent can provide the search bbox."""
    common = _FakeCommon()
    monkeypatch.setattr(hypercoast, "_load_hypercoast_common", lambda: common)
    iface = MockQGISIface()
    iface.mapCanvas().setExtent((-90.0, 30.0, -80.0, 40.0))
    tools = {tool.tool_name: tool for tool in hypercoast_tools(iface)}

    result = tools["search_hypercoast_data"].__wrapped__(
        source="pace",
        count=1,
        provider="POCLOUD",
        use_current_extent=True,
    )

    assert result["source"] == "pace"
    assert result["bbox"] == [-90.0, 30.0, -80.0, 40.0]
    assert common.calls[0][0] == "search_pace"
    assert common.calls[0][1]["provider"] == "POCLOUD"


def test_search_hypercoast_data_uses_cloud_cover_filter(monkeypatch) -> None:
    """Verify cloud-cover requests are sent through CMR-aware search."""
    captured: dict[str, Any] = {}

    def _raise_if_loaded() -> None:
        raise AssertionError("HyperCoast helpers should not run for cloud filters")

    def _fake_search_earthaccess_data(**kwargs: Any) -> list[dict[str, Any]]:
        captured.update(kwargs)
        return [
            {
                "id": "emit-clear",
                "cloud_cover": 2,
                "umm": {"GranuleUR": "EMIT_CLEAR"},
            }
        ]

    monkeypatch.setattr(hypercoast, "_load_hypercoast_common", _raise_if_loaded)
    monkeypatch.setattr(
        hypercoast,
        "_search_earthaccess_data",
        _fake_search_earthaccess_data,
    )
    tools = {tool.tool_name: tool for tool in hypercoast_tools(object())}

    result = tools["search_hypercoast_data"].__wrapped__(
        source="emit",
        bbox="-85,34,-83,36",
        count=5,
        cloud_cover_max=10,
    )

    assert result["search_backend"] == "earthaccess"
    assert result["cloud_cover"] == [0.0, 10.0]
    assert result["cloud_cover_min"] == 0.0
    assert result["cloud_cover_max"] == 10.0
    assert result["granules"][0]["cloud_cover"] == 2
    assert captured == {
        "source_key": "emit",
        "bbox": [-85.0, 34.0, -83.0, 36.0],
        "temporal": None,
        "count": 5,
        "short_name": None,
        "provider": None,
        "cloud_cover": (0.0, 10.0),
    }


def test_search_hypercoast_data_cloud_filter_falls_back_to_cmr(monkeypatch) -> None:
    """Verify cloud-cover filters are preserved in direct CMR fallback."""
    captured: dict[str, Any] = {}

    def _fake_search_earthaccess_data(**kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("earthaccess unavailable")

    def _fake_search_cmr_granules(**kwargs: Any) -> list[dict[str, Any]]:
        captured.update(kwargs)
        return [{"id": "pace-cmr", "cloud_cover": 7}]

    monkeypatch.setattr(
        hypercoast,
        "_search_earthaccess_data",
        _fake_search_earthaccess_data,
    )
    monkeypatch.setattr(
        hypercoast,
        "_search_cmr_granules",
        _fake_search_cmr_granules,
    )
    tools = {tool.tool_name: tool for tool in hypercoast_tools(object())}

    result = tools["search_hypercoast_data"].__wrapped__(
        source="pace",
        bbox="-85,34,-83,36",
        count=3,
        provider="POCLOUD",
        max_cloud_cover=15,
    )

    assert result["search_backend"] == "cmr"
    assert result["cloud_cover"] == [0.0, 15.0]
    assert result["granules"][0]["cloud_cover"] == 7
    assert captured == {
        "source_key": "pace",
        "bbox": [-85.0, 34.0, -83.0, 36.0],
        "temporal": None,
        "count": 3,
        "short_name": None,
        "provider": "POCLOUD",
        "cloud_cover": (0.0, 15.0),
    }


def test_pace_short_name_ignores_cloud_cover_detects_l3_products() -> None:
    """Verify the PACE L3 detector recognizes BGC and L3M short names."""
    assert hypercoast._pace_short_name_ignores_cloud_cover("PACE_OCI_L3M_BGC_DAILY_4KM")
    assert hypercoast._pace_short_name_ignores_cloud_cover("PACE_OCI_L3M_AOP_8DAY_4KM")
    assert hypercoast._pace_short_name_ignores_cloud_cover("PACE_OCI_L2_BGC")
    assert not hypercoast._pace_short_name_ignores_cloud_cover("PACE_OCI_L1B_SCI")
    assert not hypercoast._pace_short_name_ignores_cloud_cover(None)


def test_search_hypercoast_data_drops_cloud_cover_for_pace_bgc(monkeypatch) -> None:
    """Verify PACE BGC searches strip cloud_cover and surface a note."""
    captured: dict[str, Any] = {}

    class _FakeCommon:
        def search_pace(self, **kwargs: Any) -> list[dict[str, Any]]:
            captured.update(kwargs)
            return [{"id": "pace-bgc", "umm": {"GranuleUR": "PACE_BGC"}}]

        def search_emit(self, **kwargs: Any) -> list[dict[str, Any]]:
            raise AssertionError("EMIT search should not run for PACE source")

    monkeypatch.setattr(hypercoast, "_load_hypercoast_common", lambda: _FakeCommon())

    def _fail_filter(**kwargs: Any) -> list[dict[str, Any]]:
        raise AssertionError("cloud_cover should be dropped before filter search")

    monkeypatch.setattr(hypercoast, "_search_earthaccess_data", _fail_filter)
    monkeypatch.setattr(hypercoast, "_search_cmr_granules", _fail_filter)

    tools = {tool.tool_name: tool for tool in hypercoast_tools(object())}

    result = tools["search_hypercoast_data"].__wrapped__(
        source="pace",
        bbox="-92,28,-88,31",
        count=5,
        short_name="PACE_OCI_L3M_BGC_DAILY_4KM",
        cloud_cover_min=0,
        cloud_cover_max=0,
    )

    assert result["search_backend"] == "hypercoast"
    assert result["count"] == 1
    assert result["cloud_cover"] is None
    assert result["cloud_cover_ignored"] is True
    assert result["cloud_cover_requested"] is True
    assert any("Ignored cloud_cover" in note for note in result.get("notes", []))
    assert captured["bbox"] == [-92.0, 28.0, -88.0, 31.0]
    assert "cloud_cover" not in captured


def test_search_hypercoast_data_warns_on_degenerate_zero_filter(monkeypatch) -> None:
    """Verify a 0..0 cloud filter with no results surfaces an actionable hint."""

    def _fake_search_earthaccess_data(**kwargs: Any) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(
        hypercoast,
        "_search_earthaccess_data",
        _fake_search_earthaccess_data,
    )
    monkeypatch.setattr(
        hypercoast,
        "_search_cmr_granules",
        lambda **_: [],
    )

    tools = {tool.tool_name: tool for tool in hypercoast_tools(object())}

    result = tools["search_hypercoast_data"].__wrapped__(
        source="emit",
        bbox="-85,34,-83,36",
        count=5,
        cloud_cover_min=0,
        cloud_cover_max=0,
    )

    assert result["count"] == 0
    assert any(
        "both 0" in note or "exactly 0% cloud cover" in note
        for note in result.get("notes", [])
    )


def test_display_hypercoast_footprints_adds_geojson_layer() -> None:
    """Verify HyperCoast granule footprints can be displayed in QGIS."""
    project = MockQGISProject()
    iface = MockQGISIface(project)
    tools = {tool.tool_name: tool for tool in hypercoast_tools(iface, project)}
    granules = [
        {
            "id": "emit-cmr",
            "title": "EMIT_L2A_RFL_001",
            "bbox": ["34 -85 36 -83"],
            "cloud_cover": 2,
        }
    ]

    result = tools["display_hypercoast_footprints"].__wrapped__(
        granules=granules,
        layer_name="EMIT Footprints",
    )

    assert result["success"] is True
    assert result["feature_count"] == 1
    assert project.mapLayersByName("EMIT Footprints")
    payload = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert payload["features"][0]["properties"]["cloud_cover"] == 2
    assert "granule_json" in payload["features"][0]["properties"]


def test_display_hypercoast_footprints_uses_last_search(monkeypatch) -> None:
    """Verify footprint display can use the most recent search results."""
    project = MockQGISProject()
    iface = MockQGISIface(project)

    def _fake_search_earthaccess_data(**kwargs: Any) -> list[dict[str, Any]]:
        return [{"id": "emit-clear", "bbox": ["34 -85 36 -83"]}]

    monkeypatch.setattr(
        hypercoast,
        "_search_earthaccess_data",
        _fake_search_earthaccess_data,
    )
    tools = {tool.tool_name: tool for tool in hypercoast_tools(iface, project)}

    tools["search_hypercoast_data"].__wrapped__(
        source="emit",
        bbox="-85,34,-83,36",
        cloud_cover_max=10,
    )
    result = tools["display_hypercoast_footprints"].__wrapped__(
        layer_name="Latest EMIT Footprints",
    )

    assert result["success"] is True
    assert result["feature_count"] == 1
    assert project.mapLayersByName("Latest EMIT Footprints")


def test_get_selected_hypercoast_footprints_returns_granules() -> None:
    """Verify selected footprint metadata is returned with named fields."""
    project = MockQGISProject()
    iface = MockQGISIface(project)
    layer = MockQGISLayer("PACE BGC Footprints")
    granule = {
        "title": "PACE_OCI_L2_BGC_PACE_OCI.20240305T180543.L2.OC_BGC.V3_1.nc_3.1",
        "bbox": ["28 -92 31 -88"],
        "data_links": [
            "https://obdaac-tea.earthdatacloud.nasa.gov/ob-cumulus-prod-public/PACE_OCI.20240305T180543.L2.OC_BGC.V3_1.nc"
        ],
    }
    layer._selected = [{"granule_json": json.dumps(granule)}]
    project.addMapLayer(layer)
    tools = {tool.tool_name: tool for tool in hypercoast_tools(iface, project)}

    result = tools["get_selected_hypercoast_footprints"].__wrapped__(
        layer_name="PACE BGC Footprints"
    )

    assert result["success"] is True
    assert result["count"] == 1
    assert result["granules"][0]["title"].startswith("PACE_OCI_L2_BGC")
    assert result["granules"][0]["data_links"] == granule["data_links"]


def test_search_hypercoast_data_falls_back_to_earthaccess(monkeypatch) -> None:
    """Verify the known HyperCoast earthaccess helper bug has a fallback."""

    class _BrokenCommon:
        def search_pace(self, **kwargs: Any) -> list[dict[str, Any]]:
            """Raise the HyperCoast helper bug.

            Args:
                **kwargs: Search options.

            Raises:
                UnboundLocalError: Always raised for this test.
            """
            raise UnboundLocalError(
                "cannot access local variable 'earthaccess' where it is not "
                "associated with a value"
            )

    captured: dict[str, Any] = {}

    def _fake_search_earthaccess_data(**kwargs: Any) -> list[dict[str, Any]]:
        captured.update(kwargs)
        return [{"id": "pace-fallback", "umm": {"GranuleUR": "PACE_FALLBACK"}}]

    monkeypatch.setattr(hypercoast, "_load_hypercoast_common", lambda: _BrokenCommon())
    monkeypatch.setattr(
        hypercoast,
        "_search_earthaccess_data",
        _fake_search_earthaccess_data,
    )
    tools = {tool.tool_name: tool for tool in hypercoast_tools(object())}

    result = tools["search_hypercoast_data"].__wrapped__(
        source="pace",
        bbox="-85,34,-83,36",
        count=5,
        provider="POCLOUD",
    )

    assert result["success"] is True
    assert result["search_backend"] == "earthaccess"
    assert result["granules"][0]["granule_ur"] == "PACE_FALLBACK"
    assert captured == {
        "source_key": "pace",
        "bbox": [-85.0, 34.0, -83.0, 36.0],
        "temporal": None,
        "count": 5,
        "short_name": None,
        "provider": "POCLOUD",
    }


def test_search_hypercoast_data_falls_back_to_cmr(monkeypatch) -> None:
    """Verify search can avoid earthaccess when botocore is incompatible."""

    class _BrokenCommon:
        def search_emit(self, **kwargs: Any) -> list[dict[str, Any]]:
            """Raise a botocore compatibility import error.

            Args:
                **kwargs: Search options.

            Raises:
                ImportError: Always raised for this test.
            """
            raise ImportError("cannot import name 'EC' from 'botocore.compat'")

    captured: dict[str, Any] = {}

    def _fake_search_earthaccess_data(**kwargs: Any) -> list[dict[str, Any]]:
        raise ImportError("cannot import name 'EC' from 'botocore.compat'")

    def _fake_search_cmr_granules(**kwargs: Any) -> list[dict[str, Any]]:
        captured.update(kwargs)
        return [
            {
                "id": "emit-cmr",
                "title": "EMIT_L2A_RFL_001",
                "data_links": ["https://example.com/emit.nc"],
            }
        ]

    monkeypatch.setattr(hypercoast, "_load_hypercoast_common", lambda: _BrokenCommon())
    monkeypatch.setattr(
        hypercoast,
        "_search_earthaccess_data",
        _fake_search_earthaccess_data,
    )
    monkeypatch.setattr(
        hypercoast,
        "_search_cmr_granules",
        _fake_search_cmr_granules,
    )
    tools = {tool.tool_name: tool for tool in hypercoast_tools(object())}

    result = tools["search_hypercoast_data"].__wrapped__(
        source="emit",
        bbox="-85,34,-83,36",
        count=7,
    )

    assert result["success"] is True
    assert result["search_backend"] == "cmr"
    assert result["granules"][0]["id"] == "emit-cmr"
    assert result["granules"][0]["data_links"] == ["https://example.com/emit.nc"]
    assert captured == {
        "source_key": "emit",
        "bbox": [-85.0, 34.0, -83.0, 36.0],
        "temporal": None,
        "count": 7,
        "short_name": None,
        "provider": None,
    }


def test_download_hypercoast_data_dispatches_to_pace(monkeypatch, tmp_path) -> None:
    """Verify download delegates to the selected HyperCoast helper."""
    common = _FakeCommon()
    monkeypatch.setattr(hypercoast, "_load_hypercoast_common", lambda: common)
    tools = {tool.tool_name: tool for tool in hypercoast_tools(object())}

    result = tools["download_hypercoast_data"].__wrapped__(
        granules=[{"id": "pace-1"}],
        source="pace",
        out_dir=str(tmp_path),
        provider="POCLOUD",
        threads=3,
    )

    assert result["success"] is True
    assert result["paths"] == [str(tmp_path / "pace.nc")]
    assert common.calls == [
        (
            "download_pace",
            {
                "granules": [{"id": "pace-1"}],
                "out_dir": str(tmp_path),
                "provider": "POCLOUD",
                "threads": 3,
            },
        )
    ]


def test_download_hypercoast_data_uses_direct_links(monkeypatch, tmp_path) -> None:
    """Verify CMR search-result links can be downloaded without earthaccess."""
    captured: dict[str, Any] = {}

    def _fake_download_data_links(granules, *, out_dir, max_links_per_granule):
        captured["granules"] = granules
        captured["out_dir"] = out_dir
        captured["max_links_per_granule"] = max_links_per_granule
        return [os.path.join(out_dir, "emit.nc")]

    monkeypatch.setattr(hypercoast, "_download_data_links", _fake_download_data_links)
    tools = {tool.tool_name: tool for tool in hypercoast_tools(object())}

    granules = [{"id": "emit-cmr", "data_links": ["https://example.com/emit.nc"]}]
    result = tools["download_hypercoast_data"].__wrapped__(
        granules=granules,
        source="emit",
        out_dir=str(tmp_path),
        max_links_per_granule=2,
    )

    assert result["success"] is True
    assert result["download_backend"] == "direct_links"
    assert result["paths"] == [str(tmp_path / "emit.nc")]
    assert captured == {
        "granules": granules,
        "out_dir": str(tmp_path),
        "max_links_per_granule": 2,
    }


def test_download_hypercoast_data_uses_related_urls(monkeypatch, tmp_path) -> None:
    """Verify selected footprint metadata can provide UMM related URLs."""
    monkeypatch.setattr(hypercoast, "_download_opener", lambda: _FakeOpener(b"data"))
    granules = [
        {
            "umm": {
                "RelatedUrls": [
                    {
                        "URL": "https://oceandata.sci.gsfc.nasa.gov/browse_images/PACE_OCI.20240305T180543.L2.OC_BGC.V3_1.nc.png"
                    },
                    {
                        "URL": "https://example.com/PACE_OCI.20240305T180543.L2.OC_BGC.nc"
                    },
                ]
            }
        }
    ]
    tools = {tool.tool_name: tool for tool in hypercoast_tools(object())}

    result = tools["download_hypercoast_data"].__wrapped__(
        granules=granules,
        source="pace",
        out_dir=str(tmp_path),
    )

    assert result["success"] is True
    assert result["download_backend"] == "direct_links"
    assert result["paths"] == [str(tmp_path / "PACE_OCI.20240305T180543.L2.OC_BGC.nc")]


def test_preview_hypercoast_dataset_returns_metadata(monkeypatch, tmp_path) -> None:
    """Verify preview loads and summarizes local hyperspectral metadata."""
    source = tmp_path / "emit.nc"
    source.write_text("fake", encoding="utf-8")
    _FakeDataset.created_data_types.clear()
    monkeypatch.setattr(
        hypercoast,
        "_load_hyperspectral_runtime",
        lambda plugin=None: (_FakeDataset, None),
    )
    tools = {tool.tool_name: tool for tool in hypercoast_tools(object())}

    result = tools["preview_hypercoast_dataset"].__wrapped__(
        str(source),
        variable_name="reflectance",
    )

    assert result["success"] is True
    assert result["data_type"] == "EMIT"
    assert result["selected_variable"] == "reflectance"
    assert result["band_count"] == 3
    assert result["wavelength_min"] == 450.0
    assert result["variables"] == ["reflectance"]
    assert _FakeDataset.created_data_types == ["auto"]


def test_preview_hypercoast_dataset_normalizes_data_type(monkeypatch, tmp_path) -> None:
    """Verify lower-case data type input reaches HyperCoast as canonical PACE."""
    source = tmp_path / "PACE.nc"
    source.write_text("fake", encoding="utf-8")
    _FakeDataset.created_data_types.clear()
    monkeypatch.setattr(
        hypercoast,
        "_load_hyperspectral_runtime",
        lambda plugin=None: (_FakeDataset, None),
    )
    tools = {tool.tool_name: tool for tool in hypercoast_tools(object())}

    result = tools["preview_hypercoast_dataset"].__wrapped__(
        str(source),
        data_type="pace",
    )

    assert result["success"] is True
    assert _FakeDataset.created_data_types == ["PACE"]


def test_patch_hypercoast_top_level_exports_adds_pace_helpers(monkeypatch) -> None:
    """Verify missing top-level HyperCoast PACE helpers are patched."""
    fake_hypercoast = types.ModuleType("hypercoast")
    fake_pace = types.ModuleType("hypercoast.pace")
    fake_pace.read_pace = lambda path: path
    fake_pace.pace_to_image = lambda dataset, **kwargs: dataset

    monkeypatch.setitem(sys.modules, "hypercoast", fake_hypercoast)
    monkeypatch.setitem(sys.modules, "hypercoast.pace", fake_pace)

    hypercoast._patch_hypercoast_top_level_exports()

    assert fake_hypercoast.read_pace("x.nc") == "x.nc"
    assert fake_hypercoast.pace_to_image("dataset") == "dataset"


def test_patch_hypercoast_top_level_exports_patches_runtime_alias(
    monkeypatch,
) -> None:
    """Verify the installed QGIS plugin's cached HyperCoast module is patched."""
    fake_hypercoast = types.ModuleType("hypercoast_external")
    fake_emit = types.ModuleType("hypercoast_external.emit")
    fake_emit.read_emit = lambda path: f"read-{path}"
    fake_emit.emit_to_image = lambda dataset, **kwargs: f"image-{dataset}"
    provider_module = types.ModuleType(
        "_geoagent_hypercoast_qgis.hyperspectral_provider"
    )
    provider_module.hypercoast = fake_hypercoast
    Dataset = type("HyperspectralDataset", (), {})
    Dataset.__module__ = provider_module.__name__

    monkeypatch.setitem(sys.modules, "hypercoast_external", fake_hypercoast)
    monkeypatch.setitem(sys.modules, "hypercoast_external.emit", fake_emit)
    monkeypatch.setitem(sys.modules, provider_module.__name__, provider_module)
    monkeypatch.delitem(sys.modules, "hypercoast", raising=False)

    hypercoast._patch_hypercoast_top_level_exports(Dataset)

    assert fake_hypercoast.read_emit("x.nc") == "read-x.nc"
    assert fake_hypercoast.emit_to_image("dataset") == "image-dataset"


def test_load_hyperspectral_runtime_supports_installed_plugin_name(
    monkeypatch,
    tmp_path,
) -> None:
    """Verify runtime imports work when QGIS installs the plugin as hypercoast."""
    plugin_dir = tmp_path / "hypercoast"
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").write_text("", encoding="utf-8")
    (plugin_dir / "cache_manager.py").write_text(
        "def create_generated_raster_path(layer_name, suffix, project=None):\n"
        "    return f'/tmp/{layer_name}_{suffix}.tif'\n",
        encoding="utf-8",
    )
    (plugin_dir / "hyperspectral_provider.py").write_text(
        "class HyperspectralDataset:\n" "    pass\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(hypercoast, "_resolve_hypercoast_plugin", lambda plugin: None)
    monkeypatch.setattr(
        hypercoast,
        "_candidate_hypercoast_plugin_dirs",
        lambda plugin=None: [str(plugin_dir)],
    )
    monkeypatch.delitem(sys.modules, "hypercoast_qgis.cache_manager", raising=False)
    monkeypatch.delitem(
        sys.modules,
        "hypercoast_qgis.hyperspectral_provider",
        raising=False,
    )

    dataset_cls, path_factory = hypercoast._load_hyperspectral_runtime()

    assert dataset_cls.__name__ == "HyperspectralDataset"
    assert path_factory("Layer", "rgb") == "/tmp/Layer_rgb.tif"


def test_load_hypercoast_rgb_registers_layer(monkeypatch, tmp_path) -> None:
    """Verify RGB loading adds a raster and registers HyperCoast metadata."""
    source = tmp_path / "emit.nc"
    output = tmp_path / "rgb.tif"
    source.write_text("fake", encoding="utf-8")
    plugin = _FakePlugin()
    project = MockQGISProject()
    iface = MockQGISIface(project)
    layer = _FakeLayer("EMIT RGB", str(output))

    monkeypatch.setattr(
        hypercoast,
        "_load_hyperspectral_runtime",
        lambda plugin=None: (
            _FakeDataset,
            lambda name, suffix, project=None: str(output),
        ),
    )
    monkeypatch.setattr(
        hypercoast,
        "_add_qgis_raster_layer",
        lambda iface, project, path, layer_name: layer,
    )
    tools = {
        tool.tool_name: tool for tool in hypercoast_tools(iface, project, plugin=plugin)
    }

    result = tools["load_hypercoast_rgb"].__wrapped__(
        str(source),
        red=850,
        green=650,
        blue=550,
        layer_name="EMIT RGB",
        variable_name="reflectance",
    )

    assert result["success"] is True
    assert result["layer_name"] == "EMIT RGB"
    assert result["output_path"] == str(output)
    assert result["rgb_wavelengths"] == [850.0, 650.0, 550.0]
    assert layer.properties["hypercoast/source_path"] == str(source)
    assert layer.properties["hypercoast/selected_variable"] == "reflectance"
    assert plugin.registered["EMIT RGB"]["filepath"] == str(source)
    assert plugin.registered["EMIT RGB"]["rgb_wavelengths"] == [850.0, 650.0, 550.0]


def test_load_hypercoast_variable_registers_selected_variable(
    monkeypatch,
    tmp_path,
) -> None:
    """Verify single-variable visualization supports PACE BGC products."""
    source = tmp_path / "PACE_BGC.nc"
    output = tmp_path / "chlor_a.tif"
    source.write_text("fake", encoding="utf-8")
    plugin = _FakePlugin()
    project = MockQGISProject()
    iface = MockQGISIface(project)
    layer = _FakeLayer("PACE chlor_a", str(output))

    monkeypatch.setattr(
        hypercoast,
        "_load_hyperspectral_runtime",
        lambda plugin=None: (
            _FakeDataset,
            lambda name, suffix, project=None: str(output),
        ),
    )
    monkeypatch.setattr(
        hypercoast,
        "_add_qgis_raster_layer",
        lambda iface, project, path, layer_name: layer,
    )
    tools = {
        tool.tool_name: tool for tool in hypercoast_tools(iface, project, plugin=plugin)
    }

    result = tools["load_hypercoast_variable"].__wrapped__(
        str(source),
        data_type="pace",
        variable_name="chlor_a",
        layer_name="PACE chlor_a",
    )

    assert result["success"] is True
    assert result["layer_name"] == "PACE chlor_a"
    assert result["selected_variable"] == "chlor_a"
    assert layer.properties["hypercoast/source_path"] == str(source)
    assert layer.properties["hypercoast/selected_variable"] == "chlor_a"
    assert "hypercoast/rgb_wavelengths" not in layer.properties
    assert plugin.registered["PACE chlor_a"]["filepath"] == str(source)
    assert plugin.registered["PACE chlor_a"]["selected_variable"] == "chlor_a"


def test_open_hypercoast_panel_uses_plugin_method() -> None:
    """Verify panel opening delegates to the supplied plugin instance."""
    plugin = _FakePlugin()
    tools = {tool.tool_name: tool for tool in hypercoast_tools(object(), plugin=plugin)}

    result = tools["open_hypercoast_panel"].__wrapped__("load")

    assert result["success"] is True
    assert result["method"] == "show_load_dialog"
    assert plugin.opened == ["load"]


def test_for_hypercoast_registers_hypercoast_and_qgis_tools() -> None:
    """Verify the HyperCoast factory combines HyperCoast and QGIS tools."""
    agent = for_hypercoast(
        MockQGISIface(),
        MockQGISProject(),
        model=_MockModel(),
    )
    names = set(agent.strands_agent.tool_names)

    assert "list_hypercoast_data_types" in names
    assert "search_hypercoast_data" in names
    assert "load_hypercoast_rgb" in names
    assert "list_project_layers" in names
    assert "add_raster_layer" not in names
    assert agent.context.metadata["integration"] == "hypercoast"
