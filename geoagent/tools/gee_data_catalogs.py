"""Tool adapters for the QGIS GEE Data Catalogs plugin.

The module is import-safe outside QGIS and outside the plugin. Tool bodies
resolve ``gee_data_catalogs`` lazily so GeoAgent can be imported in ordinary
Python environments.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from geoagent.core.decorators import geo_tool
from geoagent.tools._qt_marshal import run_on_qt_gui_thread


def _on_gui(fn: Any) -> Any:
    """Run ``fn`` on the Qt GUI thread when QGIS is available."""
    return run_on_qt_gui_thread(fn)


def _project_id_from_settings() -> str | None:
    """Return the configured Earth Engine project id, if any."""
    try:
        from qgis.PyQt.QtCore import QSettings  # type: ignore[import-not-found]

        project_id = QSettings().value("GeeDataCatalogs/ee_project", "", type=str)
        project_id = project_id.strip() if project_id else ""
        if project_id:
            return project_id
    except Exception:
        pass
    return os.environ.get("EE_PROJECT_ID") or None


def _compact_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    """Return a concise, JSON-friendly catalog record."""
    keys = [
        "id",
        "name",
        "title",
        "type",
        "category",
        "provider",
        "start_date",
        "end_date",
        "source",
        "url",
        "license",
        "bigquery_table",
    ]
    out = {key: dataset.get(key) for key in keys if dataset.get(key)}
    description = str(dataset.get("description", "")).strip()
    if description:
        out["description"] = description[:500]
    keywords = dataset.get("keywords")
    if keywords:
        out["keywords"] = keywords[:15] if isinstance(keywords, list) else keywords
    return out


def _parse_list(value: str | list[str] | None) -> list[str] | None:
    """Parse comma-separated UI-style values into a list."""
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_bbox(
    bbox: str | list[float] | tuple[float, ...] | None,
) -> list[float] | None:
    """Parse west,south,east,north bounds."""
    if bbox is None or bbox == "":
        return None
    values = list(bbox) if isinstance(bbox, (list, tuple)) else str(bbox).split(",")
    if len(values) != 4:
        raise ValueError("bbox must contain west,south,east,north")
    west, south, east, north = [float(value) for value in values]
    if west >= east or south >= north:
        raise ValueError("bbox coordinates must satisfy west < east and south < north")
    return [west, south, east, north]


def _build_vis_params(
    *,
    bands: str | list[str] | None = None,
    min_value: float | int | None = None,
    max_value: float | int | None = None,
    palette: str | list[str] | None = None,
    for_feature_collection: bool = False,
) -> dict[str, Any]:
    """Build Earth Engine visualization parameters from scalar inputs."""
    vis_params: dict[str, Any] = {}
    parsed_bands = _parse_list(bands)
    if parsed_bands and not for_feature_collection:
        vis_params["bands"] = parsed_bands
    if min_value is not None and not for_feature_collection:
        vis_params["min"] = float(min_value)
    if max_value is not None and not for_feature_collection:
        vis_params["max"] = float(max_value)
    parsed_palette = _parse_list(palette)
    if parsed_palette:
        if for_feature_collection:
            vis_params["color"] = parsed_palette[0]
        else:
            vis_params["palette"] = parsed_palette
    return vis_params


def gee_data_catalogs_tools(
    iface: Any,
    project: Optional[Any] = None,
    plugin: Optional[Any] = None,
) -> list[Any]:
    """Return GeoAgent tools for the QGIS GEE Data Catalogs plugin."""
    if iface is None:
        return []

    def _project() -> Any:
        """Return the bound or singleton QGIS project."""
        if project is not None:
            return project
        from qgis.core import QgsProject  # type: ignore[import-not-found]

        return QgsProject.instance()

    def _ensure_plugin_catalog_dock() -> Any | None:
        """Return a live plugin catalog dock when a plugin instance was supplied."""
        if plugin is None:
            return None

        def _run() -> Any | None:
            dock = getattr(plugin, "_catalog_dock", None)
            if dock is None and hasattr(plugin, "_create_catalog_dock"):
                plugin._create_catalog_dock()
                dock = getattr(plugin, "_catalog_dock", None)
            if dock is not None:
                dock.show()
                dock.raise_()
            return dock

        return _on_gui(_run)

    @geo_tool(
        category="gee_data_catalogs",
        name="search_gee_datasets",
        available_in=("full", "fast"),
    )
    def search_gee_datasets(
        query: str = "",
        category: Optional[str] = None,
        data_type: Optional[str] = None,
        source: Optional[str] = None,
        max_results: int = 20,
        include_community: bool = True,
    ) -> dict[str, Any]:
        """Search official and community Google Earth Engine data catalogs."""
        from gee_data_catalogs.core.catalog_data import search_datasets

        results = search_datasets(
            query=query,
            category=category,
            data_type=data_type,
            source=source,
            include_community=include_community,
        )
        count = max(1, int(max_results))
        return {
            "count": len(results),
            "shown": min(len(results), count),
            "datasets": [_compact_dataset(item) for item in results[:count]],
        }

    @geo_tool(
        category="gee_data_catalogs",
        name="get_gee_dataset_info",
        available_in=("full", "fast"),
    )
    def get_gee_dataset_info(
        asset_id: str,
        include_community: bool = True,
    ) -> dict[str, Any]:
        """Return catalog metadata for a Google Earth Engine asset id."""
        from gee_data_catalogs.core.catalog_data import get_dataset_info

        dataset = get_dataset_info(asset_id, include_community=include_community)
        if dataset is None:
            return {"asset_id": asset_id, "found": False}
        out = _compact_dataset(dataset)
        out["found"] = True
        return out

    @geo_tool(
        category="gee_data_catalogs",
        name="summarize_gee_catalog",
        available_in=("full", "fast"),
    )
    def summarize_gee_catalog(include_community: bool = True) -> dict[str, Any]:
        """Summarize dataset counts by catalog category."""
        from gee_data_catalogs.core.catalog_data import get_catalog_data

        catalog = get_catalog_data(include_community=include_community)
        categories = []
        total = 0
        for category, payload in catalog.items():
            datasets = payload.get("datasets", [])
            total += len(datasets)
            categories.append({"category": category, "count": len(datasets)})
        return {"total_count": total, "categories": categories}

    @geo_tool(
        category="gee_data_catalogs",
        name="initialize_earth_engine",
        requires_confirmation=True,
    )
    def initialize_earth_engine(project_id: Optional[str] = None) -> dict[str, Any]:
        """Initialize Earth Engine using the plugin utility and saved project id."""
        from gee_data_catalogs.core.ee_utils import initialize_ee, is_ee_initialized

        project_to_use = project_id or _project_id_from_settings()
        initialize_ee(project=project_to_use)
        return {
            "success": is_ee_initialized(),
            "project_id": project_to_use,
        }

    @geo_tool(
        category="gee_data_catalogs",
        name="load_gee_dataset",
        requires_confirmation=True,
        long_running=True,
    )
    def load_gee_dataset(
        asset_id: str,
        layer_name: Optional[str] = None,
        asset_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        bbox: Optional[str] = None,
        cloud_cover: Optional[int] = None,
        cloud_property: Optional[str] = None,
        reducer: str = "mosaic",
        bands: Optional[str] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        palette: Optional[str] = None,
    ) -> dict[str, Any]:
        """Load a GEE Image, ImageCollection, or FeatureCollection into QGIS.

        Args:
            asset_id: Earth Engine asset id.
            layer_name: Optional QGIS layer name.
            asset_type: Optional explicit type: Image, ImageCollection, or
                FeatureCollection. When omitted, the plugin detects the type.
            start_date: Optional ImageCollection start date in YYYY-MM-DD.
            end_date: Optional ImageCollection end date in YYYY-MM-DD.
            bbox: Optional WGS84 west,south,east,north filter.
            cloud_cover: Optional maximum cloud-cover value.
            cloud_property: Optional cloud-cover property name.
            reducer: ImageCollection reducer: mosaic, median, mean, min, max, first.
            bands: Optional comma-separated visualization bands.
            min_value: Optional visualization minimum.
            max_value: Optional visualization maximum.
            palette: Optional comma-separated colors or palette entries.
        """

        def _run() -> dict[str, Any]:
            import ee

            from gee_data_catalogs.core.ee_utils import (
                add_ee_layer,
                detect_asset_type,
                filter_image_collection,
                initialize_ee,
                is_ee_initialized,
            )

            if not is_ee_initialized():
                initialize_ee(project=_project_id_from_settings())

            resolved_type = asset_type or detect_asset_type(asset_id)
            name = layer_name or asset_id.split("/")[-1]
            parsed_bbox = _parse_bbox(bbox)

            if resolved_type == "ImageCollection":
                collection = ee.ImageCollection(asset_id)
                if start_date or end_date or parsed_bbox or cloud_cover is not None:
                    collection = filter_image_collection(
                        collection,
                        start_date=start_date,
                        end_date=end_date,
                        bbox=parsed_bbox,
                        cloud_cover=cloud_cover,
                        cloud_property=cloud_property or "CLOUDY_PIXEL_PERCENTAGE",
                    )
                method = reducer.lower().strip()
                if method not in {"mosaic", "median", "mean", "min", "max", "first"}:
                    method = "mosaic"
                ee_object = getattr(collection, method)()
                vis_params = _build_vis_params(
                    bands=bands,
                    min_value=min_value,
                    max_value=max_value,
                    palette=palette,
                )
            elif resolved_type == "FeatureCollection":
                ee_object = ee.FeatureCollection(asset_id)
                vis_params = _build_vis_params(
                    palette=palette,
                    for_feature_collection=True,
                )
            else:
                ee_object = ee.Image(asset_id)
                resolved_type = "Image"
                vis_params = _build_vis_params(
                    bands=bands,
                    min_value=min_value,
                    max_value=max_value,
                    palette=palette,
                )

            layer = add_ee_layer(ee_object, vis_params, name[:50])
            try:
                iface.mapCanvas().refresh()
            except Exception:
                pass
            return {
                "success": True,
                "asset_id": asset_id,
                "asset_type": resolved_type,
                "layer_name": name[:50],
                "vis_params": vis_params,
            }

        return _on_gui(_run)

    @geo_tool(
        category="gee_data_catalogs",
        name="open_gee_catalog_panel",
        available_in=("full", "fast"),
    )
    def open_gee_catalog_panel(tab: str = "Search") -> dict[str, Any]:
        """Open the GEE Data Catalogs panel and optionally select a tab."""
        dock = _ensure_plugin_catalog_dock()
        if dock is None:
            return {"success": False, "error": "Plugin instance is not available."}

        def _run() -> dict[str, Any]:
            names = []
            tab_widget = getattr(dock, "tab_widget", None)
            if tab_widget is not None:
                for i in range(tab_widget.count()):
                    names.append(tab_widget.tabText(i))
                    if tab_widget.tabText(i).lower() == tab.lower():
                        tab_widget.setCurrentIndex(i)
            return {"success": True, "tabs": names}

        return _on_gui(_run)

    @geo_tool(
        category="gee_data_catalogs",
        name="configure_gee_dataset_load",
        available_in=("full", "fast"),
    )
    def configure_gee_dataset_load(
        asset_id: str,
        layer_name: Optional[str] = None,
        bands: Optional[str] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        palette: Optional[str] = None,
    ) -> dict[str, Any]:
        """Populate the plugin Load tab for a dataset without loading it."""
        dock = _ensure_plugin_catalog_dock()
        if dock is None:
            return {"success": False, "error": "Plugin instance is not available."}

        def _run() -> dict[str, Any]:
            dock.dataset_id_input.setText(asset_id)
            dock.layer_name_input.setText(layer_name or asset_id.split("/")[-1])
            if bands is not None:
                dock.bands_input.setText(bands)
            if min_value is not None:
                dock.vis_min_input.setText(str(min_value))
            if max_value is not None:
                dock.vis_max_input.setText(str(max_value))
            if palette is not None:
                dock.palette_input.setText(palette)
            dock.tab_widget.setCurrentIndex(3)
            return {"success": True, "asset_id": asset_id}

        return _on_gui(_run)

    return [
        search_gee_datasets,
        get_gee_dataset_info,
        summarize_gee_catalog,
        initialize_earth_engine,
        load_gee_dataset,
        open_gee_catalog_panel,
        configure_gee_dataset_load,
    ]


__all__ = ["gee_data_catalogs_tools"]
