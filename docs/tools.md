# Tools

Interactive adapters live under **`geoagent.tools`**:

- `leafmap_tools`, `anymap_tools`, `qgis_tools`, `nasa_opera_tools` — bound to live map / QGIS instances.
- Optional stubs: `stac`, `geoai`, `earthengine`, `nasa_earthdata` (expand in future releases).

Use **`@geo_tool`** ([`geoagent.core.decorators`](decorators.md)) so tools register Strands-compatible metadata for safety hooks.

## leafmap and anymap

`for_leafmap(m)` and `for_anymap(m)` expose a shared map-control surface:

- Inspect state and layers: `get_map_state`, `list_layers`.
- Navigate: `set_center`, `fly_to`, `set_zoom`, `zoom_in`, `zoom_out`, `zoom_to_bounds`, `zoom_to_layer`.
- Manage layers: `add_layer`, `remove_layer`, `clear_layers`, `set_layer_visibility`, `set_layer_opacity`.
- Add data: `add_vector_data`, `add_geojson_data`, `add_raster_data`, `add_cog_layer`, `add_stac_layer`, `add_xyz_tile_layer`, `add_pmtiles_layer`, `add_marker`.
- Change basemaps and export maps: `change_basemap`, `save_map`.

Layer lookup accepts exact names or a unique case-insensitive substring for operations such as `remove_layer`, `zoom_to_layer`, `set_layer_visibility`, and `set_layer_opacity`.

## QGIS

`for_qgis(iface, project=None)` exposes QGIS-safe tools that execute through the Qt GUI-thread marshaller:

- Inspect project and layer state: `list_project_layers`, `get_active_layer`, `get_project_state`, `get_layer_summary`, `inspect_layer_fields`, `get_selected_features`.
- Navigate: `zoom_in`, `zoom_out`, `zoom_to_layer`, `zoom_to_extent`, `zoom_to_selected`, `set_center`, `set_scale`, `refresh_canvas`.
- Manage layers and data: `add_vector_layer`, `add_raster_layer`, `add_xyz_tile_layer`, `remove_layer`, `set_layer_visibility`, `set_layer_opacity`.
- Select and process: `select_features_by_expression`, `clear_selection`, `run_processing_algorithm`.
- Open UI / save: `open_attribute_table`, `save_project`.

Destructive or persistent actions such as `remove_layer`, `clear_layers`, `save_map`, `save_project`, and `run_processing_algorithm` require confirmation through the GeoAgent safety hook.

## NASA OPERA

`for_nasa_opera(iface, project=None)` exposes native GeoAgent tools for NASA OPERA plugin workflows:

- Inspect OPERA products: `get_available_datasets`, `get_dataset_info`.
- Search Earthdata granules: `search_opera_data`.
- Display results in QGIS: `display_footprints`, `display_raster`, `create_mosaic`.

This integration replaces the plugin-local `nasa_opera.ai.tools` registry instead of wrapping it.

Example QGIS-console workflow. This direct-tool path avoids LLM/provider
initialization and reports progress in QGIS's message bar and Log Messages
panel:

```python
from geoagent.tools.nasa_opera import submit_nasa_opera_search_task

task = submit_nasa_opera_search_task(
    iface,
    dataset="OPERA_L3_DSWX-HLS_V1",
    bbox="-95.5,29.5,-95.0,30.0",
    start_date="2024-01-01",
    end_date="2024-01-31",
    max_results=5,
    display_footprints=True,
)
```

Natural-language OPERA chat is intentionally disabled inside QGIS for now.
Use direct tools or `submit_nasa_opera_search_task(...)` so QGIS task/thread
ownership remains explicit.
