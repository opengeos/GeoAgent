# Tools

Interactive adapters live under **`geoagent.tools`**:

- `leafmap_tools`, `anymap_tools`, `qgis_tools` — bound to live map / QGIS instances.
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
