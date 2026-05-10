# Browser MapLibre Example

This example is a small end-to-end browser client for
`geoagent browser`. It creates a MapLibre map, connects to the GeoAgent
WebSocket backend, sends chat prompts, and executes browser map commands sent
by the agent.

For a buildable TypeScript/Vite version, see
`examples/browser_maplibre_typescript/`.

## Run

From the repository root, install the browser backend and log in with the
default ChatGPT/Codex OAuth provider:

```bash
python -m pip install -e ".[browser]"
geoagent codex login
geoagent browser --host 127.0.0.1 --port 8765 --model gpt-5.5
```

In a second terminal, serve this directory:

```bash
cd examples/browser_maplibre
python -m http.server 8000
```

Open <http://127.0.0.1:8000>, click **Connect**, then try prompts such as:

```text
Add a red marker for Knoxville and zoom to it.
```

```text
Add an OpenStreetMap tile layer and list the layers.
```

```text
Change the basemap to dark, then get the current map state.
```

Destructive tools such as `remove_layer` and `clear_layers` are denied by the
backend's default confirmation policy. This is intentional for browser
sessions.

For trusted local sessions where the agent should be able to remove or clear
browser map layers, start the backend with:

```bash
geoagent browser --host 127.0.0.1 --port 8765 --model gpt-5.5 --auto-approve-browser-tools
```

To allow PyQGIS-style fallback code execution for local browser sessions, start
the backend with:

```bash
geoagent browser --host 127.0.0.1 --port 8765 --model gpt-5.5 --allow-browser-code
```

This exposes `run_maplibre_script`, which lets the agent execute generated
MapLibre JavaScript in the page when no dedicated browser map tool fits.
Use both flags together if you want layer removal and generated JavaScript.

## Protocol

The page handles `map_command` messages from `/geoagent/ws` and returns
`map_command_result` messages. The supported command names match
`geoagent.tools.browser_maplibre`:

- `list_layers`
- `get_map_state`
- `set_center`
- `fly_to`
- `set_zoom`
- `zoom_to_bounds`
- `change_basemap`
- `add_marker`
- `add_geojson_data`
- `add_vector_data`
- `add_xyz_tile_layer`
- `set_layer_visibility`
- `set_layer_opacity`
- `query_rendered_features`
- `screenshot_map`
- `remove_layer`
- `clear_layers`
- `run_maplibre_script` when the backend is started with `--allow-browser-code`

`add_vector_data` expects a GeoJSON URL in this example because the Python tool
does not provide enough metadata to render arbitrary vector tile sources.
