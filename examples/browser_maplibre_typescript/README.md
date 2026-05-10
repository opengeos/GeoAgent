# Browser MapLibre TypeScript Example

This example is the TypeScript/Vite version of the browser MapLibre client for
`geoagent browser`. It connects to `/geoagent/ws`, sends chat prompts, executes
browser map commands against a live MapLibre map, and returns command results to
the Python backend.

## Run

Start the GeoAgent browser backend from the repository root:

```bash
python -m pip install -e ".[browser]"
geoagent codex login
geoagent browser --host 127.0.0.1 --port 8765 --model gpt-5.5
```

In a second terminal, run the TypeScript client:

```bash
cd examples/browser_maplibre_typescript
npm ci
npm run dev
```

Open the Vite URL, usually <http://127.0.0.1:5173>, then connect to:

```text
ws://127.0.0.1:8765/geoagent/ws
```

## Prompt Examples

Basic map navigation:

```text
Add a red marker for Knoxville and zoom to it.
```

```text
Fly to Seattle at zoom level 11.
```

```text
Zoom to the bounds west -84.1, south 35.8, east -83.7, north 36.1.
```

Basemaps and layers:

```text
Change the basemap to dark, then get the current map state.
```

```text
Add an OpenStreetMap tile layer named OpenStreetMap and list the layers.
```

```text
Add the GeoJSON URL https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json as US counties.
```

```text
Hide the US counties layer, then show it again.
```

```text
Set the US counties layer opacity to 0.4.
```

Queries and screenshots:

```text
List all user-added layers.
```

```text
What features are visible at the center of the current map?
```

```text
Take a screenshot of the current map.
```

Layer removal, when the backend was started with `--auto-approve-browser-tools`:

```text
Remove the OpenStreetMap layer.
```

```text
Clear all user-added layers.
```

Generated MapLibre JavaScript, when the backend was started with
`--allow-browser-code`:

```text
Tilt the map to pitch 75 and rotate it slightly.
```

```text
Add a scale control to the lower-left corner.
```

```text
Draw a translucent circle around Knoxville with about a 25 kilometer radius.
```

To allow PyQGIS-style fallback code execution for local browser sessions, start
the backend with:

```bash
geoagent browser --host 127.0.0.1 --port 8765 --model gpt-5.5 --allow-browser-code
```

This exposes `run_maplibre_script`, which lets the agent execute generated
MapLibre JavaScript in the page when no dedicated browser map tool fits.
For trusted local sessions where the agent should be able to remove or clear
browser map layers, add `--auto-approve-browser-tools`. Use both flags together
if you want layer removal and generated JavaScript.

## Scripts

```bash
npm run dev
npm run build
npm run typecheck
```

`add_vector_data` expects a GeoJSON URL in this example because the Python tool
does not provide enough metadata to render arbitrary vector tile sources.
