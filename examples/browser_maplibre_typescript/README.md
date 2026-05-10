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

Try prompts such as:

```text
Add a red marker for Knoxville and zoom to it.
```

```text
Add an OpenStreetMap tile layer and list the layers.
```

```text
Change the basemap to dark, then get the current map state.
```

## Scripts

```bash
npm run dev
npm run build
npm run typecheck
```

`add_vector_data` expects a GeoJSON URL in this example because the Python tool
does not provide enough metadata to render arbitrary vector tile sources.
