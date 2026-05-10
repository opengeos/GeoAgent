# Examples

Runnable Jupyter notebooks live under **`docs/examples/`**:

- **`docs/examples/intro.ipynb`** — GeoAgent + Anthropic (no map).
- **`docs/examples/live_mapping.ipynb`** — leafmap MapLibre + Claude; camera comes from **`m.view_state`** (see `get_map_state` tool).
- **`docs/examples/qgis_agent.ipynb`** — QGIS-oriented tools using mock `iface` in Jupyter; snippet for real QGIS included.
- **`docs/examples/stac_workflow.ipynb`** — STAC catalog search, asset inspection, and mock QGIS raster loading.
- **`examples/browser_maplibre/`** — end-to-end browser MapLibre client for the `geoagent browser` WebSocket backend.
- **`examples/browser_maplibre_typescript/`** — TypeScript/Vite version of the browser MapLibre WebSocket client.
- **`examples/browser_maplibre_strands_typescript/`** — browser-only TypeScript/Vite MapLibre client with a Strands TypeScript agent, direct in-page map tools, and selectable model providers.
- **`examples/node_maplibre_strands_typescript/`** — Node.js TypeScript MapLibre app with a Strands backend for the GeoAgent/QGIS provider set, keeping API keys, Codex OAuth, and AWS credentials server-side.
- **`examples/nasa_opera_qgis.py`** — NASA OPERA search and footprints workflow for the QGIS Python console.

Install extras as shown in each notebook (`GeoAgent[anthropic]`, `GeoAgent[anthropic,leafmap]`, `GeoAgent[stac]`).
For NASA OPERA, install GeoAgent in the QGIS Python environment with the
`nasa-opera` extra and run the example from QGIS:

```bash
pip install "GeoAgent[nasa-opera,openai]"
```
