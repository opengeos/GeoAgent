# Examples

Runnable Jupyter notebooks live under **`docs/examples/`**:

- **`docs/examples/intro.ipynb`** — GeoAgent + Anthropic (no map).
- **`docs/examples/live_mapping.ipynb`** — leafmap MapLibre + Claude; camera comes from **`m.view_state`** (see `get_map_state` tool).
- **`docs/examples/qgis_agent.ipynb`** — QGIS-oriented tools using mock `iface` in Jupyter; snippet for real QGIS included.
- **`examples/nasa_opera_qgis.py`** — NASA OPERA search and footprints workflow for the QGIS Python console.

Install extras as shown in each notebook (`GeoAgent[anthropic]`, `GeoAgent[anthropic,leafmap]`).
For NASA OPERA, install GeoAgent in the QGIS Python environment with the
`nasa-opera` extra and run the example from QGIS:

```bash
pip install "GeoAgent[nasa-opera,openai]"
```
