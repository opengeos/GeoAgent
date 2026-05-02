# GeoAgent

[![QGIS Plugin](https://img.shields.io/badge/QGIS-Plugin-green.svg)](https://plugins.qgis.org/plugins/open_geoagent)
[![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/GeoAgent/blob/main)
[![notebook-link](https://img.shields.io/badge/notebook-link-e2d610?logo=jupyter&logoColor=white)](https://notebook.link/github/opengeos/GeoAgent/tree/main/lab/?path=docs%2Fnotebooks%2F00_key_features.ipynb)
[![image](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/opengeos/leafmap/HEAD)
[![image](https://img.shields.io/pypi/v/GeoAgent.svg)](https://pypi.python.org/pypi/GeoAgent)
[![image](https://static.pepy.tech/badge/GeoAgent)](https://pepy.tech/project/GeoAgent)
[![Conda Recipe](https://img.shields.io/badge/recipe-GeoAgent-green.svg)](https://github.com/conda-forge/GeoAgent-feedstock)
[![image](https://img.shields.io/conda/vn/conda-forge/GeoAgent.svg)](https://anaconda.org/conda-forge/GeoAgent)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/GeoAgent.svg)](https://anaconda.org/conda-forge/GeoAgent)
[![image](https://github.com/opengeos/leafmap/workflows/docs/badge.svg)](https://leafmap.org)
[![image](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![image](https://img.shields.io/badge/YouTube-Channel-red)](https://youtube.com/@giswqs)

**GeoAgent** is a shared AI agent layer for geospatial Python packages, live
map widgets, and QGIS plugins. It gives projects such as **leafmap**,
**anymap**, **geoai**, **geemap**, STAC workflows, NASA Earthdata workflows,
and QGIS plugins one consistent way to expose geospatial tools to large
language models.

GeoAgent is built on **[Strands Agents](https://strandsagents.com/)**. It
wraps Strands with geospatial context, tool metadata, optional package
adapters, provider configuration, and confirmation hooks for operations that
should not run silently.

[![](https://img.youtube.com/vi/5zkXQlHUsu8/maxresdefault.jpg)](https://youtu.be/5zkXQlHUsu8)

## Why GeoAgent?

Many geospatial libraries need the same agent features:

- bind an agent to a live map, QGIS session, dataset, or workflow object;
- expose package functions as structured tools with docstrings and metadata;
- support OpenAI, ChatGPT/Codex OAuth, Anthropic, Google Gemini, Bedrock, LiteLLM, and local
  Ollama models;
- keep optional geospatial stacks optional;
- ask for confirmation before deleting layers, saving files, or running
  expensive processing jobs;
- support multimodal plugin workflows such as pasted images and screenshots
  when the selected model provider supports vision;
- provide a stable integration point for downstream packages and plugins.

GeoAgent centralizes that layer so each package does not need to maintain its
own agent framework.

## Core Concepts

| Concept | Purpose |
| --- | --- |
| `GeoAgent` | High-level facade around a Strands `Agent`. |
| `GeoAgentConfig` | Provider, model, temperature, token, and client settings. |
| `GeoAgentContext` | Runtime objects bound to the agent, such as a map or QGIS iface. |
| `@geo_tool` | Decorator that turns a Python function into a Strands tool with GeoAgent metadata. |
| `GeoToolRegistry` | Registry for tool metadata, safety flags, categories, and fast-mode filtering. |
| `for_leafmap` | Factory that binds tools to a `leafmap.Map`-compatible object. |
| `for_anymap` | Factory that binds tools to an `anymap.Map`-compatible object. |
| `for_qgis` | Factory that binds tools to `qgis.utils.iface` and an optional `QgsProject`. |
| `for_nasa_opera` | Factory that binds NASA OPERA search/display tools plus QGIS tools. |
| `for_stac` | Factory that binds STAC catalog search/asset tools plus optional QGIS loading. |
| `create_agent` | Factory for custom tools or package-specific integrations. |

## Installation

Install the core package:

```bash
pip install GeoAgent
```

Core installs only the agent framework dependencies, mainly `strands-agents`
and `pydantic`. Geospatial packages and provider clients are optional extras:

| Extra | Purpose |
| --- | --- |
| `GeoAgent[openai]` | OpenAI model support through Strands. |
| `GeoAgent[anthropic]` | Anthropic Claude model support through Strands. |
| `GeoAgent[gemini]` | Google Gemini model support through Strands. |
| `GeoAgent[ollama]` | Local Ollama model support. |
| `GeoAgent[litellm]` | LiteLLM model support for many hosted and proxy providers. |
| `GeoAgent[leafmap]` | leafmap live map integration. |
| `GeoAgent[anymap]` | anymap live map integration. |
| `GeoAgent[stac]` | STAC client dependencies. |
| `GeoAgent[earthdata]` | NASA Earthdata dependencies. |
| `GeoAgent[nasa-opera]` | NASA OPERA search dependencies. |
| `GeoAgent[geoai]` | geoai integration dependencies. |
| `GeoAgent[earthengine]` | Google Earth Engine dependencies. |
| `GeoAgent[ui]` | Solara UI dependencies. |
| `GeoAgent[providers]` | OpenAI, Anthropic, Gemini, Ollama, and LiteLLM provider clients. |
| `GeoAgent[all]` | Most optional integrations. QGIS itself remains system-installed. |

Examples:

```bash
pip install "GeoAgent[leafmap,openai]"
pip install "GeoAgent[anymap,anthropic]"
pip install "GeoAgent[stac,earthdata,openai]"
```

For QGIS, install GeoAgent in the Python environment used by QGIS. The
`GeoAgent[qgis]` extra is a marker extra; QGIS is provided by the desktop
application or system package manager.

## Provider Configuration

GeoAgent selects a provider from environment variables when no provider is
specified:

| Provider | Environment |
| --- | --- |
| OpenAI | `OPENAI_API_KEY`, optional `OPENAI_MODEL` |
| ChatGPT/Codex OAuth | `OPENAI_CODEX_ACCESS_TOKEN`, optional `OPENAI_CODEX_MODEL` |
| Anthropic | `ANTHROPIC_API_KEY`, optional `ANTHROPIC_MODEL` |
| Google Gemini | `GEMINI_API_KEY` or `GOOGLE_API_KEY`, optional `GEMINI_MODEL` |
| LiteLLM | `LITELLM_API_KEY`, optional `LITELLM_MODEL` and `LITELLM_BASE_URL` |
| Ollama | `OLLAMA_HOST` or `USE_OLLAMA=1`, optional `OLLAMA_MODEL` |
| Bedrock | AWS credential chain and model access, optional `BEDROCK_MODEL` |

ChatGPT/Codex OAuth uses the Codex browser login flow and the Codex Responses
backend.

For notebooks and Python scripts, log in once with the CLI:

```bash
geoagent codex login
```

or from Python/Jupyter:

```python
from geoagent import login_openai_codex

login_openai_codex()
```

GeoAgent stores the refresh token in your user config directory and exports
`OPENAI_CODEX_ACCESS_TOKEN` for the current Python process. Later sessions can
reuse the stored login automatically, or explicitly call:

```python
from geoagent import ensure_openai_codex_environment

ensure_openai_codex_environment()
```

You can also configure providers explicitly:

```python
from geoagent import GeoAgentConfig, for_leafmap

agent = for_leafmap(
    m,
    config=GeoAgentConfig(
        provider="openai",
        model="gpt-5.5",
        temperature=0,
        max_tokens=4096,
    ),
)
```

Factories also accept `provider=` and `model_id=` shortcuts:

```python
agent = for_leafmap(m, provider="gemini", model_id="gemini-3.1-pro-preview")
```

## Quickstart

Use GeoAgent directly when you do not need a map or package-specific toolset:

```python
from geoagent import GeoAgent, GeoAgentConfig

agent = GeoAgent(config=GeoAgentConfig(provider="openai", model="gpt-5.5"))
resp = agent.chat("Explain STAC in two sentences.")
print(resp.answer_text)
```

Stream model output as it is generated:

```python
import asyncio
from geoagent import GeoAgent

agent = GeoAgent()

async def main():
    async for event in agent.stream_chat("Explain STAC in two sentences."):
        if "data" in event:
            print(event["data"], end="", flush=True)

asyncio.run(main())
```

Bind an agent to a live `leafmap` map:

```python
import leafmap
from geoagent import for_leafmap

m = leafmap.Map()
agent = for_leafmap(m)

resp = agent.chat("Add a marker for Knoxville and zoom to it.")
print(resp.answer_text)
m
```

Bind an agent to an `anymap` map:

```python
import anymap
from geoagent import for_anymap

m = anymap.Map()
agent = for_anymap(m)
agent.chat("Change the basemap and list the current layers.")
```

Launch the browser workspace:

```bash
pip install "GeoAgent[ui,anymap,openai]"
geoagent ui
```

The Solara UI opens directly to a map chat workspace with provider/model
controls, fast mode, session chat history, compact tool-call results, and a
conservative confirmation policy. Confirmation-required tools are denied by
default unless you enable auto-approve in the UI.

Use GeoAgent inside QGIS:

```python
from qgis.utils import iface
from geoagent import for_qgis

agent = for_qgis(iface)
resp = agent.chat("Summarize the project layers and zoom to the active layer.")
print(resp.answer_text)
```

`geoagent.tools.qgis` is import-safe outside QGIS. It imports QGIS classes only
inside tool bodies, so tests and non-QGIS environments can import the module.

The OpenGeoAgent QGIS plugin adds a dockable chat UI on top of this factory. It
supports provider and model controls, streaming responses, pasted image
attachments, screenshot capture from the map canvas, QGIS window, or selected
screen regions, image preview and save actions, Markdown transcript copying,
and copying the PyQGIS script used for an executed result.

## QGIS Plugin

OpenGeoAgent is the QGIS plugin interface for GeoAgent. It adds a dockable AI
assistant to QGIS so you can inspect projects, navigate the map canvas, load
data, run processing workflows, style layers, and execute confirmation-gated
PyQGIS scripts from natural language.

![OpenGeoAgent QGIS plugin](https://github.com/user-attachments/assets/ba33831f-3259-461a-89f3-1e9a13cac3e0)

![OpenGeoAgent QGIS plugin GUI](https://github.com/user-attachments/assets/393065b9-7c6a-4219-90e9-c8eb59b9bae7)

Key plugin features:

- provider and model controls for Bedrock, OpenAI, ChatGPT/Codex OAuth,
  Anthropic, Google Gemini, Ollama, and LiteLLM;
- project-aware QGIS tools for layers, selections, map navigation, processing,
  project saving, and attribute table actions;
- image-aware chat with clipboard paste and screenshot attachments for models
  that support vision inputs;
- screenshot capture from the map canvas, selected map regions, the QGIS
  window, and selected screen regions;
- image preview and save/export actions, plus inline rendering of image outputs
  returned by multimodal models;
- direct image generation with the `generate_image` tool when `OPENAI_API_KEY`
  is configured;
- copy Markdown transcript and copy executed PyQGIS script actions;
- confirmation-gated `run_pyqgis_script` fallback when a task needs QGIS API
  operations that are not covered by a dedicated tool;
- lazy dependency checks so opening the chat dock stays responsive.

See the [QGIS plugin documentation](docs/qgis-plugin.md) for setup and usage.

## Built-In Tool Surfaces

### leafmap and anymap

`for_leafmap(m)` and `for_anymap(m)` expose a shared map-control surface:

- inspect state and layers: `get_map_state`, `list_layers`;
- navigate: `set_center`, `fly_to`, `set_zoom`, `zoom_in`, `zoom_out`,
  `zoom_to_bounds`, `zoom_to_layer`;
- manage layers: `add_layer`, `remove_layer`, `clear_layers`,
  `set_layer_visibility`, `set_layer_opacity`;
- add data: `add_vector_data`, `add_geojson_data`, `add_raster_data`,
  `add_cog_layer`, `add_stac_layer`, `add_xyz_tile_layer`,
  `add_pmtiles_layer`, `add_marker`;
- change basemaps and export maps: `change_basemap`, `save_map`.

Layer lookup accepts exact names or a unique case-insensitive substring for
operations such as `remove_layer`, `zoom_to_layer`, `set_layer_visibility`, and
`set_layer_opacity`.

### QGIS

`for_qgis(iface, project=None)` exposes tools that run through a Qt GUI-thread
marshaller:

- inspect project and layer state: `list_project_layers`, `get_active_layer`,
  `get_project_state`, `get_layer_summary`, `inspect_layer_fields`,
  `get_selected_features`;
- navigate: `zoom_in`, `zoom_out`, `zoom_to_layer`, `zoom_to_extent`,
  `zoom_to_selected`, `set_center`, `set_scale`, `refresh_canvas`;
- manage layers and data: `add_vector_layer`, `add_raster_layer`,
  `add_xyz_tile_layer`, `remove_layer`, `set_layer_visibility`,
  `set_layer_opacity`;
- select and process: `select_features_by_expression`, `clear_selection`,
  `run_processing_algorithm`;
- open QGIS UI and save: `open_attribute_table`, `save_project`;
- run confirmation-gated PyQGIS fallback scripts with `run_pyqgis_script`
  when a task requires QGIS API operations that are not covered by a dedicated
  tool.

QGIS chat uses a sequential tool executor and GUI-thread marshalling so map
canvas and layer-tree calls are routed safely. The PyQGIS fallback receives
the current `iface`, `project`, `canvas`, and `active_layer`, making it useful
for actions such as raster band renderer changes, labeling tweaks, layer tree
updates, and other QGIS API operations.

### NASA OPERA

`for_nasa_opera(iface, project=None)` exposes NASA OPERA product tools and the
standard QGIS tool surface:

- inspect OPERA products: `get_available_datasets`, `get_dataset_info`;
- search Earthdata granules: `search_opera_data`;
- display results: `display_footprints`, `display_raster`, `create_mosaic`.

The OPERA integration is implemented as native GeoAgent tools and does not wrap
the NASA OPERA plugin's legacy `nasa_opera.ai.tools` registry.

When adding OPERA capabilities, implement the reusable tool logic in
`geoagent/tools/nasa_opera.py`. Keep the NASA OPERA QGIS plugin focused on UI,
settings, provider selection, progress display, and compatibility aliases. The
plugin consumes GeoAgent tool metadata automatically, so new GeoAgent OPERA
tools are available to the plugin AI Assistant without duplicating tool logic.

Use it from the QGIS Python console or from plugin code. For direct tool
testing, use `submit_nasa_opera_search_task(...)`; this avoids LLM/provider
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

For a longer QGIS-console script, see `examples/nasa_opera_qgis.py`.

Natural-language OPERA chat is intentionally disabled inside QGIS for now.
Use direct tools or `submit_nasa_opera_search_task(...)` so QGIS task/thread
ownership remains explicit.

### Vantor Open Data

`for_vantor(iface, project=None, plugin=None)` exposes native tools for the
QGIS Vantor plugin's public Open Data STAC catalog:

- browse event collections: `list_vantor_events`, `get_vantor_event_info`;
- search imagery: `get_current_vantor_search_extent`, `search_vantor_items`;
- display results in QGIS: `display_vantor_footprints`, `load_vantor_cog`;
- open the plugin UI when a plugin instance is supplied: `open_vantor_panel`.

Footprint display and COG loading are confirmation-gated because they add
layers to the current QGIS project.

## Direct Tool Calls

Every GeoAgent exposes the underlying Strands tool caller. This is useful for
tests, notebooks, and plugin UI actions:

```python
agent = for_leafmap(m)
agent.tool.add_marker(lat=35.9606, lon=-83.9207, name="Knoxville")
agent.tool.list_layers()
```

You can inspect registered tool names and metadata:

```python
agent.tool_names
agent.tool_registry.get_all_tools_config()
```

## Safety and Confirmation

Tools carry metadata such as `requires_confirmation`, `destructive`, and
`long_running`. Confirmation is enforced by a Strands hook before gated tools
run.

By default, confirmation-required tools are denied unless you pass a
confirmation callback:

```python
from geoagent import auto_approve_all, for_leafmap

agent = for_leafmap(m, confirm=auto_approve_all)
```

For real applications, pass a callback that opens a Qt dialog, notebook modal,
web UI prompt, or CLI prompt. Use confirmation for operations that delete data,
overwrite files, save projects, launch expensive jobs, or call external
services with cost implications.

## Custom Tools

Package integrations can expose their own functions with `@geo_tool`:

```python
from geoagent import GeoAgentContext, create_agent, geo_tool

@geo_tool(category="demo")
def buffer_distance(layer_name: str, distance: float) -> str:
    """Buffer a named layer by a distance in map units."""
    return f"Buffered {layer_name} by {distance}."

agent = create_agent(
    context=GeoAgentContext(),
    tools=[buffer_distance],
)
```

For package-specific adapters, prefer a factory that binds live objects through
closures, just like `for_leafmap`, `for_anymap`, and `for_qgis`. This keeps
widgets, clients, credentials, and session objects out of the LLM-visible
arguments.

## Fast Mode

Pass `fast=True` to reduce the exposed tool surface and cap response tokens for
lower-latency map control:

```python
agent = for_leafmap(m, fast=True)
```

Fast mode keeps common inspection, navigation, and basemap tools, filters out
heavier or more specialized tools, and limits model responses to short
post-tool replies. The cap is conservative enough for tool calls while avoiding
very long responses. The model call still dominates latency for local models,
so small prompts may not show a large timing difference on every provider.

## Examples

Runnable notebooks live under `docs/examples/`:

- `docs/examples/intro.ipynb` — basic GeoAgent usage.
- `docs/examples/openai_codex.ipynb` — ChatGPT/Codex OAuth in Python/Jupyter.
- `docs/examples/stream_chat_openai_codex.ipynb` — streamed ChatGPT/Codex output.
- `docs/examples/live_mapping.ipynb` — live map workflow.
- `docs/examples/qgis_agent.ipynb` — QGIS-oriented workflow using mocks.
- `docs/examples/stac_workflow.ipynb` — STAC catalog search and mock QGIS loading.
- `examples/nasa_opera_qgis.py` — NASA OPERA workflow for QGIS.

Prompt ideas:

- "List layers on the current map."
- "Add a marker for Seattle and zoom to it."
- "Show the QGIS project state and summarize each layer."
- "Select parcels where population is greater than 10000."
- "Add a STAC layer and set its opacity to 0.6."
- "Search for January 2024 OPERA surface water near Houston and display the footprints."

## Development

```bash
git clone https://github.com/opengeos/GeoAgent.git
cd GeoAgent
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

Run checks:

```bash
ruff check geoagent tests
pytest -q
```

See [docs/contributing.md](docs/contributing.md) for guidance on adding new
package and tool integrations.

## License

GeoAgent is released under the MIT License. See `LICENSE`.

## Links

- Documentation: <https://geoagent.gishub.org>
- Repository: <https://github.com/opengeos/GeoAgent>
- Issues: <https://github.com/opengeos/GeoAgent/issues>
- Strands Agents: <https://strandsagents.com>
