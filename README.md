# GeoAgent

[![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/GeoAgent/blob/main)
[![image](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/opengeos/GeoAgent/HEAD)
[![image](https://img.shields.io/pypi/v/geoagent.svg)](https://pypi.python.org/pypi/geoagent)
[![image](https://static.pepy.tech/badge/geoagent)](https://pepy.tech/project/geoagent)
[![Conda Recipe](https://img.shields.io/badge/recipe-geoagent-green.svg)](https://github.com/conda-forge/geoagent-feedstock)
[![image](https://img.shields.io/conda/vn/conda-forge/geoagent.svg)](https://anaconda.org/conda-forge/geoagent)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/geoagent.svg)](https://anaconda.org/conda-forge/geoagent)
[![image](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A centralized AI agent framework for Open Geospatial Python packages and QGIS plugins. GeoAgent is the shared AI layer that lets `leafmap`, `anymap`, `geoai`, `geemap`, and QGIS plugins (GeoAI, GEE Data Catalogs, NASA Earthdata, ...) plug their tools into a common agent runtime built on [DeepAgents](https://github.com/langchain-ai/deepagents) without each project re-implementing its own orchestration.

- **Documentation**: <https://geoagent.gishub.org>
- **Source code**: <https://github.com/opengeos/GeoAgent>
- **PyPI**: <https://pypi.python.org/pypi/geoagent>

## Why a centralized agent layer?

Downstream packages should focus on what they do best — mapping, ML, data access — and leave agent orchestration to a single place. GeoAgent provides:

- A **DeepAgents-based core** that handles tool calling, subagent dispatch, human-in-the-loop confirmation, and conversation state.
- **Tool adapters** that wrap live runtime objects (a `leafmap.Map`, an `anymap.Map`, the QGIS `iface`, a GeoAI workflow) into LangChain tools the agent can call directly.
- A **`@geo_tool` decorator** with safety classification so destructive operations (remove, save, export, download, run-processing) require user approval before they execute.
- A **runtime context** (`GeoAgentContext`) that carries the current map, QGIS interface, working directory, and user preferences into every tool invocation.
- **Per-package helpers** — `geoagent.for_leafmap(m)`, `geoagent.for_anymap(m)`, `geoagent.for_qgis(iface)` — that build a ready-to-use agent in one line.

Downstream packages register their own tools via `register_many(my_tools)` and instantiate an agent via `geoagent.create_geo_agent(...)`; they do not depend on `langgraph`, `langchain-openai`, or any LLM SDK directly.

## Features

- **Centralized agent runtime** built on [DeepAgents](https://github.com/langchain-ai/deepagents) (LangGraph under the hood)
- **Live map control** for `leafmap.Map` and `anymap.Map`: list/add/remove layers, zoom, pan, change basemap, save HTML
- **QGIS plugin integration**, import-safe outside QGIS — the `qgis` package is never imported at module load
- **GeoAI workflows**: segmentation, object detection, image classification (when `geoai` is installed)
- **Earth Engine** and **NASA Earthdata** tools (opt-in extras)
- **STAC search** across Earth Search, Planetary Computer, USGS, and NASA CMR catalogs
- **Spatial SQL** with DuckDB for GeoParquet and Overture Maps
- **Raster** (xarray / rioxarray / rasterio) and **vector** (geopandas) analysis tools
- **Multi-LLM support**: OpenAI, Anthropic, Google Gemini, Ollama
- **Human-in-the-loop confirmation** via a pluggable `ConfirmCallback` for destructive operations
- **Code transparency** — generated Python code is returned for reproducibility
- **Mock-friendly testing** with `geoagent.testing` mocks for leafmap, anymap, and QGIS

## Installation

GeoAgent v1.0 requires **Python 3.11+**.

```bash
pip install geoagent
```

Or install from conda-forge:

```bash
conda install -c conda-forge geoagent
```

### Optional extras

Pick the extras you need; the base install stays lightweight.

| Extra | Adds | When to install |
|---|---|---|
| `[openai]` | `langchain-openai` | OpenAI / Azure OpenAI models |
| `[anthropic]` | `langchain-anthropic` | Anthropic Claude models |
| `[google]` | `langchain-google-genai` | Google Gemini models |
| `[ollama]` | `langchain-ollama` | Local Ollama models |
| `[llm]` | all of the above |  |
| `[anymap]` | `anymap` | anymap-based maps |
| `[geoai]` | `geoai` | GeoAI segmentation / detection / classification |
| `[earthengine]` | `earthengine-api` | Google Earth Engine |
| `[nasa_earthdata]` | `earthaccess` | NASA Earthdata search & download |
| `[ui]` | `solara` | the Solara web UI |
| `[all]` | everything above |  |

```bash
# Common combos
pip install "geoagent[openai,leafmap]"       # leafmap + OpenAI agent
pip install "geoagent[anthropic,geoai]"      # Claude + GeoAI segmentation
pip install "geoagent[ollama]"               # local LLM only
pip install "geoagent[all]"                  # everything
```

QGIS itself is environment-installed (system package), so the `[qgis]` marker extra exists for completeness; the QGIS tools auto-detect QGIS at runtime.

## LLM setup

GeoAgent supports multiple LLM providers. You need at least one configured to use the agent.

| Provider       | Default Model                | API Key Env Variable | Install Extra                       |
| -------------- | ---------------------------- | -------------------- | ----------------------------------- |
| OpenAI         | `gpt-4.1`                    | `OPENAI_API_KEY`     | `pip install "geoagent[openai]"`    |
| Anthropic      | `claude-sonnet-4-5-20250929` | `ANTHROPIC_API_KEY`  | `pip install "geoagent[anthropic]"` |
| Google Gemini  | `gemini-2.5-flash`           | `GOOGLE_API_KEY`     | `pip install "geoagent[google]"`    |
| Ollama (local) | `llama3.1`                   | *(none needed)*      | `pip install "geoagent[ollama]"`    |

### Setting API keys

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

By default, GeoAgent auto-detects the first provider with credentials in the order OpenAI → Anthropic → Google → Ollama. To pick a provider and model explicitly:

```python
from geoagent import GeoAgent

agent = GeoAgent()                                          # auto-detect
agent = GeoAgent(provider="anthropic")                      # named provider
agent = GeoAgent(provider="openai", model="gpt-4o-mini")    # provider + model
```

### Local LLMs via Ollama

```bash
ollama pull llama3.1
pip install "geoagent[ollama]"
```

```python
agent = GeoAgent(provider="ollama", model="llama3.1")
```

### Custom LangChain chat model

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
agent = GeoAgent(llm=llm)
```

## Quick start

```python
from geoagent import GeoAgent

agent = GeoAgent()
result = agent.chat("Show NDVI for San Francisco in July 2024")
result.map          # interactive map widget for Jupyter
print(result.code)  # the generated Python code
```

## Working with a live map (leafmap / anymap)

```python
import leafmap
from geoagent import for_leafmap

m = leafmap.Map(center=[35.96, -83.92], zoom=10)
agent = for_leafmap(m)

agent.invoke({"messages": [
    {"role": "user", "content": "Add a Sentinel-2 layer for Knoxville and zoom to it."}
]})

m  # the agent has mutated the map in place
```

The same pattern works for `anymap`:

```python
from anymap import Map
from geoagent import for_anymap

agent = for_anymap(Map())
```

## Working from a QGIS plugin

```python
from qgis.utils import iface
from geoagent import for_qgis

agent = for_qgis(iface)
agent.invoke({"messages": [
    {"role": "user", "content": "Zoom to the active layer and list all project layers."}
]})
```

The QGIS adapter is **import-safe outside QGIS**: `import geoagent.tools.qgis` works on any Python and the tool list is empty until a real `iface` is supplied. You can also pre-build agents in headless tests with the supplied mocks.

## Safety and confirmation

Tools are classified as either safe (inspect, list, zoom, pan, get-state, generate-code) or **confirmation-required** (remove, delete, save, export, download, run long processing jobs, submit Earth Engine tasks). Confirmation-required tools are wired into deepagents' `interrupt_on` so the agent pauses before they execute.

The host application supplies a `ConfirmCallback` to bridge those interrupts to its UI:

```python
from geoagent import GeoAgent, ConfirmRequest

def confirm(request: ConfirmRequest) -> bool:
    print(f"Approve {request.tool_name}({request.args})? [y/N]")
    return input().strip().lower() == "y"

agent = GeoAgent(confirm=confirm)
```

Built-in callbacks ship for Jupyter (`input()` prompt), Solara (modal dialog), and CLI (`y/N`); the default `auto_approve_safe_only` rejects every confirmation request, so confirmation-required tools never run silently.

## Defining your own tool

Tools are plain Python functions wrapped with `@geo_tool`. The decorator produces a LangChain `BaseTool`, so the result plugs straight into `create_geo_agent(tools=[...])`.

```python
from geoagent import geo_tool, create_geo_agent

@geo_tool(category="data", requires_confirmation=False)
def search_my_catalog(query: str, limit: int = 10) -> list[dict]:
    """Search a domain-specific catalog and return matches."""
    ...

agent = create_geo_agent(tools=[search_my_catalog])
```

Confirmation-required tools auto-populate `interrupt_on`:

```python
@geo_tool(category="io", requires_confirmation=True)
def overwrite_dataset(path: str) -> str:
    """Overwrite an existing dataset."""
    ...
```

## Web UI

GeoAgent includes a Solara-based chat interface with a persistent, interactive map.

```bash
pip install "geoagent[ui]"
geoagent ui
```

Or run directly:

```bash
solara run geoagent/ui/pages
```

Features:
- **Persistent map** — layers accumulate across queries on the same interactive MapLibre map
- **Native widget rendering** — full bidirectional map interaction (zoom, pan, click)
- **Chat interface** — natural language queries with real-time status updates
- **Provider selection** — switch between OpenAI, Anthropic, Google Gemini, or Ollama
- **Code transparency** — toggle generated code display for reproducibility

You can also launch it programmatically:

```python
from geoagent.ui import launch_ui

launch_ui()
```

## Architecture

```
geoagent/
├── core/             # @geo_tool, GeoAgentContext, registry, safety, factory
├── tools/            # leafmap / anymap / qgis / geoai / earthengine /
│   └── data/         #   nasa_earthdata / stac + raster, vector, duckdb, viz
├── agents/           # subagent specs (mapping, qgis, geoai, earthdata, ...)
├── integrations/     # jupyter, qgis plugin, solara, cli helpers
├── testing/          # MockLeafmap, MockAnymap, MockQGISIface, MockQGISProject
└── ui/               # Solara pages
```

The runtime is a [DeepAgents](https://github.com/langchain-ai/deepagents) `create_deep_agent` graph configured with:

- **Tools** assembled by per-package factories (`leafmap_tools(m)`, `qgis_tools(iface)`, ...) and registered with `@geo_tool` metadata
- **Subagents** for planning, data search, raster/vector analysis, mapping, QGIS, GeoAI, and Earthdata workflows
- **`interrupt_on`** auto-built from tool metadata for human-in-the-loop confirmation
- **`context_schema=GeoAgentContext`** to carry runtime state into tools

## Migration from v0.x

GeoAgent v1.0 is a substantial rewrite. The core public API (`from geoagent import GeoAgent; agent.chat(...)`) is preserved, but internals are now built on [DeepAgents](https://github.com/langchain-ai/deepagents) and require Python 3.11+. The legacy LangChain 0.3 / LangGraph 0.2 stack is no longer used; deepagents pulls LangChain 1.x transitively.

If you held references to legacy modules under `geoagent.core.tools.*`, they continue to work as-is for now and are also re-exported (with `@geo_tool` metadata) from the new `geoagent.tools.*` and `geoagent.tools.data.*` paths.

## License

MIT
