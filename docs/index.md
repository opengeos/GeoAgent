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
[![image](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![image](https://img.shields.io/badge/YouTube-Channel-red)](https://youtube.com/@giswqs)

**GeoAgent** is a shared AI agent layer for geospatial Python packages, live
map widgets, and QGIS plugins. With one consistent interface, projects such
as **leafmap**, **anymap**, **geoai**, **geemap**, STAC, and NASA Earthdata
can expose their tools to large language models, so each library no longer
needs to build and maintain its own agent stack.

Built on **[Strands Agents](https://strandsagents.com/)**, GeoAgent adds
geospatial context, structured tool metadata, optional package adapters,
provider configuration, and confirmation hooks that pause the agent before
destructive, expensive, or otherwise irreversible operations.

See it in action:

- Short demo: [QGIS OpenGeoAgent Plugin Demo](https://youtu.be/1me0ecJ5kfk)
- Full tutorial: [OpenGeoAgent: A Multimodal AI Agent for Automated Geospatial Analysis and Visualization](https://youtu.be/5zkXQlHUsu8)

[![OpenGeoAgent video tutorial](https://img.youtube.com/vi/5zkXQlHUsu8/maxresdefault.jpg)](https://youtu.be/5zkXQlHUsu8)

## What GeoAgent Provides

- A `GeoAgent` facade around a Strands `Agent`.
- Provider configuration for OpenAI, Anthropic, Google Gemini, Bedrock,
  LiteLLM, and Ollama.
- `@geo_tool` metadata for category, safety, fast-mode filtering, and optional
  dependency checks.
- Factories for common runtime environments:
  `for_leafmap`, `for_anymap`, `for_qgis`, and `create_agent`.
- Built-in tools for live maps, QGIS projects, STAC-style workflows, Earth
  observation workflows, and optional geospatial packages.
- OpenGeoAgent QGIS plugin UI with pasted image attachments, screenshot
  capture, image preview/save actions, transcript copying, and PyQGIS script
  copying.
- Confirmation hooks for destructive, persistent, or long-running tools.
- Mock map and QGIS objects so integrations can be tested without full desktop
  or widget environments.

## Architecture

GeoAgent keeps runtime objects in Python closures and passes only structured
tool parameters through the model boundary.

| Layer             | Role                                                                            |
| ----------------- | ------------------------------------------------------------------------------- |
| `GeoAgentConfig`  | Provider, model, token, temperature, and client settings.                       |
| `GeoAgentContext` | Runtime references such as `map_obj`, `qgis_iface`, and `qgis_project`.         |
| `@geo_tool`       | Decorates package functions as Strands-compatible tools with GeoAgent metadata. |
| `GeoToolRegistry` | Stores metadata and filters tools for fast mode.                                |
| Factories         | Bind environment-specific tools to live objects.                                |
| Confirmation hook | Intercepts tools that require approval before execution.                        |

This design keeps map widgets, QGIS interfaces, authenticated clients, and
other session objects out of LLM-visible arguments while still letting tools
operate on them safely.

## Installation

```bash
pip install GeoAgent
```

Install optional integrations only when needed:

```bash
pip install "GeoAgent[leafmap,openai]"
pip install "GeoAgent[anymap,anthropic]"
pip install "GeoAgent[stac,earthdata,openai]"
```

QGIS is installed separately through the QGIS desktop application or system
package manager. Install GeoAgent into the Python environment used by QGIS.

## Provider Setup

GeoAgent can infer a provider from environment variables:

| Provider            | Environment                                                        |
| ------------------- | ------------------------------------------------------------------ |
| OpenAI              | `OPENAI_API_KEY`, optional `OPENAI_MODEL`                          |
| ChatGPT/Codex OAuth | `OPENAI_CODEX_ACCESS_TOKEN`, optional `OPENAI_CODEX_MODEL`         |
| Anthropic           | `ANTHROPIC_API_KEY`, optional `ANTHROPIC_MODEL`                    |
| Google Gemini       | `GEMINI_API_KEY` or `GOOGLE_API_KEY`, optional `GEMINI_MODEL`      |
| LiteLLM             | `LITELLM_API_KEY`, optional `LITELLM_MODEL` and `LITELLM_BASE_URL` |
| Ollama              | `OLLAMA_HOST` or `USE_OLLAMA=1`, optional `OLLAMA_MODEL`           |
| Bedrock             | AWS credentials and Bedrock model access, optional `BEDROCK_MODEL` |

Explicit configuration is also supported:

```python
from geoagent import GeoAgent, GeoAgentConfig

agent = GeoAgent(
    config=GeoAgentConfig(
        provider="openai",
        model="gpt-5.5",
        temperature=0,
    )
)
```

## Quickstart

### General Agent

```python
from geoagent import GeoAgent

agent = GeoAgent()
resp = agent.chat("Explain what STAC is in two sentences.")
print(resp.answer_text)
```

### leafmap

```python
import leafmap
from geoagent import for_leafmap

m = leafmap.Map()
agent = for_leafmap(m)
agent.chat("Add a marker for Knoxville and zoom to it.")
m
```

### anymap

```python
import anymap
from geoagent import for_anymap

m = anymap.Map()
agent = for_anymap(m)
agent.chat("List layers and change the basemap.")
```

### QGIS

```python
from qgis.utils import iface
from geoagent import for_qgis

agent = for_qgis(iface)
agent.chat("Summarize this project and zoom to the active layer.")
```

QGIS tools are import-safe outside QGIS and run through a Qt GUI-thread
marshaller when used in QGIS.

The OpenGeoAgent QGIS plugin adds a dockable multimodal chat panel. Users can
paste images from the clipboard, capture the map canvas, capture the QGIS
window, select map or screen regions, preview and save attached images, copy
the Markdown transcript, and copy the PyQGIS script that produced an executed
result.

## Built-In Tool Surfaces

### Map Widgets

`for_leafmap` and `for_anymap` provide tools for:

- inspecting map state and layer metadata;
- setting center, zoom, bounds, and layer-focused views;
- adding GeoJSON, vector, raster, COG, STAC, XYZ, PMTiles, and marker layers;
- changing basemaps;
- removing, clearing, hiding, showing, and changing layer opacity;
- saving maps to HTML.

### QGIS

`for_qgis` provides tools for:

- listing project layers and active-layer metadata;
- reading project state, layer summaries, fields, selected features, CRS,
  extents, opacity, visibility, and feature counts when available;
- zooming, centering, setting scale, and refreshing the canvas;
- adding vector, raster, and XYZ tile layers;
- setting layer visibility and opacity;
- selecting features with QGIS expressions and clearing selections;
- zooming to selected features;
- running QGIS Processing algorithms;
- opening attribute tables and saving projects;
- running confirmation-gated PyQGIS fallback scripts when the requested QGIS
  API operation is not covered by a dedicated tool.

### Custom and Package Tools

Use `create_agent` and `@geo_tool` to build tools for another package:

```python
from geoagent import GeoAgentContext, create_agent, geo_tool

@geo_tool(category="my_package")
def describe_dataset(path: str) -> dict:
    """Return basic metadata for a dataset path."""
    return {"path": path}

agent = create_agent(
    context=GeoAgentContext(),
    tools=[describe_dataset],
)
```

For integrations that need live objects, create a factory that closes over the
object and returns decorated tools. See the contributing guide for a complete
integration checklist.

## Safety

GeoAgent tools can be marked as confirmation-required, destructive, or
long-running. The confirmation hook runs before those tools execute. This is
important for operations that delete layers, overwrite files, save projects,
run processing jobs, call external services, or incur cost.

```python
from geoagent import auto_approve_all, for_leafmap

agent = for_leafmap(m, confirm=auto_approve_all)
```

Applications should usually pass their own `confirm=` callback that opens a
Qt dialog, notebook modal, web prompt, or CLI prompt.

## Learn More

- [Installation](installation.md)
- [QGIS Plugin](qgis-plugin.md)
- [Usage](usage.md)
- [Tools](tools.md)
- [Safety](safety.md)
- [Factory API](factory.md)
- [Contributing](contributing.md)
- [Examples](examples/intro.ipynb)
