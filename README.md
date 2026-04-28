# GeoAgent

**GeoAgent** is a shared AI agent layer for geospatial Python packages, live
map widgets, and QGIS plugins. It gives projects such as **leafmap**,
**anymap**, **geoai**, **geemap**, STAC workflows, NASA Earthdata workflows,
and QGIS plugins one consistent way to expose geospatial tools to large
language models.

GeoAgent is built on **[Strands Agents](https://strandsagents.com/)**. It
wraps Strands with geospatial context, tool metadata, optional package
adapters, provider configuration, and confirmation hooks for operations that
should not run silently.

## Why GeoAgent?

Many geospatial libraries need the same agent features:

- bind an agent to a live map, QGIS session, dataset, or workflow object;
- expose package functions as structured tools with docstrings and metadata;
- support OpenAI, Anthropic, Bedrock, and local Ollama models;
- keep optional geospatial stacks optional;
- ask for confirmation before deleting layers, saving files, or running
  expensive processing jobs;
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
| `GeoAgent[ollama]` | Local Ollama model support. |
| `GeoAgent[leafmap]` | leafmap live map integration. |
| `GeoAgent[anymap]` | anymap live map integration. |
| `GeoAgent[stac]` | STAC client dependencies. |
| `GeoAgent[earthdata]` | NASA Earthdata dependencies. |
| `GeoAgent[geoai]` | geoai integration dependencies. |
| `GeoAgent[earthengine]` | Google Earth Engine dependencies. |
| `GeoAgent[ui]` | Solara UI dependencies. |
| `GeoAgent[providers]` | OpenAI, Anthropic, and Ollama provider clients. |
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
| Anthropic | `ANTHROPIC_API_KEY`, optional `ANTHROPIC_MODEL` |
| Ollama | `OLLAMA_HOST` or `USE_OLLAMA=1`, optional `OLLAMA_MODEL` |
| Bedrock | AWS credential chain and model access |

You can also configure providers explicitly:

```python
from geoagent import GeoAgentConfig, for_leafmap

agent = for_leafmap(
    m,
    config=GeoAgentConfig(
        provider="openai",
        model="gpt-5.4-mini",
        temperature=0,
        max_tokens=4096,
    ),
)
```

Factories also accept `provider=` and `model_id=` shortcuts:

```python
agent = for_leafmap(m, provider="anthropic", model_id="claude-sonnet-4-6")
```

## Quickstart

Use GeoAgent directly when you do not need a map or package-specific toolset:

```python
from geoagent import GeoAgent, GeoAgentConfig

agent = GeoAgent(config=GeoAgentConfig(provider="openai", model="gpt-5.4-mini"))
resp = agent.chat("Explain STAC in two sentences.")
print(resp.answer_text)
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
- open QGIS UI and save: `open_attribute_table`, `save_project`.

QGIS chat uses a sequential tool executor and GUI-thread marshalling so map
canvas and layer-tree calls are routed safely.

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

Pass `fast=True` to reduce the exposed tool surface for lower-latency map
control:

```python
agent = for_leafmap(m, fast=True)
```

Fast mode keeps common inspection and navigation tools while filtering out
heavier or more specialized tools.

## Examples

Runnable notebooks live under `docs/examples/`:

- `docs/examples/intro.ipynb` — basic GeoAgent usage.
- `docs/examples/live_mapping.ipynb` — live map workflow.
- `docs/examples/qgis_agent.ipynb` — QGIS-oriented workflow using mocks.

Prompt ideas:

- "List layers on the current map."
- "Add a marker for Seattle and zoom to it."
- "Show the QGIS project state and summarize each layer."
- "Select parcels where population is greater than 10000."
- "Add a STAC layer and set its opacity to 0.6."

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
- Strands Agents: <https://strandsagents.com/>
