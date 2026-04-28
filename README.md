# GeoAgent

**GeoAgent** is a centralized AI agent layer for Open Geospatial Python stacks and QGIS plugins. Downstream packages (**geoai**, **leafmap**, **anymap**, **geemap**, QGIS GeoAI / GEE Catalogs / NASA Earthdata plugins, …) reuse one framework instead of each shipping duplicate orchestration code.

Version **2.x** is built on **[Strands Agents](https://strandsagents.com/)** ([sdk-python](https://github.com/strands-agents/sdk-python)): one Python `Agent`, custom `@geo_tool` wrappers, multiple model providers, hooks for safety/confirmation, and direct tool calls via `agent.strands_agent.tool.<name>(...)`.

---

## Why a shared agent layer?

- **One mental model** for tools, prompts, provider configuration, and human-in-the-loop confirmation.
- **Thin adapters** per environment (`for_leafmap`, `for_anymap`, `for_qgis`) bind live objects (`leafmap.Map`, `anymap.Map`, `qgis.utils.iface`) via closures so secrets and widgets never cross the LLM boundary incorrectly.
- **Low latency by default**: a single agent, short prompts, optional **`fast=True`** mode with a reduced tool surface for simple map actions.

---

## Installation

```bash
pip install GeoAgent
```

**Core** wheels pull in `strands-agents` and `pydantic` only. Geospatial stacks are optional extras (see `pyproject.toml`):

| Extra | Purpose |
|-------|---------|
| `GeoAgent[openai]` | OpenAI models via Strands |
| `GeoAgent[anthropic]` | Anthropic Claude |
| `GeoAgent[ollama]` | Local Ollama |
| `GeoAgent[leafmap]` | leafmap for live maps |
| `GeoAgent[anymap]` | anymap |
| `GeoAgent[stac]` | STAC client stack (Phase 2 tools) |
| `GeoAgent[earthdata]` | NASA Earthdata (`earthaccess`) |
| `GeoAgent[geoai]` | geoai |
| `GeoAgent[earthengine]` | Earth Engine API |
| `GeoAgent[ui]` | Solara UI helper deps |

Developers:

```bash
pip install -e ".[dev]"
pre-commit install
```

---

## Provider configuration

Set API keys with environment variables (no hardcoded secrets):

- **OpenAI**: `OPENAI_API_KEY`, optional `OPENAI_MODEL`
- **Anthropic**: `ANTHROPIC_API_KEY`, optional `ANTHROPIC_MODEL`
- **AWS Bedrock**: standard AWS credential chain + model access
- **Ollama**: `OLLAMA_HOST` (default `http://127.0.0.1:11434`), optional `OLLAMA_MODEL`

`GeoAgentConfig` (`provider`, `model`, `temperature`, `max_tokens`, …) overrides env defaults.

---

## Quickstart

```python
from geoagent import GeoAgent, GeoAgentConfig

# Uses provider inferred from environment (see GeoAgentConfig).
agent = GeoAgent(config=GeoAgentConfig(provider="openai", model="gpt-4o-mini"))
resp = agent.chat("Explain what STAC is in two sentences.")
print(resp.answer_text)
```

### leafmap

```bash
pip install "GeoAgent[leafmap,openai]"
```

```python
import leafmap
from geoagent import for_leafmap

m = leafmap.Map()
agent = for_leafmap(m)
agent.chat("Change the basemap to CartoDB Positron and zoom in.")
```

### anymap

```python
from geoagent import for_anymap

agent = for_anymap(m)  # m = anymap.Map(...)
```

### QGIS

```python
from qgis.utils import iface
from geoagent import for_qgis

agent = for_qgis(iface)
agent.chat("List layers in this project.")
```

`geoagent.tools.qgis` stays **import-safe** without QGIS: it never imports `qgis` at module import time.

### Advanced: custom tools

```python
from geoagent import create_agent, GeoAgentContext, geo_tool

@geo_tool(category="demo")
def hello(name: str) -> str:
    """Greet the user."""
    return f"Hello, {name}"

agent = create_agent(context=GeoAgentContext(), tools=[hello()])
```

---

## Safety and confirmation

Tools carry metadata (`requires_confirmation`, `destructive`, `long_running`). Before a gated tool runs, Strands **`BeforeToolCallEvent`** hooks call your **`ConfirmCallback`**. If you do not pass one, **`auto_approve_safe_only`** denies every confirmation request (nothing destructive runs silently).

---

## Downstream integration

1. Add `GeoAgent` as a dependency with the extras you need.
2. At runtime, build an agent with `for_leafmap` / `for_anymap` / `for_qgis` or `create_agent`.
3. Wire `confirm=` to your UI (Qt dialog in QGIS, modal in Jupyter, CLI `input()`, etc.).
4. Optionally expose **`agent.strands_agent`** for power users who want raw Strands streaming or `agent.tool.*` direct execution.

---

## Examples

Runnable Jupyter notebooks live under **`docs/examples/`** — intro, MapLibre (`live_mapping`), and QGIS mock (`qgis_agent`) — see `examples/README.md`. Prompt ideas:

- “List the layers on the current map.”
- “Zoom to Knoxville, Tennessee.”
- “Inspect the fields of the active QGIS layer.”

---

## License

MIT — see `LICENSE`.

---

## Links

- [Strands Agents documentation](https://strandsagents.com/)
- [Repository](https://github.com/opengeos/GeoAgent)
