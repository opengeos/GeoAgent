# GeoAgent

[![image](https://img.shields.io/pypi/v/geoagent.svg)](https://pypi.python.org/pypi/geoagent)
[![image](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI agent for geospatial data analysis and visualization.

- **Documentation**: <https://geoagent.gishub.org>
- **Source code**: <https://github.com/opengeos/GeoAgent>
- **PyPI**: <https://pypi.python.org/pypi/geoagent>

## Features

- Natural language interface for geospatial data workflows
- 4-agent LangGraph pipeline: Planner, Data, Analysis, Visualization
- Multi-LLM support (OpenAI, Anthropic, Google Gemini, Ollama)
- Multi-catalog STAC search (Earth Search, Planetary Computer, USGS, NASA CMR)
- Code transparency showing generated Python code at each step
- Jupyter-native with interactive MapLibre maps via leafmap
- DuckDB spatial SQL for GeoParquet and Overture Maps
- Raster analysis with xarray, rioxarray, and rasterio
- Vector operations with geopandas

## Installation

```bash
pip install geoagent
```

With all optional dependencies:

```bash
pip install "geoagent[all]"
```

## LLM Setup

GeoAgent supports multiple LLM providers. You need at least one configured to use the agent.

### Supported Providers

| Provider | Default Model | API Key Env Variable | Install Extra |
|----------|--------------|---------------------|---------------|
| OpenAI | `gpt-4.1` | `OPENAI_API_KEY` | *(included)* |
| Anthropic | `claude-sonnet-4-5-20250929` | `ANTHROPIC_API_KEY` | `pip install "geoagent[llm]"` |
| Google Gemini | `gemini-2.5-flash` | `GOOGLE_API_KEY` | `pip install "geoagent[llm]"` |
| Ollama (local) | `llama3.1` | *(none needed)* | `pip install "geoagent[ollama]"` |

### Setting API Keys

Set your API key as an environment variable:

```bash
# OpenAI (included by default)
export OPENAI_API_KEY="your-openai-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-key"

# Google Gemini
export GOOGLE_API_KEY="your-google-key"
```

You can also add these to a `.env` file or your shell profile (`~/.bashrc`, `~/.zshrc`).

### Choosing a Provider

By default, GeoAgent auto-detects available providers by checking environment variables in order: OpenAI → Anthropic → Google → Ollama. The first available provider is used.

To specify a provider and model explicitly:

```python
from geoagent import GeoAgent

# Use the default provider (auto-detected)
agent = GeoAgent()

# Use a specific provider
agent = GeoAgent(provider="anthropic")

# Use a specific provider and model
agent = GeoAgent(provider="openai", model="gpt-4o-mini")
agent = GeoAgent(provider="google", model="gemini-2.5-flash")
```

### Using Ollama (Local LLMs)

To run GeoAgent with a local LLM via [Ollama](https://ollama.com/), no API key is needed:

```bash
# Install Ollama and pull a model
ollama pull llama3.1

# Install the Ollama extra
pip install "geoagent[ollama]"
```

```python
agent = GeoAgent(provider="ollama", model="llama3.1")
```

### Using a Custom LLM Instance

You can pass any LangChain-compatible chat model directly:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
agent = GeoAgent(llm=llm)
```

## Quick Start

```python
from geoagent import GeoAgent

agent = GeoAgent()
result = agent.chat("Show NDVI for San Francisco in July 2024")
result.map   # displays interactive map in Jupyter
print(result.code)  # shows the generated Python code
```

## Architecture

GeoAgent uses a 4-agent pipeline orchestrated by LangGraph:

1. **Planner** parses natural language into structured parameters
2. **Data Agent** searches STAC catalogs and retrieves geospatial data
3. **Analysis Agent** computes indices and statistics with transparent code generation
4. **Visualization Agent** renders results on interactive leafmap MapLibre maps

## License

MIT
