# Web UI

GeoAgent includes a Solara-based chat interface for interactive geospatial analysis with a persistent, interactive map.

## Quick Start

```bash
# Install UI dependencies
pip install "geoagent[ui]"

# Launch the UI
geoagent ui
```

Or run directly:

```bash
solara run geoagent/ui/app.py
```

## Features

- **Persistent map** — layers accumulate across queries on the same interactive MapLibre map
- **Native widget rendering** — full bidirectional map interaction (zoom, pan, click)
- **Chat interface** — type natural language queries with real-time status updates
- **Provider selection** — switch between OpenAI, Anthropic, Google Gemini, or Ollama
- **Generated code** — toggle code display for transparency

## Python API

You can also launch the UI programmatically:

```python
from geoagent.ui import launch_ui

launch_ui()
```

## Module Reference

::: geoagent.ui
