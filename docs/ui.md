# Web UI

GeoAgent includes a Streamlit-based chat interface for interactive geospatial analysis.

## Quick Start

```bash
# Install UI dependencies
pip install "geoagent[ui]"

# Launch the UI
geoagent ui
```

Or run directly:

```bash
streamlit run geoagent/ui/app.py
```

## Features

- **Chat interface** — type natural language queries and get maps + code
- **Provider selection** — switch between OpenAI, Anthropic, Google Gemini, or Ollama
- **Interactive maps** — MapLibre visualizations embedded directly in the browser
- **Generated code** — expandable sections showing the Python code behind each result
- **Chat history** — previous queries and results persist during the session

## Python API

You can also launch the UI programmatically:

```python
from geoagent.ui import launch_ui

launch_ui()
```

## Module Reference

::: geoagent.ui
