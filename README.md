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
