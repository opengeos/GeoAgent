# GeoAgent

[![image](https://img.shields.io/pypi/v/geoagent.svg)](https://pypi.python.org/pypi/geoagent)
[![image](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An AI agent for geospatial data analysis and visualization.**

GeoAgent is a platform-agnostic AI agent that lets you explore, analyze, and visualize Earth observation data using natural language. It combines LLM-powered query understanding with a full geospatial Python stack.

- Free software: MIT License
- Documentation: <https://geoagent.gishub.org>
- Source code: <https://github.com/opengeos/GeoAgent>

## Key Features

- **Natural language interface** for geospatial data workflows
- **4-agent pipeline** using LangGraph: Planner, Data, Analysis, Visualization
- **Multi-LLM support** including OpenAI, Anthropic, Google Gemini, and Ollama (local)
- **Multi-catalog STAC search** across Earth Search, Planetary Computer, USGS, NASA CMR
- **Code transparency** showing generated Python code at each step
- **Jupyter-native** with inline map display
- **MapLibre visualization** via leafmap with 3D terrain, PMTiles, and vector tiles
- **DuckDB spatial SQL** for GeoParquet and Overture Maps queries
- **Full geospatial stack** with rasterio, xarray, geopandas, and rioxarray

## Quick Start

```python
from geoagent import GeoAgent

agent = GeoAgent()
result = agent.chat("Show NDVI for San Francisco in July 2024")
result.map   # displays interactive map in Jupyter
print(result.code)  # shows the generated Python code
```

## Architecture

GeoAgent uses a simplified 4-agent pipeline orchestrated by LangGraph:

1. **Planner Agent** parses natural language queries into structured parameters (intent, location, time range, dataset)
2. **Data Agent** searches STAC catalogs, queries DuckDB, and retrieves raster/vector data
3. **Analysis Agent** computes spectral indices, zonal statistics, and generates reproducible Python code
4. **Visualization Agent** renders results on interactive leafmap MapLibre maps

## Why GeoAgent?

| Feature         | Cloud-locked tools      | GeoAgent                                |
| --------------- | ----------------------- | --------------------------------------- |
| LLM provider    | Single vendor           | Any (OpenAI, Anthropic, Gemini, Ollama) |
| Map rendering   | Proprietary             | leafmap (open source, MapLibre)         |
| Data catalogs   | Single catalog          | Any STAC catalog                        |
| Deployment      | Cloud services required | `pip install geoagent`                  |
| Environment     | Web app only            | Jupyter-native + optional web UI        |
| Code visibility | Black box               | Shows generated Python code             |
