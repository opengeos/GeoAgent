# GeoAgent Core Implementation

This document describes the implemented core agent architecture for GeoAgent with MapLibre GL visualization backend.

## Implementation Status ✅

### Completed Components

1. **Core Data Models** (`geoagent/core/models.py`)
   - `PlannerOutput` - Structured query parameters
   - `DataResult` - Data retrieval results
   - `AnalysisResult` - Analysis computation results
   - `GeoAgentResponse` - Complete pipeline response

2. **Data Agent** (`geoagent/core/data_agent.py`)
   - Searches STAC catalogs and DuckDB for geospatial data
   - Supports raster (satellite), vector, and tabular data types
   - Graceful fallbacks when tools are not available
   - Mock data generation for development

3. **Analysis Agent** (`geoagent/core/analysis_agent.py`)
   - Performs geospatial analysis (NDVI, zonal stats, time series, etc.)
   - **Key feature**: Generates transparent Python code showing computations
   - Supports spectral indices, change detection, vector analysis
   - Returns visualization hints for downstream rendering

4. **Visualization Agent** (`geoagent/core/viz_agent.py`)
   - Creates MapLibre GL visualizations using leafmap's maplibregl backend
   - High-performance COG layer support with `add_cog_layer()`
   - Vector tiles, PMTiles, GeoJSON, and 3D terrain support
   - Supports single layer, multi-layer, split maps, time series
   - Graceful fallback with MockMapLibreMap when leafmap.maplibregl not available
   - Automatic visualization type selection based on data and intent

5. **Main GeoAgent Orchestrator** (`geoagent/core/agent.py`)
   - LangGraph workflow coordination (with fallback to sequential execution)
   - Simple natural language query parsing
   - Multiple entry points: `chat()`, `search()`, `analyze()`, `visualize()`
   - Complete pipeline execution with error handling

6. **LLM Utilities** (`geoagent/core/llm.py`)
   - Support for OpenAI, Anthropic, Google LLMs
   - Environment-based auto-configuration
   - Mock LLM for testing when no API keys available

## Usage Examples

### Basic Usage
```python
from geoagent import GeoAgent

# Initialize agent
agent = GeoAgent()

# Full pipeline
result = agent.chat("Show NDVI for San Francisco in July 2024")
result.map  # Display in Jupyter
print(result.code)  # Show generated Python code

# Just search for data
data = agent.search("Find Sentinel-2 imagery for California")

# Search + analysis
analysis = agent.analyze("Calculate vegetation index for New York")
```

### Response Structure
```python
response = agent.chat("query")
# response.plan - Parsed query parameters
# response.data - Retrieved data items
# response.analysis - Analysis results (optional)
# response.map - MapLibre GL visualization (optional)
# response.code - All generated Python code
# response.success - Pipeline success status
# response.execution_time - Performance timing
```

## Architecture Design

### 4-Agent Pipeline
1. **Planner Agent** (simple implementation) - Parse natural language → structured parameters
2. **Data Agent** - Search catalogs → retrieve data
3. **Analysis Agent** - Process data → compute results + generate code
4. **Visualization Agent** - Create maps → leafmap visualization

### State Management
- Uses LangGraph StateGraph when available
- Falls back to sequential execution
- Conditional workflow based on query intent

### Graceful Degradation
- Works without optional dependencies (leafmap, langgraph, LLM APIs)
- Mock implementations for development
- Tool fallbacks when external tools unavailable

## Code Transparency Feature

A key differentiator of GeoAgent is **code transparency**. The Analysis Agent generates Python code showing exactly what computations were performed:

```python
result = agent.analyze("Calculate NDVI")
print(result.analysis.code_generated)
# Outputs:
# import rasterio
# import numpy as np
#
# def calculate_ndvi(red_band, nir_band):
#     ndvi = (nir_band - red_band) / (nir_band + red_band)
#     return ndvi
# ...
```

This provides:
- **Reproducibility** - Users can re-run the exact code
- **Education** - Shows GIS/remote sensing methodology
- **Transparency** - No black box AI results
- **Customization** - Users can modify and improve the code

## Integration Points

### Tool Integration
Tools are imported dynamically from `geoagent.core.tools.*`:
- `stac.py` - STAC catalog search
- `duckdb_tool.py` - Spatial SQL queries
- `raster.py` - Raster analysis operations
- `vector.py` - Vector geometry processing
- `viz.py` - Leafmap visualization helpers

Agents gracefully handle missing tools with fallbacks and mock implementations.

### LLM Integration
- Configurable LLM provider (OpenAI, Anthropic, Google)
- Environment variable auto-detection
- Mock LLM for testing

## Testing

Run the basic example:
```bash
cd /media/hdd/Dropbox/GitHub/GeoAgent
python examples/basic_usage.py
```

Expected output shows:
- Agent initialization
- Mock data search
- Mock analysis with code generation
- Mock map creation
- Execution timing

## Next Steps

1. **Tool Implementation** - Complete the actual tool implementations in `geoagent/core/tools/`
2. **Planner Enhancement** - Improve natural language parsing with dedicated Planner Agent
3. **Dependency Management** - Create proper requirements.txt and setup.py
4. **Real Data Testing** - Test with actual STAC catalogs and geospatial data
5. **Jupyter Integration** - Optimize for notebook display and interaction

## MapLibre GL Visualization Backend

GeoAgent uses **MapLibre GL** via leafmap's `maplibregl` backend for high-performance visualization:

### MapLibre Features
- **Cloud Optimized GeoTIFF (COG)** support with `add_cog_layer()`
- **Vector tiles and PMTiles** for efficient large-scale data
- **3D terrain and elevation** visualization
- **WebGL-accelerated** rendering for smooth performance
- **Custom styling** and multiple basemap options

### API Integration
```python
from leafmap.maplibregl import Map

# GeoAgent creates MapLibre maps automatically
map = agent.visualize("Show NDVI for California")
map.add_cog_layer(url, name="NDVI", fit_bounds=True)
map.add_pmtiles(url, name="Vector Tiles")
```

### Graceful Fallbacks
- Uses `MockMapLibreMap` when leafmap.maplibregl not available
- Maintains same API for development and testing
- All map methods work consistently

## Dependencies

### Core (Required)
- `pydantic` - Data models
- `typing-extensions` - Type hints

### Optional (Graceful fallbacks)
- `leafmap[maplibregl]` - MapLibre GL visualizations
- `langgraph` - Workflow orchestration  
- `langchain-*` - LLM integrations
- `rasterio`, `geopandas` - Geospatial processing (used by tools)

## File Structure

```
geoagent/
├── core/
│   ├── __init__.py          # Core module exports
│   ├── models.py            # Pydantic data models
│   ├── agent.py             # Main GeoAgent orchestrator
│   ├── data_agent.py        # Data search and retrieval
│   ├── analysis_agent.py    # Geospatial analysis + code generation
│   ├── viz_agent.py         # Map visualization
│   ├── llm.py               # LLM utilities
│   └── tools/               # Tool implementations (separate work)
├── __init__.py              # Package exports
examples/
└── basic_usage.py           # Usage demonstration
```

This implementation provides a solid foundation for the GeoAgent platform with clear separation of concerns, graceful error handling, and the unique code transparency feature that sets it apart from other geospatial AI tools.