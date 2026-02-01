# Usage

## Basic Usage

The simplest way to use GeoAgent is through the `GeoAgent` class:

```python
from geoagent import GeoAgent

# Initialize with default LLM (auto-detects from environment)
agent = GeoAgent()

# Or specify a provider and model
agent = GeoAgent(provider="openai", model="gpt-4o")
agent = GeoAgent(provider="anthropic", model="claude-sonnet-4-20250514")
agent = GeoAgent(provider="ollama", model="llama3.1")
```

## Chat Interface

Use `chat()` for the full pipeline (plan, search, analyze, visualize):

```python
result = agent.chat("Show NDVI for San Francisco in July 2024")

# Access the results
result.plan       # PlannerOutput with parsed intent and parameters
result.data       # DataResult with retrieved data references
result.analysis   # AnalysisResult with computed values
result.map        # Interactive leafmap MapLibre map
result.code       # Generated Python code for reproducibility
```

## Targeted Operations

Use specific methods when you only need part of the pipeline:

```python
# Search for data only
data = agent.search("Find Sentinel-2 imagery of Tokyo from March 2024")

# Search and analyze
analysis = agent.analyze("Calculate NDVI for the Amazon in 2023")

# Full pipeline with visualization
result = agent.visualize("Show land cover change in Dubai from 2020 to 2024")
```

## STAC Catalog Search

Search across multiple STAC catalogs directly:

```python
from geoagent.catalogs.registry import CatalogRegistry

reg = CatalogRegistry()

# List available catalogs
for cat in reg.list_catalogs():
    print(f"{cat.name}: {cat.description}")

# Get a client for a specific catalog
client = reg.get_client("earth_search")
client = reg.get_client("planetary_computer")

# Add a custom catalog
reg.add_catalog(
    name="my_catalog",
    url="https://stac.example.com",
    description="My custom STAC catalog"
)
```

## Using Tools Directly

Each tool can be used independently:

### STAC Search

```python
from geoagent.core.tools.stac import search_stac

items = search_stac.invoke({
    "query": "Sentinel-2 imagery",
    "catalog": "earth_search",
    "bbox": [-122.5, 37.5, -122.0, 38.0],
    "datetime_range": "2024-07-01/2024-07-31",
    "collections": ["sentinel-2-l2a"],
    "max_items": 5
})
```

### DuckDB Spatial Queries

```python
from geoagent.core.tools.duckdb_tool import query_spatial_data

results = query_spatial_data.invoke({
    "sql": "SELECT * FROM read_parquet('data.parquet') LIMIT 10",
    "data_path": "path/to/data.parquet"
})
```

### Raster Analysis

```python
from geoagent.core.tools.raster import load_raster, compute_index

# Load a COG
data = load_raster.invoke({
    "url_or_path": "https://example.com/cog.tif",
    "bbox": [-122.5, 37.5, -122.0, 38.0]
})

# Compute NDVI
ndvi = compute_index.invoke({
    "dataset": data,
    "index_name": "ndvi"
})
```

### MapLibre Visualization

```python
from geoagent.core.tools.viz import show_on_map, add_cog_layer

# Create a map with layers
result = show_on_map.invoke({
    "layers": [
        {"type": "cog", "data": "https://example.com/cog.tif", "name": "Satellite"}
    ],
    "center": [37.77, -122.42],
    "zoom": 10
})
```

## LLM Provider Selection

```python
from geoagent.core.llm import get_llm, get_available_providers, check_api_keys

# Check what's available
print(get_available_providers())  # ['openai', 'ollama']
print(check_api_keys())           # {'openai': True, 'anthropic': False, ...}

# Create a specific LLM
llm = get_llm(provider="openai", model="gpt-4o", temperature=0.0)
```

## Code Transparency

One of GeoAgent's key features is showing the Python code it generates:

```python
result = agent.chat("Calculate mean NDVI for Central Park, NYC in August 2024")

# Print the generated code
print(result.code)
# Output:
# import rioxarray
# import xarray as xr
# ds = rioxarray.open_rasterio("s3://...")
# nir = ds.sel(band="B08")
# red = ds.sel(band="B04")
# ndvi = (nir - red) / (nir + red)
# mean_ndvi = float(ndvi.mean())
```

This makes workflows reproducible and educational. Researchers can take the generated code, modify it, and integrate it into their own scripts.
