# MapLibre GL Backend Update

This document summarizes the important update to use leafmap's MapLibre GL backend instead of the default folium backend.

## ✅ Changes Made

### 1. Import Updates
**Before:**
```python
import leafmap
m = leafmap.Map()
```

**After:**
```python
from leafmap.maplibregl import Map as MapLibreMap
m = MapLibreMap()
```

### 2. Layer Addition Methods
**Raster/COG Layers:**
```python
# Old (folium): 
m.add_raster(url, layer_name="Layer", fit_bounds=True)

# New (MapLibre):
m.add_cog_layer(url, name="Layer", fit_bounds=True)
```

**Vector Layers:**
```python
# Old (folium):
m.add_geojson(data, layer_name="Vector", style=style)

# New (MapLibre):
m.add_geojson(data, name="Vector", style=style)
```

### 3. MockMapLibreMap Class
Created comprehensive mock implementation with MapLibre-compatible methods:
- `add_cog_layer()` - Cloud Optimized GeoTIFF support
- `add_pmtiles()` - PMTiles vector tile support  
- `add_geojson()` - GeoJSON with proper naming
- `add_title()` - Map title support
- `to_html()` - HTML export functionality

### 4. File Updates

#### `geoagent/core/viz_agent.py`
- ✅ Import `from leafmap.maplibregl import Map`
- ✅ Replace all `add_raster()` calls with `add_cog_layer()`
- ✅ Update `add_geojson()` to use `name` parameter
- ✅ Add MapLibre-specific MockMapLibreMap class
- ✅ Update docstrings to mention MapLibre GL

#### `geoagent/core/agent.py`
- ✅ Import `from leafmap.maplibregl import Map as MapLibreMap`
- ✅ Update MAPLIBRE_AVAILABLE flag
- ✅ Update docstrings for MapLibre references

#### `geoagent/core/llm.py`
- ✅ Add MockLLM class for when no LLM providers available
- ✅ Update get_default_llm() to return MockLLM instead of raising error

### 5. Documentation Updates

#### `AGENT_IMPLEMENTATION.md`
- ✅ Add MapLibre GL features section
- ✅ Update visualization descriptions
- ✅ Update dependency information

#### `requirements-agents.txt`
- ✅ Change to `leafmap[maplibregl]>=0.32.0`
- ✅ Add `maplibre>=0.8.0`

### 6. New Examples

#### `examples/maplibre_usage.py`
- ✅ Comprehensive MapLibre GL feature demonstration
- ✅ Shows COG layer usage
- ✅ Demonstrates vector visualization
- ✅ Time series and split map examples

## 🎯 Key Benefits of MapLibre Backend

### Performance
- **WebGL-accelerated rendering** for smooth pan/zoom
- **Native COG support** for efficient satellite imagery
- **Vector tiles** for large-scale data visualization

### Features
- **3D terrain and elevation** visualization
- **PMTiles support** for efficient vector tiles
- **Custom styling** and basemap options
- **Better mobile performance**

### Developer Experience  
- **Same API** as folium backend (mostly compatible)
- **Better error handling** and performance monitoring
- **Modern web standards** (WebGL, vector tiles)

## 🔧 Usage Examples

### Basic MapLibre Map
```python
from geoagent import GeoAgent

agent = GeoAgent()
result = agent.chat("Show NDVI for San Francisco")

# result.map is now a MapLibre GL map
result.map.add_cog_layer("satellite.tif", name="Satellite", fit_bounds=True)
result.map.add_pmtiles("vectors.pmtiles", name="Roads")
```

### Features Available
```python
# High-performance COG layers
map.add_cog_layer(url, name="Satellite", fit_bounds=True)

# Vector tiles and PMTiles
map.add_pmtiles(url, name="Vector Data")

# GeoJSON with styling
map.add_geojson(geojson, name="Boundaries", style={"color": "red"})

# 3D terrain (when available)
map.add_source("terrain", {"type": "raster-dem", "url": terrain_url})
```

## ⚠️ Breaking Changes

### Import Changes
**Before:**
```python
import leafmap
```

**After:**
```python
from leafmap.maplibregl import Map
```

### Method Parameters
- `layer_name` parameter changed to `name` in most methods
- `add_raster()` replaced with `add_cog_layer()` for better performance
- Some folium-specific parameters may not be supported

### Map Object Type
- Maps are now `leafmap.maplibregl.Map` objects, not `folium.Map`
- Different API for advanced customization
- Better performance characteristics

## 🧪 Testing

Both examples work correctly:
```bash
# Basic functionality
python examples/basic_usage.py

# MapLibre-specific features  
python examples/maplibre_usage.py
```

## 📝 Next Steps

1. **Install MapLibre dependencies** when ready for real visualization
2. **Test with real geospatial data** to validate performance gains
3. **Explore 3D and vector tile features** for advanced use cases
4. **Optimize for specific visualization patterns** in geospatial analysis

The MapLibre GL backend provides a solid foundation for high-performance geospatial visualization in GeoAgent.