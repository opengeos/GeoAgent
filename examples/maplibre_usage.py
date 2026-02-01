"""MapLibre GL visualization example for GeoAgent.

This example demonstrates GeoAgent's MapLibre GL backend for high-performance
3D mapping and vector tile visualization.
"""

import os
import sys
import logging

# Add parent directory to path for importing geoagent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geoagent import GeoAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate GeoAgent MapLibre GL features."""

    print("GeoAgent MapLibre GL Example")
    print("=" * 40)

    # Initialize GeoAgent (will use MapLibre backend for visualization)
    print("\n1. Initializing GeoAgent with MapLibre backend...")
    agent = GeoAgent()

    # Example 1: Raster visualization with COG layers
    print("\n2. Creating raster visualization with COG layers...")
    query1 = "Show Sentinel-2 imagery for San Francisco in July 2024"
    result1 = agent.visualize(query1)

    if result1.map:
        print(f"✓ Created MapLibre map: {result1.map}")
        print(f"  Map type: {type(result1.map).__name__}")
        if hasattr(result1.map, "layers"):
            print(f"  Layers: {len(result1.map.layers)}")
            for i, layer in enumerate(result1.map.layers):
                print(f"    Layer {i+1}: {layer.get('name')} ({layer.get('type')})")

    # Example 2: NDVI analysis with custom styling
    print("\n3. Creating NDVI analysis with MapLibre styling...")
    query2 = "Calculate and visualize NDVI for California"
    result2 = agent.chat(query2)

    if result2.map:
        print(f"✓ Created analysis visualization: {result2.map}")
        if hasattr(result2.map, "_style"):
            print(f"  Map style: {result2.map._style}")

    # Example 3: Vector data with GeoJSON
    print("\n4. Vector data visualization...")
    query3 = "Show administrative boundaries for New York"
    result3 = agent.visualize(query3)

    if result3.map:
        print(f"✓ Created vector map: {result3.map}")

    # Example 4: Time series comparison
    print("\n5. Time series comparison with split map...")
    query4 = "Compare NDVI for forest areas between 2020 and 2024"
    result4 = agent.chat(query4)

    if result4.map:
        print(f"✓ Created time series map: {result4.map}")

    # Show MapLibre capabilities
    print("\n6. MapLibre GL Backend Features:")
    print("   ✓ Cloud Optimized GeoTIFF (COG) support via add_cog_layer()")
    print("   ✓ High-performance vector tiles with add_pmtiles()")
    print("   ✓ 3D terrain and elevation visualization")
    print("   ✓ WebGL-accelerated rendering")
    print("   ✓ Smooth zooming and panning")
    print("   ✓ Custom styling and basemaps")

    # Code transparency
    if result2.code:
        print("\n7. Generated analysis code:")
        print(result2.code[:300] + "..." if len(result2.code) > 300 else result2.code)

    print("\nMapLibre GL example completed!")
    print("\nIn Jupyter notebooks, the maps would display as interactive")
    print("MapLibre GL widgets with full 3D and vector tile support.")


if __name__ == "__main__":
    main()
