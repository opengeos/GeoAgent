"""Basic usage example for GeoAgent.

This example demonstrates how to use the GeoAgent for geospatial analysis.
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
    """Demonstrate basic GeoAgent usage."""
    
    print("GeoAgent Basic Usage Example")
    print("=" * 40)
    
    # Initialize GeoAgent
    print("\n1. Initializing GeoAgent...")
    agent = GeoAgent()
    
    # Example 1: Simple data search
    print("\n2. Searching for data...")
    query1 = "Find Sentinel-2 imagery for San Francisco"
    data_result = agent.search(query1)
    print(f"Found {data_result.total_items} items of type {data_result.data_type}")
    
    # Example 2: Analysis query  
    print("\n3. Running analysis...")
    query2 = "Calculate NDVI for San Francisco in July 2024"
    result = agent.analyze(query2)
    print(f"Analysis success: {result.success}")
    if result.analysis:
        print("Generated code:")
        print(result.analysis.code_generated[:200] + "...")
    
    # Example 3: Full pipeline with visualization
    print("\n4. Creating visualization...")
    query3 = "Show NDVI for San Francisco in July 2024"
    full_result = agent.chat(query3)
    print(f"Pipeline success: {full_result.success}")
    print(f"Execution time: {full_result.execution_time:.2f}s")
    
    if full_result.map:
        print("Map created successfully!")
        # In Jupyter, you would display the map with: full_result.map
    
    if full_result.code:
        print("\nGenerated Python code:")
        print(full_result.code)
    
    print("\nBasic usage example completed!")


if __name__ == "__main__":
    main()