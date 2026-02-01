#!/usr/bin/env python3
"""
Basic usage examples for GeoAgent core functionality.

This example demonstrates:
1. LLM provider abstraction 
2. STAC catalog registry
3. Planner agent for query parsing

Note: Requires proper API keys and langchain packages to be installed.
"""

import os
from geoagent.core import get_default_llm, create_planner
from geoagent.catalogs import list_catalogs, get_catalog_client


def example_llm_providers():
    """Example of using the LLM provider abstraction."""
    print("=== LLM Provider Examples ===")
    
    try:
        # Try to get a default LLM (will check env vars)
        llm = get_default_llm(temperature=0.1)
        print(f"✓ Default LLM obtained: {type(llm).__name__}")
    except RuntimeError as e:
        print(f"✗ No LLM providers available: {e}")
    except ImportError as e:
        print(f"✗ Missing dependencies: {e}")


def example_catalog_registry():
    """Example of using the STAC catalog registry."""
    print("\n=== STAC Catalog Registry Examples ===")
    
    # List available catalogs
    catalogs = list_catalogs()
    print(f"Available catalogs: {len(catalogs)}")
    for catalog in catalogs:
        auth_note = " (requires auth)" if catalog.requires_auth else ""
        print(f"  - {catalog.name}: {catalog.description}{auth_note}")
    
    try:
        # Get a client for the default catalog (Earth Search)
        client = get_catalog_client()
        print(f"✓ Connected to default catalog: {client.get_self_href()}")
        
        # List some collections
        collections = list(client.get_collections())[:3]  # Just first 3
        print(f"Sample collections:")
        for collection in collections:
            print(f"  - {collection.id}: {collection.title}")
            
    except Exception as e:
        print(f"✗ Failed to connect to catalog: {e}")


def example_planner():
    """Example of using the planner agent."""
    print("\n=== Planner Agent Examples ===")
    
    # Example queries
    test_queries = [
        "Show NDVI for California in summer 2023",
        "Find Landsat images of the Amazon with less than 10% cloud cover", 
        "Compare forest cover between 2020 and 2024 in Brazil",
        "Visualize Sentinel-2 data for New York City last month"
    ]
    
    try:
        planner = create_planner()
        print("✓ Planner created successfully")
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            try:
                result = planner.parse_query(query)
                print(f"  Intent: {result.intent}")
                print(f"  Location: {result.location}")
                print(f"  Time range: {result.time_range}")
                print(f"  Dataset: {result.dataset}")
                print(f"  Analysis type: {result.analysis_type}")
                if result.parameters:
                    print(f"  Parameters: {result.parameters}")
            except Exception as e:
                print(f"  ✗ Parse error: {e}")
                
    except Exception as e:
        print(f"✗ Failed to create planner: {e}")


if __name__ == "__main__":
    print("GeoAgent Basic Usage Examples")
    print("=" * 50)
    
    # Check for environment variables
    print("\nEnvironment Check:")
    api_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]
    for key in api_keys:
        status = "✓" if os.getenv(key) else "✗"
        print(f"  {status} {key}")
    
    # Run examples
    example_llm_providers()
    example_catalog_registry() 
    example_planner()
    
    print("\n" + "=" * 50)
    print("Examples completed!")