"""STAC Search Tool for GeoAgent package.

This module provides tools for searching STAC (SpatioTemporal Asset Catalog) data.
"""

from typing import List, Dict, Optional, Any
import logging

from langchain_core.tools import tool
from pystac_client import Client

try:
    from geoagent.catalogs.registry import CatalogRegistry
except ImportError:
    CatalogRegistry = None
    logging.getLogger(__name__).debug(
        "CatalogRegistry not available; using built-in catalog URLs as fallback."
    )

logger = logging.getLogger(__name__)


@tool
def search_stac(
    query: str,
    catalog: str = "microsoft-pc",
    bbox: Optional[List[float]] = None,
    datetime_range: Optional[str] = None,
    collections: Optional[List[str]] = None,
    max_items: int = 10,
    max_cloud_cover: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Search any STAC catalog for items matching the given criteria.

    This tool searches STAC catalogs to find satellite imagery and other geospatial
    assets based on location, time, and other filters.

    Args:
        query: Free-text search query to describe what you're looking for
        catalog: STAC catalog identifier (e.g., 'microsoft-pc', 'earth-search')
        bbox: Bounding box as [west, south, east, north] in WGS84 coordinates
        datetime_range: Date range in format "2023-01-01/2023-12-31" or "2023-01-01"
        collections: List of collection IDs to search within
        max_items: Maximum number of items to return (default: 10)
        max_cloud_cover: Maximum cloud cover percentage (0-100) to filter results

    Returns:
        List of dictionaries containing item metadata with keys:
        - id: Item identifier
        - datetime: Item datetime as ISO string
        - bbox: Item bounding box [west, south, east, north]
        - assets: Dictionary of available assets (e.g., bands, thumbnails)
        - thumbnail: URL to thumbnail image if available
        - collection: Collection ID
        - cloud_cover: Cloud cover percentage if available
        - geometry: Item geometry as GeoJSON

    Example:
        >>> items = search_stac(
        ...     query="Sentinel-2 imagery of San Francisco",
        ...     bbox=[-122.5, 37.7, -122.3, 37.8],
        ...     datetime_range="2023-06-01/2023-08-31",
        ...     max_cloud_cover=20
        ... )
    """
    # Validate bbox if provided
    if bbox:
        if len(bbox) != 4:
            return [
                {
                    "error": "bbox must have exactly 4 elements: [west, south, east, north]"
                }
            ]
        west_val, south_val, east_val, north_val = bbox
        if west_val >= east_val:
            return [
                {
                    "error": f"Invalid bbox: west ({west_val}) must be less than east ({east_val})"
                }
            ]
        if south_val >= north_val:
            return [
                {
                    "error": f"Invalid bbox: south ({south_val}) must be less than north ({north_val})"
                }
            ]

    try:
        # Get catalog client
        if CatalogRegistry:
            try:
                client = CatalogRegistry.get_client(catalog)
            except Exception as e:
                logger.warning(f"Failed to get catalog from registry: {e}")
                client = None
        else:
            client = None

        if not client:
            # Default catalog URLs
            catalog_urls = {
                "microsoft-pc": "https://planetarycomputer.microsoft.com/api/stac/v1",
                "earth-search": "https://earth-search.aws.element84.com/v1",
                "usgs-landsat": "https://landsatlook.usgs.gov/stac-server",
                "cop-dem": "https://copernicus-dem-30m.s3.amazonaws.com/stac/catalog.json",
            }

            catalog_url = catalog_urls.get(catalog, catalog)
            if not catalog_url.startswith("http"):
                raise ValueError(
                    f"Unknown catalog '{catalog}'. Available: {list(catalog_urls.keys())}"
                )

            client = Client.open(catalog_url)

        # Build search parameters
        search_params = {}

        if bbox:
            search_params["bbox"] = bbox

        if datetime_range:
            search_params["datetime"] = datetime_range

        if collections:
            search_params["collections"] = collections

        search_params["limit"] = max_items

        # Perform search
        search = client.search(**search_params)

        # Convert results to list of dicts
        results = []
        item_count = 0

        for item in search.items():
            if item_count >= max_items:
                break

            # Check cloud cover filter if specified
            if max_cloud_cover is not None:
                cloud_cover = None
                if "eo:cloud_cover" in item.properties:
                    cloud_cover = item.properties["eo:cloud_cover"]
                elif "view:sun_elevation" in item.properties:
                    # Some catalogs use different property names
                    cloud_cover = item.properties.get("cloudy_pixel_percentage")

                if cloud_cover is not None and cloud_cover > max_cloud_cover:
                    continue

            # Extract metadata
            item_data = {
                "id": item.id,
                "datetime": item.datetime.isoformat() if item.datetime else None,
                "bbox": list(item.bbox) if item.bbox else None,
                "collection": item.collection_id,
                "geometry": item.geometry,
                "properties": item.properties,
                "assets": {},
            }

            # Extract asset information
            for asset_key, asset in item.assets.items():
                item_data["assets"][asset_key] = {
                    "href": asset.href,
                    "type": asset.media_type,
                    "title": asset.title,
                    "description": asset.description,
                    "roles": asset.roles if asset.roles else [],
                }

            # Find thumbnail
            thumbnail_url = None
            for asset_key in ["thumbnail", "preview", "overview"]:
                if asset_key in item.assets:
                    thumbnail_url = item.assets[asset_key].href
                    break

            item_data["thumbnail"] = thumbnail_url

            # Add cloud cover if available
            cloud_cover = item.properties.get("eo:cloud_cover")
            if cloud_cover is None:
                cloud_cover = item.properties.get("cloudy_pixel_percentage")
            item_data["cloud_cover"] = cloud_cover

            results.append(item_data)
            item_count += 1

        logger.info(f"Found {len(results)} STAC items for query: {query}")
        return results

    except Exception as e:
        logger.error(f"Error searching STAC catalog: {e}")
        return [{"error": str(e), "query": query, "catalog": catalog}]


@tool
def get_stac_collections(catalog: str = "microsoft-pc") -> List[Dict[str, Any]]:
    """Get available collections from a STAC catalog.

    Args:
        catalog: STAC catalog identifier

    Returns:
        List of collection metadata dictionaries with keys:
        - id: Collection identifier
        - title: Collection title
        - description: Collection description
        - extent: Spatial and temporal extent
        - license: License information
        - keywords: List of keywords

    Example:
        >>> collections = get_stac_collections("microsoft-pc")
    """
    try:
        # Get catalog client
        if CatalogRegistry:
            try:
                client = CatalogRegistry.get_client(catalog)
            except Exception:
                client = None
        else:
            client = None

        if not client:
            # Default catalog URLs
            catalog_urls = {
                "microsoft-pc": "https://planetarycomputer.microsoft.com/api/stac/v1",
                "earth-search": "https://earth-search.aws.element84.com/v1",
                "usgs-landsat": "https://landsatlook.usgs.gov/stac-server",
            }

            catalog_url = catalog_urls.get(catalog, catalog)
            client = Client.open(catalog_url)

        collections = []
        for collection in client.get_collections():
            collection_data = {
                "id": collection.id,
                "title": collection.title or collection.id,
                "description": collection.description or "",
                "keywords": collection.keywords or [],
                "license": collection.license,
                "extent": {
                    "spatial": (
                        collection.extent.spatial.bboxes if collection.extent else None
                    ),
                    "temporal": (
                        [
                            (
                                collection.extent.temporal.intervals[0][0].isoformat()
                                if collection.extent.temporal.intervals[0][0]
                                else None
                            ),
                            (
                                collection.extent.temporal.intervals[0][1].isoformat()
                                if collection.extent.temporal.intervals[0][1]
                                else None
                            ),
                        ]
                        if collection.extent and collection.extent.temporal.intervals
                        else None
                    ),
                },
            }
            collections.append(collection_data)

        return collections

    except Exception as e:
        logger.error(f"Error getting STAC collections: {e}")
        return [{"error": str(e), "catalog": catalog}]
