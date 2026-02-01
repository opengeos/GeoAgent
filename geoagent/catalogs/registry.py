"""STAC catalog registry with pre-configured endpoints and extensibility."""

import os
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

import pystac_client


logger = logging.getLogger(__name__)


@dataclass
class CatalogInfo:
    """Information about a STAC catalog."""

    name: str
    url: str
    description: str
    requires_auth: bool = False
    auth_env_var: Optional[str] = None


# Pre-configured STAC catalogs
BUILTIN_CATALOGS = {
    "earth_search": CatalogInfo(
        name="earth_search",
        url="https://earth-search.aws.element84.com/v1",
        description="AWS Earth Search - Public Landsat and Sentinel data",
        requires_auth=False,
    ),
    "planetary_computer": CatalogInfo(
        name="planetary_computer",
        url="https://planetarycomputer.microsoft.com/api/stac/v1",
        description="Microsoft Planetary Computer - Comprehensive Earth observation data",
        requires_auth=False,  # Public access, but signing may improve performance
    ),
    "usgs": CatalogInfo(
        name="usgs",
        url="https://landsatlook.usgs.gov/stac-server",
        description="USGS Landsat Look - Official Landsat archive",
        requires_auth=False,
    ),
    "nasa_cmr": CatalogInfo(
        name="nasa_cmr",
        url="https://cmr.earthdata.nasa.gov/stac",
        description="NASA CMR-STAC - NASA Earth observation data",
        requires_auth=True,
        auth_env_var="NASA_EARTHDATA_TOKEN",
    ),
}

# Default catalog to use when none specified
DEFAULT_CATALOG = "earth_search"


class CatalogRegistry:
    """Registry for STAC catalogs with built-in and custom endpoints."""

    def __init__(self):
        """Initialize the catalog registry with built-in catalogs."""
        self._catalogs: Dict[str, CatalogInfo] = BUILTIN_CATALOGS.copy()
        # Cache for catalog collections: {catalog_name: [{"id": str, "title": str}, ...]}
        # Cached per session to avoid repeated network calls
        self.__class__._collection_cache = getattr(
            self.__class__, "_collection_cache", {}
        )

    def list_catalogs(self) -> List[CatalogInfo]:
        """
        List all available catalogs.

        Returns:
            List of CatalogInfo objects
        """
        return list(self._catalogs.values())

    def get_catalog(self, name: str) -> Optional[CatalogInfo]:
        """
        Get catalog information by name.

        Args:
            name: Catalog name

        Returns:
            CatalogInfo object or None if not found
        """
        return self._catalogs.get(name)

    def add_catalog(
        self,
        name: str,
        url: str,
        description: str,
        requires_auth: bool = False,
        auth_env_var: Optional[str] = None,
    ) -> None:
        """
        Add a custom catalog to the registry.

        Args:
            name: Unique catalog name
            url: STAC API endpoint URL
            description: Human-readable description
            requires_auth: Whether authentication is required
            auth_env_var: Environment variable name for auth token
        """
        self._catalogs[name] = CatalogInfo(
            name=name,
            url=url,
            description=description,
            requires_auth=requires_auth,
            auth_env_var=auth_env_var,
        )
        logger.info(f"Added catalog '{name}' to registry")

    def get_client(self, name: Optional[str] = None) -> pystac_client.Client:
        """
        Get a pystac_client.Client for the specified catalog.

        Args:
            name: Catalog name. Uses default if None.

        Returns:
            Configured pystac_client.Client instance

        Raises:
            ValueError: If catalog not found
            RuntimeError: If authentication required but credentials missing
        """
        if name is None:
            name = DEFAULT_CATALOG

        catalog = self.get_catalog(name)
        if not catalog:
            available = ", ".join(self._catalogs.keys())
            raise ValueError(f"Catalog '{name}' not found. Available: {available}")

        # Check authentication
        if catalog.requires_auth and catalog.auth_env_var:
            if not os.getenv(catalog.auth_env_var):
                raise RuntimeError(
                    f"Catalog '{name}' requires authentication. "
                    f"Set environment variable: {catalog.auth_env_var}"
                )

        client_kwargs = {}

        # Handle authentication for specific catalogs
        if name == "nasa_cmr" and catalog.auth_env_var:
            token = os.getenv(catalog.auth_env_var)
            if token:
                client_kwargs["headers"] = {"Authorization": f"Bearer {token}"}

        # Create the client
        try:
            client = pystac_client.Client.open(catalog.url, **client_kwargs)

            # Special handling for Planetary Computer signing
            if name == "planetary_computer":
                try:
                    import planetary_computer

                    # Wrap the client with signing capability
                    original_search = client.search

                    def signed_search(*args, **kwargs):
                        """Search wrapper that signs items for access."""
                        search_result = original_search(*args, **kwargs)
                        # Sign the items in the search result for direct access
                        search_result.sign = lambda: planetary_computer.sign(
                            search_result
                        )
                        return search_result

                    client.search = signed_search
                    logger.info("Planetary Computer signing enabled")

                except ImportError:
                    logger.warning(
                        "planetary-computer package not available. "
                        "Some data access may be slower. "
                        "Install with: pip install planetary-computer"
                    )

            return client

        except Exception as e:
            raise RuntimeError(f"Failed to connect to catalog '{name}': {str(e)}")

    def get_collection_index(
        self, catalog_name: str = "planetary_computer"
    ) -> List[Dict[str, str]]:
        """Fetch and cache a simple index of collections for a catalog.

        The index contains dicts with collection id and title, e.g.:
        [{"id": "sentinel-2-l2a", "title": "Sentinel-2 Level-2A"}, ...]

        Results are cached in-memory for the duration of the session.

        Args:
            catalog_name: Name of the catalog as registered in this registry

        Returns:
            List of dicts with keys "id" and "title"
        """
        # Return cached if available
        cache = self.__class__._collection_cache
        if catalog_name in cache:
            return cache[catalog_name]

        client = self.get_client(catalog_name)
        collections: List[Dict[str, str]] = []
        try:
            for col in client.get_collections():
                col_id = getattr(col, "id", None) or col.to_dict().get("id")
                title = (
                    getattr(col, "title", None) or col.to_dict().get("title") or col_id
                )
                if col_id:
                    collections.append({"id": col_id, "title": title})
        except Exception as e:
            raise RuntimeError(f"Failed to list collections for '{catalog_name}': {e}")

        # Cache and return
        cache[catalog_name] = collections
        return collections

    def remove_catalog(self, name: str) -> bool:
        """
        Remove a catalog from the registry.

        Args:
            name: Catalog name to remove

        Returns:
            True if catalog was removed, False if not found

        Note:
            Built-in catalogs cannot be removed, only custom ones
        """
        if name in BUILTIN_CATALOGS:
            logger.warning(f"Cannot remove built-in catalog '{name}'")
            return False

        if name in self._catalogs:
            del self._catalogs[name]
            logger.info(f"Removed catalog '{name}' from registry")
            return True

        return False


# Global registry instance
_global_registry = CatalogRegistry()


def get_registry() -> CatalogRegistry:
    """Get the global catalog registry instance."""
    return _global_registry


def list_catalogs() -> List[CatalogInfo]:
    """List all available catalogs from the global registry."""
    return _global_registry.list_catalogs()


def get_catalog_client(name: Optional[str] = None) -> pystac_client.Client:
    """
    Get a STAC client for the specified catalog.

    Args:
        name: Catalog name. Uses default if None.

    Returns:
        Configured pystac_client.Client instance
    """
    return _global_registry.get_client(name)


def get_collection_index(catalog_name: str = "planetary_computer") -> list:
    """Convenience accessor for a catalog's collection index."""
    return _global_registry.get_collection_index(catalog_name)
