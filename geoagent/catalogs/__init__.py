"""STAC catalog registry and management."""

from .registry import (
    CatalogRegistry,
    CatalogInfo,
    get_registry,
    list_catalogs,
    get_catalog_client,
    BUILTIN_CATALOGS,
    DEFAULT_CATALOG,
)

__all__ = [
    "CatalogRegistry",
    "CatalogInfo",
    "get_registry",
    "list_catalogs",
    "get_catalog_client",
    "BUILTIN_CATALOGS",
    "DEFAULT_CATALOG",
]
