"""STAC search tool, re-exported with GeoAgent metadata.

This is a thin wrapper around the v0.x ``geoagent.core.tools.stac`` module
that stamps :func:`geoagent.core.decorators.geo_tool` metadata onto the
existing LangChain tools so they participate in the new registry, safety,
and category-based filtering machinery.

The underlying implementation (pystac-client + the catalog registry) is
unchanged.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from geoagent.core.decorators import stamp_geo_meta


def stac_tools() -> list[BaseTool]:
    """Build the STAC tool set.

    Returns:
        A list of LangChain ``BaseTool`` instances exposing ``search_stac``
        and ``get_stac_collections``. Empty when ``pystac_client`` is
        unavailable.
    """
    try:
        from geoagent.core.tools.stac import search_stac, get_stac_collections
    except ImportError:
        return []

    stamp_geo_meta(
        search_stac,
        category="data",
        requires_confirmation=False,
        requires_packages=["pystac_client"],
    )
    stamp_geo_meta(
        get_stac_collections,
        category="data",
        requires_confirmation=False,
        requires_packages=["pystac_client"],
    )

    return [search_stac, get_stac_collections]


__all__ = ["stac_tools"]
