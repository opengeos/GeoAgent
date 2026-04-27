"""STAC search tool, re-exported with GeoAgent metadata.

This is a thin wrapper around the v0.x ``geoagent.core.tools.stac`` module
that stamps :func:`geoagent.core.decorators.geo_tool` metadata onto the
existing LangChain tools so they participate in the new registry, safety,
and category-based filtering machinery.

The underlying implementation (pystac-client + the catalog registry) is
unchanged.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from geoagent.core.decorators import GEO_META_KEY


def _stamp(tool: BaseTool, **geo_meta: Any) -> BaseTool:
    """Stamp GeoAgent metadata onto an already-built LangChain tool.

    Args:
        tool: A LangChain ``BaseTool``.
        **geo_meta: Keys to merge into ``tool.metadata["geo"]``.

    Returns:
        The same tool, with metadata mutated.
    """
    meta = dict(tool.metadata or {})
    geo = dict(meta.get(GEO_META_KEY, {}))
    geo.update(geo_meta)
    meta[GEO_META_KEY] = geo
    tool.metadata = meta
    return tool


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

    _stamp(
        search_stac,
        category="data",
        requires_confirmation=False,
        requires_packages=["pystac_client"],
    )
    _stamp(
        get_stac_collections,
        category="data",
        requires_confirmation=False,
        requires_packages=["pystac_client"],
    )

    return [search_stac, get_stac_collections]


__all__ = ["stac_tools"]
