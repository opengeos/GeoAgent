"""DuckDB spatial SQL tools.

Wraps the v0.x DuckDB tools (:mod:`geoagent.core.tools.duckdb_tool`) with
GeoAgent metadata.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from geoagent.tools.stac import _stamp


def duckdb_tools() -> list[BaseTool]:
    """Build the DuckDB spatial SQL tool set.

    Returns:
        ``query_spatial_data``, ``query_overture``, ``analyze_spatial_data``.
    """
    try:
        from geoagent.core.tools.duckdb_tool import (
            query_spatial_data,
            query_overture,
            analyze_spatial_data,
        )
    except ImportError:
        return []

    requirements = ["duckdb"]
    tools = (query_spatial_data, query_overture, analyze_spatial_data)
    for tool in tools:
        _stamp(
            tool,
            category="data",
            requires_confirmation=False,
            requires_packages=requirements,
        )
    return list(tools)


__all__ = ["duckdb_tools"]
