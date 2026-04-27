"""Data subagent — STAC search and DuckDB spatial SQL.

Replaces the legacy ``geoagent/core/data_agent.py`` ``DataAgent`` class
with a deepagents subagent whose tool set is the union of STAC and
DuckDB tool factories.
"""

from __future__ import annotations

from typing import Any


def data_subagent() -> dict[str, Any]:
    """Build the Data :class:`deepagents.SubAgent` spec.

    Returns:
        A subagent dict with ``name``, ``description``, ``system_prompt``,
        and a ``tools`` list combining STAC search and DuckDB spatial SQL
        tools. Tools whose required packages (pystac_client, duckdb) are
        unavailable are silently dropped by the registry.
    """
    from geoagent.core.prompts import DATA_PROMPT
    from geoagent.tools.data.duckdb import duckdb_tools
    from geoagent.tools.stac import stac_tools

    return {
        "name": "data",
        "description": (
            "Search STAC catalogs (Planetary Computer, Earth Search, "
            "USGS, NASA CMR) and DuckDB spatial sources (Overture, "
            "GeoParquet) for the requested datasets. Reports URLs and "
            "metadata; does not download large rasters."
        ),
        "system_prompt": DATA_PROMPT,
        "tools": stac_tools() + duckdb_tools(),
    }


__all__ = ["data_subagent"]
