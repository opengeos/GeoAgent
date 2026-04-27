"""Analysis subagent — raster + vector analysis with code generation.

Replaces the legacy 1541-line ``geoagent/core/analysis_agent.py``
``AnalysisAgent`` class. The handler patterns (NDVI/EVI/SAVI, zonal
stats, change detection, time series, land_cover/elevation/water_mapping/
fire_detection/snow_cover/surface_temperature/event_impact) live in
:data:`geoagent.core.prompts.ANALYSIS_PROMPT`; the actual computation is
performed by the raster and vector tool factories.
"""

from __future__ import annotations

from typing import Any


def analysis_subagent() -> dict[str, Any]:
    """Build the Analysis :class:`deepagents.SubAgent` spec.

    Returns:
        A subagent dict combining the raster and vector tool factories.
    """
    from geoagent.core.prompts import ANALYSIS_PROMPT
    from geoagent.tools.data.raster import raster_tools
    from geoagent.tools.data.vector import vector_tools

    return {
        "name": "analysis",
        "description": (
            "Compute spectral indices (NDVI, EVI, SAVI, NDWI), zonal "
            "statistics, change detection, time-series summaries, and "
            "vector operations (buffer, spatial join, filter, geometry "
            "summaries). Generates reproducible Python code for every "
            "computation."
        ),
        "system_prompt": ANALYSIS_PROMPT,
        "tools": raster_tools() + vector_tools(),
    }


__all__ = ["analysis_subagent"]
