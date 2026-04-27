"""QGIS subagent — bound to a live ``QgisInterface`` and ``QgsProject``.

This subagent only exists when the runtime context includes a real (or
mocked) QGIS interface. Module import is safe outside QGIS because all
QGIS-specific imports happen lazily inside
:mod:`geoagent.tools.qgis`.
"""

from __future__ import annotations

from typing import Any, Optional


def qgis_subagent(iface: Any, project: Any = None) -> Optional[dict[str, Any]]:
    """Build the QGIS :class:`deepagents.SubAgent` spec.

    Args:
        iface: A QGIS ``QgisInterface`` instance (or
            :class:`geoagent.testing.MockQGISIface`). ``None`` returns
            ``None`` so the caller can skip the spec.
        project: Optional ``QgsProject`` instance.

    Returns:
        A subagent dict, or ``None`` when ``iface`` is ``None``.
    """
    if iface is None:
        return None

    from geoagent.core.prompts import QGIS_PROMPT
    from geoagent.tools.qgis import qgis_tools

    return {
        "name": "qgis",
        "description": (
            "Operate inside a running QGIS Python: list and inspect "
            "project layers, navigate the map canvas, add vector / "
            "raster data, run processing algorithms (with confirmation), "
            "and produce PyQGIS code snippets."
        ),
        "system_prompt": QGIS_PROMPT,
        "tools": qgis_tools(iface, project),
    }


__all__ = ["qgis_subagent"]
