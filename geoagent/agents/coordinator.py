"""Coordinator helper — assemble the active subagent list for a run.

The coordinator's *system prompt* lives in
:data:`geoagent.core.prompts.COORDINATOR_PROMPT` and is wired into the
top-level deepagents agent by :func:`geoagent.core.factory.create_geo_agent`.

This module exposes :func:`default_subagents`, which the factory calls
with the active :class:`geoagent.core.context.GeoAgentContext` to build
the list of declarative ``SubAgent`` dicts. Subagents are added based on
runtime state: planner / data / analysis / context always run; mapping
appears when a live map widget is provided; qgis when an iface is
provided; geoai and earthdata when their optional packages are
importable.
"""

from __future__ import annotations

from typing import Any

from geoagent.core.context import GeoAgentContext

from .analysis import analysis_subagent
from .context import context_subagent
from .data import data_subagent
from .earthdata import earthdata_subagent
from .geoai import geoai_subagent
from .mapping import mapping_subagent
from .planner import PLANNER_SUBAGENT
from .qgis import qgis_subagent


def default_subagents(ctx: GeoAgentContext) -> list[dict[str, Any]]:
    """Return the active subagent list for ``ctx``.

    Args:
        ctx: The runtime context. Inspecting ``ctx.map_obj`` and
            ``ctx.qgis_iface`` decides which runtime-bound subagents to
            include.

    Returns:
        A list of declarative ``SubAgent`` dicts ready to pass to
        :func:`deepagents.create_deep_agent` as ``subagents=``.
    """
    subagents: list[dict[str, Any]] = [
        PLANNER_SUBAGENT,
        data_subagent(),
        analysis_subagent(),
        context_subagent(),
    ]
    mapping = mapping_subagent(ctx.map_obj)
    if mapping is not None:
        subagents.append(mapping)
    qgis = qgis_subagent(ctx.qgis_iface, ctx.qgis_project)
    if qgis is not None:
        subagents.append(qgis)
    for builder in (geoai_subagent, earthdata_subagent):
        sub = builder()
        if sub is not None:
            subagents.append(sub)
    return subagents


__all__ = ["default_subagents"]
