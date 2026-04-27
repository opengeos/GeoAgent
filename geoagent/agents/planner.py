"""Planner subagent — LLM-only, no tools.

Decomposes a natural-language geospatial query into a structured plan
(intent / location / time range / dataset / analysis type) the
coordinator can use to delegate. Replaces the legacy
``geoagent/core/planner.py`` ``Planner`` class without re-introducing
its Pydantic ``PlannerOutput`` model — the coordinator consumes the
planner's output as free-form text now.
"""

from __future__ import annotations

from geoagent.core.prompts import PLANNER_PROMPT

PLANNER_SUBAGENT: dict = {
    "name": "planner",
    "description": (
        "Decompose a geospatial query into intent (search / analyze / "
        "visualize / compare / explain / monitor), location, time range, "
        "dataset hint, and analysis type. Returns a structured summary "
        "the coordinator uses to delegate to data, analysis, or mapping."
    ),
    "system_prompt": PLANNER_PROMPT,
    "tools": [],
}
"""Planner :class:`deepagents.SubAgent` spec — LLM-only, no tools."""


__all__ = ["PLANNER_SUBAGENT"]
