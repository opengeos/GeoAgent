"""Public response object returned from :meth:`GeoAgent.chat`.

The field names are kept compatible with the v0.x ``GeoAgentResponse`` type
so downstream code holding the dataclass continues to work, but the fields
are no longer typed against the deleted v0.x ``PlannerOutput`` /
``DataResult`` / ``AnalysisResult`` models. Callers should treat ``plan``,
``data``, and ``analysis`` as opaque dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class GeoAgentResponse:
    """Result of a single :meth:`GeoAgent.chat` call.

    Attributes:
        plan: Structured plan produced by the planner subagent, if any.
        data: Data-fetch result (e.g. STAC items, GeoDataFrame metadata).
        analysis: Analysis output (e.g. zonal stats, computed indices).
        map: The interactive map widget rendered for this query, when
            applicable. May be the same object passed in as ``target_map``.
        code: Python code generated during the run for reproducibility.
        answer_text: A plain-text answer for explanatory queries that did
            not produce structured data.
        success: Whether the query completed without error.
        error_message: Error string when ``success`` is ``False``.
        execution_time: Wall-clock seconds spent in the agent run.
        executed_tools: Names of tools the agent actually called, in order.
        cancelled_tools: Names of tools the user cancelled via the
            :class:`ConfirmCallback`.
        messages: The raw deepagents message log (LangChain messages) for
            advanced inspection. Optional and may be omitted for brevity.
    """

    plan: Any = None
    data: Any = None
    analysis: Any = None
    map: Any = None
    code: str = ""
    answer_text: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    execution_time: float = 0.0
    executed_tools: list[str] = field(default_factory=list)
    cancelled_tools: list[str] = field(default_factory=list)
    messages: Optional[list[Any]] = None
