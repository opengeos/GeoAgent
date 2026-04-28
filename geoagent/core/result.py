"""Response object for :meth:`geoagent.GeoAgent.chat`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    pass


@dataclass
class GeoAgentResponse:
    """Normalized result after a GeoAgent conversational turn."""

    answer_text: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    execution_time: float = 0.0
    executed_tools: list[str] = field(default_factory=list)
    cancelled_tools: list[str] = field(default_factory=list)
    map: Any = None
    raw: Any = None
    """Underlying :class:`strands.agent.agent_result.AgentResult` when available."""
