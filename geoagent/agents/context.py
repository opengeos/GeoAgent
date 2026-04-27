"""Context subagent — conversational Q&A.

Replaces the legacy ``geoagent/core/context_agent.py`` ``ContextAgent``
class. EXPLAIN-style queries (``what is X?``, ``why does Y happen?``)
are answered here without retrieving or rendering data.
"""

from __future__ import annotations

from typing import Any


def context_subagent() -> dict[str, Any]:
    """Build the Context :class:`deepagents.SubAgent` spec.

    Returns:
        A subagent dict with no tools — answers from LLM knowledge alone.
    """
    from geoagent.core.prompts import CONTEXT_PROMPT

    return {
        "name": "context",
        "description": (
            "Answer EXPLAIN-style geospatial / earth-science questions "
            "in prose. No data retrieval or map rendering."
        ),
        "system_prompt": CONTEXT_PROMPT,
        "tools": [],
    }


__all__ = ["context_subagent"]
