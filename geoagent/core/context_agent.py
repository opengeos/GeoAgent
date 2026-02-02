"""Context Agent for answering general and earth-science questions via LLM.

The ContextAgent is the conversational backbone of GeoAgent.  Any query that
asks for *information or explanation* rather than *data retrieval or
visualization* is routed here.  It can answer questions about anything —
earth science, programming, history, etc. — while naturally weaving in
geospatial context when relevant.
"""

from typing import Any, Dict, Optional
import logging

from langchain_core.prompts import ChatPromptTemplate

from .models import PlannerOutput, DataResult, AnalysisResult

logger = logging.getLogger(__name__)

CONTEXT_SYSTEM_PROMPT = """You are GeoAgent, an AI-powered geospatial assistant. \
You can answer any question, but you specialize in earth science, geospatial \
analysis, and environmental topics.

For earth science and geospatial questions:
- Provide accurate, scientific information
- Reference specific datasets, satellites, or data sources when relevant
- Mention time periods and locations
- Suggest data visualizations when appropriate

For general questions:
- Answer helpfully and accurately
- If the question could benefit from geospatial data, mention that capability

Keep responses concise, clear, and informative."""


class ContextAgent:
    """Agent for answering general and contextual questions via LLM."""

    def __init__(self, llm: Any):
        """Initialize the Context Agent.

        Args:
            llm: Language model instance for generating answers.
        """
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", CONTEXT_SYSTEM_PROMPT), ("human", "{query}")]
        )
        self.chain = self.prompt | self.llm

    def answer(
        self,
        plan: PlannerOutput,
        data: Optional[DataResult] = None,
        *,
        query: Optional[str] = None,
    ) -> AnalysisResult:
        """Answer a question using LLM knowledge.

        Args:
            plan: The planner output containing parsed intent and context.
            data: Optional data result that may provide supporting information.
            query: The original user query.  When supplied this is sent to the
                LLM verbatim (with optional location / time enrichment).
                Falls back to ``plan.intent`` for backward compatibility.

        Returns:
            An :class:`AnalysisResult` whose ``result_data["answer"]``
            contains the generated answer text.
        """
        try:
            # Prefer the original query so the LLM sees the full question
            # rather than the single-word intent label ("explain").
            prompt_query = query or plan.intent

            # Enrich with location / time when available
            if plan.location:
                loc_name = plan.location.get("name", "")
                if loc_name:
                    prompt_query += f" (Location: {loc_name})"
            if plan.time_range:
                start = plan.time_range.get("start_date", "")
                end = plan.time_range.get("end_date", "")
                if start and end:
                    prompt_query += f" (Time period: {start} to {end})"

            response = self.chain.invoke({"query": prompt_query})
            answer_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            viz_hints: Dict[str, Any] = {}
            if data and data.total_items > 0:
                viz_hints = {
                    "type": "contextual",
                    "show_data": True,
                    "title": f"Context: {prompt_query[:50]}...",
                }

            result_data: Dict[str, Any] = {
                "analysis_type": "contextual",
                "answer": answer_text,
                "has_supporting_data": data is not None and data.total_items > 0,
                "location": plan.location,
                "time_range": plan.time_range,
            }

            # Escape triple-quotes inside the answer for safe embedding
            safe_answer = answer_text.replace('"""', r"\"\"\"")
            code = (
                f"# GeoAgent contextual answer\n"
                f"# Query: {prompt_query}\n"
                f'answer = """{safe_answer}"""\n'
                f"print(answer)\n"
            )

            return AnalysisResult(
                result_data=result_data,
                code_generated=code,
                visualization_hints=viz_hints,
                success=True,
            )
        except Exception as e:
            logger.error(f"Context agent failed: {e}")
            return AnalysisResult(
                result_data={"error": str(e)},
                code_generated=f"# Context query failed: {e}",
                success=False,
                error_message=str(e),
            )
