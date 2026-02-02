"""Context Agent for answering earth science questions using LLM knowledge."""

from typing import Any, Optional
import logging

from langchain_core.prompts import ChatPromptTemplate

from .models import PlannerOutput, DataResult, AnalysisResult

logger = logging.getLogger(__name__)

CONTEXT_SYSTEM_PROMPT = """You are an expert earth scientist and geospatial analyst.
Answer questions about earth science, natural disasters, climate, and environmental
phenomena.

When answering:
1. Provide accurate, scientific information
2. Reference specific datasets, satellites, or data sources when relevant
3. Mention time periods and locations specifically
4. Suggest data that could be visualized to support the answer
5. Be concise but thorough

Format your response with:
- A brief summary (2-3 sentences)
- Key findings or facts (bullet points)
- Relevant data sources for visualization
- Recommendations for further analysis"""


class ContextAgent:
    """Agent for answering contextual earth science questions."""

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
    ) -> AnalysisResult:
        """Answer a contextual earth-science question.

        Args:
            plan: The planner output containing the user's intent and context.
            data: Optional data result that may provide supporting information.

        Returns:
            An :class:`AnalysisResult` with the generated answer text and
            optional visualization hints.
        """
        try:
            query = plan.intent
            if plan.location:
                loc_name = plan.location.get("name", "")
                if loc_name:
                    query += f" (Location: {loc_name})"
            if plan.time_range:
                start = plan.time_range.get("start_date", "")
                end = plan.time_range.get("end_date", "")
                if start and end:
                    query += f" (Time period: {start} to {end})"

            response = self.chain.invoke({"query": query})
            answer_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            viz_hints: dict[str, Any] = {}
            if data and data.total_items > 0:
                viz_hints = {
                    "type": "contextual",
                    "show_data": True,
                    "title": f"Context: {plan.intent[:50]}...",
                }

            result_data = {
                "analysis_type": "contextual",
                "answer": answer_text,
                "has_supporting_data": data is not None and data.total_items > 0,
                "location": plan.location,
                "time_range": plan.time_range,
            }

            code = (
                f"# Contextual Earth Science Analysis\n"
                f"# Query: {plan.intent}\n"
                f'answer = """{answer_text}"""\n'
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
