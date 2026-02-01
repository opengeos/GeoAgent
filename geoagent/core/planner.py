"""Planner agent for parsing natural language queries into structured parameters."""

from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from .llm import get_default_llm
from .models import PlannerOutput, Intent


class _PlannerLLMSchema(BaseModel):
    """Internal schema for LLM structured output parsing.

    Uses simple types that work well with LLM structured output,
    then converts to the canonical PlannerOutput model.
    """

    intent: Intent = Field(description="The primary intent of the query")
    location: Optional[str] = Field(
        default=None,
        description="Location name (e.g. 'California') or bounding box as 'west,south,east,north'",
    )
    time_range: Optional[List[str]] = Field(
        default=None,
        description="Start and end dates as a two-element list [YYYY-MM-DD, YYYY-MM-DD]",
    )
    dataset: Optional[str] = Field(
        default=None,
        description="Collection name (e.g. 'sentinel-2-l2a') or description of desired dataset",
    )
    analysis_type: Optional[str] = Field(
        default=None,
        description="Type of analysis requested (e.g. 'ndvi', 'change_detection', 'time_series')",
    )
    max_cloud_cover: Optional[int] = Field(
        default=None, description="Maximum cloud cover percentage (0-100)"
    )
    max_items: Optional[int] = Field(
        default=None, description="Maximum number of items to return"
    )


SYSTEM_PROMPT = """You are an expert at parsing natural language queries about Earth observation and geospatial data.

Extract structured information from user queries and return it in the specified format.

Intent mapping:
- SEARCH: Finding or discovering datasets, collections, or imagery
- ANALYZE: Computing indices, statistics, or performing analysis on data
- VISUALIZE: Creating maps, plots, or visual representations
- COMPARE: Comparing different time periods, locations, or datasets

Location can be:
- Named places: "California", "Amazon rainforest", "Lagos Nigeria"
- Bounding box coordinates: "west,south,east,north" (e.g. "-120.5,34.0,-118.0,35.5")

Time ranges should be converted to YYYY-MM-DD format:
- "summer 2023" -> ("2023-06-01", "2023-08-31")
- "last year" -> ("2022-01-01", "2022-12-31")
- "March 2024" -> ("2024-03-01", "2024-03-31")

Choose the most appropriate collection from the catalog list below.
Set the `dataset` field to the exact collection ID.
If no collection fits, leave dataset as None.

{collections}

CRITICAL RULES:
- Only use "sentinel-2-l2a" when the user explicitly asks for satellite imagery, spectral indices (NDVI, EVI), or Sentinel-2
- Do NOT set analysis_type to "ndvi" unless the user specifically asks for NDVI or vegetation index
- For land cover queries, set analysis_type to "land_cover"
- For DEM/elevation queries, set analysis_type to "elevation"

Analysis types include:
- Vegetation indices: "ndvi", "evi", "savi"
- Land cover: "land_cover"
- Elevation / DEM: "elevation"
- Change detection: "change_detection"
- Time series: "time_series"
- Water indices: "ndwi", "mndwi"

Additional parameters can include:
- Cloud cover thresholds
- Spatial resolution requirements
- Specific bands or wavelengths
- Analysis parameters

Examples:

Query: "Show NDVI for California in summer 2023"
Output: {{
    "intent": "analyze",
    "location": "California",
    "time_range": ["2023-06-01", "2023-08-31"],
    "dataset": "sentinel-2-l2a",
    "analysis_type": "ndvi"
}}

Query: "Find Landsat images of the Amazon with less than 10% cloud cover"
Output: {{
    "intent": "search",
    "location": "Amazon rainforest",
    "dataset": "landsat-c2-l2",
    "parameters": {{"cloud_cover": 10}}
}}

Query: "Compare forest cover between 2020 and 2024 in Brazil"
Output: {{
    "intent": "compare",
    "location": "Brazil",
    "time_range": ["2020-01-01", "2024-12-31"],
    "analysis_type": "land_cover",
    "parameters": {{"comparison_type": "temporal"}}
}}

Query: "Show land cover for California"
Output: {{
    "intent": "visualize",
    "location": "California",
    "dataset": "io-lulc-9-class",
    "analysis_type": "land_cover"
}}

Query: "Show DEM for Yellowstone"
Output: {{
    "intent": "visualize",
    "location": "Yellowstone",
    "dataset": "cop-dem-glo-30",
    "analysis_type": "elevation"
}}

Query: "Display elevation map of the Grand Canyon"
Output: {{
    "intent": "visualize",
    "location": "Grand Canyon",
    "dataset": "cop-dem-glo-30",
    "analysis_type": "elevation"
}}

Query: "Show land use in Tokyo"
Output: {{
    "intent": "visualize",
    "location": "Tokyo",
    "dataset": "io-lulc-9-class",
    "analysis_type": "land_cover"
}}

Extract information accurately and conservatively. If something is unclear, leave it as None rather than guessing."""


class Planner:
    """Agent for parsing natural language queries into structured parameters."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        collections: Optional[List[Dict[str, str]]] = None,
    ):
        """
        Initialize the planner agent.

        Args:
            llm: Language model to use. Uses default if None.
        """
        self.llm = llm or get_default_llm(temperature=0.0)

        # Format collections into a readable list for the system prompt
        collections_text = ""
        if collections:
            lines = ["Available collections in the STAC catalog:"]
            for c in collections:
                cid = c.get("id", "")
                title = c.get("title", "")
                if title and title != cid:
                    lines.append(f"- {cid}: {title}")
                else:
                    lines.append(f"- {cid}")
            collections_text = "\n".join(lines)

        # Use replace instead of format to avoid conflicts with {{ }} in examples
        system_prompt = SYSTEM_PROMPT.replace("{collections}", collections_text)
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{query}")]
        )

        # Build structured output chains — try strict first, json_mode as fallback
        self._chain_strict = None
        self._chain_json = None
        try:
            self._chain_strict = self.prompt | self.llm.with_structured_output(
                _PlannerLLMSchema
            )
        except Exception:
            pass
        try:
            self._chain_json = self.prompt | self.llm.with_structured_output(
                _PlannerLLMSchema, method="json_mode"
            )
        except Exception:
            pass

    @staticmethod
    def _convert_to_planner_output(result: _PlannerLLMSchema) -> PlannerOutput:
        """Convert LLM schema output to the canonical PlannerOutput model."""
        location = None
        if result.location:
            try:
                parts = [float(x) for x in result.location.split(",")]
                if len(parts) == 4:
                    location = {"bbox": parts}
                else:
                    location = {"name": result.location}
            except ValueError:
                location = {"name": result.location}

        time_range = None
        if result.time_range:
            time_range = {
                "start_date": result.time_range[0],
                "end_date": result.time_range[1],
            }

        # Build parameters dict from explicit fields
        parameters: Dict[str, Any] = {}
        if result.max_cloud_cover is not None:
            parameters["max_cloud_cover"] = result.max_cloud_cover
        if result.max_items is not None:
            parameters["max_items"] = result.max_items

        return PlannerOutput(
            intent=result.intent.value,
            location=location,
            time_range=time_range,
            dataset=result.dataset,
            analysis_type=result.analysis_type,
            parameters=parameters,
            confidence=1.0,
        )

    def parse_query(self, query: str) -> PlannerOutput:
        """
        Parse a natural language query into structured parameters.

        Args:
            query: Natural language query about Earth observation data

        Returns:
            PlannerOutput with extracted structured information

        Raises:
            Exception: If LLM fails to parse the query
        """
        last_err = None
        for chain in (self._chain_strict, self._chain_json):
            if chain is None:
                continue
            try:
                result = chain.invoke({"query": query})
                if isinstance(result, _PlannerLLMSchema):
                    return self._convert_to_planner_output(result)
            except Exception as e:
                last_err = e
                logger.debug(f"Structured output attempt failed: {e}")
                continue

        raise Exception(
            f"Failed to parse query: {last_err or 'no structured output chain available'}"
        )

    def parse_batch(self, queries: List[str]) -> List[PlannerOutput]:
        """
        Parse multiple queries in batch.

        Args:
            queries: List of natural language queries

        Returns:
            List of PlannerOutput objects
        """
        results = []
        for query in queries:
            try:
                result = self.parse_query(query)
                results.append(result)
            except Exception as e:
                # Create a minimal output for failed queries
                fallback = PlannerOutput(
                    intent=Intent.SEARCH.value,
                    parameters={"error": str(e), "original_query": query},
                )
                results.append(fallback)

        return results


def create_planner(
    llm: Optional[BaseChatModel] = None,
    collections: Optional[List[Dict[str, str]]] = None,
) -> Planner:
    """
    Create a planner instance.

    Args:
        llm: Language model to use. Uses default if None.

    Returns:
        Configured Planner instance
    """
    return Planner(llm=llm, collections=collections)


def parse_query(
    query: str,
    llm: Optional[BaseChatModel] = None,
    collections: Optional[List[Dict[str, str]]] = None,
) -> PlannerOutput:
    """
    Convenience function to parse a single query.

    Args:
        query: Natural language query
        llm: Language model to use. Uses default if None.

    Returns:
        PlannerOutput with extracted information
    """
    planner = create_planner(llm=llm, collections=collections)
    return planner.parse_query(query)
