"""Planner agent for parsing natural language queries into structured parameters."""

from datetime import datetime
from typing import Optional, Dict, Any, List, Union, Tuple
from enum import Enum

from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from .llm import get_default_llm


class Intent(str, Enum):
    """Supported query intents."""

    SEARCH = "search"
    ANALYZE = "analyze"
    VISUALIZE = "visualize"
    COMPARE = "compare"


class PlannerOutput(BaseModel):
    """Structured output from the planner agent."""

    intent: Intent = Field(description="The primary intent of the query")
    location: Optional[str] = Field(
        default=None,
        description="Location name (e.g. 'California') or bounding box as 'west,south,east,north'",
    )
    time_range: Optional[Tuple[str, str]] = Field(
        default=None, description="Start and end dates in YYYY-MM-DD format"
    )
    dataset: Optional[str] = Field(
        default=None,
        description="Collection name (e.g. 'sentinel-2-l2a') or description of desired dataset",
    )
    analysis_type: Optional[str] = Field(
        default=None,
        description="Type of analysis requested (e.g. 'ndvi', 'change_detection', 'time_series')",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Additional parameters specific to the query"
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

Common datasets and their collection names:
- Landsat: "landsat-c2-l2"
- Sentinel-2: "sentinel-2-l2a"
- MODIS: "modis-*" (various products)
- Sentinel-1: "sentinel-1-grd"

Analysis types include:
- Vegetation indices: "ndvi", "evi", "savi"
- Land cover: "land_cover", "classification"
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

Extract information accurately and conservatively. If something is unclear, leave it as None rather than guessing."""


class Planner:
    """Agent for parsing natural language queries into structured parameters."""

    def __init__(self, llm: Optional[BaseChatModel] = None):
        """
        Initialize the planner agent.

        Args:
            llm: Language model to use. Uses default if None.
        """
        self.llm = llm or get_default_llm(temperature=0.0)
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", "{query}")]
        )

        # Create the structured output chain
        self.chain = self.prompt | self.llm.with_structured_output(PlannerOutput)

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
        try:
            result = self.chain.invoke({"query": query})

            # Post-process and validate the result
            if isinstance(result, PlannerOutput):
                return result
            else:
                # Fallback if structured output fails
                raise ValueError("Failed to get structured output from LLM")

        except Exception as e:
            raise Exception(f"Failed to parse query: {str(e)}")

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
                    intent=Intent.SEARCH,
                    parameters={"error": str(e), "original_query": query},
                )
                results.append(fallback)

        return results


def create_planner(llm: Optional[BaseChatModel] = None) -> Planner:
    """
    Create a planner instance.

    Args:
        llm: Language model to use. Uses default if None.

    Returns:
        Configured Planner instance
    """
    return Planner(llm=llm)


def parse_query(query: str, llm: Optional[BaseChatModel] = None) -> PlannerOutput:
    """
    Convenience function to parse a single query.

    Args:
        query: Natural language query
        llm: Language model to use. Uses default if None.

    Returns:
        PlannerOutput with extracted information
    """
    planner = create_planner(llm=llm)
    return planner.parse_query(query)
