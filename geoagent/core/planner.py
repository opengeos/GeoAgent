"""Planner agent for parsing natural language queries into structured parameters."""

from typing import Optional, Dict, Any, List
import logging

from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from .llm import get_default_llm
from .models import PlannerOutput, Intent

logger = logging.getLogger(__name__)

# Comprehensive mapping from query keywords/topics to Planetary Computer
# collection IDs.  Used by the Planner as a fallback when the LLM does not
# set a dataset explicitly.
COLLECTION_MAPPING: Dict[str, str] = {
    # Optical Imagery
    "sentinel-2": "sentinel-2-l2a",
    "sentinel 2": "sentinel-2-l2a",
    "landsat": "landsat-c2-l2",
    "hls": "hls-l30",
    "naip": "naip",
    "aster": "aster-l1t",
    # SAR / Radar
    "sentinel-1": "sentinel-1-grd",
    "sentinel 1": "sentinel-1-grd",
    "radar": "sentinel-1-grd",
    "sar": "sentinel-1-grd",
    "sentinel-1 rtc": "sentinel-1-rtc",
    "alos palsar": "alos-dem",
    # Elevation / DEM / Terrain
    "elevation": "cop-dem-glo-30",
    "dem": "cop-dem-glo-30",
    "terrain": "cop-dem-glo-30",
    "copernicus dem": "cop-dem-glo-30",
    "nasadem": "nasadem",
    "alos dem": "alos-dem",
    "alos world": "alos-dem",
    "lidar": "3dep-lidar-dsm",
    "lidar height": "3dep-lidar-hag",
    "3dep": "3dep-lidar-dsm",
    # Land Cover
    "land cover": "io-lulc-9-class",
    "land use": "io-lulc-9-class",
    "lulc": "io-lulc-9-class",
    "cropland": "usda-cdl",
    "crop": "usda-cdl",
    # Vegetation
    "ndvi": "sentinel-2-l2a",
    "vegetation index": "modis-13Q1-061",
    "vegetation indices": "modis-13Q1-061",
    "modis vegetation": "modis-13Q1-061",
    "evi": "modis-13Q1-061",
    "leaf area": "modis-15A2H-061",
    "lai": "modis-15A2H-061",
    "net production": "modis-17A2H-061",
    "npp": "modis-17A2H-061",
    "gross primary": "modis-17A2H-061",
    "gpp": "modis-17A2H-061",
    # Water
    "surface water": "jrc-gsw",
    "water": "jrc-gsw",
    "flood": "sentinel-1-grd",
    "ndwi": "sentinel-2-l2a",
    "mndwi": "sentinel-2-l2a",
    # Fire / Thermal
    "fire": "modis-14A1-061",
    "wildfire": "modis-14A1-061",
    "thermal anomaly": "modis-14A1-061",
    "thermal anomalies": "modis-14A1-061",
    "burn": "modis-14A1-061",
    "burned area": "modis-14A1-061",
    "burn severity": "modis-14A1-061",
    # Temperature
    "temperature": "modis-11A1-061",
    "surface temperature": "modis-11A1-061",
    "land surface temperature": "modis-11A1-061",
    "lst": "modis-11A1-061",
    "sea surface temperature": "modis-11A1-061",
    "sst": "modis-11A1-061",
    # Snow / Ice
    "snow": "modis-10A1-061",
    "snow cover": "modis-10A1-061",
    "ice": "modis-10A1-061",
    # Atmosphere
    "albedo": "modis-43A3-061",
    "brdf": "modis-43A4-061",
    "nadir brdf": "modis-43A4-061",
    "reflectance": "modis-09A1-061",
    "surface reflectance": "modis-09A1-061",
    # Other
    "nightlight": "viirs-nighttime-lights",
    "night light": "viirs-nighttime-lights",
    "population": "gridded-pop",
    "buildings": "ms-buildings",
}


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
- EXPLAIN: Answering ANY question that asks for information or explanation rather than data retrieval/visualization. This includes earth science questions, general knowledge, greetings, coding questions, definitions, how-things-work, opinions, etc.
- MONITOR: Tracking ongoing events like wildfires, floods, deforestation over time

IMPORTANT — choosing between EXPLAIN and data intents:
- If the user is asking a QUESTION (what, why, how, explain, describe, tell me about, etc.) → EXPLAIN
- If the user wants to SEE, SHOW, MAP, FIND, COMPUTE, DISPLAY, or DOWNLOAD actual data → SEARCH/ANALYZE/VISUALIZE/COMPARE
- Conversational messages (greetings, jokes, general chat) → EXPLAIN
- "What is NDVI?" → EXPLAIN (asking for information)
- "Show NDVI for California" → ANALYZE (requesting data computation)

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

Collection mapping guidance (use these when the query mentions a topic but
not a specific collection):
- Surface water / flood mapping → "jrc-gsw" or "sentinel-1-grd"
- Fire / wildfire / burn / thermal anomaly → "modis-14A1-061"
- Snow / ice cover → "modis-10A1-061"
- Surface temperature / LST / SST → "modis-11A1-061"
- Vegetation indices (MODIS) → "modis-13Q1-061"
- Leaf area index → "modis-15A2H-061"
- Net primary production / GPP → "modis-17A2H-061"
- Nighttime lights → "viirs-nighttime-lights"
- Cropland / crop type → "usda-cdl"
- Population density → "gridded-pop"
- Building footprints → "ms-buildings"
- LIDAR / 3DEP → "3dep-lidar-dsm"
- NAIP aerial imagery → "naip"
- HLS (Harmonized Landsat Sentinel) → "hls-l30"

CRITICAL RULES:
- Only use "sentinel-2-l2a" when the user explicitly asks for satellite imagery, spectral indices (NDVI, EVI), or Sentinel-2
- Do NOT set analysis_type to "ndvi" unless the user specifically asks for NDVI or vegetation index
- For land cover queries, set analysis_type to "land_cover"
- For DEM/elevation queries, set analysis_type to "elevation"
- For water mapping, set analysis_type to "water_mapping"
- For fire/burn detection, set analysis_type to "fire_detection"
- For snow/ice queries, set analysis_type to "snow_cover"
- For temperature queries, set analysis_type to "surface_temperature"
- For disaster impact assessment, set analysis_type to "event_impact"
- For contextual questions ("why", "explain", "what causes"), use EXPLAIN intent
- For monitoring queries ("track", "monitor", "ongoing"), use MONITOR intent

Analysis types include:
- Vegetation indices: "ndvi", "evi", "savi"
- Land cover: "land_cover"
- Elevation / DEM: "elevation"
- Change detection: "change_detection"
- Time series: "time_series"
- Water indices: "ndwi", "mndwi"
- Water mapping: "water_mapping"
- Fire detection: "fire_detection"
- Snow cover: "snow_cover"
- Surface temperature: "surface_temperature"
- Event impact: "event_impact"

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

Query: "Show surface water changes in Lake Chad"
Output: {{
    "intent": "analyze",
    "location": "Lake Chad",
    "dataset": "jrc-gsw",
    "analysis_type": "water_mapping"
}}

Query: "Detect active fires in Australia in January 2020"
Output: {{
    "intent": "analyze",
    "location": "Australia",
    "time_range": ["2020-01-01", "2020-01-31"],
    "dataset": "modis-14A1-061",
    "analysis_type": "fire_detection"
}}

Query: "Show snow cover in the Alps in winter 2023"
Output: {{
    "intent": "visualize",
    "location": "Alps",
    "time_range": ["2023-12-01", "2024-02-28"],
    "dataset": "modis-10A1-061",
    "analysis_type": "snow_cover"
}}

Query: "Map land surface temperature in Phoenix during summer 2024"
Output: {{
    "intent": "analyze",
    "location": "Phoenix",
    "time_range": ["2024-06-01", "2024-08-31"],
    "dataset": "modis-11A1-061",
    "analysis_type": "surface_temperature"
}}

Query: "Show nighttime lights of India"
Output: {{
    "intent": "visualize",
    "location": "India",
    "dataset": "viirs-nighttime-lights"
}}

Query: "What caused the 2023 Turkey earthquake and how did it affect the region?"
Output: {{
    "intent": "explain",
    "location": "Turkey",
    "time_range": ["2023-02-01", "2023-03-31"],
    "analysis_type": "event_impact"
}}

Query: "Monitor deforestation in the Amazon over the past 5 years"
Output: {{
    "intent": "monitor",
    "location": "Amazon rainforest",
    "time_range": ["2019-01-01", "2024-12-31"],
    "dataset": "sentinel-2-l2a",
    "analysis_type": "change_detection"
}}

Query: "Assess the impact of Hurricane Ian on Florida"
Output: {{
    "intent": "analyze",
    "location": "Florida",
    "time_range": ["2022-09-20", "2022-10-15"],
    "dataset": "sentinel-1-grd",
    "analysis_type": "event_impact"
}}

Query: "Show cropland map of Iowa"
Output: {{
    "intent": "visualize",
    "location": "Iowa",
    "dataset": "usda-cdl",
    "analysis_type": "land_cover"
}}

Query: "Explain how NDVI is used to monitor drought"
Output: {{
    "intent": "explain",
    "analysis_type": "ndvi"
}}

Query: "What is NDVI?"
Output: {{
    "intent": "explain"
}}

Query: "How do satellites capture images of Earth?"
Output: {{
    "intent": "explain"
}}

Query: "What is climate change?"
Output: {{
    "intent": "explain"
}}

Query: "Tell me a joke"
Output: {{
    "intent": "explain"
}}

Query: "What is Python?"
Output: {{
    "intent": "explain"
}}

Query: "Hello, what can you do?"
Output: {{
    "intent": "explain"
}}

Query: "How was NYC impacted by Hurricane Sandy?"
Output: {{
    "intent": "explain",
    "location": "New York City",
    "analysis_type": "event_impact"
}}

Query: "What datasets are available for monitoring deforestation?"
Output: {{
    "intent": "explain"
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
    def _resolve_collection(
        intent: str,
        analysis_type: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve a Planetary Computer collection ID from query context.

        Uses :data:`COLLECTION_MAPPING` to find the best-matching collection
        when the LLM did not set one explicitly.

        Args:
            intent: The raw intent / query string.
            analysis_type: Optional analysis type hint (e.g. ``"fire_detection"``).

        Returns:
            A collection ID string, or ``None`` if no match was found.
        """
        text = f"{intent} {analysis_type or ''}".lower()

        # Try longest keys first for more specific matches (e.g. "land cover"
        # before "land").
        for key in sorted(COLLECTION_MAPPING, key=len, reverse=True):
            if key in text:
                return COLLECTION_MAPPING[key]
        return None

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

        # Resolve collection via COLLECTION_MAPPING when LLM left dataset empty
        dataset = result.dataset
        if not dataset:
            dataset = Planner._resolve_collection(
                result.intent.value,
                result.analysis_type,
            )

        return PlannerOutput(
            intent=result.intent.value,
            location=location,
            time_range=time_range,
            dataset=dataset,
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
