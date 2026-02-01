"""Main GeoAgent orchestrator using LangGraph for agent coordination.

This module contains the main GeoAgent class that orchestrates the entire
geospatial analysis pipeline using multiple specialized agents.
"""

from typing import Any, Dict, List, Optional, TypedDict
import logging
import time

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import StateGraph, END

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning(
        "LangGraph not available. GeoAgent will use simple sequential execution."
    )

from .models import (  # noqa: E402
    PlannerOutput,
    DataResult,
    AnalysisResult,
    GeoAgentResponse,
)
from .data_agent import DataAgent  # noqa: E402
from .analysis_agent import AnalysisAgent  # noqa: E402
from .viz_agent import VizAgent  # noqa: E402
from .planner import Planner  # noqa: E402
from .llm import get_default_llm  # noqa: E402
from geoagent.catalogs.registry import get_collection_index  # noqa: E402


class AgentState(TypedDict):
    """State passed between agents in the LangGraph workflow."""

    query: str
    plan: Optional[PlannerOutput]
    data: Optional[DataResult]
    analysis: Optional[AnalysisResult]
    map: Optional[Any]
    code: str
    error: Optional[str]
    should_analyze: bool
    should_visualize: bool


class GeoAgent:
    """Main GeoAgent orchestrator for geospatial analysis workflows.

    GeoAgent coordinates multiple specialized agents to perform end-to-end
    geospatial data analysis from natural language queries.
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        catalogs: Optional[List[str]] = None,
    ):
        """Initialize GeoAgent with LLM and configuration.

        Args:
            llm: Language model instance. If None, uses get_default_llm()
            provider: LLM provider name (e.g., 'openai', 'anthropic')
            model: Specific model name
            catalogs: List of STAC catalog URLs to search
        """
        self.llm = llm or get_default_llm()
        self.provider = provider
        self.model = model
        self.catalogs = catalogs or []

        # Fetch available collections from the Planetary Computer STAC (with fallback)
        try:
            self.collection_index = get_collection_index()
        except Exception as e:
            logger.warning(
                f"Failed to fetch collection index: {e}. Proceeding without it."
            )
            self.collection_index = []

        # Initialize specialized agents
        self.planner = Planner(self.llm, collections=self.collection_index)
        self.data_agent = DataAgent(self.llm)
        self.analysis_agent = AnalysisAgent(self.llm)
        self.viz_agent = VizAgent(self.llm)

        # Initialize workflow graph
        self.workflow = self._create_workflow()

        logger.info("GeoAgent initialized successfully")

    def chat(self, query: str) -> GeoAgentResponse:
        """Main method to process a natural language query.

        Args:
            query: Natural language geospatial analysis query

        Returns:
            GeoAgentResponse with complete pipeline results
        """
        logger.info(f"Processing query: {query}")
        start_time = time.time()

        try:
            # Initialize state
            initial_state = AgentState(
                query=query,
                plan=None,
                data=None,
                analysis=None,
                map=None,
                code="",
                error=None,
                should_analyze=True,
                should_visualize=True,
            )

            # Execute workflow
            if LANGGRAPH_AVAILABLE and self.workflow:
                final_state = self.workflow.invoke(initial_state)
            else:
                # Fallback to sequential execution
                final_state = self._sequential_execution(initial_state)

            # Create response
            execution_time = time.time() - start_time

            response = GeoAgentResponse(
                plan=final_state["plan"],
                data=final_state["data"],
                analysis=final_state["analysis"],
                map=final_state["map"],
                code=final_state["code"],
                success=final_state["error"] is None,
                error_message=final_state["error"],
                execution_time=execution_time,
            )

            logger.info(f"Query processed successfully in {execution_time:.2f}s")
            return response

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query processing failed: {e}")

            return GeoAgentResponse(
                plan=PlannerOutput(intent=query, confidence=0.0),
                success=False,
                error_message=str(e),
                execution_time=execution_time,
            )

    def search(self, query: str) -> DataResult:
        """Shortcut method to just search for data without analysis.

        Args:
            query: Natural language data search query

        Returns:
            DataResult with found data
        """
        logger.info(f"Data search for: {query}")

        try:
            # Parse query into plan
            plan = self._parse_query(query)

            # Search for data
            data_result = self.data_agent.search_data(plan)

            logger.info(f"Found {data_result.total_items} data items")
            return data_result

        except Exception as e:
            logger.error(f"Data search failed: {e}")
            return DataResult(
                items=[], metadata={"error": str(e)}, data_type="unknown", total_items=0
            )

    def analyze(self, query: str) -> GeoAgentResponse:
        """Shortcut method for search + analysis without visualization.

        Args:
            query: Natural language analysis query

        Returns:
            GeoAgentResponse with data and analysis results
        """
        logger.info(f"Analysis for: {query}")

        try:
            # Parse query
            plan = self._parse_query(query)

            # Search data
            data = self.data_agent.search_data(plan)

            # Perform analysis
            analysis = self.analysis_agent.analyze(plan, data)

            response = GeoAgentResponse(
                plan=plan,
                data=data,
                analysis=analysis,
                code=analysis.code_generated,
                success=analysis.success,
                error_message=analysis.error_message,
            )

            logger.info("Analysis completed")
            return response

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return GeoAgentResponse(
                plan=PlannerOutput(intent=query, confidence=0.0),
                success=False,
                error_message=str(e),
            )

    def visualize(self, query: str) -> GeoAgentResponse:
        """Run full pipeline including MapLibre GL visualization.

        Args:
            query: Natural language query for complete analysis

        Returns:
            GeoAgentResponse with MapLibre map visualization
        """
        return self.chat(query)  # Full pipeline is the same as chat

    def _create_workflow(self) -> Optional[Any]:
        """Create LangGraph workflow for agent coordination.

        Returns:
            Compiled LangGraph workflow or None if LangGraph unavailable
        """
        if not LANGGRAPH_AVAILABLE:
            return None

        try:
            # Create state graph
            workflow = StateGraph(AgentState)

            # Add nodes
            workflow.add_node("plan", self._plan_node)
            workflow.add_node("fetch_data", self._fetch_data_node)
            workflow.add_node("analyze", self._analyze_node)
            workflow.add_node("visualize", self._visualize_node)

            # Define edges
            workflow.set_entry_point("plan")
            workflow.add_edge("plan", "fetch_data")
            workflow.add_conditional_edges(
                "fetch_data",
                self._should_analyze,
                {True: "analyze", False: "visualize"},
            )
            workflow.add_conditional_edges(
                "analyze", self._should_visualize, {True: "visualize", False: END}
            )
            workflow.add_edge("visualize", END)

            return workflow.compile()

        except Exception as e:
            logger.warning(f"Could not create LangGraph workflow: {e}")
            return None

    def _sequential_execution(self, state: AgentState) -> AgentState:
        """Fallback sequential execution when LangGraph is not available.

        Args:
            state: Initial agent state

        Returns:
            Final agent state
        """
        logger.info("Using sequential execution (LangGraph not available)")

        try:
            # Step 1: Plan
            state = self._plan_node(state)

            # Step 2: Fetch data
            state = self._fetch_data_node(state)

            # Step 3: Analyze (if needed)
            if (
                state["should_analyze"]
                and state["data"]
                and state["data"].total_items > 0
            ):
                state = self._analyze_node(state)

            # Step 4: Visualize (if needed)
            if state["should_visualize"]:
                state = self._visualize_node(state)

            return state

        except Exception as e:
            state["error"] = str(e)
            logger.error(f"Sequential execution failed: {e}")
            return state

    def _plan_node(self, state: AgentState) -> AgentState:
        """Planning node - parse natural language query into structured parameters.

        Args:
            state: Current agent state

        Returns:
            Updated state with plan
        """
        logger.debug("Executing planning node")

        try:
            plan = self._parse_query(state["query"])
            state["plan"] = plan

            # Determine if we need analysis and visualization
            intent_lower = plan.intent.lower()

            # Analysis is needed for computational tasks
            analysis_keywords = [
                "calculate",
                "compute",
                "analyze",
                "ndvi",
                "evi",
                "index",
                "statistics",
                "mean",
                "median",
                "change",
                "trend",
                "zonal",
            ]
            needs_analysis = any(kw in intent_lower for kw in analysis_keywords)
            # Land cover and elevation need analysis routing for proper viz hints
            analysis_type_hint = (plan.analysis_type or "").lower()
            if analysis_type_hint in (
                "land_cover",
                "classification",
                "lulc",
                "elevation",
                "dem",
                "terrain",
            ):
                needs_analysis = True
            state["should_analyze"] = needs_analysis

            # Visualization is usually desired unless explicitly asking for just data
            viz_skip_keywords = ["download", "list", "count", "metadata"]
            state["should_visualize"] = not any(
                kw in intent_lower for kw in viz_skip_keywords
            )

            logger.debug(
                f"Plan created: analyze={state['should_analyze']}, visualize={state['should_visualize']}"
            )

        except Exception as e:
            state["error"] = f"Planning failed: {e}"
            logger.error(state["error"])

        return state

    def _fetch_data_node(self, state: AgentState) -> AgentState:
        """Data fetching node - search and retrieve geospatial data.

        Args:
            state: Current agent state

        Returns:
            Updated state with data
        """
        logger.debug("Executing data fetching node")

        try:
            if state["plan"]:
                data = self.data_agent.search_data(state["plan"])
                state["data"] = data

                # Generate reproducible search code
                state["code"] += self._generate_search_code(state["plan"], data)

                logger.debug(
                    f"Data fetched: {data.total_items} items of type {data.data_type}"
                )
            else:
                state["error"] = "No plan available for data fetching"

        except Exception as e:
            state["error"] = f"Data fetching failed: {e}"
            logger.error(state["error"])

        return state

    def _generate_search_code(self, plan: PlannerOutput, data: Any) -> str:
        """Generate reproducible Python code for the STAC search.

        Args:
            plan: Query plan used for the search
            data: Search results

        Returns:
            Python code string
        """
        bbox = plan.location.get("bbox") if plan.location else None
        location_name = plan.location.get("name", "") if plan.location else ""
        time_range = plan.time_range
        datetime_str = ""
        if time_range:
            datetime_str = (
                f"{time_range.get('start_date', '')}/{time_range.get('end_date', '')}"
            )

        # Build collection: use planner output directly; fallback to Sentinel-2 if absent
        dataset = plan.dataset
        collection = dataset if dataset else "sentinel-2-l2a"

        # Cloud cover filter only for imagery collections
        cloud_filter = ""
        imagery_collections = {
            "sentinel-2-l2a",
            "landsat-c2-l2",
            "naip",
            "sentinel-1-grd",
        }
        max_cc = plan.parameters.get("max_cloud_cover")
        if max_cc is not None and collection in imagery_collections:
            cloud_filter = f'\n    query={{"eo:cloud_cover": {{"lt": {max_cc}}}}},'

        code = f'''import planetary_computer
from pystac_client import Client

# Search Planetary Computer STAC catalog{f" - {location_name}" if location_name else ""}
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

search = catalog.search(
    collections=["{collection}"],
    bbox={bbox},{f"""
    datetime="{datetime_str}",""" if datetime_str else ""}{cloud_filter}
    max_items=10,
)

items = list(search.items())
print(f"Found {{len(items)}} items")
for item in items:
    cc = item.properties.get("eo:cloud_cover", "N/A")
    print(f"  {{item.id}} - cloud cover: {{cc}}%")
'''
        return code

    def _analyze_node(self, state: AgentState) -> AgentState:
        """Analysis node - perform geospatial analysis on data.

        Args:
            state: Current agent state

        Returns:
            Updated state with analysis results
        """
        logger.debug("Executing analysis node")

        try:
            if state["plan"] and state["data"]:
                analysis = self.analysis_agent.analyze(state["plan"], state["data"])
                state["analysis"] = analysis
                state["code"] += analysis.code_generated + "\n"

                if not analysis.success:
                    state["error"] = analysis.error_message

                logger.debug(f"Analysis completed: success={analysis.success}")
            else:
                state["error"] = "Missing plan or data for analysis"

        except Exception as e:
            state["error"] = f"Analysis failed: {e}"
            logger.error(state["error"])

        return state

    def _visualize_node(self, state: AgentState) -> AgentState:
        """Visualization node - create map visualization.

        Args:
            state: Current agent state

        Returns:
            Updated state with map
        """
        logger.debug("Executing visualization node")

        try:
            if state["plan"]:
                viz_map = self.viz_agent.create_visualization(
                    state["plan"], state["data"], state["analysis"]
                )
                state["map"] = viz_map

                # Add visualization code
                state["code"] += self._generate_viz_code(state["plan"], state["data"])

                logger.debug("Map visualization created")
            else:
                state["error"] = "Missing plan for visualization"

        except Exception as e:
            state["error"] = f"Visualization failed: {e}"
            logger.error(state["error"])

        return state

    def _generate_viz_code(self, plan: PlannerOutput, data: Any) -> str:
        """Generate reproducible visualization code.

        Args:
            plan: Query plan
            data: Data result

        Returns:
            Python code string for visualization
        """
        if not data or not data.items:
            return ""

        item = data.items[0]
        item_id = item.get("id", "")
        collection = item.get("collection", "")

        if not collection:
            return ""

        # Determine assets
        assets = item.get("assets", {})
        if "visual" in assets:
            assets_str = '"visual"'
        elif "B04" in assets and "B03" in assets and "B02" in assets:
            assets_str = '["B04", "B03", "B02"]'
        elif "red" in assets and "green" in assets and "blue" in assets:
            assets_str = '["red", "green", "blue"]'
        else:
            assets_str = '"visual"'

        code = f"""
# Visualize on an interactive map
import leafmap.maplibregl as leafmap

m = leafmap.Map()
m.add_stac_layer(
    collection="{collection}",
    item="{item_id}",
    assets={assets_str},
    titiler_endpoint="planetary-computer",
    name="{item_id}",
    fit_bounds=True,
)
m
"""
        return code

    def _should_analyze(self, state: AgentState) -> bool:
        """Conditional edge function to determine if analysis is needed.

        Args:
            state: Current agent state

        Returns:
            True if analysis should be performed
        """
        return (
            state["should_analyze"]
            and state["data"] is not None
            and state["data"].total_items > 0
            and state["error"] is None
        )

    def _should_visualize(self, state: AgentState) -> bool:
        """Conditional edge function to determine if visualization is needed.

        Args:
            state: Current agent state

        Returns:
            True if visualization should be performed
        """
        return state["should_visualize"] and state["error"] is None

    def _parse_query(self, query: str) -> PlannerOutput:
        """Parse natural language query into structured plan.

        Uses LLM for intent/parameter extraction, then geocodes the location.
        Falls back to regex-based parsing if LLM fails.

        Args:
            query: Natural language query

        Returns:
            PlannerOutput with parsed parameters
        """
        logger.debug(f"Parsing query: {query}")

        try:
            # Use LLM-based planner for robust parsing
            plan = self.planner.parse_query(query)
            logger.info(
                f"LLM parsed: location={plan.location}, time={plan.time_range}, "
                f"dataset={plan.dataset}, params={plan.parameters}"
            )

            # Geocode location name to bbox if needed
            if (
                plan.location
                and "name" in plan.location
                and "bbox" not in plan.location
            ):
                geocoded = self._geocode_location(plan.location["name"])
                if geocoded:
                    plan.location = geocoded
                else:
                    logger.warning(f"Could not geocode: {plan.location['name']}")

            # Post-process: LLM sometimes puts time_range in parameters
            if plan.time_range is None and "time_range" in plan.parameters:
                tr = plan.parameters.pop("time_range")
                if isinstance(tr, (list, tuple)) and len(tr) == 2:
                    plan.time_range = {
                        "start_date": tr[0],
                        "end_date": tr[1],
                    }

            # Post-process: normalize cloud cover thresholds
            cc = plan.parameters.get("cloud_cover")
            if cc is not None:
                # "cloud-free" (0) is unrealistic; use 10% threshold
                if cc <= 0:
                    plan.parameters["cloud_cover"] = 10
                plan.parameters["max_cloud_cover"] = plan.parameters.pop("cloud_cover")

            return plan

        except Exception as e:
            logger.warning(f"LLM planner failed ({e}), falling back to regex parser")
            return self._parse_query_fallback(query)

    def _parse_query_fallback(self, query: str) -> PlannerOutput:
        """Fallback regex-based query parser when LLM is unavailable.

        Args:
            query: Natural language query

        Returns:
            PlannerOutput with parsed parameters
        """
        query_lower = query.lower()
        intent = query.strip()

        location = self._extract_location(query)
        time_range = self._extract_time_range(query_lower)

        dataset = None
        if "sentinel" in query_lower or "sentinel-2" in query_lower:
            dataset = "sentinel-2"
        elif "landsat" in query_lower:
            dataset = "landsat"
        elif "modis" in query_lower:
            dataset = "modis"

        parameters = {}
        if "cloud-free" in query_lower or "cloud free" in query_lower:
            parameters["max_cloud_cover"] = 10
        elif "low cloud" in query_lower or "low-cloud" in query_lower:
            parameters["max_cloud_cover"] = 20
        elif "cloud cover" in query_lower or "cloudy" in query_lower:
            parameters["max_cloud_cover"] = 20

        return PlannerOutput(
            intent=intent,
            location=location,
            time_range=time_range,
            dataset=dataset,
            parameters=parameters,
            confidence=0.5,
        )

    def _geocode_location(self, place_name: str) -> Optional[Dict[str, Any]]:
        """Geocode a place name to a bbox.

        Args:
            place_name: Place name string (e.g., "Knoxville", "San Francisco")

        Returns:
            Location dict with bbox and name, or None
        """
        try:
            from geopy.geocoders import Nominatim

            geolocator = Nominatim(user_agent="geoagent")
            result = geolocator.geocode(place_name, exactly_one=True, timeout=5)

            if result:
                lat, lon = result.latitude, result.longitude
                bbox = [lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1]
                name = result.address.split(",")[0]
                logger.info(f"Geocoded '{place_name}' -> {name} ({lat:.4f}, {lon:.4f})")
                return {"bbox": bbox, "name": name}
        except ImportError:
            logger.warning("geopy not installed")
        except Exception as e:
            logger.warning(f"Geocoding failed for '{place_name}': {e}")

        # Try fallback city lookup
        return self._extract_location_fallback(place_name.lower())

    def _extract_location(self, query: str) -> Optional[Dict[str, Any]]:
        """Extract location from query using geocoding (for regex fallback parser).

        Tries to find a place name in the query and geocode it to a bbox.

        Args:
            query: Natural language query

        Returns:
            Location dict with bbox and name, or None
        """
        try:
            from geopy.geocoders import Nominatim

            geolocator = Nominatim(user_agent="geoagent")

            # Try to extract place name - remove common non-location words
            import re

            # Remove analysis terms to isolate location
            cleaned = re.sub(
                r"\b(show|display|compute|calculate|analyze|find|get|plot|map|"
                r"ndvi|evi|savi|imagery|image|images|satellite|sentinel-?\d*|landsat|"
                r"modis|for|in|of|the|from|during|between|and|with|using|"
                r"cloud[- ]?free|low[- ]?cloud|cloud\s*cover|cloudy|"
                r"recent|latest|best|high[- ]?resolution|"
                r"january|february|march|april|may|june|july|august|"
                r"september|october|november|december|"
                r"jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec|"
                r"\d{4})\b",
                "",
                query,
                flags=re.IGNORECASE,
            ).strip()

            # Clean up extra whitespace
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")

            if not cleaned or len(cleaned) < 2:
                logger.debug("No location found in query")
                return None

            logger.debug(f"Geocoding: '{cleaned}'")
            result = geolocator.geocode(cleaned, exactly_one=True, timeout=5)

            if result:
                lat, lon = result.latitude, result.longitude
                # Create bbox around the point (~0.1 degrees ≈ 10km)
                bbox = [lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1]
                name = result.address.split(",")[0]
                logger.info(f"Geocoded '{cleaned}' -> {name} ({lat:.4f}, {lon:.4f})")
                return {"bbox": bbox, "name": name}
            else:
                logger.warning(f"Could not geocode: '{cleaned}'")
                return None

        except ImportError:
            logger.warning("geopy not installed, using fallback location parsing")
            return self._extract_location_fallback(query.lower())
        except Exception as e:
            logger.warning(f"Geocoding failed: {e}")
            return self._extract_location_fallback(query.lower())

    def _extract_location_fallback(self, query_lower: str) -> Optional[Dict[str, Any]]:
        """Fallback location extraction using hardcoded city lookups.

        Args:
            query_lower: Lowercased query string

        Returns:
            Location dict or None
        """
        cities = {
            "san francisco": {
                "bbox": [-122.5, 37.7, -122.3, 37.8],
                "name": "San Francisco",
            },
            "new york": {"bbox": [-74.1, 40.6, -73.9, 40.8], "name": "New York"},
            "los angeles": {
                "bbox": [-118.4, 33.9, -118.1, 34.1],
                "name": "Los Angeles",
            },
            "chicago": {"bbox": [-87.8, 41.7, -87.5, 42.0], "name": "Chicago"},
            "seattle": {"bbox": [-122.4, 47.5, -122.2, 47.7], "name": "Seattle"},
            "denver": {"bbox": [-105.1, 39.6, -104.8, 39.8], "name": "Denver"},
            "houston": {"bbox": [-95.5, 29.6, -95.2, 29.9], "name": "Houston"},
            "miami": {"bbox": [-80.3, 25.7, -80.1, 25.9], "name": "Miami"},
            "california": {"bbox": [-124.4, 32.5, -114.1, 42.0], "name": "California"},
        }
        for city, loc in cities.items():
            if city in query_lower:
                return loc
        return None

    def _extract_time_range(self, query_lower: str) -> Optional[Dict[str, str]]:
        """Extract time range from query text.

        Handles patterns like 'July 2024', 'in 2025', 'June 2023', etc.

        Args:
            query_lower: Lowercased query string

        Returns:
            Dict with start_date and end_date, or None
        """
        import re

        months = {
            "january": ("01", "31"),
            "jan": ("01", "31"),
            "february": ("02", "28"),
            "feb": ("02", "28"),
            "march": ("03", "31"),
            "mar": ("03", "31"),
            "april": ("04", "30"),
            "apr": ("04", "30"),
            "may": ("05", "31"),
            "june": ("06", "30"),
            "jun": ("06", "30"),
            "july": ("07", "31"),
            "jul": ("07", "31"),
            "august": ("08", "31"),
            "aug": ("08", "31"),
            "september": ("09", "30"),
            "sep": ("09", "30"),
            "october": ("10", "31"),
            "oct": ("10", "31"),
            "november": ("11", "30"),
            "nov": ("11", "30"),
            "december": ("12", "31"),
            "dec": ("12", "31"),
        }

        # Match "Month YYYY" or "YYYY" patterns
        for month_name, (month_num, last_day) in months.items():
            pattern = rf"\b{month_name}\s+(\d{{4}})\b"
            match = re.search(pattern, query_lower)
            if match:
                year = match.group(1)
                return {
                    "start_date": f"{year}-{month_num}-01",
                    "end_date": f"{year}-{month_num}-{last_day}",
                }

        # Match bare year
        year_match = re.search(r"\b(20\d{2})\b", query_lower)
        if year_match:
            year = year_match.group(1)
            return {"start_date": f"{year}-01-01", "end_date": f"{year}-12-31"}

        return None
