"""Main GeoAgent orchestrator using LangGraph for agent coordination.

This module contains the main GeoAgent class that orchestrates the entire
geospatial analysis pipeline using multiple specialized agents.
"""

from typing import Any, Dict, List, Optional, TypedDict
import logging
import time

try:
    from langgraph.graph import StateGraph, END

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(
        "LangGraph not available. GeoAgent will use simple sequential execution."
    )

try:
    import leafmap

    LEAFMAP_AVAILABLE = True
except ImportError:
    LEAFMAP_AVAILABLE = False
    logger.warning("leafmap not available. Maps will be created as placeholders.")

from .models import PlannerOutput, DataResult, AnalysisResult, GeoAgentResponse
from .data_agent import DataAgent
from .analysis_agent import AnalysisAgent
from .viz_agent import VizAgent
from .llm import get_default_llm

logger = logging.getLogger(__name__)


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

        # Initialize specialized agents
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
        """Run full pipeline including visualization.

        Args:
            query: Natural language query for complete analysis

        Returns:
            GeoAgentResponse with map visualization
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
            state["should_analyze"] = any(
                kw in intent_lower for kw in analysis_keywords
            )

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

                logger.debug(
                    f"Data fetched: {data.total_items} items of type {data.data_type}"
                )
            else:
                state["error"] = "No plan available for data fetching"

        except Exception as e:
            state["error"] = f"Data fetching failed: {e}"
            logger.error(state["error"])

        return state

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

                logger.debug("Map visualization created")
            else:
                state["error"] = "Missing plan for visualization"

        except Exception as e:
            state["error"] = f"Visualization failed: {e}"
            logger.error(state["error"])

        return state

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

        This is a simplified parser. In a full implementation, this would
        use a dedicated Planner Agent with more sophisticated NL understanding.

        Args:
            query: Natural language query

        Returns:
            PlannerOutput with parsed parameters
        """
        logger.debug(f"Parsing query: {query}")

        query_lower = query.lower()

        # Extract intent
        intent = query.strip()

        # Extract location (simple keyword matching)
        location = None
        if "san francisco" in query_lower:
            location = {"bbox": [-122.5, 37.7, -122.3, 37.8], "name": "San Francisco"}
        elif "new york" in query_lower:
            location = {"bbox": [-74.1, 40.6, -73.9, 40.8], "name": "New York"}
        elif "california" in query_lower:
            location = {"bbox": [-124.4, 32.5, -114.1, 42.0], "name": "California"}

        # Extract time range (simple patterns)
        time_range = None
        if "2024" in query:
            if "july" in query_lower or "jul" in query_lower:
                time_range = {"start_date": "2024-07-01", "end_date": "2024-07-31"}
            elif "june" in query_lower or "jun" in query_lower:
                time_range = {"start_date": "2024-06-01", "end_date": "2024-06-30"}
            else:
                time_range = {"start_date": "2024-01-01", "end_date": "2024-12-31"}

        # Extract dataset preference
        dataset = None
        if "sentinel" in query_lower or "sentinel-2" in query_lower:
            dataset = "sentinel-2"
        elif "landsat" in query_lower:
            dataset = "landsat"
        elif "modis" in query_lower:
            dataset = "modis"

        # Extract additional parameters
        parameters = {}
        if "cloud cover" in query_lower or "cloudy" in query_lower:
            parameters["max_cloud_cover"] = 20

        plan = PlannerOutput(
            intent=intent,
            location=location,
            time_range=time_range,
            dataset=dataset,
            parameters=parameters,
            confidence=0.8,  # Simple confidence score
        )

        logger.debug(f"Plan created: {plan.intent}")
        return plan
