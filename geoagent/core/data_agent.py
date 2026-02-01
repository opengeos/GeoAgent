"""Data Agent for fetching geospatial data from various sources.

The Data Agent is responsible for searching and retrieving geospatial data
based on structured query parameters from the Planner Agent.
"""

from typing import Any, Dict, Optional
import logging

from .models import PlannerOutput, DataResult

logger = logging.getLogger(__name__)


class STACSearchWrapper:
    """Wrapper around pystac_client for STAC searches."""

    def __init__(self, catalog_url: str = None):
        self.catalog_url = (
            catalog_url or "https://planetarycomputer.microsoft.com/api/stac/v1"
        )
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from pystac_client import Client

            # Use planetary_computer modifier for signed URLs if available
            modifier = None
            if "planetarycomputer" in self.catalog_url:
                try:
                    import planetary_computer

                    modifier = planetary_computer.sign_inplace
                except ImportError:
                    logger.warning(
                        "planetary_computer not installed. "
                        "Run: pip install planetary-computer"
                    )

            self._client = Client.open(self.catalog_url, modifier=modifier)
        return self._client

    def search(
        self,
        bbox=None,
        datetime=None,
        collections=None,
        max_items=10,
        **kwargs,
    ):
        """Search STAC catalog and return list of item dicts."""
        search_params = {}
        if bbox:
            search_params["bbox"] = bbox
        if datetime:
            search_params["datetime"] = datetime
        if collections:
            search_params["collections"] = collections
        if "query" in kwargs:
            search_params["query"] = kwargs.pop("query")
        search_params["limit"] = max_items

        search = self.client.search(**search_params)

        results = []
        for item in search.items():
            if len(results) >= max_items:
                break

            item_data = {
                "id": item.id,
                "collection": item.collection_id,
                "geometry": item.geometry,
                "bbox": list(item.bbox) if item.bbox else None,
                "properties": item.properties,
                "assets": {},
            }

            for asset_key, asset in item.assets.items():
                item_data["assets"][asset_key] = {
                    "href": asset.href,
                    "type": asset.media_type,
                    "title": asset.title,
                    "roles": asset.roles if asset.roles else [],
                }

            results.append(item_data)

        return results


class DataAgent:
    """Agent responsible for fetching geospatial data from various sources.

    The Data Agent takes structured query parameters and searches for relevant
    geospatial data using STAC catalogs, DuckDB queries, and other data sources.
    """

    def __init__(self, llm: Any, tools: Optional[Dict[str, Any]] = None):
        """Initialize the Data Agent.

        Args:
            llm: Language model instance for decision making
            tools: Dictionary of available data tools (stac, duckdb, etc.)
        """
        self.llm = llm
        self.tools = tools or {}
        self._setup_tools()

    def _setup_tools(self):
        """Setup and initialize data tools."""
        try:
            if "stac" not in self.tools:
                self.tools["stac"] = STACSearchWrapper()
            logger.info("Data tools initialized (STAC search available)")
        except Exception as e:
            logger.warning(f"Some data tools not available: {e}")

    def search_data(self, plan: PlannerOutput) -> DataResult:
        """Search for geospatial data based on the query plan.

        Args:
            plan: Structured query parameters from Planner Agent

        Returns:
            DataResult containing found data items and metadata
        """
        logger.info(f"Searching data for intent: {plan.intent}")

        try:
            # Determine data type based on intent and dataset
            data_type = self._determine_data_type(plan)

            if data_type == "raster":
                return self._search_raster_data(plan)
            elif data_type == "vector":
                return self._search_vector_data(plan)
            elif data_type == "tabular":
                return self._search_tabular_data(plan)
            else:
                # Fallback: try multiple sources
                return self._search_multi_source(plan)

        except Exception as e:
            logger.error(f"Error searching data: {e}")
            return DataResult(
                items=[], metadata={"error": str(e)}, data_type="unknown", total_items=0
            )

    def _determine_data_type(self, plan: PlannerOutput) -> str:
        """Determine the primary data type needed based on the query plan.

        Args:
            plan: Query plan with intent and parameters

        Returns:
            Data type: "raster", "vector", or "tabular"
        """
        intent = plan.intent.lower()
        dataset = (plan.dataset or "").lower()

        # Raster analysis indicators
        raster_keywords = [
            "ndvi",
            "spectral",
            "index",
            "satellite",
            "imagery",
            "sentinel",
            "landsat",
            "modis",
            "pixel",
            "band",
        ]

        # Vector analysis indicators
        vector_keywords = [
            "boundary",
            "polygon",
            "point",
            "line",
            "geometry",
            "administrative",
            "road",
            "building",
            "parcel",
        ]

        # Check for explicit raster datasets
        if any(sat in dataset for sat in ["sentinel", "landsat", "modis", "aster"]):
            return "raster"

        # Check intent for raster operations
        if any(keyword in intent for keyword in raster_keywords):
            return "raster"

        # Check intent for vector operations
        if any(keyword in intent for keyword in vector_keywords):
            return "vector"

        # Default to raster for satellite-based analysis
        return "raster"

    def _search_raster_data(self, plan: PlannerOutput) -> DataResult:
        """Search for raster/satellite imagery using STAC catalogs.

        Args:
            plan: Query plan with spatial/temporal parameters

        Returns:
            DataResult with STAC items
        """
        if "stac" not in self.tools:
            logger.warning("STAC tool not available")
            return self._create_mock_result("raster", plan)

        try:
            # Build STAC search parameters
            search_params = self._build_stac_params(plan)

            # Execute STAC search
            stac_tool = self.tools["stac"]
            items = stac_tool.search(**search_params)

            return DataResult(
                items=items,
                metadata={
                    "search_params": search_params,
                    "catalog": getattr(stac_tool, "catalog_url", "unknown"),
                },
                data_type="raster",
                total_items=len(items),
                search_query=search_params,
            )

        except Exception as e:
            logger.error(f"STAC search failed: {e}")
            return self._create_mock_result("raster", plan)

    def _search_vector_data(self, plan: PlannerOutput) -> DataResult:
        """Search for vector data using various sources.

        Args:
            plan: Query plan with spatial parameters

        Returns:
            DataResult with vector data references
        """
        if "duckdb" not in self.tools:
            logger.warning("DuckDB tool not available")
            return self._create_mock_result("vector", plan)

        try:
            # Build query for vector data
            query_params = self._build_vector_params(plan)

            # Execute DuckDB query
            duckdb_tool = self.tools["duckdb"]
            results = duckdb_tool.query(**query_params)

            return DataResult(
                items=results,
                metadata={"query_params": query_params, "source": "duckdb"},
                data_type="vector",
                total_items=len(results),
                search_query=query_params,
            )

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return self._create_mock_result("vector", plan)

    def _search_tabular_data(self, plan: PlannerOutput) -> DataResult:
        """Search for tabular data using DuckDB.

        Args:
            plan: Query plan with data requirements

        Returns:
            DataResult with tabular data
        """
        if "duckdb" not in self.tools:
            logger.warning("DuckDB tool not available")
            return self._create_mock_result("tabular", plan)

        try:
            # Build tabular query
            query_params = self._build_tabular_params(plan)

            # Execute query
            duckdb_tool = self.tools["duckdb"]
            results = duckdb_tool.query(**query_params)

            return DataResult(
                items=results,
                metadata={"query_params": query_params, "source": "duckdb"},
                data_type="tabular",
                total_items=len(results),
                search_query=query_params,
            )

        except Exception as e:
            logger.error(f"Tabular search failed: {e}")
            return self._create_mock_result("tabular", plan)

    def _search_multi_source(self, plan: PlannerOutput) -> DataResult:
        """Search across multiple data sources and combine results.

        Args:
            plan: Query plan

        Returns:
            Combined DataResult
        """
        # Try raster first, then vector if no results
        raster_result = self._search_raster_data(plan)

        if raster_result.total_items > 0:
            return raster_result

        vector_result = self._search_vector_data(plan)
        if vector_result.total_items > 0:
            return vector_result

        # Return empty result if nothing found
        return DataResult(
            items=[],
            metadata={"searched_sources": ["raster", "vector"]},
            data_type="unknown",
            total_items=0,
        )

    def _build_stac_params(self, plan: PlannerOutput) -> Dict[str, Any]:
        """Build STAC search parameters from query plan.

        Args:
            plan: Query plan with spatial/temporal info

        Returns:
            STAC search parameters dictionary
        """
        params = {}

        # Add spatial parameters
        if plan.location:
            if "bbox" in plan.location:
                params["bbox"] = plan.location["bbox"]
            elif "geometry" in plan.location:
                params["intersects"] = plan.location["geometry"]

        # Add temporal parameters
        if plan.time_range:
            start_date = plan.time_range.get("start_date")
            end_date = plan.time_range.get("end_date")
            if start_date and end_date:
                params["datetime"] = f"{start_date}/{end_date}"

        # Add collection/dataset filters
        if plan.dataset:
            # Use the planner-provided dataset directly as the collection ID
            params["collections"] = [plan.dataset]

        # Resolve collection from analysis_type or intent if not already set
        if "collections" not in params:
            analysis_type = (plan.analysis_type or "").lower()
            intent_lower = plan.intent.lower()

            # Check analysis_type first (most reliable signal from planner)
            if analysis_type in ("land_cover", "classification", "lulc"):
                params["collections"] = ["io-lulc-9-class"]
            elif analysis_type in (
                "elevation",
                "dem",
                "terrain",
                "slope",
                "hillshade",
            ):
                params["collections"] = ["cop-dem-glo-30"]
            # Then check intent keywords
            elif any(
                kw in intent_lower
                for kw in ["land cover", "landcover", "lulc", "land use"]
            ):
                params["collections"] = ["io-lulc-9-class"]
            elif any(
                kw in intent_lower
                for kw in ["dem", "elevation", "terrain", "height", "topograph"]
            ):
                params["collections"] = ["cop-dem-glo-30"]
            elif any(
                kw in intent_lower
                for kw in [
                    "ndvi",
                    "evi",
                    "vegetation",
                    "spectral",
                    "band",
                    "imagery",
                ]
            ):
                params["collections"] = ["sentinel-2-l2a"]

        # Add cloud cover filter only for imagery collections (not DEM/land cover)
        # Heuristic: imagery collections often contain these keywords
        current_collections = set(params.get("collections", []))
        imagery_keywords = ("sentinel", "landsat", "naip", "modis")
        is_imagery = any(
            any(kw in (c or "").lower() for kw in imagery_keywords)
            for c in current_collections
        )
        if is_imagery:
            max_cloud = plan.parameters.get("max_cloud_cover") or plan.parameters.get(
                "cloud_cover"
            )
            if max_cloud is not None:
                params["query"] = {"eo:cloud_cover": {"lt": max_cloud}}

        # Add limit to prevent too many results
        params["max_items"] = plan.parameters.get("max_items", 10)

        return params

    def _build_vector_params(self, plan: PlannerOutput) -> Dict[str, Any]:
        """Build vector query parameters from plan.

        Args:
            plan: Query plan

        Returns:
            Vector query parameters
        """
        params = {
            "intent": plan.intent,
            "location": plan.location,
            "parameters": plan.parameters,
        }
        return params

    def _build_tabular_params(self, plan: PlannerOutput) -> Dict[str, Any]:
        """Build tabular query parameters from plan.

        Args:
            plan: Query plan

        Returns:
            Tabular query parameters
        """
        params = {
            "intent": plan.intent,
            "dataset": plan.dataset,
            "parameters": plan.parameters,
        }
        return params

    def _create_mock_result(self, data_type: str, plan: PlannerOutput) -> DataResult:
        """Create a mock result when tools are not available.

        Args:
            data_type: Type of data that was requested
            plan: Original query plan

        Returns:
            Mock DataResult for development/testing
        """
        logger.info(f"Creating mock {data_type} result for development")

        mock_items = []
        if data_type == "raster":
            mock_items = [
                {
                    "id": "mock_sentinel2_item",
                    "collection": "sentinel-2-l2a",
                    "geometry": (
                        plan.location.get("geometry") if plan.location else None
                    ),
                    "properties": {
                        "datetime": "2024-07-15T10:30:00Z",
                        "cloud_cover": 5.2,
                    },
                    "assets": {
                        "red": {"href": "mock://red.tif"},
                        "nir": {"href": "mock://nir.tif"},
                    },
                }
            ]

        return DataResult(
            items=mock_items,
            metadata={"mock": True, "reason": "tools_not_available"},
            data_type=data_type,
            total_items=len(mock_items),
        )
