"""Visualization Agent for creating geospatial maps and visualizations.

The Visualization Agent creates interactive leafmap visualizations based on
data and analysis results from other agents in the pipeline.
"""

from typing import Any, Dict, List, Optional, Union
import logging

try:
    import leafmap

    LEAFMAP_AVAILABLE = True
except ImportError:
    LEAFMAP_AVAILABLE = False


class MockMap:
    """Mock map object when leafmap is not available."""

    def __init__(self):
        self.layers = []
        self.center = [0, 0]
        self.zoom = 5
        self.title = ""

    def set_center(self, lon, lat, zoom=None):
        self.center = [lon, lat]
        if zoom:
            self.zoom = zoom

    def add_raster(self, url, layer_name=None, fit_bounds=False):
        self.layers.append({"type": "raster", "url": url, "name": layer_name})

    def add_geojson(self, data, layer_name=None, style=None):
        self.layers.append(
            {"type": "geojson", "data": data, "name": layer_name, "style": style}
        )

    def add_basemap(self, basemap):
        self.layers.append({"type": "basemap", "name": basemap})

    def __repr__(self):
        return f"MockMap(center={self.center}, zoom={self.zoom}, layers={len(self.layers)})"


def create_map():
    """Create a map object (leafmap if available, otherwise mock)."""
    if LEAFMAP_AVAILABLE:
        return create_map()
    else:
        return MockMap()


from .models import DataResult, AnalysisResult, PlannerOutput

logger = logging.getLogger(__name__)


class VizAgent:
    """Agent responsible for creating geospatial visualizations.
    
    The Visualization Agent takes data and analysis results and creates
    appropriate MapLibre GL visualizations using leafmap's maplibregl backend
    for high-performance 3D mapping and vector tile support.
    """

    def __init__(self, llm: Any, tools: Optional[Dict[str, Any]] = None):
        """Initialize the Visualization Agent.

        Args:
            llm: Language model instance for visualization decisions
            tools: Dictionary of available visualization tools
        """
        self.llm = llm
        self.tools = tools or {}
        self._setup_tools()

    def _setup_tools(self):
        """Setup and initialize visualization tools."""
        try:
            # Import visualization tools
            # TODO: Enable when actual tools are implemented
            # from ..tools.viz import VizTool

            # if 'viz' not in self.tools:
            #     self.tools['viz'] = VizTool()

            logger.info("Visualization tools setup (using placeholders)")

        except ImportError as e:
            logger.warning(f"Visualization tools not available: {e}")
            # Graceful fallback - use leafmap directly

    def create_visualization(
        self,
        plan: PlannerOutput,
        data: Optional[DataResult] = None,
        analysis: Optional[AnalysisResult] = None,
    ) -> Any:
        """Create appropriate visualization based on available data and analysis.

        Args:
            plan: Original query plan for context
            data: Data retrieved by Data Agent
            analysis: Analysis results from Analysis Agent

        Returns:
            Leafmap Map object ready for display
        """
        logger.info("Creating visualization")

        try:
            # Determine visualization type based on available data and analysis
            viz_type = self._determine_viz_type(plan, data, analysis)

            if viz_type == "raster_layer":
                return self._create_raster_visualization(plan, data, analysis)
            elif viz_type == "vector_layer":
                return self._create_vector_visualization(plan, data, analysis)
            elif viz_type == "analysis_result":
                return self._create_analysis_visualization(plan, data, analysis)
            elif viz_type == "time_series":
                return self._create_time_series_visualization(plan, data, analysis)
            elif viz_type == "split_map":
                return self._create_split_map_visualization(plan, data, analysis)
            else:
                return self._create_default_visualization(plan, data, analysis)

        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            return self._create_error_visualization(str(e))

    def _determine_viz_type(
        self,
        plan: PlannerOutput,
        data: Optional[DataResult] = None,
        analysis: Optional[AnalysisResult] = None,
    ) -> str:
        """Determine the appropriate visualization type.

        Args:
            plan: Query plan with intent
            data: Available data
            analysis: Analysis results

        Returns:
            Visualization type string
        """
        # If analysis has specific visualization hints, use those
        if analysis and analysis.visualization_hints:
            viz_hints = analysis.visualization_hints
            if viz_hints.get("type") == "time_series":
                return "time_series"
            elif viz_hints.get("type") == "split_map":
                return "split_map"

        # Check for change detection (typically needs split map)
        if analysis and "change" in plan.intent.lower():
            return "split_map"

        # Check data type
        if data:
            if data.data_type == "raster":
                if analysis:
                    return "analysis_result"  # Processed raster
                else:
                    return "raster_layer"  # Raw raster
            elif data.data_type == "vector":
                return "vector_layer"

        # Check for time series in intent
        if any(
            term in plan.intent.lower() for term in ["time series", "temporal", "trend"]
        ):
            return "time_series"

        return "default"

    def _create_raster_visualization(
        self,
        plan: PlannerOutput,
        data: DataResult,
        analysis: Optional[AnalysisResult] = None,
    ) -> Any:
        """Create visualization for raster data.

        Args:
            plan: Query plan
            data: Raster data
            analysis: Optional analysis results

        Returns:
            Leafmap Map with raster layers
        """
        m = create_map()

        # Set map center based on data location
        if plan.location and "bbox" in plan.location:
            bbox = plan.location["bbox"]
            center_lat = (bbox[1] + bbox[3]) / 2
            center_lon = (bbox[0] + bbox[2]) / 2
            m.set_center(center_lon, center_lat, zoom=10)

        # Add raster layers
        for i, item in enumerate(data.items[:5]):  # Limit to 5 items
            if "assets" in item:
                # Determine best asset to visualize
                asset_key = self._select_best_asset(item["assets"], plan.intent)
                if asset_key and asset_key in item["assets"]:
                    asset_url = item["assets"][asset_key]["href"]

                    # Add layer with appropriate styling
                    layer_name = f"{item.get('id', f'Layer {i+1}')}"

                    try:
                        if "viz" in self.tools:
                            viz_tool = self.tools["viz"]
                            viz_tool.add_raster_layer(
                                m, asset_url, layer_name, plan.intent
                            )
                        else:
                            # Fallback: use leafmap directly
                            m.add_raster(
                                asset_url, layer_name=layer_name, fit_bounds=True
                            )

                    except Exception as e:
                        logger.warning(f"Could not add raster layer {layer_name}: {e}")

        # Add title
        title = f"Raster Visualization: {plan.intent}"
        self._add_title_to_map(m, title)

        return m

    def _create_vector_visualization(
        self,
        plan: PlannerOutput,
        data: DataResult,
        analysis: Optional[AnalysisResult] = None,
    ) -> Any:
        """Create visualization for vector data.

        Args:
            plan: Query plan
            data: Vector data
            analysis: Optional analysis results

        Returns:
            Leafmap Map with vector layers
        """
        m = create_map()

        # Set map center
        if plan.location and "bbox" in plan.location:
            bbox = plan.location["bbox"]
            center_lat = (bbox[1] + bbox[3]) / 2
            center_lon = (bbox[0] + bbox[2]) / 2
            m.set_center(center_lon, center_lat, zoom=10)

        # Add vector layers
        for i, item in enumerate(data.items):
            if "geometry" in item:
                layer_name = f"Vector Layer {i+1}"

                # Determine styling based on analysis
                style = {"color": "blue", "weight": 2, "fillOpacity": 0.3}
                if analysis and analysis.visualization_hints:
                    style.update(analysis.visualization_hints.get("style", {}))

                try:
                    if "viz" in self.tools:
                        viz_tool = self.tools["viz"]
                        viz_tool.add_vector_layer(m, item, layer_name, style)
                    else:
                        # Fallback: add as GeoJSON
                        m.add_geojson(item, layer_name=layer_name, style=style)

                except Exception as e:
                    logger.warning(f"Could not add vector layer {layer_name}: {e}")

        title = f"Vector Visualization: {plan.intent}"
        self._add_title_to_map(m, title)

        return m

    def _create_analysis_visualization(
        self, plan: PlannerOutput, data: DataResult, analysis: AnalysisResult
    ) -> Any:
        """Create visualization for analysis results.

        Args:
            plan: Query plan
            data: Source data
            analysis: Analysis results with visualization hints

        Returns:
            Leafmap Map showing analysis results
        """
        m = create_map()

        # Set map center
        if plan.location and "bbox" in plan.location:
            bbox = plan.location["bbox"]
            center_lat = (bbox[1] + bbox[3]) / 2
            center_lon = (bbox[0] + bbox[2]) / 2
            m.set_center(center_lon, center_lat, zoom=10)

        # Use visualization hints from analysis
        viz_hints = analysis.visualization_hints

        try:
            if "viz" in self.tools:
                viz_tool = self.tools["viz"]
                viz_tool.add_analysis_layer(m, analysis.result_data, viz_hints)
            else:
                # Fallback visualization based on analysis type
                self._add_analysis_fallback(m, data, analysis)

        except Exception as e:
            logger.warning(f"Could not create analysis visualization: {e}")
            # Fall back to data visualization
            return self._create_raster_visualization(plan, data)

        # Add analysis info
        title = f"Analysis Results: {plan.intent}"
        self._add_title_to_map(m, title)
        self._add_analysis_legend(m, analysis)

        return m

    def _create_time_series_visualization(
        self,
        plan: PlannerOutput,
        data: DataResult,
        analysis: Optional[AnalysisResult] = None,
    ) -> Any:
        """Create time series visualization.

        Args:
            plan: Query plan
            data: Time series data
            analysis: Optional analysis results

        Returns:
            Leafmap Map with time series visualization
        """
        m = create_map()

        # Set map center
        if plan.location:
            if "bbox" in plan.location:
                bbox = plan.location["bbox"]
                center_lat = (bbox[1] + bbox[3]) / 2
                center_lon = (bbox[0] + bbox[2]) / 2
            elif "lat" in plan.location and "lon" in plan.location:
                center_lat = plan.location["lat"]
                center_lon = plan.location["lon"]
            else:
                center_lat, center_lon = 0, 0

            m.set_center(center_lon, center_lat, zoom=10)

        # Add time series layers
        try:
            if "viz" in self.tools:
                viz_tool = self.tools["viz"]
                viz_tool.add_time_series_layers(m, data.items)
            else:
                # Fallback: add first and last items
                if len(data.items) >= 2:
                    first_item = data.items[0]
                    last_item = data.items[-1]

                    # Add layers if they have assets
                    if "assets" in first_item and "assets" in last_item:
                        asset_key = self._select_best_asset(
                            first_item["assets"], plan.intent
                        )
                        if asset_key:
                            first_url = first_item["assets"][asset_key]["href"]
                            last_url = last_item["assets"][asset_key]["href"]

                            m.add_raster(
                                first_url,
                                layer_name="Time Series Start",
                                fit_bounds=True,
                            )
                            m.add_raster(
                                last_url, layer_name="Time Series End", fit_bounds=False
                            )

        except Exception as e:
            logger.warning(f"Could not create time series visualization: {e}")

        title = f"Time Series Visualization: {plan.intent}"
        self._add_title_to_map(m, title)

        return m

    def _create_split_map_visualization(
        self,
        plan: PlannerOutput,
        data: DataResult,
        analysis: Optional[AnalysisResult] = None,
    ) -> Any:
        """Create split map for before/after comparisons.

        Args:
            plan: Query plan
            data: Comparison data
            analysis: Optional analysis results

        Returns:
            Split-panel leafmap Map
        """
        try:
            # Create split map if we have multiple time periods
            if len(data.items) >= 2:
                first_item = data.items[0]
                last_item = data.items[-1]

                if "assets" in first_item and "assets" in last_item:
                    asset_key = self._select_best_asset(
                        first_item["assets"], plan.intent
                    )
                    if asset_key:
                        left_url = first_item["assets"][asset_key]["href"]
                        right_url = last_item["assets"][asset_key]["href"]

                        # Create split map
                        m = create_map()

                        # Set center
                        if plan.location and "bbox" in plan.location:
                            bbox = plan.location["bbox"]
                            center_lat = (bbox[1] + bbox[3]) / 2
                            center_lon = (bbox[0] + bbox[2]) / 2
                            m.set_center(center_lon, center_lat, zoom=10)

                        # Add layers to both sides
                        m.add_raster(left_url, layer_name="Before", fit_bounds=True)
                        m.add_raster(right_url, layer_name="After", fit_bounds=False)

                        title = f"Before/After Comparison: {plan.intent}"
                        self._add_title_to_map(m, title)

                        return m

            # Fallback if split map cannot be created
            return self._create_raster_visualization(plan, data, analysis)

        except Exception as e:
            logger.warning(f"Could not create split map: {e}")
            return self._create_default_visualization(plan, data, analysis)

    def _create_default_visualization(
        self,
        plan: PlannerOutput,
        data: Optional[DataResult] = None,
        analysis: Optional[AnalysisResult] = None,
    ) -> Any:
        """Create default visualization when specific type cannot be determined.

        Args:
            plan: Query plan
            data: Available data
            analysis: Available analysis

        Returns:
            Basic leafmap Map
        """
        m = create_map()

        # Set center based on location if available
        if plan.location:
            if "bbox" in plan.location:
                bbox = plan.location["bbox"]
                center_lat = (bbox[1] + bbox[3]) / 2
                center_lon = (bbox[0] + bbox[2]) / 2
                m.set_center(center_lon, center_lat, zoom=10)
            elif "geometry" in plan.location:
                # Try to get centroid from geometry
                try:
                    import shapely.geometry as sg

                    geom = sg.shape(plan.location["geometry"])
                    centroid = geom.centroid
                    m.set_center(centroid.x, centroid.y, zoom=10)
                except:
                    pass

        # Add basemap
        m.add_basemap("OpenTopoMap")

        # Add simple data visualization if available
        if data and data.items:
            try:
                if data.data_type == "raster":
                    # Try to add first raster item
                    item = data.items[0]
                    if "assets" in item:
                        asset_key = self._select_best_asset(item["assets"], plan.intent)
                        if asset_key and asset_key in item["assets"]:
                            asset_url = item["assets"][asset_key]["href"]
                            m.add_raster(
                                asset_url, layer_name="Data Layer", fit_bounds=True
                            )

                elif data.data_type == "vector":
                    # Try to add vector data
                    for item in data.items[:3]:  # Limit to 3 items
                        if "geometry" in item:
                            m.add_geojson(item, layer_name="Vector Data")

            except Exception as e:
                logger.warning(f"Could not add data to default visualization: {e}")

        title = f"GeoAgent Map: {plan.intent}"
        self._add_title_to_map(m, title)

        return m

    def _create_error_visualization(self, error_message: str) -> Any:
        """Create error visualization when something goes wrong.

        Args:
            error_message: Error description

        Returns:
            Basic leafmap Map with error information
        """
        m = create_map()
        m.add_basemap("OpenStreetMap")

        # Add error message
        self._add_title_to_map(m, f"Visualization Error: {error_message}")

        return m

    def _select_best_asset(self, assets: Dict[str, Any], intent: str) -> Optional[str]:
        """Select the best asset for visualization based on intent.

        Args:
            assets: Available STAC assets
            intent: Analysis intent

        Returns:
            Best asset key or None
        """
        intent_lower = intent.lower()

        # For NDVI and vegetation analysis, prefer red or nir
        if any(term in intent_lower for term in ["ndvi", "vegetation", "green"]):
            for key in ["nir", "red", "B04", "B08"]:
                if key in assets:
                    return key

        # For RGB visualization
        if any(term in intent_lower for term in ["rgb", "color", "visual"]):
            for key in ["visual", "rgb", "red"]:
                if key in assets:
                    return key

        # Default preference order
        preference_order = [
            "visual",
            "rgb",
            "red",
            "nir",
            "B04",
            "B03",
            "B02",
            "B08",
            "swir",
            "B11",
            "B12",
        ]

        for key in preference_order:
            if key in assets:
                return key

        # Return first available asset
        return list(assets.keys())[0] if assets else None

    def _add_analysis_fallback(
        self, m: Any, data: DataResult, analysis: AnalysisResult
    ):
        """Add analysis visualization fallback when viz tools are not available.

        Args:
            m: Map to add layers to
            data: Source data
            analysis: Analysis results
        """
        # Simple fallback based on result data
        if "ndvi" in str(analysis.result_data).lower():
            # For NDVI results, try to add the first raster with appropriate styling
            if data.items and "assets" in data.items[0]:
                asset_key = self._select_best_asset(data.items[0]["assets"], "ndvi")
                if asset_key:
                    asset_url = data.items[0]["assets"][asset_key]["href"]
                    m.add_raster(asset_url, layer_name="NDVI Analysis", fit_bounds=True)

    def _add_title_to_map(self, m: Any, title: str):
        """Add title to the map.

        Args:
            m: Map to add title to
            title: Title text
        """
        try:
            # Try to add title using leafmap functionality
            if hasattr(m, "add_title"):
                m.add_title(title)
            else:
                # Fallback: add as a simple text control
                logger.info(f"Map title: {title}")
        except Exception as e:
            logger.debug(f"Could not add title to map: {e}")

    def _add_analysis_legend(self, m: Any, analysis: AnalysisResult):
        """Add legend for analysis results.

        Args:
            m: Map to add legend to
            analysis: Analysis results with visualization hints
        """
        try:
            viz_hints = analysis.visualization_hints

            if "colormap" in viz_hints and "vmin" in viz_hints and "vmax" in viz_hints:
                # Add colorbar legend
                if hasattr(m, "add_colorbar"):
                    m.add_colorbar(
                        colors=viz_hints["colormap"],
                        vmin=viz_hints["vmin"],
                        vmax=viz_hints["vmax"],
                        caption=viz_hints.get("title", "Analysis Result"),
                    )

        except Exception as e:
            logger.debug(f"Could not add legend to map: {e}")
