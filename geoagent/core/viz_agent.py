"""Visualization Agent for creating geospatial maps and visualizations.

The Visualization Agent creates interactive MapLibre GL visualizations using
leafmap's maplibregl backend for high-performance 3D mapping.
"""

from typing import Any, Dict, Optional
import logging
import os

# Planetary Computer TiTiler endpoint for fast tile serving
PC_TITILER_ENDPOINT = "https://planetarycomputer.microsoft.com/api/data/v1"

# Enable anonymous access for public S3 COG data (fallback)
if "AWS_NO_SIGN_REQUEST" not in os.environ:
    os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

try:
    from leafmap.maplibregl import Map as MapLibreMap

    MAPLIBRE_AVAILABLE = True
except ImportError:
    MAPLIBRE_AVAILABLE = False


class MockMapLibreMap:
    """Mock MapLibre map object when leafmap.maplibregl is not available."""

    def __init__(self, center=[0, 0], zoom=5, height="600px", **kwargs):
        self.layers = []
        self.center = center
        self.zoom = zoom
        self.height = height
        self.title = ""
        self._style = "open-street-map"

    def set_center(self, lon, lat, zoom=None):
        self.center = [lon, lat]
        if zoom is not None:
            self.zoom = zoom

    def add_cog_layer(self, url, name=None, fit_bounds=False, **kwargs):
        """Add Cloud Optimized GeoTIFF layer."""
        self.layers.append(
            {
                "type": "cog",
                "url": url,
                "name": name or f"COG Layer {len(self.layers)+1}",
                "fit_bounds": fit_bounds,
                **kwargs,
            }
        )

    def add_raster(self, url, layer_name=None, fit_bounds=False, **kwargs):
        """Add raster layer (fallback to COG)."""
        self.add_cog_layer(url, name=layer_name, fit_bounds=fit_bounds, **kwargs)

    def add_geojson(self, data, layer_name=None, style=None, **kwargs):
        """Add GeoJSON layer."""
        self.layers.append(
            {
                "type": "geojson",
                "data": data,
                "name": layer_name or f"GeoJSON Layer {len(self.layers)+1}",
                "style": style,
                **kwargs,
            }
        )

    def add_pmtiles(self, url, name=None, **kwargs):
        """Add PMTiles vector layer."""
        self.layers.append(
            {
                "type": "pmtiles",
                "url": url,
                "name": name or f"PMTiles Layer {len(self.layers)+1}",
                **kwargs,
            }
        )

    def add_basemap(self, basemap="open-street-map"):
        """Set basemap style."""
        self._style = basemap

    def add_layer(self, layer_dict):
        """Add generic layer."""
        self.layers.append(layer_dict)

    def add_source(self, source_id, source_dict):
        """Add data source."""
        # Mock implementation
        pass

    def add_title(self, title):
        """Add title to map."""
        self.title = title

    def to_html(self, filename=None):
        """Export map to HTML."""
        html = f"""
        <div style="text-align: center;">
            <h3>{self.title}</h3>
            <p>Mock MapLibre Map</p>
            <p>Center: {self.center}, Zoom: {self.zoom}</p>
            <p>Layers: {len(self.layers)}</p>
            <ul>
        """
        for layer in self.layers:
            html += f"<li>{layer.get('name', 'Unnamed')} ({layer.get('type', 'unknown')})</li>"
        html += "</ul></div>"

        if filename:
            with open(filename, "w") as f:
                f.write(html)
        return html

    def __repr__(self):
        return f"MockMapLibreMap(center={self.center}, zoom={self.zoom}, layers={len(self.layers)})"


def create_map(**kwargs):
    """Create a MapLibre map object (real if available, otherwise mock)."""
    if MAPLIBRE_AVAILABLE:
        return MapLibreMap(**kwargs)
    else:
        return MockMapLibreMap(**kwargs)


from .models import DataResult, AnalysisResult, PlannerOutput  # noqa: E402

logger = logging.getLogger(__name__)


class VizAgent:
    """Agent responsible for creating geospatial visualizations.

    The Visualization Agent takes data and analysis results and creates
    appropriate leafmap visualizations for display in Jupyter notebooks.
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
            MapLibre Map object ready for display
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
            MapLibre Map with raster layers
        """
        m = create_map()

        # Set map center based on data location
        if plan.location and "bbox" in plan.location:
            bbox = plan.location["bbox"]
            center_lat = (bbox[1] + bbox[3]) / 2
            center_lon = (bbox[0] + bbox[2]) / 2
            m.set_center(center_lon, center_lat, zoom=10)

        # Add raster layers using Planetary Computer TiTiler
        for i, item in enumerate(data.items[:1]):
            item_id = item.get("id", f"Layer {i+1}")
            collection = item.get("collection", "")

            # Skip mock items
            assets = item.get("assets", {})
            if not assets or any(
                v.get("href", "").startswith("mock://") for v in assets.values()
            ):
                logger.debug(f"Skipping mock item {item_id}")
                continue

            try:
                # Use add_stac_layer with PC TiTiler for best performance
                if MAPLIBRE_AVAILABLE and collection:
                    viz_assets = self._select_viz_assets(assets, plan.intent)
                    m.add_stac_layer(
                        collection=collection,
                        item=item_id,
                        assets=viz_assets,
                        titiler_endpoint="planetary-computer",
                        name=item_id,
                        fit_bounds=True,
                    )
                else:
                    # Fallback to COG layer with signed URL
                    asset_key = self._select_best_asset(assets, plan.intent)
                    if asset_key and asset_key in assets:
                        asset_url = assets[asset_key]["href"]
                        m.add_cog_layer(
                            asset_url,
                            name=item_id,
                            titiler_endpoint=PC_TITILER_ENDPOINT,
                            fit_bounds=True,
                        )
            except Exception as e:
                logger.warning(f"Could not add raster layer {item_id}: {e}")

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
            MapLibre Map with vector layers
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
                        m.add_geojson(item, name=layer_name, style=style)

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
            MapLibre Map showing analysis results
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
            MapLibre Map with time series visualization
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

                            m.add_cog_layer(
                                first_url, name="Time Series Start", fit_bounds=True
                            )
                            m.add_cog_layer(
                                last_url, name="Time Series End", fit_bounds=False
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
                        m.add_cog_layer(left_url, name="Before", fit_bounds=True)
                        m.add_cog_layer(right_url, name="After", fit_bounds=False)

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
                except Exception:
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
                            if not asset_url.startswith("mock://"):
                                m.add_cog_layer(
                                    asset_url, name="Data Layer", fit_bounds=True
                                )

                elif data.data_type == "vector":
                    # Try to add vector data
                    for item in data.items[:3]:  # Limit to 3 items
                        if "geometry" in item:
                            m.add_geojson(item, name="Vector Data")

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

    def _select_viz_assets(self, assets: Dict[str, Any], intent: str) -> list:
        """Select asset names for STAC layer visualization.

        Returns a list of asset keys suitable for add_stac_layer.

        Args:
            assets: Available STAC assets
            intent: Analysis intent

        Returns:
            List of asset key strings (e.g., ["visual"] or ["B04", "B03", "B02"])
        """
        intent_lower = intent.lower()

        # DEM and land cover collections use "data" or "map" asset
        if any(
            term in intent_lower
            for term in [
                "dem",
                "elevation",
                "terrain",
                "land_cover",
                "land cover",
                "landcover",
                "lulc",
                "land use",
            ]
        ):
            if "data" in assets:
                return ["data"]
            if "map" in assets:
                return ["map"]

        # For imagery/visual requests, prefer true color composite
        if "visual" in assets:
            return ["visual"]

        # For NDVI, show NIR-Red false color or just visual
        if any(term in intent_lower for term in ["ndvi", "vegetation"]):
            if "nir" in assets and "red" in assets and "green" in assets:
                return ["nir", "red", "green"]
            if "B08" in assets and "B04" in assets and "B03" in assets:
                return ["B08", "B04", "B03"]

        # RGB composite
        if "red" in assets and "green" in assets and "blue" in assets:
            return ["red", "green", "blue"]
        if "B04" in assets and "B03" in assets and "B02" in assets:
            return ["B04", "B03", "B02"]

        # Fallback to first available data asset
        best = self._select_best_asset(assets, intent)
        return [best] if best else []

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
        viz_hints = analysis.visualization_hints
        viz_type = viz_hints.get("type", "") if viz_hints else ""

        # Handle land cover and elevation data via STAC layer
        if viz_type in ("land_cover", "elevation") and data and data.items:
            item = data.items[0]
            item_id = item.get("id", "")
            collection = item.get("collection", "")

            if MAPLIBRE_AVAILABLE and collection:
                asset_key = viz_hints.get("asset_key", "data")
                try:
                    m.add_stac_layer(
                        collection=collection,
                        item=item_id,
                        assets=[asset_key],
                        titiler_endpoint="planetary-computer",
                        name=viz_hints.get("title", item_id),
                        fit_bounds=True,
                    )
                    logger.info(f"Added {viz_type} STAC layer: {collection}/{item_id}")
                    return
                except Exception as e:
                    logger.warning(f"Could not add {viz_type} STAC layer: {e}")

        # Check if we have a computed NDVI raster to display
        ndvi_path = None
        if viz_hints and "ndvi_path" in viz_hints:
            ndvi_path = viz_hints["ndvi_path"]
        elif isinstance(analysis.result_data, dict):
            ndvi_path = analysis.result_data.get("ndvi_path")

        if ndvi_path and os.path.exists(ndvi_path):
            try:
                m.add_raster(
                    ndvi_path,
                    layer_name="NDVI",
                    colormap="RdYlGn",
                    vmin=-0.2,
                    vmax=0.8,
                    fit_bounds=True,
                )
                logger.info(f"Added NDVI raster layer from {ndvi_path}")
                return
            except Exception as e:
                logger.warning(f"Could not add NDVI raster: {e}")

        # Fallback: try to add raw band from data
        if data.items and "assets" in data.items[0]:
            asset_key = self._select_best_asset(data.items[0]["assets"], "ndvi")
            if asset_key:
                asset_url = data.items[0]["assets"][asset_key]["href"]
                if not asset_url.startswith("mock://"):
                    try:
                        m.add_cog_layer(
                            asset_url, name="NDVI Analysis", fit_bounds=True
                        )
                    except Exception as e:
                        logger.warning(f"Could not add COG layer: {e}")

    def _add_title_to_map(self, m: Any, title: str):
        """Add title to the map.

        Args:
            m: MapLibre map to add title to
            title: Title text
        """
        try:
            # Try to add title using MapLibre functionality
            if hasattr(m, "add_title"):
                m.add_title(title)
            elif hasattr(m, "title"):
                # For MockMapLibreMap
                m.title = title
            else:
                # Fallback: log the title
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
                # Add colorbar legend (leafmap maplibregl uses cmap/label)
                if hasattr(m, "add_colorbar"):
                    m.add_colorbar(
                        cmap=viz_hints["colormap"],
                        vmin=viz_hints["vmin"],
                        vmax=viz_hints["vmax"],
                        label=viz_hints.get("title", "Analysis Result"),
                    )

        except Exception as e:
            logger.debug(f"Could not add legend to map: {e}")
