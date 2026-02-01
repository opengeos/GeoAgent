"""Analysis Agent for performing geospatial analysis operations.

The Analysis Agent takes data references and performs various geospatial
analyses while generating transparent Python code for reproducibility.
"""

from typing import Any, Dict, List, Optional
import logging
import json
import os
import tempfile

from .models import DataResult, AnalysisResult, PlannerOutput

logger = logging.getLogger(__name__)

# Ensure anonymous S3 access for public COGs
if "AWS_NO_SIGN_REQUEST" not in os.environ:
    os.environ["AWS_NO_SIGN_REQUEST"] = "YES"


class AnalysisAgent:
    """Agent responsible for geospatial analysis operations.

    The Analysis Agent performs raster and vector analysis operations
    and generates Python code showing exactly what was computed.
    """

    def __init__(self, llm: Any, tools: Optional[Dict[str, Any]] = None):
        """Initialize the Analysis Agent.

        Args:
            llm: Language model instance for analysis planning
            tools: Dictionary of available analysis tools (raster, vector, etc.)
        """
        self.llm = llm
        self.tools = tools or {}
        self._setup_tools()

    def _setup_tools(self):
        """Setup and initialize analysis tools."""
        try:
            # TODO: Import and register RasterAnalysisTool and VectorAnalysisTool
            # once the tool implementations are finalized.

            logger.info("Analysis tools setup (using placeholders)")

        except ImportError as e:
            logger.warning(f"Some analysis tools not available: {e}")
            # Graceful fallback - tools will be added when available

    def analyze(self, plan: PlannerOutput, data: DataResult) -> AnalysisResult:
        """Perform analysis on the provided data.

        Args:
            plan: Original query plan with analysis intent
            data: Data retrieved by the Data Agent

        Returns:
            AnalysisResult with computed results and generated code
        """
        logger.info(f"Starting analysis for intent: {plan.intent}")

        try:
            # Determine analysis type based on intent and data type
            analysis_type = self._determine_analysis_type(plan, data)

            if analysis_type == "land_cover":
                return self._handle_land_cover(plan, data)
            elif analysis_type == "elevation":
                return self._handle_elevation(plan, data)
            elif analysis_type == "spectral_index":
                return self._compute_spectral_index(plan, data)
            elif analysis_type == "zonal_statistics":
                return self._compute_zonal_statistics(plan, data)
            elif analysis_type == "time_series":
                return self._compute_time_series(plan, data)
            elif analysis_type == "change_detection":
                return self._compute_change_detection(plan, data)
            elif analysis_type == "vector_analysis":
                return self._perform_vector_analysis(plan, data)
            else:
                return self._perform_general_analysis(plan, data)

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return AnalysisResult(
                result_data={"error": str(e)},
                code_generated=f"# Analysis failed: {e}",
                success=False,
                error_message=str(e),
            )

    def _determine_analysis_type(self, plan: PlannerOutput, data: DataResult) -> str:
        """Determine the type of analysis to perform.

        Args:
            plan: Query plan with intent
            data: Available data

        Returns:
            Analysis type string
        """
        intent = plan.intent.lower()
        analysis_type_hint = (plan.analysis_type or "").lower()

        # Check plan.analysis_type first (most reliable signal from planner)
        if analysis_type_hint in ("land_cover", "classification", "lulc"):
            return "land_cover"
        if analysis_type_hint in ("elevation", "dem", "terrain", "slope", "hillshade"):
            return "elevation"
        if analysis_type_hint in ("ndvi", "evi", "savi", "ndbi", "ndwi", "mndwi"):
            return "spectral_index"

        # Land cover analysis (from intent keywords)
        if any(
            term in intent
            for term in [
                "land_cover",
                "land cover",
                "landcover",
                "lulc",
                "land use",
                "classification",
            ]
        ):
            return "land_cover"

        # DEM / elevation analysis (from intent keywords)
        if any(
            term in intent
            for term in [
                "dem",
                "elevation",
                "terrain",
                "slope",
                "hillshade",
                "topograph",
            ]
        ):
            return "elevation"

        # Spectral index analysis
        if any(
            index in intent
            for index in ["ndvi", "evi", "savi", "ndbi", "spectral", "index"]
        ):
            return "spectral_index"

        # Zonal statistics
        if any(
            term in intent
            for term in ["zonal", "statistics", "mean", "median", "sum", "aggregate"]
        ):
            return "zonal_statistics"

        # Time series analysis
        if any(
            term in intent
            for term in ["time series", "temporal", "trend", "change over time"]
        ):
            return "time_series"

        # Change detection
        if any(
            term in intent
            for term in ["change", "detection", "difference", "before", "after"]
        ):
            return "change_detection"

        # Vector operations
        if data.data_type == "vector" or any(
            term in intent for term in ["buffer", "intersect", "union", "clip"]
        ):
            return "vector_analysis"

        return "general"

    def _compute_spectral_index(
        self, plan: PlannerOutput, data: DataResult
    ) -> AnalysisResult:
        """Compute spectral indices from satellite imagery.

        Args:
            plan: Query plan with index specifications
            data: Satellite imagery data

        Returns:
            AnalysisResult with computed indices
        """
        intent = plan.intent.lower()

        # Determine which index to compute
        if "ndvi" in intent:
            index_type = "ndvi"
        elif "evi" in intent:
            index_type = "evi"
        elif "savi" in intent:
            index_type = "savi"
        else:
            index_type = "ndvi"  # Default

        # Try real computation first
        if index_type == "ndvi" and data.items:
            try:
                return self._compute_ndvi_real(plan, data)
            except Exception as e:
                logger.warning(f"Real NDVI computation failed: {e}")

        # Fall back to mock if real computation fails
        return self._create_mock_analysis(index_type, data)

    def _compute_ndvi_real(
        self, plan: PlannerOutput, data: DataResult
    ) -> AnalysisResult:
        """Actually compute NDVI from STAC item COG bands.

        Uses rasterio to read red/NIR bands and compute NDVI,
        then saves result as a temporary GeoTIFF for visualization.
        """
        import numpy as np
        import rasterio
        from rasterio.windows import from_bounds

        # Find first item with red and nir bands
        # Planetary Computer uses B04 (red) and B08 (NIR); other catalogs use
        # lowercase "red"/"nir"
        RED_KEYS = ["red", "B04", "b04", "Red"]
        NIR_KEYS = ["nir", "B08", "b08", "Nir", "NIR"]

        item = None
        red_key = None
        nir_key = None
        for candidate in data.items:
            assets = candidate.get("assets", {})
            found_red = next((k for k in RED_KEYS if k in assets), None)
            found_nir = next((k for k in NIR_KEYS if k in assets), None)
            if found_red and found_nir:
                red_href = assets[found_red].get("href", "")
                nir_href = assets[found_nir].get("href", "")
                if red_href.startswith("http") and nir_href.startswith("http"):
                    item = candidate
                    red_key = found_red
                    nir_key = found_nir
                    break

        if item is None:
            raise ValueError("No STAC item found with red and nir COG bands")

        red_url = item["assets"][red_key]["href"]
        nir_url = item["assets"][nir_key]["href"]
        item_id = item.get("id", "unknown")

        logger.info(f"Computing NDVI for item {item_id}")

        # Read within bbox if available (windowed read for performance)
        bbox_wgs84 = None
        if plan.location and "bbox" in plan.location:
            bbox_wgs84 = plan.location["bbox"]

        with rasterio.open(red_url) as red_src:
            if bbox_wgs84:
                # Transform bbox from WGS84 to raster CRS if needed
                from pyproj import Transformer

                raster_crs = red_src.crs
                if raster_crs and str(raster_crs) != "EPSG:4326":
                    transformer = Transformer.from_crs(
                        "EPSG:4326", raster_crs, always_xy=True
                    )
                    x_min, y_min = transformer.transform(bbox_wgs84[0], bbox_wgs84[1])
                    x_max, y_max = transformer.transform(bbox_wgs84[2], bbox_wgs84[3])
                    bbox_native = [x_min, y_min, x_max, y_max]
                else:
                    bbox_native = bbox_wgs84

                window = from_bounds(*bbox_native, red_src.transform)
                # Ensure window has positive dimensions
                window = window.round_offsets().round_lengths()
                if window.width < 1 or window.height < 1:
                    raise ValueError("Bbox too small for raster resolution")
                red = red_src.read(1, window=window).astype("float32")
                win_transform = red_src.window_transform(window)
            else:
                red = red_src.read(1).astype("float32")
                win_transform = red_src.transform
                window = None
            red_profile = red_src.profile.copy()

        with rasterio.open(nir_url) as nir_src:
            if window is not None:
                nir = nir_src.read(1, window=window).astype("float32")
            else:
                nir = nir_src.read(1).astype("float32")

        # Compute NDVI
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = (nir - red) / (nir + red)
            ndvi = np.where(np.isfinite(ndvi), ndvi, np.nan)

        # Compute statistics
        valid = ndvi[np.isfinite(ndvi)]
        stats = {
            "mean": float(np.nanmean(valid)) if len(valid) > 0 else None,
            "min": float(np.nanmin(valid)) if len(valid) > 0 else None,
            "max": float(np.nanmax(valid)) if len(valid) > 0 else None,
            "std": float(np.nanstd(valid)) if len(valid) > 0 else None,
            "pixels": int(len(valid)),
        }

        # Save NDVI as temporary GeoTIFF
        ndvi_path = os.path.join(tempfile.gettempdir(), f"ndvi_{item_id}.tif")
        out_profile = red_profile.copy()
        out_profile.update(
            dtype="float32",
            count=1,
            driver="GTiff",
            compress="deflate",
            height=ndvi.shape[0],
            width=ndvi.shape[1],
            transform=win_transform,
            nodata=np.nan,
        )
        with rasterio.open(ndvi_path, "w", **out_profile) as dst:
            dst.write(ndvi, 1)

        logger.info(f"NDVI saved to {ndvi_path}, mean={stats['mean']:.3f}")

        # Generate reproducible code
        code = f"""import os
import rasterio
import numpy as np
from rasterio.windows import from_bounds
from pyproj import Transformer

os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

red_url = "{red_url}"
nir_url = "{nir_url}"
bbox_wgs84 = {bbox_wgs84}  # [west, south, east, north] in EPSG:4326

# Read red and NIR bands (windowed read within bbox)
with rasterio.open(red_url) as red_src:
    # Transform bbox to raster CRS
    transformer = Transformer.from_crs("EPSG:4326", red_src.crs, always_xy=True)
    x_min, y_min = transformer.transform(bbox_wgs84[0], bbox_wgs84[1])
    x_max, y_max = transformer.transform(bbox_wgs84[2], bbox_wgs84[3])
    window = from_bounds(x_min, y_min, x_max, y_max, red_src.transform)
    red = red_src.read(1, window=window).astype("float32")
    profile = red_src.profile.copy()
    transform = red_src.window_transform(window)

with rasterio.open(nir_url) as nir_src:
    nir = nir_src.read(1, window=window).astype("float32")

# Compute NDVI = (NIR - Red) / (NIR + Red)
ndvi = (nir - red) / (nir + red)
ndvi = np.where(np.isfinite(ndvi), ndvi, np.nan)

print(f"NDVI shape: {{ndvi.shape}}")
print(f"NDVI range: {{np.nanmin(ndvi):.3f}} to {{np.nanmax(ndvi):.3f}}")
print(f"NDVI mean: {{np.nanmean(ndvi):.3f}}")
"""

        result_data = {
            "analysis_type": "ndvi",
            "item_id": item_id,
            "ndvi_stats": stats,
            "ndvi_path": ndvi_path,
        }

        viz_hints = {
            "type": "ndvi",
            "ndvi_path": ndvi_path,
            "colormap": "RdYlGn",
            "vmin": -0.2,
            "vmax": 0.8,
            "title": f"NDVI - {item_id}",
        }

        return AnalysisResult(
            result_data=result_data,
            code_generated=code,
            visualization_hints=viz_hints,
        )

    def _handle_land_cover(
        self, plan: PlannerOutput, data: DataResult
    ) -> AnalysisResult:
        """Handle land cover data - passthrough with appropriate viz hints.

        Args:
            plan: Query plan
            data: Land cover data from STAC

        Returns:
            AnalysisResult with land cover viz configuration
        """
        location_name = ""
        if plan.location:
            location_name = plan.location.get("name", "")

        result_data = {
            "analysis_type": "land_cover",
            "data_type": "categorical",
            "items_found": data.total_items,
            "collection": (data.items[0].get("collection", "") if data.items else ""),
        }

        bbox = plan.location.get("bbox") if plan.location else None
        code = f"""import planetary_computer
import leafmap.maplibregl as leafmap
from pystac_client import Client

# Search for land cover data - {location_name}
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

search = catalog.search(
    collections=["io-lulc-9-class"],
    bbox={bbox},
    max_items=1,
)

items = list(search.items())
print(f"Found {{len(items)}} land cover items")

# Visualize
m = leafmap.Map()
if items:
    item = items[0]
    m.add_stac_layer(
        collection="io-lulc-9-class",
        item=item.id,
        assets=["data"],
        titiler_endpoint="planetary-computer",
        name="Land Cover",
        fit_bounds=True,
    )
m
"""

        viz_hints = {
            "type": "land_cover",
            "asset_key": "data",
            "title": f"Land Cover - {location_name}" if location_name else "Land Cover",
        }

        return AnalysisResult(
            result_data=result_data,
            code_generated=code,
            visualization_hints=viz_hints,
        )

    def _handle_elevation(
        self, plan: PlannerOutput, data: DataResult
    ) -> AnalysisResult:
        """Handle DEM/elevation data - passthrough with appropriate viz hints.

        Args:
            plan: Query plan
            data: DEM data from STAC

        Returns:
            AnalysisResult with elevation viz configuration
        """
        location_name = ""
        if plan.location:
            location_name = plan.location.get("name", "")

        result_data = {
            "analysis_type": "elevation",
            "data_type": "continuous",
            "items_found": data.total_items,
            "collection": (data.items[0].get("collection", "") if data.items else ""),
        }

        bbox = plan.location.get("bbox") if plan.location else None
        code = f"""import planetary_computer
import leafmap.maplibregl as leafmap
from pystac_client import Client

# Search for DEM data - {location_name}
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

search = catalog.search(
    collections=["cop-dem-glo-30"],
    bbox={bbox},
    max_items=1,
)

items = list(search.items())
print(f"Found {{len(items)}} DEM items")

# Visualize
m = leafmap.Map()
if items:
    item = items[0]
    m.add_stac_layer(
        collection="cop-dem-glo-30",
        item=item.id,
        assets=["data"],
        titiler_endpoint="planetary-computer",
        name="Elevation (DEM)",
        fit_bounds=True,
    )
m
"""

        viz_hints = {
            "type": "elevation",
            "colormap": "terrain",
            "asset_key": "data",
            "title": (
                f"Elevation (DEM) - {location_name}"
                if location_name
                else "Elevation (DEM)"
            ),
        }

        return AnalysisResult(
            result_data=result_data,
            code_generated=code,
            visualization_hints=viz_hints,
        )

    def _compute_zonal_statistics(
        self, plan: PlannerOutput, data: DataResult
    ) -> AnalysisResult:
        """Compute zonal statistics for areas of interest.

        Args:
            plan: Query plan with zones specification
            data: Raster or vector data

        Returns:
            AnalysisResult with computed statistics
        """
        if "raster" in self.tools:
            try:
                raster_tool = self.tools["raster"]

                # Extract zones from plan parameters or location
                zones = plan.parameters.get("zones") or plan.location

                result = raster_tool.zonal_statistics(data.items, zones)
                code = self._generate_zonal_code(data.items, zones)

                viz_hints = {
                    "type": "choropleth",
                    "color_column": "mean",
                    "colormap": "viridis",
                }

                return AnalysisResult(
                    result_data=result,
                    code_generated=code,
                    visualization_hints=viz_hints,
                )

            except Exception as e:
                logger.error(f"Zonal statistics failed: {e}")
                return self._create_mock_analysis("zonal", data)
        else:
            return self._create_mock_analysis("zonal", data)

    def _compute_time_series(
        self, plan: PlannerOutput, data: DataResult
    ) -> AnalysisResult:
        """Compute time series analysis from temporal data.

        Args:
            plan: Query plan with temporal specifications
            data: Time series data

        Returns:
            AnalysisResult with time series analysis
        """
        if "raster" in self.tools:
            try:
                raster_tool = self.tools["raster"]

                # Extract location for time series
                location = plan.location

                result = raster_tool.time_series_analysis(data.items, location)
                code = self._generate_timeseries_code(data.items, location)

                viz_hints = {
                    "type": "time_series",
                    "x_column": "date",
                    "y_column": "value",
                    "title": "Time Series Analysis",
                }

                return AnalysisResult(
                    result_data=result,
                    code_generated=code,
                    visualization_hints=viz_hints,
                )

            except Exception as e:
                logger.error(f"Time series analysis failed: {e}")
                return self._create_mock_analysis("time_series", data)
        else:
            return self._create_mock_analysis("time_series", data)

    def _compute_change_detection(
        self, plan: PlannerOutput, data: DataResult
    ) -> AnalysisResult:
        """Perform change detection analysis.

        Args:
            plan: Query plan with change detection parameters
            data: Multi-temporal data

        Returns:
            AnalysisResult with change analysis
        """
        if "raster" in self.tools:
            try:
                raster_tool = self.tools["raster"]

                result = raster_tool.change_detection(data.items)
                code = self._generate_change_code(data.items)

                viz_hints = {
                    "type": "change_map",
                    "colormap": "RdYlBu",
                    "center_zero": True,
                }

                return AnalysisResult(
                    result_data=result,
                    code_generated=code,
                    visualization_hints=viz_hints,
                )

            except Exception as e:
                logger.error(f"Change detection failed: {e}")
                return self._create_mock_analysis("change", data)
        else:
            return self._create_mock_analysis("change", data)

    def _perform_vector_analysis(
        self, plan: PlannerOutput, data: DataResult
    ) -> AnalysisResult:
        """Perform vector analysis operations.

        Args:
            plan: Query plan with vector operations
            data: Vector data

        Returns:
            AnalysisResult with vector analysis results
        """
        if "vector" in self.tools:
            try:
                vector_tool = self.tools["vector"]

                # Determine vector operation
                intent = plan.intent.lower()
                if "buffer" in intent:
                    distance = plan.parameters.get("distance", 1000)
                    result = vector_tool.buffer(data.items, distance)
                    code = self._generate_buffer_code(data.items, distance)
                elif "intersect" in intent:
                    result = vector_tool.intersect(data.items)
                    code = self._generate_intersect_code(data.items)
                else:
                    result = vector_tool.general_analysis(data.items, plan.intent)
                    code = self._generate_general_vector_code(data.items, plan.intent)

                viz_hints = {
                    "type": "vector_map",
                    "style": {"color": "blue", "weight": 2},
                }

                return AnalysisResult(
                    result_data=result,
                    code_generated=code,
                    visualization_hints=viz_hints,
                )

            except Exception as e:
                logger.error(f"Vector analysis failed: {e}")
                return self._create_mock_analysis("vector", data)
        else:
            return self._create_mock_analysis("vector", data)

    def _perform_general_analysis(
        self, plan: PlannerOutput, data: DataResult
    ) -> AnalysisResult:
        """Perform general analysis when specific type cannot be determined.

        Args:
            plan: Query plan
            data: Available data

        Returns:
            AnalysisResult with general analysis
        """
        # Basic data summary
        result = {
            "data_summary": {
                "total_items": data.total_items,
                "data_type": data.data_type,
                "intent": plan.intent,
            },
            "items_preview": data.items[:5] if data.items else [],
        }

        code = f"""# General analysis for: {plan.intent}
# Data type: {data.data_type}
# Total items: {data.total_items}

import json
data_summary = {json.dumps(result, indent=2)}
print("Analysis Summary:", data_summary)
"""

        viz_hints = {"type": "summary", "show_data_info": True}

        return AnalysisResult(
            result_data=result, code_generated=code, visualization_hints=viz_hints
        )

    def _generate_index_code(self, items: List[Dict], index_type: str) -> str:
        """Generate Python code for spectral index calculation.

        Args:
            items: STAC items or data references
            index_type: Type of spectral index

        Returns:
            Generated Python code string
        """
        if index_type == "ndvi":
            return f"""# NDVI Calculation
import rasterio
import numpy as np

# Load red and NIR bands from STAC items
items = {json.dumps(items[:2], indent=2)}

def calculate_ndvi(red_band, nir_band):
    \"\"\"Calculate NDVI from red and NIR bands.\"\"\"
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    return ndvi

# Process each item
for item in items:
    red_asset = item['assets']['red']['href']
    nir_asset = item['assets']['nir']['href']

    with rasterio.open(red_asset) as red_src:
        red = red_src.read(1).astype(float)

    with rasterio.open(nir_asset) as nir_src:
        nir = nir_src.read(1).astype(float)

    ndvi = calculate_ndvi(red, nir)
    print(f"NDVI calculated for {{item['id']}}")
    print(f"NDVI range: {{np.nanmin(ndvi):.3f}} to {{np.nanmax(ndvi):.3f}}")
"""

        elif index_type == "evi":
            return f"""# EVI Calculation
import rasterio
import numpy as np

# Enhanced Vegetation Index calculation
items = {json.dumps(items[:2], indent=2)}

def calculate_evi(red, nir, blue, G=2.5, C1=6.0, C2=7.5, L=1.0):
    \"\"\"Calculate EVI from RGB and NIR bands.\"\"\"
    evi = G * ((nir - red) / (nir + C1*red - C2*blue + L))
    return evi

# Process items for EVI
for item in items:
    # Load required bands
    red_asset = item['assets']['red']['href']
    nir_asset = item['assets']['nir']['href']
    blue_asset = item['assets']['blue']['href']

    # Calculate EVI
    evi = calculate_evi(red, nir, blue)
    print(f"EVI calculated for {{item['id']}}")
"""

        return f"# {index_type.upper()} calculation code would be generated here"

    def _generate_zonal_code(self, items: List[Dict], zones: Any) -> str:
        """Generate code for zonal statistics."""
        return f"""# Zonal Statistics Calculation
import rasterio
import geopandas as gpd
from rasterio.mask import mask
import numpy as np

# Load raster data
items = {json.dumps(items[:1], indent=2)}
zones = {json.dumps(zones, default=str)}

def zonal_statistics(raster_path, zones_geom):
    \"\"\"Calculate zonal statistics for geometries.\"\"\"
    with rasterio.open(raster_path) as src:
        # Mask raster with each zone
        stats = []
        for zone in zones_geom:
            masked, transform = mask(src, [zone], crop=True)
            data = masked[0]
            valid_data = data[data != src.nodata]

            if len(valid_data) > 0:
                zone_stats = {{
                    'mean': float(np.mean(valid_data)),
                    'median': float(np.median(valid_data)),
                    'std': float(np.std(valid_data)),
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data))
                }}
            else:
                zone_stats = {{'mean': None, 'median': None}}

            stats.append(zone_stats)

    return stats

# Calculate statistics for each zone
print("Computing zonal statistics...")
"""

    def _generate_timeseries_code(self, items: List[Dict], location: Any) -> str:
        """Generate code for time series analysis."""
        return f"""# Time Series Analysis
import rasterio
import pandas as pd
import numpy as np
from datetime import datetime

# Time series data
items = {json.dumps(items[:5], indent=2)}
location = {json.dumps(location, default=str)}

def extract_time_series(items, point_location):
    \"\"\"Extract time series values at a location.\"\"\"
    ts_data = []

    for item in items:
        # Get date from item
        date_str = item['properties']['datetime']
        date = pd.to_datetime(date_str)

        # Extract value at location
        asset_href = item['assets']['red']['href']  # or chosen band
        with rasterio.open(asset_href) as src:
            # Convert location to raster coordinates
            row, col = src.index(location['lon'], location['lat'])
            value = src.read(1)[row, col]

            ts_data.append({{
                'date': date,
                'value': float(value) if value != src.nodata else None
            }})

    return pd.DataFrame(ts_data)

# Extract time series
ts_df = extract_time_series(items, location)
print("Time series extracted:")
print(ts_df.head())
"""

    def _generate_change_code(self, items: List[Dict]) -> str:
        """Generate code for change detection."""
        return f"""# Change Detection Analysis
import rasterio
import numpy as np

# Multi-temporal data
items = {json.dumps(items[:2], indent=2)}

def detect_change(before_raster, after_raster):
    \"\"\"Compute change between two raster datasets.\"\"\"
    with rasterio.open(before_raster) as src1:
        before = src1.read(1).astype(float)

    with rasterio.open(after_raster) as src2:
        after = src2.read(1).astype(float)

    # Calculate change
    change = after - before
    percent_change = ((after - before) / before) * 100

    return {{
        'absolute_change': change,
        'percent_change': percent_change,
        'change_stats': {{
            'mean_change': float(np.nanmean(change)),
            'std_change': float(np.nanstd(change))
        }}
    }}

# Detect changes between time periods
if len(items) >= 2:
    before_asset = items[0]['assets']['red']['href']
    after_asset = items[-1]['assets']['red']['href']

    change_result = detect_change(before_asset, after_asset)
    print("Change detection completed")
    print(f"Mean change: {{change_result['change_stats']['mean_change']:.3f}}")
"""

    def _generate_buffer_code(self, items: List[Dict], distance: float) -> str:
        """Generate code for buffer analysis."""
        return f"""# Buffer Analysis
import geopandas as gpd

# Vector data
items = {json.dumps(items[:1], indent=2)}
buffer_distance = {distance}

def create_buffer(geometries, distance):
    \"\"\"Create buffer around geometries.\"\"\"
    gdf = gpd.GeoDataFrame(items)
    buffered = gdf.buffer(distance)
    return buffered

# Create buffers
buffered_geoms = create_buffer(items, buffer_distance)
print(f"Created buffers with {{buffer_distance}}m radius")
"""

    def _generate_intersect_code(self, items: List[Dict]) -> str:
        """Generate code for intersection analysis."""
        return f"""# Intersection Analysis
import geopandas as gpd

# Vector data for intersection
items = {json.dumps(items[:2], indent=2)}

def intersect_geometries(geom_list):
    \"\"\"Find intersections between geometries.\"\"\"
    gdf1 = gpd.GeoDataFrame([geom_list[0]])
    gdf2 = gpd.GeoDataFrame([geom_list[1]])

    intersection = gpd.overlay(gdf1, gdf2, how='intersection')
    return intersection

# Perform intersection
intersections = intersect_geometries(items)
print("Intersection analysis completed")
"""

    def _generate_general_vector_code(self, items: List[Dict], intent: str) -> str:
        """Generate general vector analysis code."""
        return f"""# General Vector Analysis: {intent}
import geopandas as gpd
import pandas as pd

# Vector data
items = {json.dumps(items[:3], indent=2)}

def analyze_vector_data(data, intent):
    \"\"\"Perform general vector analysis.\"\"\"
    gdf = gpd.GeoDataFrame(data)

    analysis_result = {{
        'total_features': len(gdf),
        'geometry_types': gdf.geom_type.value_counts().to_dict(),
        'bounds': gdf.total_bounds.tolist(),
        'area_stats': {{
            'mean_area': float(gdf.area.mean()),
            'total_area': float(gdf.area.sum())
        }} if 'Polygon' in gdf.geom_type.values else None
    }}

    return analysis_result

# Analyze vector data
result = analyze_vector_data(items, "{intent}")
print("Vector analysis completed:")
print(result)
"""

    def _get_index_viz_hints(self, index_type: str, result: Dict) -> Dict[str, Any]:
        """Get visualization hints for spectral indices."""
        index_configs = {
            "ndvi": {"colormap": "RdYlGn", "vmin": -1, "vmax": 1, "title": "NDVI"},
            "evi": {"colormap": "Greens", "vmin": -1, "vmax": 1, "title": "EVI"},
            "savi": {"colormap": "YlOrRd", "vmin": -1, "vmax": 1, "title": "SAVI"},
        }

        return index_configs.get(
            index_type, {"colormap": "viridis", "title": index_type.upper()}
        )

    def _create_mock_analysis(
        self, analysis_type: str, data: DataResult
    ) -> AnalysisResult:
        """Create mock analysis result when tools are not available.

        Args:
            analysis_type: Type of analysis attempted
            data: Input data

        Returns:
            Mock AnalysisResult for development
        """
        logger.info(f"Creating mock {analysis_type} analysis result")

        mock_result = {
            "analysis_type": analysis_type,
            "data_processed": data.total_items,
            "mock": True,
            "summary": f"Mock {analysis_type} analysis completed",
        }

        if analysis_type == "ndvi":
            mock_result.update(
                {"ndvi_stats": {"mean": 0.65, "min": -0.2, "max": 0.95, "std": 0.18}}
            )

        mock_code = f"""# Mock {analysis_type} analysis
# This is a placeholder while tools are being developed

import numpy as np
import matplotlib.pyplot as plt

# Mock analysis for {analysis_type}
print("Mock {analysis_type} analysis completed")
result = {mock_result}
"""

        return AnalysisResult(
            result_data=mock_result,
            code_generated=mock_code,
            visualization_hints={"type": analysis_type, "mock": True},
        )
