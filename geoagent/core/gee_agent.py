"""Google Earth Engine Agent for GeoAgent.

Handles searching, filtering, and preparing GEE datasets for visualization.
Uses the Earth Engine Python API and geemap for map rendering.
"""

import logging
from typing import Any, Dict, List, Optional

from .models import PlannerOutput, DataResult

logger = logging.getLogger(__name__)

# Common GEE dataset mappings
GEE_COLLECTIONS: Dict[str, str] = {
    # Optical
    "sentinel-2": "COPERNICUS/S2_SR_HARMONIZED",
    "landsat-9": "LANDSAT/LC09/C02/T1_L2",
    "landsat-8": "LANDSAT/LC08/C02/T1_L2",
    "landsat-7": "LANDSAT/LE07/C02/T1_L2",
    "landsat-5": "LANDSAT/LT05/C02/T1_L2",
    "naip": "USDA/NAIP/DOQQ",
    # SAR
    "sentinel-1": "COPERNICUS/S1_GRD",
    # Elevation / DEM
    "srtm": "USGS/SRTMGL1_003",
    "alos-dem": "JAXA/ALOS/AW3D30/V3_2",
    "copernicus-dem": "COPERNICUS/DEM/GLO30",
    "nasadem": "NASA/NASADEM_HGT/001",
    # Land Cover
    "dynamic-world": "GOOGLE/DYNAMICWORLD/V1",
    "esri-lulc": (
        "projects/sat-io/open-datasets/land-cover/ESRI_Global-LULC_10m_TS"
    ),
    "esa-worldcover": "ESA/WorldCover/v200",
    "nlcd": "USGS/NLCD_RELEASES/2021_REL/NLCD",
    "modis-landcover": "MODIS/061/MCD12Q1",
    "globcover": "ESA/GLOBCOVER_L4_200901_200912_V2_3",
    # Vegetation
    "modis-ndvi": "MODIS/061/MOD13Q1",
    "modis-evi": "MODIS/061/MOD13Q1",
    "modis-lai": "MODIS/061/MOD15A2H",
    "modis-npp": "MODIS/061/MOD17A2H",
    "modis-gpp": "MODIS/061/MOD17A2H",
    # Water
    "jrc-water": "JRC/GSW1_4/GlobalSurfaceWater",
    "jrc-monthly": "JRC/GSW1_4/MonthlyHistory",
    # Fire
    "modis-fire": "MODIS/061/MOD14A1",
    "firms": "FIRMS",
    "modis-burned": "MODIS/061/MCD64A1",
    # Temperature
    "modis-lst": "MODIS/061/MOD11A1",
    "era5-temperature": "ECMWF/ERA5_LAND/DAILY_AGGR",
    # Snow
    "modis-snow": "MODIS/061/MOD10A1",
    # Nightlights
    "viirs-nightlights": "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG",
    "dmsp-nightlights": "NOAA/DMSP-OLS/NIGHTTIME_LIGHTS",
    # Climate / Weather
    "era5-land": "ECMWF/ERA5_LAND/DAILY_AGGR",
    "chirps-rainfall": "UCSB-CHG/CHIRPS/DAILY",
    "gridmet": "IDAHO_EPSCOR/GRIDMET",
    # Population
    "worldpop": "WorldPop/GP/100m/pop",
    "ghsl-population": "JRC/GHSL/P2023A/GHS_POP",
    # Atmosphere
    "modis-aerosol": "MODIS/061/MOD04_L2",
    "sentinel-5p-no2": "COPERNICUS/S5P/OFFL/L3_NO2",
    "sentinel-5p-co": "COPERNICUS/S5P/OFFL/L3_CO",
    # Soil
    "soilgrids": "projects/soilgrids-isric/ocd_mean",
    # Buildings
    "google-buildings": "GOOGLE/Research/open-buildings/v3/polygons",
    "ms-buildings": "projects/sat-io/open-datasets/MSBuildings",
}

# Visualization parameters for common GEE datasets
GEE_VIS_PARAMS: Dict[str, Dict[str, Any]] = {
    "COPERNICUS/S2_SR_HARMONIZED": {
        "bands": ["B4", "B3", "B2"],
        "min": 0,
        "max": 3000,
        "name": "Sentinel-2 True Color",
    },
    "LANDSAT/LC09/C02/T1_L2": {
        "bands": ["SR_B4", "SR_B3", "SR_B2"],
        "min": 7000,
        "max": 30000,
        "name": "Landsat 9 True Color",
    },
    "LANDSAT/LC08/C02/T1_L2": {
        "bands": ["SR_B4", "SR_B3", "SR_B2"],
        "min": 7000,
        "max": 30000,
        "name": "Landsat 8 True Color",
    },
    "USGS/SRTMGL1_003": {
        "bands": ["elevation"],
        "min": 0,
        "max": 4000,
        "palette": ["006633", "E5FFCC", "662A00", "D8D8D8", "F5F5F5"],
        "name": "SRTM Elevation",
    },
    "COPERNICUS/DEM/GLO30": {
        "bands": ["DEM"],
        "min": 0,
        "max": 4000,
        "palette": ["006633", "E5FFCC", "662A00", "D8D8D8", "F5F5F5"],
        "name": "Copernicus DEM",
    },
    "NASA/NASADEM_HGT/001": {
        "bands": ["elevation"],
        "min": 0,
        "max": 4000,
        "palette": ["006633", "E5FFCC", "662A00", "D8D8D8", "F5F5F5"],
        "name": "NASADEM Elevation",
    },
    "GOOGLE/DYNAMICWORLD/V1": {
        "bands": ["label"],
        "min": 0,
        "max": 8,
        "palette": [
            "419BDF",
            "397D49",
            "88B053",
            "7A87C6",
            "E49635",
            "DFC35A",
            "C4281B",
            "A59B8F",
            "B39FE1",
        ],
        "name": "Dynamic World Land Cover",
    },
    "ESA/WorldCover/v200": {
        "bands": ["Map"],
        "name": "ESA WorldCover",
    },
    "MODIS/061/MOD13Q1": {
        "bands": ["NDVI"],
        "min": -2000,
        "max": 9000,
        "palette": [
            "CE7E45",
            "DF923D",
            "F1B555",
            "FCD163",
            "99B718",
            "74A901",
            "66A000",
            "529400",
            "3E8601",
            "207401",
            "056201",
            "004C00",
            "023B01",
            "012E01",
            "011D01",
            "011301",
        ],
        "name": "MODIS NDVI",
    },
    "JRC/GSW1_4/GlobalSurfaceWater": {
        "bands": ["occurrence"],
        "min": 0,
        "max": 100,
        "palette": ["ffffff", "ffbbbb", "0000ff"],
        "name": "JRC Global Surface Water",
    },
    "MODIS/061/MOD14A1": {
        "bands": ["MaxFRP"],
        "min": 0,
        "max": 200,
        "palette": ["FFFF00", "FF0000"],
        "name": "MODIS Fire",
    },
    "MODIS/061/MOD11A1": {
        "bands": ["LST_Day_1km"],
        "min": 13000,
        "max": 16500,
        "palette": [
            "040274",
            "040281",
            "0502a3",
            "0502b8",
            "0502ce",
            "0502e6",
            "0602ff",
            "235cb1",
            "307ef3",
            "269db1",
            "30c8e2",
            "32d3ef",
            "3be285",
            "3ff38f",
            "86e26f",
            "3ae237",
            "b5e22e",
            "d6e21f",
            "fff705",
            "ffd611",
            "ffb613",
            "ff8b13",
            "ff6e08",
            "ff500d",
            "ff0000",
            "de0101",
            "c21301",
            "a71001",
            "911003",
        ],
        "name": "MODIS LST Day",
    },
    "MODIS/061/MOD10A1": {
        "bands": ["NDSI_Snow_Cover"],
        "min": 0,
        "max": 100,
        "palette": ["000000", "0000ff", "00ffff", "ffffff"],
        "name": "MODIS Snow Cover",
    },
    "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG": {
        "bands": ["avg_rad"],
        "min": 0,
        "max": 60,
        "name": "VIIRS Nightlights",
    },
    "COPERNICUS/S1_GRD": {
        "bands": ["VV"],
        "min": -25,
        "max": 0,
        "name": "Sentinel-1 SAR",
    },
    "UCSB-CHG/CHIRPS/DAILY": {
        "bands": ["precipitation"],
        "min": 0,
        "max": 50,
        "palette": [
            "FFFFFF",
            "00FFFF",
            "0080FF",
            "DA00FF",
            "FFA400",
            "FF0000",
        ],
        "name": "CHIRPS Rainfall",
    },
    "ECMWF/ERA5_LAND/DAILY_AGGR": {
        "bands": ["temperature_2m"],
        "min": 250,
        "max": 320,
        "palette": [
            "000080",
            "0000D9",
            "4000FF",
            "8000FF",
            "0080FF",
            "00FFFF",
            "00FF80",
            "80FF00",
            "DAFF00",
            "FFFF00",
            "FFF500",
            "FFDA00",
            "FFB000",
            "FFA400",
            "FF4F00",
            "FF2500",
            "FF0A00",
            "FF00FF",
        ],
        "name": "ERA5 Temperature",
    },
    "MODIS/061/MCD64A1": {
        "bands": ["BurnDate"],
        "min": 1,
        "max": 366,
        "palette": ["FF0000", "FFA500", "FFFF00"],
        "name": "MODIS Burned Area",
    },
    "COPERNICUS/S5P/OFFL/L3_NO2": {
        "bands": ["tropospheric_NO2_column_number_density"],
        "min": 0,
        "max": 0.0002,
        "palette": [
            "black",
            "blue",
            "purple",
            "cyan",
            "green",
            "yellow",
            "red",
        ],
        "name": "Sentinel-5P NO2",
    },
    "WorldPop/GP/100m/pop": {
        "bands": ["population"],
        "min": 0,
        "max": 50,
        "palette": ["24126c", "1fff6f", "d4ff50"],
        "name": "WorldPop Population",
    },
    "USGS/NLCD_RELEASES/2021_REL/NLCD": {
        "bands": ["landcover"],
        "name": "NLCD Land Cover",
    },
    "MODIS/061/MOD17A2H": {
        "bands": ["Gpp"],
        "min": 0,
        "max": 600,
        "palette": ["bbe029", "0a9501", "074b03"],
        "name": "MODIS GPP",
    },
    "MODIS/061/MOD15A2H": {
        "bands": ["Lai_500m"],
        "min": 0,
        "max": 100,
        "palette": ["e1e4b4", "99b718", "207401"],
        "name": "MODIS LAI",
    },
}


def _initialize_ee() -> bool:
    """Initialize Earth Engine if not already done."""
    try:
        import ee

        try:
            # Check if already initialized
            ee.Number(1).getInfo()
        except Exception:
            ee.Initialize(project="ee-giswqs")
        return True
    except Exception as e:
        logger.warning(f"Failed to initialize Earth Engine: {e}")
        return False


class GEEAgent:
    """Agent for searching and preparing Google Earth Engine datasets."""

    def __init__(self) -> None:
        self._ee_initialized = _initialize_ee()
        if self._ee_initialized:
            logger.info("GEE Agent initialized successfully")
        else:
            logger.warning("GEE Agent: Earth Engine not available")

    @property
    def available(self) -> bool:
        """Return True if Earth Engine is initialized and ready."""
        return self._ee_initialized

    def search_data(self, plan: PlannerOutput) -> DataResult:
        """Search GEE for data matching the plan.

        Args:
            plan: Structured query plan from the planner.

        Returns:
            DataResult with GEE search results.
        """
        import ee

        if not self.available:
            return DataResult(
                items=[],
                metadata={"error": "Earth Engine not initialized"},
                data_type="gee",
                total_items=0,
            )

        try:
            gee_collection_id = self._resolve_gee_collection(plan)
            if not gee_collection_id:
                return DataResult(
                    items=[],
                    metadata={"error": "No matching GEE collection found"},
                    data_type="gee",
                    total_items=0,
                )

            # Try as ImageCollection first, fall back to Image
            is_image_collection = True
            size = 0
            items_info: List[Dict[str, Any]] = []
            try:
                collection = ee.ImageCollection(gee_collection_id)

                # Apply spatial filter
                if plan.location and "bbox" in plan.location:
                    bbox = plan.location["bbox"]
                    geometry = ee.Geometry.Rectangle(bbox)
                    collection = collection.filterBounds(geometry)

                # Apply temporal filter
                if plan.time_range:
                    start = plan.time_range.get("start_date")
                    end = plan.time_range.get("end_date")
                    if start and end:
                        collection = collection.filterDate(
                            str(start), str(end)
                        )

                # Cloud filter for optical imagery
                max_cloud = plan.parameters.get("max_cloud_cover")
                if max_cloud is not None:
                    cloud_prop = "CLOUDY_PIXEL_PERCENTAGE"
                    if "LANDSAT" in gee_collection_id:
                        cloud_prop = "CLOUD_COVER"
                    collection = collection.filter(
                        ee.Filter.lt(cloud_prop, max_cloud)
                    )

                size = collection.size().getInfo()

                # Get first few items info
                if size > 0:
                    limit = min(size, 10)
                    sample = collection.limit(limit).toList(limit)
                    for i in range(limit):
                        try:
                            img = ee.Image(sample.get(i))
                            props = img.getInfo()["properties"]
                            item: Dict[str, Any] = {
                                "id": props.get(
                                    "system:index", f"item_{i}"
                                ),
                                "collection": gee_collection_id,
                                "properties": {
                                    "datetime": props.get(
                                        "system:time_start"
                                    ),
                                    "cloud_cover": props.get(
                                        "CLOUDY_PIXEL_PERCENTAGE",
                                        props.get("CLOUD_COVER"),
                                    ),
                                },
                                "data_source": "gee",
                            }
                            items_info.append(item)
                        except Exception:
                            break

            except Exception:
                # It's a single Image (like SRTM, WorldCover, etc.)
                is_image_collection = False
                try:
                    image = ee.Image(gee_collection_id)
                    info = image.getInfo()
                    size = 1
                    items_info = [
                        {
                            "id": gee_collection_id.split("/")[-1],
                            "collection": gee_collection_id,
                            "properties": info.get("properties", {}),
                            "data_source": "gee",
                        }
                    ]
                except Exception as e2:
                    return DataResult(
                        items=[],
                        metadata={"error": str(e2)},
                        data_type="gee",
                        total_items=0,
                    )

            vis_params = dict(GEE_VIS_PARAMS.get(gee_collection_id, {}))

            return DataResult(
                items=items_info,
                metadata={
                    "gee_collection": gee_collection_id,
                    "is_image_collection": is_image_collection,
                    "vis_params": vis_params,
                    "total_in_catalog": size,
                    "data_source": "gee",
                },
                data_type="gee",
                total_items=min(size, 10),
                search_query={
                    "collection": gee_collection_id,
                    "bbox": (
                        plan.location.get("bbox") if plan.location else None
                    ),
                    "time_range": plan.time_range,
                },
            )

        except Exception as e:
            logger.error(f"GEE search failed: {e}")
            return DataResult(
                items=[],
                metadata={"error": str(e)},
                data_type="gee",
                total_items=0,
            )

    def _resolve_gee_collection(
        self, plan: PlannerOutput
    ) -> Optional[str]:
        """Resolve a GEE collection ID from the plan.

        Args:
            plan: Structured query plan.

        Returns:
            GEE collection ID string or None.
        """
        # Check if plan.dataset is already a GEE collection ID
        if plan.dataset and "/" in plan.dataset:
            return plan.dataset

        # Check direct mapping
        dataset_lower = (plan.dataset or "").lower()
        for key, gee_id in GEE_COLLECTIONS.items():
            if key == dataset_lower or dataset_lower.startswith(key):
                return gee_id

        # Check by analysis type / intent keywords
        analysis = (plan.analysis_type or "").lower()
        intent = plan.intent.lower()
        query_lower = f"{intent} {analysis} {dataset_lower}"

        mapping: Dict[str, str] = {
            "dynamic world": "GOOGLE/DYNAMICWORLD/V1",
            "worldcover": "ESA/WorldCover/v200",
            "world cover": "ESA/WorldCover/v200",
            "nlcd": "USGS/NLCD_RELEASES/2021_REL/NLCD",
            "srtm": "USGS/SRTMGL1_003",
            "chirps": "UCSB-CHG/CHIRPS/DAILY",
            "era5": "ECMWF/ERA5_LAND/DAILY_AGGR",
            "no2": "COPERNICUS/S5P/OFFL/L3_NO2",
            "air quality": "COPERNICUS/S5P/OFFL/L3_NO2",
            "population": "WorldPop/GP/100m/pop",
            "rainfall": "UCSB-CHG/CHIRPS/DAILY",
            "precipitation": "UCSB-CHG/CHIRPS/DAILY",
        }

        for keyword, gee_id in mapping.items():
            if keyword in query_lower:
                return gee_id

        # Fallback mapping from analysis_type
        type_mapping: Dict[str, str] = {
            "land_cover": "GOOGLE/DYNAMICWORLD/V1",
            "elevation": "USGS/SRTMGL1_003",
            "ndvi": "MODIS/061/MOD13Q1",
            "fire_detection": "MODIS/061/MOD14A1",
            "water_mapping": "JRC/GSW1_4/GlobalSurfaceWater",
            "snow_cover": "MODIS/061/MOD10A1",
            "surface_temperature": "MODIS/061/MOD11A1",
            "nightlights": "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG",
        }
        if analysis in type_mapping:
            return type_mapping[analysis]

        return None

    def create_visualization(
        self,
        plan: PlannerOutput,
        data: DataResult,
        target_map: Any = None,
    ) -> Any:
        """Create a geemap visualization for GEE data.

        Args:
            plan: Structured query plan.
            data: GEE data result from search_data().
            target_map: Optional existing map to add layers to.

        Returns:
            A geemap Map instance with the GEE layer added.
        """
        import ee
        import geemap.maplibregl as geemap_mod

        gee_collection_id = data.metadata.get("gee_collection")
        if not gee_collection_id:
            return None

        vis_params = dict(data.metadata.get("vis_params", {}))
        vis_name = vis_params.pop(
            "name", gee_collection_id.split("/")[-1]
        )
        is_image_collection = data.metadata.get(
            "is_image_collection", True
        )

        # Create or reuse map
        m = target_map if target_map is not None else geemap_mod.Map()

        # Build the EE image to display
        try:
            if is_image_collection:
                collection = ee.ImageCollection(gee_collection_id)

                if plan.location and "bbox" in plan.location:
                    bbox = plan.location["bbox"]
                    geometry = ee.Geometry.Rectangle(bbox)
                    collection = collection.filterBounds(geometry)

                if plan.time_range:
                    start = plan.time_range.get("start_date")
                    end = plan.time_range.get("end_date")
                    if start and end:
                        collection = collection.filterDate(
                            str(start), str(end)
                        )

                max_cloud = plan.parameters.get("max_cloud_cover")
                if max_cloud is not None:
                    cloud_prop = "CLOUDY_PIXEL_PERCENTAGE"
                    if "LANDSAT" in gee_collection_id:
                        cloud_prop = "CLOUD_COVER"
                    collection = collection.filter(
                        ee.Filter.lt(cloud_prop, max_cloud)
                    )

                # Use median composite for visualization
                image = collection.median()
            else:
                image = ee.Image(gee_collection_id)

            m.add_ee_layer(image, vis_params, vis_name)

            # Center map on location
            if plan.location and "bbox" in plan.location:
                bbox = plan.location["bbox"]
                center_lon = (bbox[0] + bbox[2]) / 2
                center_lat = (bbox[1] + bbox[3]) / 2
                m.set_center(center_lon, center_lat, zoom=10)

        except Exception as e:
            logger.error(f"GEE visualization failed: {e}")

        return m

    def generate_code(
        self, plan: PlannerOutput, data: DataResult
    ) -> str:
        """Generate reproducible Python code for the GEE operation.

        Args:
            plan: Structured query plan.
            data: GEE data result.

        Returns:
            Executable Python code as a string.
        """
        gee_collection_id = data.metadata.get("gee_collection", "")
        vis_params = dict(data.metadata.get("vis_params", {}))
        is_ic = data.metadata.get("is_image_collection", True)

        location_name = ""
        if plan.location:
            location_name = plan.location.get("name", "")

        bbox = plan.location.get("bbox") if plan.location else None

        vis_name = vis_params.pop(
            "name", gee_collection_id.split("/")[-1]
        )

        code_lines = [
            "import ee",
            "import geemap.maplibregl as geemap",
            "",
            "# Initialize Earth Engine",
            "ee.Initialize(project='ee-giswqs')",
            "",
        ]

        if is_ic:
            header = f"# Load and filter {vis_name}"
            if location_name:
                header += f" - {location_name}"
            code_lines += [
                header,
                f'collection = ee.ImageCollection("{gee_collection_id}")',
            ]
            if bbox:
                code_lines.append(
                    f"geometry = ee.Geometry.Rectangle({bbox})"
                )
                code_lines.append(
                    "collection = collection.filterBounds(geometry)"
                )
            if plan.time_range:
                start = plan.time_range.get("start_date", "")
                end = plan.time_range.get("end_date", "")
                if start and end:
                    code_lines.append(
                        f'collection = collection.filterDate("{start}", "{end}")'
                    )
            code_lines += [
                "",
                'print(f"Found {collection.size().getInfo()} images")',
                "",
                "# Create median composite",
                "image = collection.median()",
            ]
        else:
            code_lines += [
                f"# Load {vis_name}",
                f'image = ee.Image("{gee_collection_id}")',
            ]

        # Add visualization
        vis_str = repr(vis_params)
        code_lines += [
            "",
            "# Visualize on map",
            "m = geemap.Map()",
            f"vis_params = {vis_str}",
            f'm.add_ee_layer(image, vis_params, "{vis_name}")',
        ]
        if bbox:
            center_lon = (bbox[0] + bbox[2]) / 2
            center_lat = (bbox[1] + bbox[3]) / 2
            code_lines.append(
                f"m.set_center({center_lon}, {center_lat}, zoom=10)"
            )
        code_lines.append("m")

        return "\n".join(code_lines) + "\n"
