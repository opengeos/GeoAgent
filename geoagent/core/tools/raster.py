"""Raster Analysis Tool for GeoAgent package.

This module provides tools for raster data analysis using xarray and rioxarray.
"""

from typing import List, Dict, Optional, Any
import logging
import numpy as np

from langchain_core.tools import tool

try:
    import xarray as xr
    import rioxarray as rxr
    import rasterio
except ImportError:
    xr = rxr = rasterio = None

try:
    import geopandas as gpd
except ImportError:
    gpd = None

logger = logging.getLogger(__name__)


@tool
def load_raster(
    url_or_path: str,
    bbox: Optional[List[float]] = None,
    crs: Optional[str] = None,
    bands: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Load a Cloud Optimized GeoTIFF (COG) or other raster file via rioxarray.

    This tool loads raster data from local files or remote URLs, with optional
    spatial and band subsetting.

    Args:
        url_or_path: Path or URL to raster file (COG, GeoTIFF, NetCDF, etc.)
        bbox: Bounding box [west, south, east, north] to clip data spatially
        crs: Target CRS to reproject to (e.g., 'EPSG:4326', 'EPSG:3857')
        bands: List of band numbers to load (1-indexed). If None, loads all bands

    Returns:
        Dictionary containing:
        - dataset_info: Basic information about the dataset
        - spatial_ref: CRS and geotransform information
        - bands_info: Information about each band
        - data_summary: Summary statistics
        - success: Whether loading succeeded

    Example:
        >>> raster = load_raster(
        ...     url_or_path="https://example.com/landsat.tif",
        ...     bbox=[-120.5, 35.5, -120.0, 36.0],
        ...     bands=[4, 3, 2]  # NIR, Red, Green
        ... )
    """
    if xr is None or rxr is None:
        return {
            "error": "rioxarray and xarray are required. Install with: pip install rioxarray xarray"
        }

    try:
        # Load the dataset
        if bands:
            # Load specific bands (convert to 0-indexed)
            band_indices = [b - 1 for b in bands]
            ds = rxr.open_rasterio(url_or_path, chunks=True)[band_indices]
        else:
            ds = rxr.open_rasterio(url_or_path, chunks=True)

        # Clip to bounding box if provided
        if bbox:
            west, south, east, north = bbox
            ds = ds.rio.clip_box(minx=west, miny=south, maxx=east, maxy=north)

        # Reproject if CRS specified
        if crs and ds.rio.crs != crs:
            ds = ds.rio.reproject(crs)

        # Get dataset information
        dataset_info = {
            "shape": ds.shape,
            "dims": list(ds.dims),
            "coords": {
                k: {"size": v.size, "dtype": str(v.dtype)} for k, v in ds.coords.items()
            },
            "attrs": dict(ds.attrs),
            "dtype": str(ds.dtype),
            "nbytes": ds.nbytes,
        }

        # Spatial reference info
        spatial_ref = {
            "crs": str(ds.rio.crs) if ds.rio.crs else None,
            "transform": list(ds.rio.transform()) if ds.rio.transform() else None,
            "bounds": ds.rio.bounds() if ds.rio.bounds() else None,
            "resolution": ds.rio.resolution() if ds.rio.resolution() else None,
        }

        # Band information
        bands_info = []
        if "band" in ds.dims:
            for i, band_num in enumerate(ds.band.values):
                band_data = ds.isel(band=i)
                # Compute statistics (on a sample if data is large)
                sample_data = band_data.values.flatten()
                if len(sample_data) > 100000:
                    sample_data = np.random.choice(sample_data, 100000, replace=False)

                # Remove NaN values for statistics
                valid_data = sample_data[~np.isnan(sample_data)]

                if len(valid_data) > 0:
                    stats = {
                        "min": float(np.min(valid_data)),
                        "max": float(np.max(valid_data)),
                        "mean": float(np.mean(valid_data)),
                        "std": float(np.std(valid_data)),
                        "valid_pixels": int(len(valid_data)),
                        "total_pixels": int(len(sample_data)),
                    }
                else:
                    stats = {"error": "No valid pixels found"}

                bands_info.append({"band": int(band_num), "stats": stats})
        else:
            # Single band
            sample_data = ds.values.flatten()
            if len(sample_data) > 100000:
                sample_data = np.random.choice(sample_data, 100000, replace=False)

            valid_data = sample_data[~np.isnan(sample_data)]
            if len(valid_data) > 0:
                stats = {
                    "min": float(np.min(valid_data)),
                    "max": float(np.max(valid_data)),
                    "mean": float(np.mean(valid_data)),
                    "std": float(np.std(valid_data)),
                    "valid_pixels": int(len(valid_data)),
                    "total_pixels": int(len(sample_data)),
                }
            else:
                stats = {"error": "No valid pixels found"}

            bands_info.append({"band": 1, "stats": stats})

        # Overall data summary
        data_summary = {
            "total_bands": len(bands_info),
            "spatial_extent": spatial_ref["bounds"],
            "pixel_count": (
                dataset_info["shape"][-2] * dataset_info["shape"][-1]
                if len(dataset_info["shape"]) >= 2
                else 0
            ),
            "file_size_mb": dataset_info["nbytes"] / (1024 * 1024),
        }

        logger.info(f"Successfully loaded raster: {url_or_path}")

        return {
            "dataset_info": dataset_info,
            "spatial_ref": spatial_ref,
            "bands_info": bands_info,
            "data_summary": data_summary,
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error loading raster: {e}")
        return {"error": str(e), "url_or_path": url_or_path, "success": False}


@tool
def compute_index(
    url_or_path: str,
    index_name: str,
    band_mapping: Optional[Dict[str, int]] = None,
    bbox: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Compute spectral indices (NDVI, NDWI, EVI, NDBI, etc.) from raster band data.

    This tool computes common remote sensing indices from multi-band raster data.

    Args:
        url_or_path: Path or URL to multi-band raster file
        index_name: Name of index to compute (NDVI, NDWI, EVI, NDBI, SAVI, MSAVI, GNDVI)
        band_mapping: Dictionary mapping band names to band numbers.
                     Default Landsat-8 mapping: {"red": 4, "green": 3, "blue": 2, "nir": 5, "swir1": 6, "swir2": 7}
        bbox: Optional bounding box [west, south, east, north] to clip data

    Returns:
        Dictionary containing:
        - index_stats: Statistics of computed index values
        - spatial_ref: CRS and geotransform of result
        - formula_used: Formula that was applied
        - success: Whether computation succeeded

    Example:
        >>> ndvi = compute_index(
        ...     url_or_path="landsat8.tif",
        ...     index_name="NDVI",
        ...     band_mapping={"red": 4, "nir": 5}
        ... )
    """
    if xr is None or rxr is None:
        return {"error": "rioxarray and xarray are required"}

    # Default band mapping (Landsat-8 style)
    if band_mapping is None:
        band_mapping = {
            "red": 4,
            "green": 3,
            "blue": 2,
            "nir": 5,
            "swir1": 6,
            "swir2": 7,
        }

    # Index formulas
    index_formulas = {
        "NDVI": ("nir", "red", lambda nir, red: (nir - red) / (nir + red)),
        "NDWI": ("green", "nir", lambda green, nir: (green - nir) / (green + nir)),
        "EVI": (
            "nir",
            "red",
            "blue",
            lambda nir, red, blue: 2.5
            * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1)),
        ),
        "NDBI": ("swir1", "nir", lambda swir1, nir: (swir1 - nir) / (swir1 + nir)),
        "SAVI": ("nir", "red", lambda nir, red: 1.5 * (nir - red) / (nir + red + 0.5)),
        "MSAVI": (
            "nir",
            "red",
            lambda nir, red: (
                2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))
            )
            / 2,
        ),
        "GNDVI": ("nir", "green", lambda nir, green: (nir - green) / (nir + green)),
    }

    try:
        index_name_upper = index_name.upper()
        if index_name_upper not in index_formulas:
            return {
                "error": f"Unknown index: {index_name}. Available: {list(index_formulas.keys())}",
                "success": False,
            }

        # Load raster data
        ds = rxr.open_rasterio(url_or_path, chunks=True)

        # Clip if bbox provided
        if bbox:
            west, south, east, north = bbox
            ds = ds.rio.clip_box(minx=west, miny=south, maxx=east, maxy=north)

        # Get formula components
        formula_info = index_formulas[index_name_upper]
        required_bands = formula_info[:-1]  # All but the last element (function)
        formula_func = formula_info[-1]  # Last element is the function

        # Extract required bands
        band_arrays = []
        missing_bands = []

        for band_name in required_bands:
            if band_name in band_mapping:
                band_num = band_mapping[band_name]
                try:
                    # Convert to 0-indexed and extract band
                    band_array = ds.isel(band=band_num - 1)
                    band_arrays.append(band_array)
                except (IndexError, KeyError):
                    missing_bands.append(f"{band_name} (band {band_num})")
            else:
                missing_bands.append(band_name)

        if missing_bands:
            return {
                "error": f"Missing required bands for {index_name}: {missing_bands}",
                "available_bands": f"1 to {ds.sizes.get('band', 'unknown')}",
                "success": False,
            }

        # Compute index
        if len(band_arrays) == 2:
            index_result = formula_func(band_arrays[0], band_arrays[1])
        elif len(band_arrays) == 3:
            index_result = formula_func(band_arrays[0], band_arrays[1], band_arrays[2])
        else:
            return {
                "error": f"Unexpected number of bands required: {len(band_arrays)}",
                "success": False,
            }

        # Compute statistics
        valid_pixels = index_result.where(np.isfinite(index_result))
        valid_data = valid_pixels.values.flatten()
        valid_data = valid_data[~np.isnan(valid_data)]

        if len(valid_data) > 0:
            index_stats = {
                "min": float(np.min(valid_data)),
                "max": float(np.max(valid_data)),
                "mean": float(np.mean(valid_data)),
                "std": float(np.std(valid_data)),
                "valid_pixels": int(len(valid_data)),
                "total_pixels": int(index_result.size),
            }
        else:
            index_stats = {"error": "No valid pixels in computed index"}

        # Spatial reference info
        spatial_ref = {
            "crs": str(index_result.rio.crs) if index_result.rio.crs else None,
            "transform": (
                list(index_result.rio.transform())
                if index_result.rio.transform()
                else None
            ),
            "bounds": index_result.rio.bounds() if index_result.rio.bounds() else None,
            "shape": index_result.shape,
        }

        # Formula description
        formula_descriptions = {
            "NDVI": "(NIR - Red) / (NIR + Red)",
            "NDWI": "(Green - NIR) / (Green + NIR)",
            "EVI": "2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))",
            "NDBI": "(SWIR1 - NIR) / (SWIR1 + NIR)",
            "SAVI": "1.5 * (NIR - Red) / (NIR + Red + 0.5)",
            "MSAVI": "(2*NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR - Red))) / 2",
            "GNDVI": "(NIR - Green) / (NIR + Green)",
        }

        logger.info(f"Successfully computed {index_name} index")

        return {
            "index_stats": index_stats,
            "spatial_ref": spatial_ref,
            "formula_used": formula_descriptions.get(index_name_upper, "Unknown"),
            "bands_used": {
                name: band_mapping[name]
                for name in required_bands
                if name in band_mapping
            },
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error computing index {index_name}: {e}")
        return {"error": str(e), "index_name": index_name, "success": False}


@tool
def raster_to_array(
    url: str, bbox: Optional[List[float]] = None, max_size: int = 1000000
) -> Dict[str, Any]:
    """Simple fetch of raster data as numpy array with metadata.

    Args:
        url: URL or path to raster file
        bbox: Optional bounding box [west, south, east, north]
        max_size: Maximum number of pixels to load (for memory safety)

    Returns:
        Dictionary with numpy array data and metadata
    """
    if xr is None or rxr is None:
        return {"error": "rioxarray required"}

    try:
        # Load data
        ds = rxr.open_rasterio(url, chunks=True)

        # Clip if needed
        if bbox:
            west, south, east, north = bbox
            ds = ds.rio.clip_box(minx=west, miny=south, maxx=east, maxy=north)

        # Check size
        total_pixels = np.prod(ds.shape)
        if total_pixels > max_size:
            return {
                "error": f"Raster too large: {total_pixels} pixels (max: {max_size})",
                "shape": ds.shape,
                "suggestion": "Use a smaller bbox or increase max_size",
            }

        # Convert to numpy
        array_data = ds.values

        return {
            "array_shape": array_data.shape,
            "dtype": str(array_data.dtype),
            "crs": str(ds.rio.crs) if ds.rio.crs else None,
            "transform": list(ds.rio.transform()) if ds.rio.transform() else None,
            "bounds": ds.rio.bounds() if ds.rio.bounds() else None,
            "success": True,
            "message": f"Loaded {total_pixels:,} pixels",
        }

    except Exception as e:
        return {"error": str(e), "success": False}


@tool
def zonal_stats(
    raster_path: str,
    zones_path: str,
    stats: List[str] = ["mean", "max", "min", "std", "count"],
    raster_band: int = 1,
) -> List[Dict[str, Any]]:
    """Compute zonal statistics from raster data using vector zones.

    Args:
        raster_path: Path to raster file
        zones_path: Path to vector file with polygon zones
        stats: List of statistics to compute ('mean', 'max', 'min', 'std', 'count', 'sum')
        raster_band: Raster band number to analyze (1-indexed)

    Returns:
        List of dictionaries with zonal statistics for each polygon
    """
    if not all([xr, rxr, gpd, rasterio]):
        return [
            {
                "error": "Missing required packages: xarray, rioxarray, geopandas, rasterio"
            }
        ]

    try:
        # Load vector zones
        zones = gpd.read_file(zones_path)

        # Load raster metadata
        with rasterio.open(raster_path) as src:
            nodata_value = src.nodata
            raster_crs = src.crs

        # Reproject zones to raster CRS if needed
        if zones.crs != raster_crs:
            zones = zones.to_crs(raster_crs)

        results = []

        for idx, zone in zones.iterrows():
            try:
                # Mask raster to polygon
                from rasterio.mask import mask

                zone_geom = [zone.geometry.__geo_interface__]
                with rasterio.open(raster_path) as src:
                    masked_data, masked_transform = mask(
                        src,
                        zone_geom,
                        crop=True,
                        indexes=[raster_band],
                    )

                # Flatten and remove nodata
                values = masked_data.flatten()
                values = (
                    values[values != nodata_value]
                    if nodata_value is not None
                    else values
                )
                values = values[~np.isnan(values)]

                # Compute statistics
                zone_stats = {"zone_id": idx}

                if len(values) > 0:
                    if "mean" in stats:
                        zone_stats["mean"] = float(np.mean(values))
                    if "max" in stats:
                        zone_stats["max"] = float(np.max(values))
                    if "min" in stats:
                        zone_stats["min"] = float(np.min(values))
                    if "std" in stats:
                        zone_stats["std"] = float(np.std(values))
                    if "sum" in stats:
                        zone_stats["sum"] = float(np.sum(values))
                    if "count" in stats:
                        zone_stats["count"] = int(len(values))
                else:
                    # No valid pixels
                    for stat in stats:
                        zone_stats[stat] = None
                    zone_stats["count"] = 0

                # Add zone attributes if they exist
                for col in zones.columns:
                    if col != "geometry":
                        zone_stats[col] = zone[col]

                results.append(zone_stats)

            except Exception as e:
                results.append({"zone_id": idx, "error": str(e)})

        return results

    except Exception as e:
        logger.error(f"Error computing zonal statistics: {e}")
        return [{"error": str(e)}]
