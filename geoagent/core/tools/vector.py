"""Vector Analysis Tool for GeoAgent package.

This module provides tools for vector data analysis using geopandas.
"""

from typing import List, Dict, Optional, Any
import logging

from langchain_core.tools import tool

try:
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point, Polygon, LineString, MultiPolygon, box
    from shapely.ops import unary_union
    import fiona
except ImportError:
    gpd = pd = Point = Polygon = LineString = MultiPolygon = box = unary_union = (
        fiona
    ) = None

logger = logging.getLogger(__name__)


@tool
def read_vector(
    path_or_url: str,
    bbox: Optional[List[float]] = None,
    crs: Optional[str] = None,
    layer: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Read vector data from various formats (GeoJSON, GeoParquet, Shapefile, GPKG).

    This tool reads vector data and provides summary information about the dataset
    including geometry types, attribute schema, and spatial extent.

    Args:
        path_or_url: Path or URL to vector data file
        bbox: Bounding box [west, south, east, north] to filter features spatially
        crs: Target CRS to reproject to (e.g., 'EPSG:4326', 'EPSG:3857')
        layer: Layer name for multi-layer formats like GPKG
        columns: List of specific columns to load (None = all columns)

    Returns:
        Dictionary containing:
        - summary: Basic dataset information (row count, columns, geometry types)
        - schema: Information about each column
        - spatial_info: CRS, bounds, and geometry statistics
        - sample_data: First few rows as examples
        - success: Whether reading succeeded

    Example:
        >>> data_info = read_vector(
        ...     path_or_url="/path/to/buildings.geojson",
        ...     bbox=[-122.5, 37.7, -122.3, 37.8],
        ...     crs="EPSG:3857"
        ... )
    """
    if gpd is None:
        return {"error": "geopandas is required. Install with: pip install geopandas"}

    try:
        # Read vector data
        read_kwargs = {}
        if bbox:
            read_kwargs["bbox"] = bbox
        if layer:
            read_kwargs["layer"] = layer
        if columns:
            # For some formats, we need to handle column selection differently
            pass

        gdf = gpd.read_file(path_or_url, **read_kwargs)

        # Filter columns if specified
        if columns:
            # Always keep geometry column
            cols_to_keep = [col for col in columns if col in gdf.columns]
            if gdf.geometry.name not in cols_to_keep:
                cols_to_keep.append(gdf.geometry.name)
            gdf = gdf[cols_to_keep]

        # Reproject if CRS specified
        if crs and gdf.crs != crs:
            gdf = gdf.to_crs(crs)

        # Basic summary
        summary = {
            "row_count": len(gdf),
            "column_count": len(gdf.columns),
            "geometry_column": gdf.geometry.name,
            "geometry_types": gdf.geometry.type.value_counts().to_dict(),
            "has_empty_geometries": gdf.geometry.is_empty.any(),
            "has_null_geometries": gdf.geometry.isnull().any(),
        }

        # Schema information
        schema = []
        for col in gdf.columns:
            if col == gdf.geometry.name:
                col_info = {
                    "name": col,
                    "type": "geometry",
                    "geometry_type": (
                        gdf.geometry.type.mode().iloc[0] if not gdf.empty else "unknown"
                    ),
                }
            else:
                col_info = {
                    "name": col,
                    "type": str(gdf[col].dtype),
                    "null_count": gdf[col].isnull().sum(),
                    "unique_count": (
                        gdf[col].nunique() if gdf[col].dtype == "object" else None
                    ),
                }
                # Add sample values for categorical data
                if gdf[col].dtype == "object" and gdf[col].nunique() <= 10:
                    col_info["unique_values"] = gdf[col].unique().tolist()
            schema.append(col_info)

        # Spatial information
        spatial_info = {
            "crs": str(gdf.crs) if gdf.crs else None,
            "bounds": gdf.total_bounds.tolist() if not gdf.empty else None,
        }

        # Add more detailed geometry statistics
        if not gdf.empty:
            geom_stats = {}
            if gdf.geometry.type.iloc[0] in ["Polygon", "MultiPolygon"]:
                areas = gdf.geometry.area
                geom_stats.update(
                    {
                        "area_min": float(areas.min()),
                        "area_max": float(areas.max()),
                        "area_mean": float(areas.mean()),
                        "total_area": float(areas.sum()),
                    }
                )
            elif gdf.geometry.type.iloc[0] in ["LineString", "MultiLineString"]:
                lengths = gdf.geometry.length
                geom_stats.update(
                    {
                        "length_min": float(lengths.min()),
                        "length_max": float(lengths.max()),
                        "length_mean": float(lengths.mean()),
                        "total_length": float(lengths.sum()),
                    }
                )

            spatial_info["geometry_stats"] = geom_stats

        # Sample data (first 3 rows, excluding geometry for readability)
        sample_data = []
        if not gdf.empty:
            sample_df = gdf.head(3).copy()
            for idx, row in sample_df.iterrows():
                sample_row = {}
                for col in sample_df.columns:
                    if col == gdf.geometry.name:
                        # Simplify geometry representation
                        geom = row[col]
                        if geom is not None:
                            sample_row[col] = (
                                f"{geom.geom_type} (bounds: {list(geom.bounds)})"
                            )
                        else:
                            sample_row[col] = None
                    else:
                        sample_row[col] = row[col]
                sample_data.append(sample_row)

        logger.info(
            f"Successfully read vector data: {len(gdf)} features from {path_or_url}"
        )

        return {
            "summary": summary,
            "schema": schema,
            "spatial_info": spatial_info,
            "sample_data": sample_data,
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error reading vector data: {e}")
        return {"error": str(e), "path_or_url": path_or_url, "success": False}


@tool
def spatial_filter(
    data_path: str,
    geometry_wkt: str,
    predicate: str = "intersects",
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Filter vector data by spatial relationship with a geometry.

    Args:
        data_path: Path to input vector data file
        geometry_wkt: Well-Known Text representation of filter geometry
        predicate: Spatial predicate ('intersects', 'within', 'contains', 'crosses', 'touches')
        output_path: Optional path to save filtered results

    Returns:
        Dictionary with filtering results and summary statistics

    Example:
        >>> result = spatial_filter(
        ...     data_path="/path/to/buildings.geojson",
        ...     geometry_wkt="POLYGON((-122.5 37.7, -122.3 37.7, -122.3 37.8, -122.5 37.8, -122.5 37.7))",
        ...     predicate="within"
        ... )
    """
    if gpd is None:
        return {"error": "geopandas is required"}

    try:
        from shapely import wkt

        # Read input data
        gdf = gpd.read_file(data_path)
        original_count = len(gdf)

        # Parse geometry
        filter_geom = wkt.loads(geometry_wkt)

        # Convert to same CRS as data
        filter_gdf = gpd.GeoDataFrame([1], geometry=[filter_geom], crs="EPSG:4326")
        if gdf.crs != filter_gdf.crs:
            filter_gdf = filter_gdf.to_crs(gdf.crs)
        filter_geom = filter_gdf.geometry.iloc[0]

        # Apply spatial filter
        if predicate == "intersects":
            mask = gdf.geometry.intersects(filter_geom)
        elif predicate == "within":
            mask = gdf.geometry.within(filter_geom)
        elif predicate == "contains":
            mask = gdf.geometry.contains(filter_geom)
        elif predicate == "crosses":
            mask = gdf.geometry.crosses(filter_geom)
        elif predicate == "touches":
            mask = gdf.geometry.touches(filter_geom)
        else:
            return {"error": f"Unknown predicate: {predicate}"}

        filtered_gdf = gdf[mask]
        filtered_count = len(filtered_gdf)

        # Save if output path provided
        if output_path and filtered_count > 0:
            filtered_gdf.to_file(output_path)

        return {
            "original_count": original_count,
            "filtered_count": filtered_count,
            "features_removed": original_count - filtered_count,
            "filter_geometry": geometry_wkt,
            "predicate": predicate,
            "output_saved": output_path is not None and filtered_count > 0,
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error in spatial filter: {e}")
        return {"error": str(e), "success": False}


@tool
def buffer_analysis(
    data_path: str,
    distance: float,
    unit: str = "meters",
    output_path: Optional[str] = None,
    dissolve: bool = False,
) -> Dict[str, Any]:
    """Create buffer polygons around geometries.

    Args:
        data_path: Path to input vector data
        distance: Buffer distance
        unit: Distance unit ('meters', 'feet', 'degrees')
        output_path: Optional path to save buffered geometries
        dissolve: Whether to dissolve overlapping buffers into single polygons

    Returns:
        Dictionary with buffer analysis results

    Example:
        >>> buffers = buffer_analysis(
        ...     data_path="/path/to/points.geojson",
        ...     distance=1000,
        ...     unit="meters",
        ...     dissolve=True
        ... )
    """
    if gpd is None:
        return {"error": "geopandas is required"}

    try:
        # Read input data
        gdf = gpd.read_file(data_path)

        # Convert distance to appropriate units
        if unit == "meters" and gdf.crs and not gdf.crs.is_geographic:
            # Already in projected coordinates
            buffer_distance = distance
        elif unit == "meters" and gdf.crs and gdf.crs.is_geographic:
            # Convert to meters (approximate)
            # Use UTM zone for better accuracy
            # Use UTM zone for better accuracy
            utm_crs = gdf.estimate_utm_crs()
            buffer_distance = distance
        elif unit == "degrees":
            buffer_distance = distance
        elif unit == "feet":
            buffer_distance = distance * 0.3048  # Convert feet to meters
        else:
            return {"error": f"Unknown unit: {unit}"}

        # Project to appropriate CRS if needed
        original_crs = gdf.crs
        if original_crs and original_crs.is_geographic and unit != "degrees":
            # Project to UTM for metric buffers
            utm_crs = gdf.estimate_utm_crs()
            gdf = gdf.to_crs(utm_crs)

        # Create buffers
        buffered_gdf = gdf.copy()
        buffered_gdf["geometry"] = gdf.geometry.buffer(buffer_distance)

        # Dissolve if requested
        if dissolve:
            dissolved = buffered_gdf.dissolve()
            buffered_gdf = dissolved.reset_index(drop=True)

        # Project back to original CRS
        if original_crs and gdf.crs != original_crs:
            buffered_gdf = buffered_gdf.to_crs(original_crs)

        # Calculate statistics
        if not buffered_gdf.empty:
            total_area = buffered_gdf.geometry.area.sum()
            area_stats = {
                "total_area": float(total_area),
                "mean_area": float(buffered_gdf.geometry.area.mean()),
                "max_area": float(buffered_gdf.geometry.area.max()),
                "min_area": float(buffered_gdf.geometry.area.min()),
            }
        else:
            area_stats = {}

        # Save if requested
        if output_path and not buffered_gdf.empty:
            buffered_gdf.to_file(output_path)

        return {
            "input_features": len(gdf),
            "output_features": len(buffered_gdf),
            "buffer_distance": distance,
            "unit": unit,
            "dissolved": dissolve,
            "area_stats": area_stats,
            "output_saved": output_path is not None and not buffered_gdf.empty,
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error in buffer analysis: {e}")
        return {"error": str(e), "success": False}


@tool
def spatial_join(
    left_path: str,
    right_path: str,
    predicate: str = "intersects",
    how: str = "inner",
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Perform spatial join between two vector datasets.

    Args:
        left_path: Path to left (target) vector dataset
        right_path: Path to right (join) vector dataset
        predicate: Spatial predicate for join ('intersects', 'within', 'contains')
        how: Join type ('inner', 'left', 'right')
        output_path: Optional path to save joined results

    Returns:
        Dictionary with join results and statistics

    Example:
        >>> join_result = spatial_join(
        ...     left_path="/path/to/parcels.geojson",
        ...     right_path="/path/to/zoning.geojson",
        ...     predicate="within",
        ...     how="left"
        ... )
    """
    if gpd is None:
        return {"error": "geopandas is required"}

    try:
        # Read datasets
        left_gdf = gpd.read_file(left_path)
        right_gdf = gpd.read_file(right_path)

        left_count = len(left_gdf)
        right_count = len(right_gdf)

        # Ensure same CRS
        if left_gdf.crs != right_gdf.crs:
            right_gdf = right_gdf.to_crs(left_gdf.crs)

        # Perform spatial join
        joined_gdf = gpd.sjoin(left_gdf, right_gdf, predicate=predicate, how=how)

        # Calculate statistics
        joined_count = len(joined_gdf)

        # Count matches per left feature
        if how in ["inner", "left"]:
            matches_per_feature = joined_gdf.groupby(joined_gdf.index).size()
            match_stats = {
                "features_with_matches": len(matches_per_feature),
                "features_without_matches": (
                    left_count - len(matches_per_feature) if how == "left" else None
                ),
                "max_matches_per_feature": (
                    int(matches_per_feature.max())
                    if not matches_per_feature.empty
                    else 0
                ),
                "mean_matches_per_feature": (
                    float(matches_per_feature.mean())
                    if not matches_per_feature.empty
                    else 0
                ),
            }
        else:
            match_stats = {}

        # Remove spatial index columns if present
        cols_to_drop = [col for col in joined_gdf.columns if col.startswith("index_")]
        if cols_to_drop:
            joined_gdf = joined_gdf.drop(columns=cols_to_drop)

        # Save if requested
        if output_path and not joined_gdf.empty:
            joined_gdf.to_file(output_path)

        return {
            "left_features": left_count,
            "right_features": right_count,
            "joined_features": joined_count,
            "predicate": predicate,
            "join_type": how,
            "match_stats": match_stats,
            "output_columns": list(joined_gdf.columns),
            "output_saved": output_path is not None and not joined_gdf.empty,
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error in spatial join: {e}")
        return {"error": str(e), "success": False}


@tool
def analyze_geometries(
    data_path: str, analysis_type: str = "summary"
) -> Dict[str, Any]:
    """Analyze geometric properties of vector data.

    Args:
        data_path: Path to vector data file
        analysis_type: Type of analysis ('summary', 'validity', 'topology')

    Returns:
        Dictionary with geometric analysis results
    """
    if gpd is None:
        return {"error": "geopandas is required"}

    try:
        gdf = gpd.read_file(data_path)

        if analysis_type == "summary":
            # Basic geometric summary
            results = {
                "feature_count": len(gdf),
                "geometry_types": gdf.geometry.type.value_counts().to_dict(),
                "bounds": gdf.total_bounds.tolist() if not gdf.empty else None,
                "crs": str(gdf.crs) if gdf.crs else None,
            }

            # Type-specific statistics
            if not gdf.empty:
                geom_type = gdf.geometry.type.iloc[0]

                if geom_type in ["Polygon", "MultiPolygon"]:
                    areas = gdf.geometry.area
                    results["area_stats"] = {
                        "min": float(areas.min()),
                        "max": float(areas.max()),
                        "mean": float(areas.mean()),
                        "total": float(areas.sum()),
                    }

                    perimeters = gdf.geometry.length
                    results["perimeter_stats"] = {
                        "min": float(perimeters.min()),
                        "max": float(perimeters.max()),
                        "mean": float(perimeters.mean()),
                        "total": float(perimeters.sum()),
                    }

                elif geom_type in ["LineString", "MultiLineString"]:
                    lengths = gdf.geometry.length
                    results["length_stats"] = {
                        "min": float(lengths.min()),
                        "max": float(lengths.max()),
                        "mean": float(lengths.mean()),
                        "total": float(lengths.sum()),
                    }

        elif analysis_type == "validity":
            # Check geometry validity
            valid_mask = gdf.geometry.is_valid
            results = {
                "total_features": len(gdf),
                "valid_features": int(valid_mask.sum()),
                "invalid_features": int((~valid_mask).sum()),
                "validity_ratio": float(valid_mask.mean()),
            }

            # Get validation errors for invalid geometries
            if not valid_mask.all():
                invalid_gdf = gdf[~valid_mask]
                validation_errors = []
                for idx, geom in invalid_gdf.geometry.items():
                    try:
                        from shapely.validation import explain_validity

                        error = explain_validity(geom)
                        validation_errors.append({"index": idx, "error": error})
                    except Exception:
                        validation_errors.append(
                            {"index": idx, "error": "Unknown validation error"}
                        )

                results["validation_errors"] = validation_errors[
                    :10
                ]  # Limit to first 10

        elif analysis_type == "topology":
            # Topological analysis
            results = {
                "empty_geometries": int(gdf.geometry.is_empty.sum()),
                "null_geometries": int(gdf.geometry.isnull().sum()),
                "simple_geometries": (
                    int(gdf.geometry.is_simple.sum())
                    if hasattr(gdf.geometry, "is_simple")
                    else None
                ),
                "ring_geometries": (
                    int(gdf.geometry.is_ring.sum())
                    if hasattr(gdf.geometry, "is_ring")
                    else None
                ),
            }

            # Check for overlaps (expensive operation, so limit to small datasets)
            if len(gdf) < 100:
                overlap_count = 0
                for i in range(len(gdf)):
                    for j in range(i + 1, len(gdf)):
                        if gdf.geometry.iloc[i].overlaps(gdf.geometry.iloc[j]):
                            overlap_count += 1
                results["overlapping_pairs"] = overlap_count
            else:
                results["overlapping_pairs"] = "Not computed (dataset too large)"

        else:
            return {"error": f"Unknown analysis type: {analysis_type}"}

        return {"analysis_type": analysis_type, "results": results, "success": True}

    except Exception as e:
        logger.error(f"Error analyzing geometries: {e}")
        return {"error": str(e), "success": False}
