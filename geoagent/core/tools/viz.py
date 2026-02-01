"""Visualization Tool for GeoAgent package.

This module provides tools for creating interactive maps using leafmap's MapLibre backend.
"""

from typing import List, Dict, Optional, Any
import logging
import tempfile
import os
import json

from langchain_core.tools import tool

try:
    from leafmap.maplibregl import Map
    import leafmap
except ImportError:
    Map = leafmap = None

try:
    import geopandas as gpd
    import pandas as pd
except ImportError:
    gpd = pd = None

logger = logging.getLogger(__name__)


@tool
def show_on_map(
    layers: List[Dict[str, Any]],
    center: Optional[List[float]] = None,
    zoom: Optional[int] = None,
    basemap: str = "liberty",
    map_width: str = "100%",
    map_height: str = "600px",
    style: str = "liberty",
) -> Dict[str, Any]:
    """Create an interactive MapLibre map with multiple data layers.

    This tool creates a web map using leafmap's MapLibre backend with vector and raster layers
    for spatial data visualization with 3D terrain support.

    Args:
        layers: List of layer dictionaries, each containing:
               - type: 'vector', 'raster', 'cog', 'pmtiles', or 'stac'
               - data: Path/URL to data or GeoDataFrame
               - name: Display name for layer
               - style: Optional styling dict for vector layers
               - colormap: Optional colormap for raster layers
        center: Map center coordinates [latitude, longitude]
        zoom: Initial zoom level (1-20)
        basemap: Base map style ('liberty', 'positron', 'dark-matter', 'satellite', 'terrain')
        map_width: Map width (CSS format, e.g., '800px', '100%')
        map_height: Map height (CSS format, e.g., '600px', '80vh')
        style: MapLibre map style ('liberty', 'positron', 'dark-matter')

    Returns:
        Dictionary containing:
        - map_html: HTML representation of the map
        - success: Whether map creation succeeded
        - layer_count: Number of layers added
        - map_bounds: Final map bounds

    Example:
        >>> map_result = show_on_map([
        ...     {
        ...         "type": "vector",
        ...         "data": "/path/to/buildings.geojson",
        ...         "name": "Buildings",
        ...         "style": {"color": "red", "fillOpacity": 0.7}
        ...     },
        ...     {
        ...         "type": "cog",
        ...         "data": "https://example.com/satellite.tif",
        ...         "name": "Satellite Imagery",
        ...         "colormap": "viridis"
        ...     }
        ... ])
    """
    if Map is None:
        return {
            "error": "leafmap with MapLibre backend is required. Install with: pip install leafmap maplibre-gl-jupyter"
        }

    try:
        # Create base map with MapLibre
        if center:
            lat, lon = center
        else:
            lat, lon = 40.0, -100.0  # Default center (US)

        if zoom is None:
            zoom = 10

        # Create MapLibre Map
        m = Map(
            center=[lon, lat],  # MapLibre uses [lon, lat] format
            zoom=zoom,
            style=style,
            width=map_width,
            height=map_height,
        )

        # Add layers
        layers_added = 0
        all_bounds = []

        for layer in layers:
            try:
                layer_type = layer.get("type", "vector")
                data = layer.get("data")
                name = layer.get("name", f"Layer {layers_added + 1}")

                if layer_type == "vector":
                    # Add vector layer using add_geojson
                    style_dict = layer.get(
                        "style",
                        {
                            "color": "#3388ff",
                            "weight": 2,
                            "fillColor": "#3388ff",
                            "fillOpacity": 0.5,
                        },
                    )

                    if isinstance(data, str):
                        # Load from file/URL and convert to GeoJSON
                        if gpd is not None:
                            gdf = gpd.read_file(data)
                            if not gdf.empty:
                                # Convert to GeoJSON format for MapLibre
                                geojson_data = json.loads(gdf.to_json())
                                m.add_geojson(geojson_data, layer_id=name, **style_dict)
                                bounds = gdf.total_bounds
                                all_bounds.append(bounds)
                            else:
                                logger.warning(f"Empty dataset for layer {name}")
                                continue
                    elif gpd is not None and isinstance(data, gpd.GeoDataFrame):
                        # Use GeoDataFrame directly
                        if not data.empty:
                            geojson_data = json.loads(data.to_json())
                            m.add_geojson(geojson_data, layer_id=name, **style_dict)
                            bounds = data.total_bounds
                            all_bounds.append(bounds)
                        else:
                            logger.warning(f"Empty GeoDataFrame for layer {name}")
                            continue
                    else:
                        logger.warning(
                            f"Cannot add vector layer {name}: unsupported data type"
                        )
                        continue

                elif layer_type in ["raster", "cog"]:
                    # Add raster/COG layer
                    colormap = layer.get("colormap", "viridis")
                    rescale = layer.get("rescale")
                    opacity = layer.get("opacity", 0.8)

                    if isinstance(data, str):
                        m.add_cog_layer(
                            url=data,
                            name=name,
                            colormap=colormap,
                            rescale=rescale,
                            opacity=opacity,
                        )
                    else:
                        logger.warning(
                            f"Cannot add raster layer {name}: data must be URL/path"
                        )
                        continue

                elif layer_type == "pmtiles":
                    # Add PMTiles layer
                    style_dict = layer.get("style", {})

                    if isinstance(data, str):
                        m.add_pmtiles(url=data, name=name, style=style_dict)
                    else:
                        logger.warning(
                            f"Cannot add PMTiles layer {name}: data must be URL"
                        )
                        continue

                elif layer_type == "stac":
                    # Add STAC item layer using COG
                    colormap = layer.get("colormap", "viridis")

                    if isinstance(data, str):
                        # Assume data is the direct STAC item COG URL
                        m.add_cog_layer(url=data, name=name, colormap=colormap)
                    else:
                        logger.warning(
                            f"Cannot add STAC layer {name}: data must be COG URL"
                        )
                        continue

                layers_added += 1
                logger.info(f"Added layer: {name} ({layer_type})")

            except Exception as e:
                logger.error(f"Error adding layer {layer.get('name', 'unknown')}: {e}")
                continue

        # Auto-zoom to data extent if bounds available
        if all_bounds and not center:
            # Calculate overall bounds
            min_x = min(bounds[0] for bounds in all_bounds)
            min_y = min(bounds[1] for bounds in all_bounds)
            max_x = max(bounds[2] for bounds in all_bounds)
            max_y = max(bounds[3] for bounds in all_bounds)

            # Fit bounds for MapLibre (uses different format)
            m.fit_bounds([[min_x, min_y], [max_x, max_y]])

        # Get map HTML
        try:
            # For MapLibre, we use the widget representation
            map_html = m._repr_html_()
            if not map_html:
                # Fallback method for MapLibre
                temp_file = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
                m.to_html(temp_file.name)
                with open(temp_file.name, "r") as f:
                    map_html = f.read()
                os.unlink(temp_file.name)
        except Exception as e:
            logger.warning(f"Could not generate HTML representation: {e}")
            map_html = f"<p>MapLibre map created successfully with {layers_added} layers, but HTML export failed.</p>"

        # Calculate final bounds
        final_bounds = None
        if all_bounds:
            min_x = min(bounds[0] for bounds in all_bounds)
            min_y = min(bounds[1] for bounds in all_bounds)
            max_x = max(bounds[2] for bounds in all_bounds)
            max_y = max(bounds[3] for bounds in all_bounds)
            final_bounds = [min_x, min_y, max_x, max_y]

        logger.info(f"Successfully created MapLibre map with {layers_added} layers")

        return {
            "map_html": map_html,
            "success": True,
            "layer_count": layers_added,
            "map_bounds": final_bounds,
            "center": [lat, lon],
            "zoom": zoom,
            "style": style,
            "backend": "maplibre",
        }

    except Exception as e:
        logger.error(f"Error creating MapLibre map: {e}")
        return {"error": str(e), "success": False, "layer_count": 0}


@tool
def add_cog_layer(
    url: str,
    name: str = "COG Layer",
    colormap: str = "viridis",
    rescale: Optional[List[float]] = None,
    opacity: float = 0.8,
    nodata: Optional[float] = None,
    zoom_to_layer: bool = True,
) -> Dict[str, Any]:
    """Add a Cloud Optimized GeoTIFF (COG) tile layer using MapLibre backend.

    Args:
        url: URL to COG file
        name: Display name for the layer
        colormap: Colormap name ('viridis', 'plasma', 'terrain', 'magma', etc.)
        rescale: Value range to rescale data [min, max]
        opacity: Layer opacity (0.0 to 1.0)
        nodata: NoData value to treat as transparent
        zoom_to_layer: Whether to zoom to layer extent

    Returns:
        Dictionary with layer addition results
    """
    if Map is None:
        return {"error": "leafmap with MapLibre backend is required"}

    try:
        # Create new MapLibre map
        m = Map(style="liberty")

        # Add COG layer using MapLibre backend
        m.add_cog_layer(
            url=url,
            name=name,
            colormap=colormap,
            rescale=rescale,
            opacity=opacity,
            nodata=nodata,
            zoom_to_layer=zoom_to_layer,
        )

        # Generate map HTML
        try:
            map_html = m._repr_html_()
        except Exception:
            map_html = (
                f"<p>COG layer '{name}' added successfully but HTML export failed.</p>"
            )

        # Get layer info
        layer_info = {
            "name": name,
            "url": url,
            "colormap": colormap,
            "rescale": rescale,
            "opacity": opacity,
            "nodata": nodata,
            "type": "cog",
            "backend": "maplibre",
        }

        return {"layer_added": layer_info, "map_html": map_html, "success": True}

    except Exception as e:
        logger.error(f"Error adding COG layer: {e}")
        return {"error": str(e), "success": False}


@tool
def add_vector_layer(
    data_path: str,
    name: str = "Vector Layer",
    style: Optional[Dict[str, Any]] = None,
    popup_fields: Optional[List[str]] = None,
    zoom_to_layer: bool = True,
) -> Dict[str, Any]:
    """Add vector data as a layer to a MapLibre map.

    Args:
        data_path: Path to vector data file
        name: Display name for layer
        style: Styling dictionary with MapLibre paint properties
        popup_fields: List of attribute fields to show in feature popups
        zoom_to_layer: Whether to zoom to layer extent

    Returns:
        Dictionary with layer addition results
    """
    if Map is None or gpd is None:
        return {"error": "leafmap with MapLibre backend and geopandas are required"}

    try:
        # Load vector data
        gdf = gpd.read_file(data_path)

        if gdf.empty:
            return {"error": "Vector data is empty", "success": False}

        # Create MapLibre map
        m = Map(style="liberty")

        # Default style for MapLibre
        if style is None:
            style = {
                "color": "#3388ff",
                "weight": 2,
                "fillColor": "#3388ff",
                "fillOpacity": 0.5,
            }

        # Convert to GeoJSON for MapLibre
        geojson_data = json.loads(gdf.to_json())

        # Add vector layer using add_geojson
        m.add_geojson(data=geojson_data, layer_id=name, **style)

        # Zoom to layer extent if requested
        if zoom_to_layer and not gdf.empty:
            bounds = gdf.total_bounds
            # MapLibre uses [[west, south], [east, north]] format
            m.fit_bounds([[bounds[0], bounds[1]], [bounds[2], bounds[3]]])

        # Generate map HTML
        try:
            map_html = m._repr_html_()
        except Exception:
            map_html = f"<p>Vector layer '{name}' added successfully but HTML export failed.</p>"

        layer_info = {
            "name": name,
            "feature_count": len(gdf),
            "geometry_types": gdf.geometry.type.unique().tolist(),
            "bounds": gdf.total_bounds.tolist() if not gdf.empty else None,
            "style": style,
            "popup_fields": popup_fields,
            "type": "vector",
            "backend": "maplibre",
        }

        return {"layer_added": layer_info, "map_html": map_html, "success": True}

    except Exception as e:
        logger.error(f"Error adding vector layer: {e}")
        return {"error": str(e), "success": False}


@tool
def split_map(
    left_url: str,
    right_url: str,
    left_label: str = "Left Layer",
    right_label: str = "Right Layer",
    center: Optional[List[float]] = None,
    zoom: int = 10,
    left_colormap: str = "viridis",
    right_colormap: str = "viridis",
) -> Dict[str, Any]:
    """Create a split-panel comparison map using MapLibre backend.

    Args:
        left_url: URL to left panel raster (COG)
        right_url: URL to right panel raster (COG)
        left_label: Label for left panel
        right_label: Label for right panel
        center: Map center [latitude, longitude]
        zoom: Initial zoom level
        left_colormap: Colormap for left layer
        right_colormap: Colormap for right layer

    Returns:
        Dictionary with split map results

    Example:
        >>> split_result = split_map(
        ...     left_url="https://example.com/before.tif",
        ...     right_url="https://example.com/after.tif",
        ...     left_label="Before",
        ...     right_label="After"
        ... )
    """
    if Map is None:
        return {"error": "leafmap with MapLibre backend is required"}

    try:
        # Set default center if not provided
        if center is None:
            center = [40.0, -100.0]

        lat, lon = center

        # Create MapLibre split/compare map
        m = Map(
            center=[lon, lat], zoom=zoom, style="liberty"  # MapLibre uses [lon, lat]
        )

        # Add compare/swipe functionality with two COG layers
        m.add_cog_layer(
            url=left_url,
            name=left_label,
            colormap=left_colormap,
            compare=True,
            position="left",
        )

        m.add_cog_layer(
            url=right_url,
            name=right_label,
            colormap=right_colormap,
            compare=True,
            position="right",
        )

        # Enable split/swipe control
        try:
            m.add_compare_control()
        except AttributeError:
            # Fallback if compare control method doesn't exist
            logger.warning("Compare control not available, created side-by-side layers")

        # Generate HTML
        try:
            map_html = m._repr_html_()
        except Exception:
            map_html = (
                "<p>MapLibre split map created successfully but HTML export failed.</p>"
            )

        return {
            "map_html": map_html,
            "left_url": left_url,
            "right_url": right_url,
            "left_label": left_label,
            "right_label": right_label,
            "left_colormap": left_colormap,
            "right_colormap": right_colormap,
            "center": center,
            "zoom": zoom,
            "type": "split_map",
            "backend": "maplibre",
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error creating MapLibre split map: {e}")
        return {"error": str(e), "success": False}


@tool
def create_choropleth_map(
    data_path: str,
    column: str,
    scheme: str = "quantiles",
    k: int = 5,
    colormap: str = "viridis",
    legend_title: Optional[str] = None,
    opacity: float = 0.8,
) -> Dict[str, Any]:
    """Create a choropleth map using MapLibre backend from vector data with numeric attributes.

    Args:
        data_path: Path to vector data file
        column: Name of numeric column to visualize
        scheme: Classification scheme ('quantiles', 'equal_interval', 'natural_breaks')
        k: Number of classes
        colormap: Color scheme name
        legend_title: Custom legend title
        opacity: Fill opacity (0.0 to 1.0)

    Returns:
        Dictionary with choropleth map results
    """
    if not all([Map, gpd]):
        return {"error": "leafmap with MapLibre backend and geopandas are required"}

    try:
        # Load data
        gdf = gpd.read_file(data_path)

        if column not in gdf.columns:
            return {
                "error": f"Column '{column}' not found in data",
                "available_columns": list(gdf.columns),
                "success": False,
            }

        # Check if column is numeric
        if not pd.api.types.is_numeric_dtype(gdf[column]):
            return {
                "error": f"Column '{column}' is not numeric",
                "column_type": str(gdf[column].dtype),
                "success": False,
            }

        # Create MapLibre map
        m = Map(style="liberty")

        # Create choropleth using MapLibre's data-driven styling
        try:
            # Use leafmap's choropleth functionality for MapLibre
            m.add_choropleth(
                data=gdf,
                column=column,
                scheme=scheme,
                k=k,
                colormap=colormap,
                legend_title=legend_title or column,
                opacity=opacity,
            )
        except AttributeError:
            # Fallback: add as styled GeoJSON
            from matplotlib import cm
            import numpy as np

            # Classify data
            if scheme == "quantiles":
                breaks = gdf[column].quantile(np.linspace(0, 1, k + 1)).tolist()
            elif scheme == "equal_interval":
                min_val, max_val = gdf[column].min(), gdf[column].max()
                breaks = np.linspace(min_val, max_val, k + 1).tolist()
            else:  # natural_breaks fallback to quantiles
                breaks = gdf[column].quantile(np.linspace(0, 1, k + 1)).tolist()

            # Get colormap
            cmap = cm.get_cmap(colormap)
            colors = [cmap(i / (k - 1)) for i in range(k)]
            hex_colors = [
                f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
                for c in colors
            ]

            # Classify features and assign colors
            gdf_copy = gdf.copy()
            gdf_copy["color_class"] = pd.cut(
                gdf_copy[column], bins=breaks, labels=hex_colors, include_lowest=True
            )
            gdf_copy["color_class"] = gdf_copy["color_class"].astype(str)

            # Convert to GeoJSON with color styling
            geojson_data = json.loads(gdf_copy.to_json())

            # Add styled layer
            m.add_geojson(
                data=geojson_data,
                layer_id=f"Choropleth: {column}",
                paint={
                    "fill-color": ["get", "color_class"],
                    "fill-opacity": opacity,
                    "stroke-color": "#333333",
                    "stroke-width": 1,
                },
            )

        # Zoom to data extent
        if not gdf.empty:
            bounds = gdf.total_bounds
            m.fit_bounds([[bounds[0], bounds[1]], [bounds[2], bounds[3]]])

        # Calculate statistics
        stats = {
            "min": float(gdf[column].min()),
            "max": float(gdf[column].max()),
            "mean": float(gdf[column].mean()),
            "median": float(gdf[column].median()),
            "std": float(gdf[column].std()),
        }

        # Generate HTML
        try:
            map_html = m._repr_html_()
        except Exception:
            map_html = "<p>MapLibre choropleth map created successfully but HTML export failed.</p>"

        return {
            "map_html": map_html,
            "column": column,
            "scheme": scheme,
            "classes": k,
            "colormap": colormap,
            "statistics": stats,
            "feature_count": len(gdf),
            "backend": "maplibre",
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error creating MapLibre choropleth map: {e}")
        return {"error": str(e), "success": False}


@tool
def add_pmtiles_layer(
    url: str,
    name: str = "PMTiles Layer",
    style: Optional[Dict[str, Any]] = None,
    source_layer: Optional[str] = None,
    zoom_to_layer: bool = True,
) -> Dict[str, Any]:
    """Add a PMTiles layer using MapLibre backend.

    PMTiles is a single-file archive format for storing and serving map tiles,
    optimized for cloud storage and efficient vector tile delivery.

    Args:
        url: URL to PMTiles archive (.pmtiles file)
        name: Display name for the layer
        style: MapLibre style specification for the layer
        source_layer: Source layer name within the PMTiles archive
        zoom_to_layer: Whether to zoom to layer extent

    Returns:
        Dictionary with layer addition results

    Example:
        >>> pmtiles_result = add_pmtiles_layer(
        ...     url="https://example.com/buildings.pmtiles",
        ...     name="Buildings",
        ...     style={
        ...         "fill-color": "#ff6b35",
        ...         "fill-opacity": 0.7,
        ...         "stroke-color": "#333333"
        ...     }
        ... )
    """
    if Map is None:
        return {"error": "leafmap with MapLibre backend is required"}

    try:
        # Create MapLibre map
        m = Map(style="liberty")

        # Default style for PMTiles
        if style is None:
            style = {
                "fill-color": "#3388ff",
                "fill-opacity": 0.6,
                "stroke-color": "#333333",
                "stroke-width": 1,
            }

        # Add PMTiles layer
        m.add_pmtiles(
            url=url,
            name=name,
            style=style,
            source_layer=source_layer,
            zoom_to_layer=zoom_to_layer,
        )

        # Generate map HTML
        try:
            map_html = m._repr_html_()
        except Exception:
            map_html = f"<p>PMTiles layer '{name}' added successfully but HTML export failed.</p>"

        layer_info = {
            "name": name,
            "url": url,
            "style": style,
            "source_layer": source_layer,
            "type": "pmtiles",
            "backend": "maplibre",
        }

        return {"layer_added": layer_info, "map_html": map_html, "success": True}

    except Exception as e:
        logger.error(f"Error adding PMTiles layer: {e}")
        return {"error": str(e), "success": False}


@tool
def create_3d_terrain_map(
    center: Optional[List[float]] = None,
    zoom: int = 10,
    pitch: int = 60,
    bearing: int = 0,
    terrain_source: str = "mapbox",
) -> Dict[str, Any]:
    """Create a 3D terrain map using MapLibre backend.

    Args:
        center: Map center [latitude, longitude]
        zoom: Initial zoom level
        pitch: Map pitch angle (0-60 degrees)
        bearing: Map bearing/rotation (0-360 degrees)
        terrain_source: Terrain data source ("mapbox", "terrarium")

    Returns:
        Dictionary with 3D map results
    """
    if Map is None:
        return {"error": "leafmap with MapLibre backend is required"}

    try:
        # Set default center if not provided
        if center is None:
            center = [46.0, 8.0]  # Swiss Alps default

        lat, lon = center

        # Create MapLibre map with 3D terrain
        m = Map(
            center=[lon, lat], zoom=zoom, pitch=pitch, bearing=bearing, style="liberty"
        )

        # Enable 3D terrain
        try:
            m.add_terrain_source(terrain_source)
            m.set_terrain(exaggeration=1.5)
        except AttributeError:
            logger.warning("3D terrain not supported in this version of leafmap")

        # Generate map HTML
        try:
            map_html = m._repr_html_()
        except Exception:
            map_html = (
                "<p>3D terrain map created successfully but HTML export failed.</p>"
            )

        return {
            "map_html": map_html,
            "center": center,
            "zoom": zoom,
            "pitch": pitch,
            "bearing": bearing,
            "terrain_source": terrain_source,
            "type": "3d_terrain",
            "backend": "maplibre",
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error creating 3D terrain map: {e}")
        return {"error": str(e), "success": False}


@tool
def save_map(
    map_html: str, output_path: str, title: str = "GeoAgent MapLibre Map"
) -> Dict[str, Any]:
    """Save a MapLibre map to an HTML file.

    Args:
        map_html: HTML content of the map
        output_path: Path to save HTML file
        title: HTML page title

    Returns:
        Dictionary with save results
    """
    try:
        # Create full HTML document
        full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    {map_html}
    <p style="text-align: center; color: #666; margin-top: 20px;">
        Created with GeoAgent using MapLibre GL JS
    </p>
</body>
</html>
"""

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_html)

        # Get file info
        file_size = os.path.getsize(output_path)

        return {
            "output_path": output_path,
            "file_size_bytes": file_size,
            "title": title,
            "backend": "maplibre",
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error saving map: {e}")
        return {"error": str(e), "success": False}
