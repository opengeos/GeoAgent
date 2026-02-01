"""DuckDB Spatial SQL Tool for GeoAgent package.

This module provides tools for running spatial SQL queries using DuckDB.
"""

from typing import List, Dict, Optional, Any, Union
import logging

from langchain_core.tools import tool

try:
    import duckdb
except ImportError:
    duckdb = None

logger = logging.getLogger(__name__)


def _ensure_spatial_extension(conn):
    """Ensure spatial extension is installed and loaded."""
    try:
        # Install spatial extension if not already installed
        conn.execute("INSTALL spatial")
    except Exception:
        # Extension might already be installed
        pass

    # Load spatial extension
    conn.execute("LOAD spatial")


@tool
def query_spatial_data(
    sql: str, data_path: Optional[str] = None, output_format: str = "dict"
) -> Union[List[Dict[str, Any]], str]:
    """Run DuckDB spatial SQL queries on GeoParquet, CSV, or other spatial data formats.

    This tool allows you to execute spatial SQL queries using DuckDB's spatial extension.
    It supports reading from local files, URLs (S3, HTTP), and various formats including
    GeoParquet, Parquet, CSV, and JSON.

    Args:
        sql: SQL query string to execute. Use standard SQL with spatial functions
        data_path: Path or URL to data file (GeoParquet, CSV, etc.). Can be None if
                  SQL contains complete file references
        output_format: Format for results - 'dict' for list of dicts, 'summary' for summary stats

    Returns:
        List of dictionaries with query results, or summary string if output_format='summary'

    Example:
        >>> results = query_spatial_data(
        ...     sql="SELECT * FROM read_parquet(?) WHERE ST_Contains(ST_MakeEnvelope(-122.5, 37.7, -122.3, 37.8), geom)",
        ...     data_path="s3://overturemaps-us-west-2/release/2024-07-22.0/theme=buildings/type=building/*.parquet"
        ... )

        >>> results = query_spatial_data(
        ...     sql="SELECT ST_Area(geom) as area, name FROM buildings WHERE area > 1000 ORDER BY area DESC LIMIT 10",
        ...     data_path="/path/to/buildings.geojson"
        ... )
    """
    if duckdb is None:
        return [
            {
                "error": "DuckDB is not installed. Please install with: pip install duckdb"
            }
        ]

    try:
        # Create DuckDB connection
        conn = duckdb.connect()

        # Ensure spatial extension is loaded
        _ensure_spatial_extension(conn)

        # If data_path is provided, create a temporary view
        if data_path:
            # Sanitize data_path to prevent SQL injection
            safe_path = data_path.replace("'", "''")

            # Determine file format and create appropriate read statement
            if data_path.endswith((".parquet", ".pq")):
                view_sql = (
                    f"CREATE VIEW data AS SELECT * FROM read_parquet('{safe_path}')"
                )
            elif data_path.endswith(".geojson"):
                view_sql = f"CREATE VIEW data AS SELECT * FROM st_read('{safe_path}')"
            elif data_path.endswith(".csv"):
                view_sql = (
                    f"CREATE VIEW data AS SELECT * FROM read_csv_auto('{safe_path}')"
                )
            elif data_path.endswith(".json"):
                view_sql = (
                    f"CREATE VIEW data AS SELECT * FROM read_json_auto('{safe_path}')"
                )
            elif data_path.endswith(".gpkg"):
                view_sql = f"CREATE VIEW data AS SELECT * FROM st_read('{safe_path}')"
            elif data_path.endswith(".shp"):
                view_sql = f"CREATE VIEW data AS SELECT * FROM st_read('{safe_path}')"
            else:
                # Try to auto-detect
                view_sql = f"CREATE VIEW data AS SELECT * FROM st_read('{safe_path}')"

            conn.execute(view_sql)

        # Execute the main query
        result = conn.execute(sql).fetchall()
        columns = [desc[0] for desc in conn.description]

        # Convert to list of dictionaries
        if output_format == "dict":
            results = []
            for row in result:
                row_dict = {}
                for i, col in enumerate(columns):
                    value = row[i]
                    # Convert special types to serializable formats
                    if hasattr(value, "__geo_interface__"):
                        # Geometry objects
                        row_dict[col] = value.__geo_interface__
                    elif isinstance(value, (bytes, memoryview)):
                        # Binary data (WKB)
                        row_dict[col] = (
                            value.hex() if hasattr(value, "hex") else str(value)
                        )
                    else:
                        row_dict[col] = value
                results.append(row_dict)

            logger.info(f"Query returned {len(results)} rows")
            return results

        elif output_format == "summary":
            summary = "Query executed successfully.\n"
            summary += f"Columns: {', '.join(columns)}\n"
            summary += f"Rows returned: {len(result)}\n"
            if result:
                summary += f"Sample row: {dict(zip(columns, result[0]))}"
            return summary

        else:
            return [{"error": f"Unknown output format: {output_format}"}]

    except Exception as e:
        logger.error(f"Error executing spatial SQL query: {e}")
        return [{"error": str(e), "sql": sql}]

    finally:
        try:
            conn.close()
        except Exception:
            pass


@tool
def query_overture(
    sql: str,
    bbox: Optional[List[float]] = None,
    theme: str = "buildings",
    release: str = "2024-07-22.0",  # TODO: Update to latest Overture Maps release or fetch dynamically
) -> List[Dict[str, Any]]:
    """Query Overture Maps data using DuckDB spatial SQL.

    This tool provides easy access to Overture Maps datasets including buildings,
    places, transportation, and more using spatial SQL queries.

    Args:
        sql: SQL query to execute against Overture data. Use 'overture_data' as table name
        bbox: Optional bounding box [west, south, east, north] to filter results spatially
        theme: Overture theme to query ('buildings', 'places', 'transportation', 'base')
        release: Overture release version (default: '2024-07-22.0')

    Returns:
        List of dictionaries containing query results with Overture features

    Example:
        >>> buildings = query_overture(
        ...     sql="SELECT * FROM overture_data WHERE height > 50 LIMIT 100",
        ...     bbox=[-122.5, 37.7, -122.3, 37.8],
        ...     theme="buildings"
        ... )

        >>> places = query_overture(
        ...     sql="SELECT name, categories, confidence FROM overture_data WHERE 'restaurant' = ANY(categories) LIMIT 50",
        ...     theme="places"
        ... )
    """
    if duckdb is None:
        return [
            {
                "error": "DuckDB is not installed. Please install with: pip install duckdb"
            }
        ]

    try:
        # Create DuckDB connection
        conn = duckdb.connect()

        # Ensure spatial extension is loaded
        _ensure_spatial_extension(conn)

        # Build Overture data URL
        base_url = f"s3://overturemaps-us-west-2/release/{release}/theme={theme}"

        # Map themes to their type patterns
        type_patterns = {
            "buildings": "type=building",
            "places": "type=place",
            "transportation": "type=segment",
            "base": "type=land",
        }

        type_pattern = type_patterns.get(theme, "type=*")
        data_url = f"{base_url}/{type_pattern}/*.parquet"

        # Create view with spatial filtering if bbox provided
        if bbox:
            west, south, east, north = bbox
            bbox_wkt = f"POLYGON(({west} {south}, {east} {south}, {east} {north}, {west} {north}, {west} {south}))"

            view_sql = f"""
            CREATE VIEW overture_data AS
            SELECT * FROM read_parquet('{data_url}')
            WHERE ST_Intersects(geometry, ST_GeomFromText('{bbox_wkt}'))
            """
        else:
            view_sql = (
                f"CREATE VIEW overture_data AS SELECT * FROM read_parquet('{data_url}')"
            )

        conn.execute(view_sql)

        # Execute the query
        result = conn.execute(sql).fetchall()
        columns = [desc[0] for desc in conn.description]

        # Convert to list of dictionaries
        results = []
        for row in result:
            row_dict = {}
            for i, col in enumerate(columns):
                value = row[i]
                # Handle geometry and other special types
                if hasattr(value, "__geo_interface__"):
                    row_dict[col] = value.__geo_interface__
                elif isinstance(value, (bytes, memoryview)):
                    row_dict[col] = value.hex() if hasattr(value, "hex") else str(value)
                elif isinstance(value, list) and col == "categories":
                    # Handle Overture categories array
                    row_dict[col] = list(value)
                else:
                    row_dict[col] = value
            results.append(row_dict)

        logger.info(f"Overture query returned {len(results)} features")
        return results

    except Exception as e:
        logger.error(f"Error querying Overture data: {e}")
        return [{"error": str(e), "sql": sql, "theme": theme}]

    finally:
        try:
            conn.close()
        except Exception:
            pass


@tool
def analyze_spatial_data(
    data_path: str, analysis_type: str = "summary", geometry_column: str = "geom"
) -> Dict[str, Any]:
    """Analyze spatial data to get summary statistics and schema information.

    Args:
        data_path: Path or URL to spatial data file
        analysis_type: Type of analysis ('summary', 'schema', 'bounds', 'sample')
        geometry_column: Name of geometry column (default: 'geom')

    Returns:
        Dictionary with analysis results

    Example:
        >>> info = analyze_spatial_data("/path/to/data.geojson", "summary")
    """
    if duckdb is None:
        return {"error": "DuckDB is not installed"}

    try:
        conn = duckdb.connect()
        _ensure_spatial_extension(conn)

        # Create view of the data
        if data_path.endswith((".parquet", ".pq")):
            view_sql = f"CREATE VIEW data AS SELECT * FROM read_parquet('{data_path}')"
        else:
            view_sql = f"CREATE VIEW data AS SELECT * FROM st_read('{data_path}')"

        conn.execute(view_sql)

        results = {}

        if analysis_type == "schema":
            schema = conn.execute("DESCRIBE data").fetchall()
            results["schema"] = [{"column": row[0], "type": row[1]} for row in schema]

        elif analysis_type == "summary":
            # Get row count
            count = conn.execute("SELECT COUNT(*) FROM data").fetchone()[0]
            results["row_count"] = count

            # Get schema
            schema = conn.execute("DESCRIBE data").fetchall()
            results["columns"] = [row[0] for row in schema]

            # Get spatial bounds if geometry column exists
            try:
                bounds_query = f"""
                SELECT
                    ST_XMin(ST_Envelope(ST_Union({geometry_column}))) as minx,
                    ST_YMin(ST_Envelope(ST_Union({geometry_column}))) as miny,
                    ST_XMax(ST_Envelope(ST_Union({geometry_column}))) as maxx,
                    ST_YMax(ST_Envelope(ST_Union({geometry_column}))) as maxy
                FROM data
                """
                bounds = conn.execute(bounds_query).fetchone()
                if bounds:
                    results["bounds"] = {
                        "minx": bounds[0],
                        "miny": bounds[1],
                        "maxx": bounds[2],
                        "maxy": bounds[3],
                    }
            except Exception:
                # Geometry column might not exist or be named differently
                pass

        elif analysis_type == "bounds":
            try:
                bounds_query = f"""
                SELECT
                    ST_XMin(ST_Envelope(ST_Union({geometry_column}))) as minx,
                    ST_YMin(ST_Envelope(ST_Union({geometry_column}))) as miny,
                    ST_XMax(ST_Envelope(ST_Union({geometry_column}))) as maxx,
                    ST_YMax(ST_Envelope(ST_Union({geometry_column}))) as maxy
                FROM data
                """
                bounds = conn.execute(bounds_query).fetchone()
                if bounds:
                    results = {
                        "minx": bounds[0],
                        "miny": bounds[1],
                        "maxx": bounds[2],
                        "maxy": bounds[3],
                    }
            except Exception as e:
                results = {"error": f"Could not compute bounds: {e}"}

        elif analysis_type == "sample":
            sample = conn.execute("SELECT * FROM data LIMIT 5").fetchall()
            columns = [desc[0] for desc in conn.description]
            results["sample"] = []
            for row in sample:
                row_dict = dict(zip(columns, row))
                results["sample"].append(row_dict)

        return results

    except Exception as e:
        logger.error(f"Error analyzing spatial data: {e}")
        return {"error": str(e)}

    finally:
        try:
            conn.close()
        except Exception:
            pass
