"""Shared Pydantic models for GeoAgent components.

This module contains all shared data models used across the GeoAgent pipeline
to avoid circular dependencies between modules.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Intent(str, Enum):
    """Supported query intents."""

    SEARCH = "search"
    ANALYZE = "analyze"
    VISUALIZE = "visualize"
    COMPARE = "compare"


class PlannerOutput(BaseModel):
    """Output from the Planner Agent containing structured query parameters."""

    intent: str = Field(
        description="The analysis intent (e.g., 'compute_ndvi', 'find_deforestation')"
    )
    location: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Location information (bounding box, geometry, place name)",
    )
    time_range: Optional[Dict[str, Union[str, datetime]]] = Field(
        default=None, description="Temporal range with start_date and end_date"
    )
    dataset: Optional[str] = Field(
        default=None,
        description="Preferred dataset or catalog (e.g., 'sentinel-2', 'landsat')",
    )
    analysis_type: Optional[str] = Field(
        default=None,
        description="Type of analysis requested (e.g., 'ndvi', 'change_detection')",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Additional analysis parameters"
    )
    confidence: float = Field(
        default=1.0, description="Confidence score of the parsing (0-1)"
    )


class DataResult(BaseModel):
    """Result from the Data Agent containing retrieved geospatial data."""

    items: List[Dict[str, Any]] = Field(
        description="List of STAC items or data references"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the data search"
    )
    data_type: str = Field(
        description="Type of data retrieved: 'raster', 'vector', or 'tabular'"
    )
    total_items: int = Field(description="Total number of items found")
    search_query: Optional[Dict[str, Any]] = Field(
        default=None, description="The search query that was executed"
    )


class AnalysisResult(BaseModel):
    """Result from the Analysis Agent containing computed analysis."""

    result_data: Union[Dict[str, Any], List[Any]] = Field(
        description="The computed analysis results (summary statistics, arrays, etc.)"
    )
    code_generated: str = Field(
        description="Python code that was generated and executed for transparency"
    )
    visualization_hints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Suggested visualization parameters (colormap, ranges, etc.)",
    )
    success: bool = Field(
        default=True, description="Whether the analysis completed successfully"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if analysis failed"
    )


class GeoAgentResponse(BaseModel):
    """Complete response from GeoAgent containing all pipeline results."""

    plan: PlannerOutput = Field(description="The parsed query plan")
    data: Optional[DataResult] = Field(
        default=None, description="Data retrieval results"
    )
    analysis: Optional[AnalysisResult] = Field(
        default=None, description="Analysis results if analysis was performed"
    )
    map: Optional[Any] = Field(  # leafmap.Map - using Any to avoid import issues
        default=None, description="Generated leafmap visualization"
    )
    code: str = Field(
        default="", description="All generated Python code from the pipeline"
    )
    success: bool = Field(
        default=True, description="Whether the overall pipeline succeeded"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if pipeline failed"
    )
    execution_time: Optional[float] = Field(
        default=None, description="Total execution time in seconds"
    )
