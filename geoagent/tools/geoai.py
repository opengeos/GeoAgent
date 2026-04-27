"""Tool adapters for the ``geoai`` package.

This module is import-safe when ``geoai`` is not installed: the top-level
import is guarded, and :func:`geoai_tools` returns an empty list in that
case so the registry transparently filters these tools out.

The tool functions wrap the most commonly used ``geoai`` entry points:
segmentation, object detection, and image classification.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import BaseTool

from geoagent.core.decorators import geo_tool

try:
    import geoai  # type: ignore[import-not-found]

    _GEOAI_AVAILABLE = True
except ImportError:
    geoai = None  # type: ignore[assignment]
    _GEOAI_AVAILABLE = False


def geoai_tools() -> list[BaseTool]:
    """Build the geoai tool set.

    Returns:
        A list of LangChain ``BaseTool`` instances. Empty when ``geoai`` is
        not importable.
    """
    if not _GEOAI_AVAILABLE:
        return []

    @geo_tool(
        category="ai",
        requires_packages=("geoai",),
        context_keys=("workdir",),
    )
    def segment_image(
        image_path: str,
        model: str = "geodeep",
        output_path: Optional[str] = None,
        output_format: str = "raster",
    ) -> dict[str, Any]:
        """Run instance / semantic segmentation on a raster image.

        Args:
            image_path: Source raster path or URL.
            model: GeoAI model identifier (default ``"geodeep"``).
            output_path: Optional destination path. If omitted, geoai
                chooses a default in the working directory.
            output_format: ``"raster"`` or ``"vector"``.

        Returns:
            A dict with ``"output_path"`` and any model-reported metrics.
        """
        kwargs: dict[str, Any] = {"output_format": output_format}
        if output_path is not None:
            kwargs["output_path"] = output_path
        if hasattr(geoai, "geodeep_segment"):
            result = geoai.geodeep_segment(image_path, model=model, **kwargs)
        elif hasattr(geoai, "semantic_segmentation"):
            result = geoai.semantic_segmentation(image_path, model=model, **kwargs)
        else:
            raise RuntimeError(
                "geoai does not expose geodeep_segment or semantic_segmentation."
            )
        return {"output_path": output_path, "result": result}

    @geo_tool(
        category="ai",
        requires_packages=("geoai",),
        context_keys=("workdir",),
    )
    def detect_objects(
        image_path: str,
        model: str = "geodeep",
        labels: Optional[list[str]] = None,
        output_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """Detect objects in a raster image.

        Args:
            image_path: Source raster path or URL.
            model: Model identifier.
            labels: Optional class label list to restrict detection.
            output_path: Optional vector output path.

        Returns:
            A dict with ``"output_path"`` and the raw detection result.
        """
        kwargs: dict[str, Any] = {}
        if labels is not None:
            kwargs["labels"] = labels
        if output_path is not None:
            kwargs["output_path"] = output_path
        if hasattr(geoai, "geodeep_detect"):
            result = geoai.geodeep_detect(image_path, model=model, **kwargs)
        elif hasattr(geoai, "object_detection"):
            result = geoai.object_detection(image_path, model=model, **kwargs)
        else:
            raise RuntimeError(
                "geoai does not expose geodeep_detect or object_detection."
            )
        return {"output_path": output_path, "result": result}

    @geo_tool(
        category="ai",
        requires_packages=("geoai",),
        context_keys=("workdir",),
    )
    def classify_image(
        image_path: str,
        model: str = "geodeep",
    ) -> dict[str, Any]:
        """Classify an entire image to a single category.

        Args:
            image_path: Source raster path or URL.
            model: Model identifier.

        Returns:
            A dict with ``"label"`` and ``"score"`` keys.
        """
        if hasattr(geoai, "image_classification"):
            return geoai.image_classification(image_path, model=model)
        raise RuntimeError("geoai does not expose image_classification.")

    return [segment_image, detect_objects, classify_image]


__all__ = ["geoai_tools"]
