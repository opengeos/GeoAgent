"""Tool adapters for Google Earth Engine.

Import-safe when ``earthengine-api`` (``ee``) is not installed.
:func:`earthengine_tools` returns an empty list when EE is unavailable or
not initialised.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import BaseTool

from geoagent.core.decorators import geo_tool

try:
    import ee  # type: ignore[import-not-found]

    _EE_AVAILABLE = True
except ImportError:
    ee = None  # type: ignore[assignment]
    _EE_AVAILABLE = False


def earthengine_tools(ee_initialized: bool = True) -> list[BaseTool]:
    """Build the Earth Engine tool set.

    Args:
        ee_initialized: Pass ``False`` to skip building tools even when ``ee``
            is importable, e.g. when ``ee.Initialize()`` has not been called
            yet. Useful for plugins that defer EE auth until the user
            chooses to use it.

    Returns:
        A list of LangChain ``BaseTool`` instances, or empty if Earth Engine
        is unavailable.
    """
    if not _EE_AVAILABLE or not ee_initialized:
        return []

    @geo_tool(
        category="data",
        requires_packages=("ee",),
    )
    def search_collection(query: str, max_results: int = 10) -> list[dict[str, Any]]:
        """Search the Earth Engine catalog for collections matching a query.

        Args:
            query: Free-text search string.
            max_results: Maximum number of collections to return.

        Returns:
            A list of dicts with ``id``, ``title``, and ``description`` keys
            (when available).
        """
        try:
            data = ee.data.listAssets(  # type: ignore[union-attr]
                {"parent": "projects/earthengine-public/assets", "filter": query}
            )
        except Exception as exc:
            return [{"error": str(exc)}]
        assets = data.get("assets", [])[:max_results]
        return [{"id": a.get("name", ""), "type": a.get("type", "")} for a in assets]

    @geo_tool(
        category="data",
        requires_packages=("ee",),
    )
    def get_image_metadata(asset_id: str) -> dict[str, Any]:
        """Return key metadata for an EE Image asset.

        Args:
            asset_id: Earth Engine asset ID.

        Returns:
            A metadata dict from ``ee.Image.getInfo()``.
        """
        return ee.Image(asset_id).getInfo()  # type: ignore[union-attr]

    @geo_tool(
        category="data",
        requires_packages=("ee",),
    )
    def compute_index(
        asset_id: str,
        index: str = "NDVI",
        red_band: str = "B4",
        nir_band: str = "B8",
    ) -> dict[str, Any]:
        """Compute a spectral index on an EE Image.

        Args:
            asset_id: Earth Engine asset ID for the source image.
            index: Index name (``"NDVI"``, ``"NDWI"``, ``"EVI"``).
            red_band: Red band name.
            nir_band: NIR band name.

        Returns:
            A dict with ``"asset_id"`` and ``"index"`` keys describing the
            computed image (left as a server-side reference; serialise via
            ``getInfo`` only on small AOIs).
        """
        image = ee.Image(asset_id)  # type: ignore[union-attr]
        if index.upper() == "NDVI":
            result = image.normalizedDifference([nir_band, red_band]).rename("NDVI")
        elif index.upper() == "NDWI":
            result = image.normalizedDifference([nir_band, red_band]).rename("NDWI")
        else:
            raise ValueError(f"Unsupported index {index!r}.")
        return {"asset_id": asset_id, "index": index, "image_id": str(result)}

    @geo_tool(
        category="io",
        requires_confirmation=True,
        requires_packages=("ee",),
    )
    def export_to_drive(
        asset_id: str,
        description: str,
        folder: str = "GeoAgentExports",
        scale: int = 30,
    ) -> dict[str, Any]:
        """Submit an Earth Engine export task to Google Drive.

        Args:
            asset_id: Earth Engine image asset ID.
            description: Task description (used as the filename prefix).
            folder: Destination Drive folder.
            scale: Pixel scale in metres.

        Returns:
            A dict with the task id and status.
        """
        image = ee.Image(asset_id)  # type: ignore[union-attr]
        task = ee.batch.Export.image.toDrive(  # type: ignore[union-attr]
            image=image,
            description=description,
            folder=folder,
            scale=scale,
        )
        task.start()
        return {"task_id": task.id, "status": "STARTED", "description": description}

    @geo_tool(
        category="io",
        requires_confirmation=True,
        requires_packages=("ee",),
    )
    def submit_task(asset_id: str, description: str) -> dict[str, Any]:
        """Submit a generic export task for an EE asset.

        Args:
            asset_id: Earth Engine asset ID.
            description: Task description.

        Returns:
            A dict with the task id.
        """
        image = ee.Image(asset_id)  # type: ignore[union-attr]
        task = ee.batch.Export.image.toAsset(  # type: ignore[union-attr]
            image=image, description=description
        )
        task.start()
        return {"task_id": task.id, "status": "STARTED"}

    @geo_tool(
        category="data",
        requires_packages=("ee",),
    )
    def get_task_status(task_id: str) -> dict[str, Any]:
        """Return the status of a submitted EE task.

        Args:
            task_id: Task id from a previous export call.

        Returns:
            The task status dict reported by Earth Engine.
        """
        for task in ee.batch.Task.list():  # type: ignore[union-attr]
            if getattr(task, "id", None) == task_id:
                return task.status()
        return {"task_id": task_id, "status": "UNKNOWN"}

    return [
        search_collection,
        get_image_metadata,
        compute_index,
        export_to_drive,
        submit_task,
        get_task_status,
    ]


__all__ = ["earthengine_tools"]
