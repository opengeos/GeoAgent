"""Tool adapters for NASA Earthdata via the ``earthaccess`` library.

Import-safe when ``earthaccess`` is not installed; :func:`earthdata_tools`
returns an empty list in that case.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import BaseTool

from geoagent.core.decorators import geo_tool

try:
    import earthaccess  # type: ignore[import-not-found]

    _EARTHACCESS_AVAILABLE = True
except ImportError:
    earthaccess = None  # type: ignore[assignment]
    _EARTHACCESS_AVAILABLE = False


def earthdata_tools() -> list[BaseTool]:
    """Build the NASA Earthdata tool set.

    Returns:
        A list of LangChain ``BaseTool`` instances. Empty when
        ``earthaccess`` is unavailable.
    """
    if not _EARTHACCESS_AVAILABLE:
        return []

    @geo_tool(
        category="data",
        requires_packages=("earthaccess",),
    )
    def search_granules(
        short_name: str,
        bbox: Optional[list[float]] = None,
        temporal: Optional[list[str]] = None,
        max_results: int = 25,
    ) -> list[dict[str, Any]]:
        """Search the NASA Earthdata CMR for granules.

        Args:
            short_name: Collection short name (e.g. ``"HLSL30"``).
            bbox: Bounding box ``[west, south, east, north]`` in WGS84.
            temporal: ``[start, end]`` ISO datetimes (e.g.
                ``["2024-06-01", "2024-06-30"]``).
            max_results: Maximum number of granules to return.

        Returns:
            A list of granule metadata dicts.
        """
        kwargs: dict[str, Any] = {"short_name": short_name, "count": max_results}
        if bbox is not None:
            kwargs["bounding_box"] = tuple(bbox)
        if temporal is not None:
            kwargs["temporal"] = tuple(temporal)
        results = earthaccess.search_data(**kwargs)
        out: list[dict[str, Any]] = []
        for granule in results:
            try:
                out.append(
                    {
                        "id": str(granule.get("meta", {}).get("concept-id", "")),
                        "title": str(granule.get("umm", {}).get("GranuleUR", "")),
                        "links": [link for link in granule.data_links()],
                    }
                )
            except Exception:
                out.append({"granule": str(granule)})
        return out

    @geo_tool(
        category="data",
        requires_packages=("earthaccess",),
    )
    def list_collections(
        keyword: str,
        max_results: int = 25,
    ) -> list[dict[str, Any]]:
        """List NASA Earthdata collections matching a keyword.

        Args:
            keyword: Free-text search keyword.
            max_results: Maximum number of collections to return.

        Returns:
            A list of collection metadata dicts.
        """
        results = earthaccess.search_datasets(keyword=keyword, count=max_results)
        out: list[dict[str, Any]] = []
        for dataset in results:
            try:
                umm = dataset.get("umm", {}) if isinstance(dataset, dict) else {}
                out.append(
                    {
                        "short_name": umm.get("ShortName", ""),
                        "version": umm.get("Version", ""),
                        "title": umm.get("EntryTitle", ""),
                    }
                )
            except Exception:
                out.append({"dataset": str(dataset)})
        return out

    @geo_tool(
        category="data",
        requires_packages=("earthaccess",),
    )
    def get_granule_metadata(concept_id: str) -> dict[str, Any]:
        """Fetch full metadata for a granule by CMR concept-id.

        Args:
            concept_id: CMR concept id (``G123-PROVIDER``).

        Returns:
            The granule metadata dict.
        """
        results = earthaccess.search_data(concept_id=concept_id, count=1)
        if not results:
            return {"concept_id": concept_id, "found": False}
        granule = results[0]
        return granule if isinstance(granule, dict) else {"granule": str(granule)}

    @geo_tool(
        category="io",
        requires_confirmation=True,
        requires_packages=("earthaccess",),
        context_keys=("workdir",),
    )
    def download_granules(
        concept_ids: list[str],
        destination: str,
    ) -> dict[str, Any]:
        """Download granules from NASA Earthdata.

        Args:
            concept_ids: CMR concept ids to download.
            destination: Local directory.

        Returns:
            A dict with ``"downloaded"`` (list of paths) and ``"errors"``.
        """
        granules = []
        for concept_id in concept_ids:
            results = earthaccess.search_data(concept_id=concept_id, count=1)
            if results:
                granules.extend(results)
        if not granules:
            return {"downloaded": [], "errors": ["No granules found."]}
        try:
            paths = earthaccess.download(granules, destination)
            return {"downloaded": list(paths), "errors": []}
        except Exception as exc:
            return {"downloaded": [], "errors": [str(exc)]}

    return [
        search_granules,
        list_collections,
        get_granule_metadata,
        download_granules,
    ]


__all__ = ["earthdata_tools"]
