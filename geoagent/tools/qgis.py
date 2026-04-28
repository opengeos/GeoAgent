"""Tool adapters for QGIS via the ``QgisInterface`` (typically ``iface``).

This module is **import-safe** outside a QGIS Python environment: the
top-level imports do NOT pull in the ``qgis`` package, and the
:func:`qgis_tools` factory returns an empty list when no ``iface`` is
provided. Tool bodies that need QGIS classes (``QgsProject``,
``QgsRectangle``, ``QgsVectorLayer``, ``QgsRasterLayer``) import them
lazily so the module can still be imported in CI without QGIS.

Typical usage from inside a QGIS plugin's Python console::

    from qgis.utils import iface
    from geoagent import for_qgis
    agent = for_qgis(iface)
    agent.chat("Zoom to the active layer.")
"""

from __future__ import annotations

from typing import Any, Optional

from geoagent.core.decorators import geo_tool
from geoagent.tools._qt_marshal import run_on_qt_gui_thread


def _transform_extent_to_canvas_crs(layer: Any, canvas: Any, extent: Any) -> Any:
    """Re-project a layer's extent into the canvas / project CRS.

    ``QgsMapLayer.extent()`` returns the extent in the layer's native
    CRS. ``QgsMapCanvas.setExtent()`` expects coordinates in the
    canvas's destination CRS. When a layer is loaded as EPSG:4326
    (typical for GeoJSON) but the project canvas is EPSG:3857 (Web
    Mercator), passing the layer's lat/lon extent directly to
    ``setExtent`` zooms the canvas to a sliver near (0, 0) in
    Mercator, which renders as a blank view. Transforming the bbox
    through ``QgsCoordinateTransform`` first puts it in the right
    CRS so the canvas zooms to the actual layer.

    The helper degrades gracefully when called with mocks (no
    ``crs()`` / ``mapSettings()`` methods) or outside QGIS (no
    ``QgsCoordinateTransform`` import) — in those cases it returns
    the original extent so existing tests keep passing.

    Args:
        layer: The source layer whose ``extent`` was just fetched.
        canvas: The map canvas the extent will be applied to.
        extent: The layer-CRS extent (a ``QgsRectangle`` in real QGIS;
            anything else from a mock).

    Returns:
        The extent re-projected into the canvas's destination CRS,
        or the original extent when the transform cannot be set up.
    """
    if not (hasattr(layer, "crs") and hasattr(canvas, "mapSettings")):
        return extent
    try:
        from qgis.core import (  # type: ignore[import-not-found]
            QgsCoordinateTransform,
            QgsProject,
        )
    except ImportError:
        return extent
    try:
        src_crs = layer.crs()
        dst_crs = canvas.mapSettings().destinationCrs()
    except Exception:
        return extent
    if src_crs is None or dst_crs is None:
        return extent
    # ``QgsCoordinateReferenceSystem`` defines ``__eq__`` so this works
    # for both authority-id and proj-string-defined CRSes.
    try:
        if src_crs == dst_crs:
            return extent
    except Exception:
        pass
    try:
        transform = QgsCoordinateTransform(src_crs, dst_crs, QgsProject.instance())
        return transform.transformBoundingBox(extent)
    except Exception:
        return extent


def _transform_bbox_to_canvas_crs(
    canvas: Any,
    west: float,
    south: float,
    east: float,
    north: float,
    src_crs: str,
) -> Any:
    """Re-project a [west, south, east, north] bbox into the canvas CRS.

    LLMs naturally produce place-name extents in lat/lon (EPSG:4326)
    even when the project canvas is Web Mercator (EPSG:3857). Without a
    transform, ``canvas.setExtent`` interprets the lat/lon coordinates
    as the canvas's metres, zooming to a sliver near (0, 0) and
    rendering blank.

    Returns a ``QgsRectangle`` in the canvas CRS when the transform
    succeeds, falls back to a ``QgsRectangle`` (or 4-tuple, outside
    QGIS) in the source CRS when any step is unavailable.

    Args:
        canvas: The map canvas the bbox will be applied to.
        west: Western coordinate in ``src_crs``.
        south: Southern coordinate in ``src_crs``.
        east: Eastern coordinate in ``src_crs``.
        north: Northern coordinate in ``src_crs``.
        src_crs: Authority ID of the bbox's CRS, e.g. ``"EPSG:4326"``.

    Returns:
        A ``QgsRectangle`` in the canvas's destination CRS when both
        QGIS and the canvas are available; otherwise a ``QgsRectangle``
        (or 4-tuple if QGIS is missing entirely) in the original CRS.
    """
    try:
        from qgis.core import (  # type: ignore[import-not-found]
            QgsCoordinateReferenceSystem,
            QgsCoordinateTransform,
            QgsProject,
            QgsRectangle,
        )
    except ImportError:
        return (west, south, east, north)

    rect = QgsRectangle(west, south, east, north)
    if not hasattr(canvas, "mapSettings"):
        return rect
    try:
        src = QgsCoordinateReferenceSystem(src_crs)
        dst = canvas.mapSettings().destinationCrs()
    except Exception:
        return rect
    if dst is None:
        return rect
    try:
        if src == dst:
            return rect
    except Exception:
        pass
    try:
        transform = QgsCoordinateTransform(src, dst, QgsProject.instance())
        return transform.transformBoundingBox(rect)
    except Exception:
        return rect


def _resolve_layer(project: Any, layer_name: str) -> Any:
    """Resolve a layer by name from a project.

    Args:
        project: A ``QgsProject`` (or :class:`MockQGISProject`) instance.
        layer_name: Layer name to look up.

    Returns:
        The first matching layer.

    Raises:
        LookupError: If no layer with that name exists in the project.
    """
    layers = project.mapLayersByName(layer_name)
    if not layers:
        raise LookupError(f"No layer named {layer_name!r} in the project.")
    return layers[0]


def qgis_tools(iface: Any, project: Optional[Any] = None) -> list[Any]:
    """Build the QGIS tool set bound to a live ``QgisInterface``.

    Args:
        iface: The QGIS ``QgisInterface`` (``qgis.utils.iface``) or a mock.
            Passing ``None`` returns an empty list, so callers may safely do
            ``qgis_tools(getattr(some_ctx, 'qgis_iface', None))`` outside
            QGIS.
        project: Optional ``QgsProject`` instance. If omitted, the tools fall
            back to ``QgsProject.instance()``.

    Returns:
        A list of Strands tool objects. Empty when ``iface`` is
        ``None``.
    """
    if iface is None:
        return []

    def _on_gui(func: Any) -> Any:
        return run_on_qt_gui_thread(func)

    def _project() -> Any:
        if project is not None:
            return project
        if hasattr(iface, "project"):
            try:
                proj = iface.project()
                if proj is not None:
                    return proj
            except Exception:
                pass
        try:
            from qgis.core import QgsProject  # type: ignore[import-not-found]

            return QgsProject.instance()
        except Exception as exc:  # pragma: no cover - QGIS-only path
            raise RuntimeError(
                "QGIS is not available; cannot resolve QgsProject."
            ) from exc

    @geo_tool(
        category="qgis",
    )
    def list_project_layers() -> list[dict[str, Any]]:
        """List all layers in the active QGIS project.

        Returns:
            A list of dicts ``{"name": ..., "type": ..., "source": ...}``.
        """

        def _run() -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for layer in _project().mapLayers().values():
                out.append(
                    {
                        "name": layer.name(),
                        "type": str(layer.type()),
                        "source": layer.source(),
                    }
                )
            return out

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def get_active_layer() -> dict[str, Any]:
        """Return metadata for the currently active layer.

        Returns:
            A dict with ``name``, ``type``, and ``source`` keys, or a single
            ``{"active_layer": None}`` if no layer is active.
        """

        def _run() -> dict[str, Any]:
            layer = iface.activeLayer()
            if layer is None:
                return {"active_layer": None}
            return {
                "name": layer.name(),
                "type": str(layer.type()),
                "source": layer.source(),
            }

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def zoom_in() -> str:
        """Zoom the QGIS map canvas in by one step.

        Returns:
            A status string.
        """

        def _run() -> str:
            iface.mapCanvas().zoomIn()
            return "Zoomed in."

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def zoom_out() -> str:
        """Zoom the QGIS map canvas out by one step.

        Returns:
            A status string.
        """

        def _run() -> str:
            iface.mapCanvas().zoomOut()
            return "Zoomed out."

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def zoom_to_layer(layer_name: str) -> str:
        """Zoom the canvas to the extent of a named layer.

        Args:
            layer_name: Display name of the layer.

        Returns:
            A status string.
        """

        def _run() -> str:
            layer = _resolve_layer(_project(), layer_name)
            iface.setActiveLayer(layer)
            canvas = iface.mapCanvas()
            # Prefer the explicit ``setExtent`` + ``refresh`` path over
            # ``iface.zoomToActiveLayer()``. The latter updates the extent
            # but does not always trigger XYZ tile providers (Google
            # Satellite, OSM, ESRI) to refetch tiles at the new zoom-pyramid
            # level, leaving basemaps stuck on upscaled lower-resolution
            # tiles. ``setExtent`` + ``refresh`` mirrors the path QGIS uses
            # for user-driven zoom and resolves the tile pyramid correctly.
            # ``iface.zoomToActiveLayer`` stays as a fallback for canvas
            # types where ``setExtent`` is unavailable (e.g. test mocks
            # without the method) — call it AFTER setExtent so the iface
            # path runs only when needed.
            extent = layer.extent() if hasattr(layer, "extent") else None
            if extent is not None and hasattr(canvas, "setExtent"):
                # Re-project from the layer's CRS to the canvas / project
                # CRS before applying the extent. A GeoJSON loaded as
                # EPSG:4326 has a lat/lon extent that ``setExtent`` would
                # otherwise interpret as Web-Mercator metres, zooming to
                # ~(0, 0) and rendering as a blank canvas.
                extent = _transform_extent_to_canvas_crs(layer, canvas, extent)
                canvas.setExtent(extent)
                if hasattr(canvas, "refresh"):
                    canvas.refresh()
            elif hasattr(iface, "zoomToActiveLayer"):
                iface.zoomToActiveLayer()
            return f"Zoomed to layer {layer_name!r}."

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def zoom_to_extent(
        west: float,
        south: float,
        east: float,
        north: float,
        crs: str = "EPSG:4326",
    ) -> str:
        """Zoom the canvas to a geographic extent.

        The bbox is interpreted in ``crs`` (lat/lon by default — the
        natural CRS for resolving place names). The tool re-projects
        into the canvas / project CRS before applying ``setExtent``,
        so passing ``[-122.5, 47.5, -122.2, 47.7]`` for Seattle works
        regardless of whether the project canvas is EPSG:3857 (Web
        Mercator), EPSG:4326, or anything else.

        Args:
            west: Western coordinate (in ``crs``).
            south: Southern coordinate (in ``crs``).
            east: Eastern coordinate (in ``crs``).
            north: Northern coordinate (in ``crs``).
            crs: Authority ID of the bbox's CRS. Defaults to
                ``"EPSG:4326"`` (WGS84 lat/lon). Use ``"EPSG:3857"``
                if you already have Web-Mercator metres, or any other
                authority ID supported by ``QgsCoordinateReferenceSystem``.

        Returns:
            A status string.
        """

        def _run() -> str:
            canvas = iface.mapCanvas()
            rect = _transform_bbox_to_canvas_crs(canvas, west, south, east, north, crs)
            canvas.setExtent(rect)
            canvas.refresh()
            return f"Zoomed to extent [{west}, {south}, {east}, {north}] ({crs})."

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def add_vector_layer(
        path_or_uri: str,
        name: str,
        provider: str = "ogr",
    ) -> str:
        """Add a vector layer to the project.

        Args:
            path_or_uri: Path or provider URI for the data source.
            name: Display name.
            provider: QGIS data provider key (default ``"ogr"``).

        Returns:
            A status string.
        """

        def _run() -> str:
            layer = iface.addVectorLayer(path_or_uri, name, provider)
            if layer is None or (hasattr(layer, "isValid") and not layer.isValid()):
                return f"Failed to load vector layer from {path_or_uri!r}."
            return f"Added vector layer {name!r}."

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def add_raster_layer(path_or_uri: str, name: str) -> str:
        """Add a raster layer to the project.

        Args:
            path_or_uri: Path or provider URI for the raster.
            name: Display name.

        Returns:
            A status string.
        """

        def _run() -> str:
            layer = iface.addRasterLayer(path_or_uri, name)
            if layer is None or (hasattr(layer, "isValid") and not layer.isValid()):
                return f"Failed to load raster layer from {path_or_uri!r}."
            return f"Added raster layer {name!r}."

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
        requires_confirmation=True,
    )
    def remove_layer(layer_name: str) -> str:
        """Remove a layer from the project.

        Args:
            layer_name: Display name of the layer to remove.

        Returns:
            A status string.
        """

        def _run() -> str:
            proj = _project()
            layers = proj.mapLayersByName(layer_name)
            if not layers:
                return f"Layer {layer_name!r} not found."
            for layer in layers:
                try:
                    proj.removeMapLayer(layer.id())
                except Exception:
                    proj.removeMapLayer(layer)
            return f"Removed layer {layer_name!r}."

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def set_layer_visibility(layer_name: str, visible: bool) -> str:
        """Show or hide a layer in the layer panel.

        Args:
            layer_name: Display name of the layer.
            visible: ``True`` to show, ``False`` to hide.

        Returns:
            A status string.
        """

        def _run() -> str:
            proj = _project()
            layer = _resolve_layer(proj, layer_name)
            # Try the layer-tree-based path first; fall back to a simple attribute.
            try:
                tree_layer = proj.layerTreeRoot().findLayer(layer.id())  # type: ignore[attr-defined]
                tree_layer.setItemVisibilityChecked(bool(visible))
            except Exception:
                try:
                    layer.visible = bool(visible)
                except Exception as exc:
                    return f"Could not set visibility on {layer_name!r}: {exc}"
            return f"Layer {layer_name!r} visibility set to {visible}."

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def inspect_layer_fields(layer_name: str) -> list[dict[str, Any]]:
        """List fields and types for a vector layer.

        Args:
            layer_name: Display name of the layer.

        Returns:
            A list of ``{"name": ..., "type": ...}`` per field.
        """

        def _run() -> list[dict[str, Any]]:
            layer = _resolve_layer(_project(), layer_name)
            out: list[dict[str, Any]] = []
            for field in layer.fields():
                if isinstance(field, dict):
                    out.append(
                        {"name": field.get("name", ""), "type": field.get("type", "")}
                    )
                else:
                    out.append(
                        {
                            "name": getattr(field, "name", lambda: "")(),
                            "type": str(
                                getattr(
                                    field, "typeName", lambda: type(field).__name__
                                )()
                            ),
                        }
                    )
            return out

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def get_selected_features(
        layer_name: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Return selected features from a layer (or the active layer).

        Args:
            layer_name: Display name of the layer; if omitted, uses the
                active layer.

        Returns:
            A list of feature dicts (mock-friendly; the QGIS path returns a
            simple ``{"id": ..., "attributes": ...}``).
        """

        def _run() -> list[dict[str, Any]]:
            if layer_name is None:
                layer = iface.activeLayer()
                if layer is None:
                    return []
            else:
                layer = _resolve_layer(_project(), layer_name)
            features = layer.selectedFeatures()
            out: list[dict[str, Any]] = []
            for feature in features:
                if isinstance(feature, dict):
                    out.append(feature)
                else:
                    attrs = getattr(feature, "attributes", lambda: [])()
                    fid = getattr(feature, "id", lambda: None)()
                    out.append({"id": fid, "attributes": list(attrs)})
            return out

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
        requires_confirmation=True,
    )
    def run_processing_algorithm(
        algorithm_id: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Run a QGIS processing algorithm.

        Args:
            algorithm_id: Algorithm identifier (e.g. ``"native:buffer"``).
            parameters: Algorithm parameters as a dict.

        Returns:
            The algorithm's result dictionary.
        """

        def _run() -> dict[str, Any]:
            try:
                import processing  # type: ignore[import-not-found]
            except Exception as exc:  # pragma: no cover - QGIS-only path
                raise RuntimeError(
                    "QGIS Processing framework is not available."
                ) from exc
            return processing.run(algorithm_id, parameters)

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def open_attribute_table(layer_name: str) -> str:
        """Open the attribute table for a layer.

        Args:
            layer_name: Display name of the layer.

        Returns:
            A status string.
        """

        def _run() -> str:
            layer = _resolve_layer(_project(), layer_name)
            iface.setActiveLayer(layer)
            if hasattr(iface, "showAttributeTable"):
                iface.showAttributeTable(layer)  # type: ignore[attr-defined]
            return f"Attribute table requested for {layer_name!r}."

        return _on_gui(_run)

    @geo_tool(
        category="qgis",
    )
    def refresh_canvas() -> str:
        """Refresh the QGIS map canvas.

        Returns:
            A status string.
        """

        def _run() -> str:
            iface.mapCanvas().refresh()
            return "Canvas refreshed."

        return _on_gui(_run)

    return [
        list_project_layers,
        get_active_layer,
        zoom_in,
        zoom_out,
        zoom_to_layer,
        zoom_to_extent,
        add_vector_layer,
        add_raster_layer,
        remove_layer,
        set_layer_visibility,
        inspect_layer_fields,
        get_selected_features,
        run_processing_algorithm,
        open_attribute_table,
        refresh_canvas,
    ]


__all__ = ["qgis_tools"]
