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

from langchain_core.tools import BaseTool

from geoagent.core.decorators import geo_tool


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


def qgis_tools(iface: Any, project: Optional[Any] = None) -> list[BaseTool]:
    """Build the QGIS tool set bound to a live ``QgisInterface``.

    Args:
        iface: The QGIS ``QgisInterface`` (``qgis.utils.iface``) or a mock.
            Passing ``None`` returns an empty list, so callers may safely do
            ``qgis_tools(getattr(some_ctx, 'qgis_iface', None))`` outside
            QGIS.
        project: Optional ``QgsProject`` instance. If omitted, the tools fall
            back to ``QgsProject.instance()``.

    Returns:
        A list of LangChain ``BaseTool`` instances. Empty when ``iface`` is
        ``None``.
    """
    if iface is None:
        return []

    def _project() -> Any:
        if project is not None:
            return project
        try:
            from qgis.core import QgsProject  # type: ignore[import-not-found]

            return QgsProject.instance()
        except Exception as exc:  # pragma: no cover - QGIS-only path
            raise RuntimeError(
                "QGIS is not available; cannot resolve QgsProject."
            ) from exc

    @geo_tool(
        category="qgis",
        requires_packages=("qgis",),
        context_keys=("qgis_iface", "qgis_project"),
    )
    def list_project_layers() -> list[dict[str, Any]]:
        """List all layers in the active QGIS project.

        Returns:
            A list of dicts ``{"name": ..., "type": ..., "source": ...}``.
        """
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

    @geo_tool(
        category="qgis",
        requires_packages=("qgis",),
        context_keys=("qgis_iface",),
    )
    def get_active_layer() -> dict[str, Any]:
        """Return metadata for the currently active layer.

        Returns:
            A dict with ``name``, ``type``, and ``source`` keys, or a single
            ``{"active_layer": None}`` if no layer is active.
        """
        layer = iface.activeLayer()
        if layer is None:
            return {"active_layer": None}
        return {
            "name": layer.name(),
            "type": str(layer.type()),
            "source": layer.source(),
        }

    @geo_tool(
        category="qgis",
        requires_packages=("qgis",),
        context_keys=("qgis_iface",),
    )
    def zoom_in() -> str:
        """Zoom the QGIS map canvas in by one step.

        Returns:
            A status string.
        """
        canvas = iface.mapCanvas()
        canvas.zoomIn()
        # ``QgsMapCanvas`` defers redraws after a programmatic extent
        # change — XYZ tile layers (Google Satellite, OSM, etc.) won't
        # request fresh tiles until ``refresh()`` is called explicitly.
        if hasattr(canvas, "refresh"):
            canvas.refresh()
        return "Zoomed in."

    @geo_tool(
        category="qgis",
        requires_packages=("qgis",),
        context_keys=("qgis_iface",),
    )
    def zoom_out() -> str:
        """Zoom the QGIS map canvas out by one step.

        Returns:
            A status string.
        """
        canvas = iface.mapCanvas()
        canvas.zoomOut()
        if hasattr(canvas, "refresh"):
            canvas.refresh()
        return "Zoomed out."

    @geo_tool(
        category="qgis",
        requires_packages=("qgis",),
        context_keys=("qgis_iface", "qgis_project"),
    )
    def zoom_to_layer(layer_name: str) -> str:
        """Zoom the canvas to the extent of a named layer.

        Args:
            layer_name: Display name of the layer.

        Returns:
            A status string.
        """
        layer = _resolve_layer(_project(), layer_name)
        iface.setActiveLayer(layer)
        canvas = iface.mapCanvas()
        if hasattr(iface, "zoomToActiveLayer"):
            iface.zoomToActiveLayer()
        else:
            extent = layer.extent() if hasattr(layer, "extent") else None
            if extent is not None:
                canvas.setExtent(extent)
        # ``zoomToActiveLayer`` updates the canvas extent but does not
        # always trigger a redraw of XYZ tile layers (Google Satellite,
        # OSM, etc.) — without an explicit ``refresh()`` the basemap can
        # appear blank at the new extent until the user pans manually.
        if hasattr(canvas, "refresh"):
            canvas.refresh()
        return f"Zoomed to layer {layer_name!r}."

    @geo_tool(
        category="qgis",
        requires_packages=("qgis",),
        context_keys=("qgis_iface",),
    )
    def zoom_to_extent(west: float, south: float, east: float, north: float) -> str:
        """Zoom the canvas to a geographic extent (in the project CRS).

        Args:
            west: Western coordinate.
            south: Southern coordinate.
            east: Eastern coordinate.
            north: Northern coordinate.

        Returns:
            A status string.
        """
        try:
            from qgis.core import QgsRectangle  # type: ignore[import-not-found]

            rect = QgsRectangle(west, south, east, north)
        except Exception:
            rect = (west, south, east, north)
        iface.mapCanvas().setExtent(rect)
        iface.mapCanvas().refresh()
        return f"Zoomed to extent [{west}, {south}, {east}, {north}]."

    @geo_tool(
        category="qgis",
        requires_packages=("qgis",),
        context_keys=("qgis_iface",),
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
        layer = iface.addVectorLayer(path_or_uri, name, provider)
        if layer is None or (hasattr(layer, "isValid") and not layer.isValid()):
            return f"Failed to load vector layer from {path_or_uri!r}."
        return f"Added vector layer {name!r}."

    @geo_tool(
        category="qgis",
        requires_packages=("qgis",),
        context_keys=("qgis_iface",),
    )
    def add_raster_layer(path_or_uri: str, name: str) -> str:
        """Add a raster layer to the project.

        Args:
            path_or_uri: Path or provider URI for the raster.
            name: Display name.

        Returns:
            A status string.
        """
        layer = iface.addRasterLayer(path_or_uri, name)
        if layer is None or (hasattr(layer, "isValid") and not layer.isValid()):
            return f"Failed to load raster layer from {path_or_uri!r}."
        return f"Added raster layer {name!r}."

    @geo_tool(
        category="qgis",
        requires_confirmation=True,
        requires_packages=("qgis",),
        context_keys=("qgis_project",),
    )
    def remove_layer(layer_name: str) -> str:
        """Remove a layer from the project.

        Args:
            layer_name: Display name of the layer to remove.

        Returns:
            A status string.
        """
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

    @geo_tool(
        category="qgis",
        requires_packages=("qgis",),
        context_keys=("qgis_project",),
    )
    def set_layer_visibility(layer_name: str, visible: bool) -> str:
        """Show or hide a layer in the layer panel.

        Args:
            layer_name: Display name of the layer.
            visible: ``True`` to show, ``False`` to hide.

        Returns:
            A status string.
        """
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

    @geo_tool(
        category="qgis",
        requires_packages=("qgis",),
        context_keys=("qgis_project",),
    )
    def inspect_layer_fields(layer_name: str) -> list[dict[str, Any]]:
        """List fields and types for a vector layer.

        Args:
            layer_name: Display name of the layer.

        Returns:
            A list of ``{"name": ..., "type": ...}`` per field.
        """
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
                            getattr(field, "typeName", lambda: type(field).__name__)()
                        ),
                    }
                )
        return out

    @geo_tool(
        category="qgis",
        requires_packages=("qgis",),
        context_keys=("qgis_iface", "qgis_project"),
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

    @geo_tool(
        category="qgis",
        requires_confirmation=True,
        requires_packages=("qgis",),
        context_keys=("qgis_iface",),
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
        try:
            import processing  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - QGIS-only path
            raise RuntimeError("QGIS Processing framework is not available.") from exc
        return processing.run(algorithm_id, parameters)

    @geo_tool(
        category="qgis",
        requires_packages=("qgis",),
        context_keys=("qgis_iface",),
    )
    def open_attribute_table(layer_name: str) -> str:
        """Open the attribute table for a layer.

        Args:
            layer_name: Display name of the layer.

        Returns:
            A status string.
        """
        layer = _resolve_layer(_project(), layer_name)
        iface.setActiveLayer(layer)
        if hasattr(iface, "showAttributeTable"):
            iface.showAttributeTable(layer)  # type: ignore[attr-defined]
        return f"Attribute table requested for {layer_name!r}."

    @geo_tool(
        category="qgis",
        requires_packages=("qgis",),
        context_keys=("qgis_iface",),
    )
    def refresh_canvas() -> str:
        """Refresh the QGIS map canvas.

        Returns:
            A status string.
        """
        iface.mapCanvas().refresh()
        return "Canvas refreshed."

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
