"""Mock objects for GeoAgent tests.

These mocks let the tool adapters in :mod:`geoagent.tools` be exercised in
CI without leafmap, anymap, or QGIS installed. Each mock implements the
minimum API surface that GeoAgent's tools call.

They are deliberately simple and not intended as full-fidelity emulators of
the real packages — they exist to verify that GeoAgent invokes the right
methods with the right arguments, not to validate real geospatial behaviour.
"""

from __future__ import annotations

from typing import Any, Optional


class MockLeafmap:
    """Stand-in for ``leafmap.Map`` / ``leafmap.maplibregl.Map``.

    Stores added layers as plain dicts on ``self.layers`` so tests can
    assert on layer composition. Keeps the simplest possible state for
    center, zoom, basemap, and title.
    """

    def __init__(
        self,
        center: list[float] | None = None,
        zoom: int = 5,
        height: str = "600px",
        **kwargs: Any,
    ) -> None:
        self.layers: list[dict[str, Any]] = []
        self.center: list[float] = list(center) if center else [0.0, 0.0]
        self.zoom: int = zoom
        self.height: str = height
        self.title: str = ""
        self._style: str = "open-street-map"
        self._bounds: Optional[list[list[float]]] = None
        self.kwargs = kwargs

    # ----- viewport -----
    def set_center(self, lon: float, lat: float, zoom: Optional[int] = None) -> None:
        self.center = [lon, lat]
        if zoom is not None:
            self.zoom = zoom

    def set_zoom(self, zoom: int) -> None:
        self.zoom = zoom

    def fit_bounds(self, bounds: list[list[float]]) -> None:
        self._bounds = bounds

    def fly_to(self, lon: float, lat: float, zoom: Optional[int] = None) -> None:
        self.set_center(lon, lat, zoom)

    def get_center(self) -> list[float]:
        return list(self.center)

    def get_zoom(self) -> int:
        return self.zoom

    def get_bounds(self) -> Optional[list[list[float]]]:
        return self._bounds

    # ----- basemap -----
    def add_basemap(self, basemap: str = "open-street-map") -> None:
        self._style = basemap

    # ----- layers -----
    def add_layer(self, layer_dict: dict[str, Any]) -> None:
        self.layers.append(dict(layer_dict))

    def add_geojson(
        self,
        data: Any,
        layer_name: str | None = None,
        style: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.layers.append(
            {
                "type": "geojson",
                "data": data,
                "name": layer_name or f"GeoJSON Layer {len(self.layers) + 1}",
                "style": style,
                **kwargs,
            }
        )

    def add_cog_layer(
        self,
        url: str,
        name: str | None = None,
        fit_bounds: bool = False,
        **kwargs: Any,
    ) -> None:
        self.layers.append(
            {
                "type": "cog",
                "url": url,
                "name": name or f"COG Layer {len(self.layers) + 1}",
                "fit_bounds": fit_bounds,
                **kwargs,
            }
        )

    def add_raster(
        self,
        url: str,
        layer_name: str | None = None,
        fit_bounds: bool = False,
        **kwargs: Any,
    ) -> None:
        self.add_cog_layer(url, name=layer_name, fit_bounds=fit_bounds, **kwargs)

    def add_pmtiles(self, url: str, name: str | None = None, **kwargs: Any) -> None:
        self.layers.append(
            {
                "type": "pmtiles",
                "url": url,
                "name": name or f"PMTiles Layer {len(self.layers) + 1}",
                **kwargs,
            }
        )

    def add_xyz_tile_layer(
        self,
        url: str,
        name: str | None = None,
        attribution: str = "",
        **kwargs: Any,
    ) -> None:
        self.layers.append(
            {
                "type": "xyz",
                "url": url,
                "name": name or f"XYZ Layer {len(self.layers) + 1}",
                "attribution": attribution,
                **kwargs,
            }
        )

    def add_marker(
        self,
        location: tuple[float, float] | list[float],
        popup: str | None = None,
        tooltip: str | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.layers.append(
            {
                "type": "marker",
                "location": list(location),
                "popup": popup,
                "tooltip": tooltip,
                "name": name or f"Marker {len(self.layers) + 1}",
                **kwargs,
            }
        )

    def add_stac_layer(
        self,
        collection: str,
        item: str | None = None,
        assets: list[str] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.layers.append(
            {
                "type": "stac",
                "collection": collection,
                "item": item,
                "assets": list(assets) if assets else [],
                "name": name or f"STAC {collection}",
                **kwargs,
            }
        )

    def remove_layer(self, name: str) -> bool:
        before = len(self.layers)
        self.layers = [layer for layer in self.layers if layer.get("name") != name]
        return len(self.layers) != before

    def clear_layers(self) -> None:
        self.layers = []

    def set_layer_visibility(self, name: str, visible: bool) -> bool:
        for layer in self.layers:
            if layer.get("name") == name:
                layer["visible"] = bool(visible)
                return True
        return False

    def set_layer_opacity(self, name: str, opacity: float) -> bool:
        for layer in self.layers:
            if layer.get("name") == name:
                layer["opacity"] = float(opacity)
                return True
        return False

    # ----- annotations / IO -----
    def add_title(self, title: str) -> None:
        self.title = title

    def to_html(self, filename: str | None = None) -> str:
        layer_list = "".join(
            f"<li>{layer.get('name', 'Unnamed')} ({layer.get('type', 'unknown')})</li>"
            for layer in self.layers
        )
        html = (
            f"<div><h3>{self.title}</h3>"
            f"<p>Mock map. Center: {self.center}, Zoom: {self.zoom}</p>"
            f"<ul>{layer_list}</ul></div>"
        )
        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(html)
        return html

    def __repr__(self) -> str:
        return (
            f"MockLeafmap(center={self.center}, zoom={self.zoom}, "
            f"layers={len(self.layers)})"
        )


class MockAnymap(MockLeafmap):
    """Stand-in for ``anymap.Map``.

    The two libraries' high-level APIs are similar enough that for testing
    purposes :class:`MockAnymap` reuses :class:`MockLeafmap`. The class
    name is what tools use to detect "anymap-style" map objects via
    ``type(map_obj).__module__``.
    """

    __module__ = "anymap.testing"


class MockQGISLayer:
    """Minimal stand-in for ``QgsMapLayer`` (vector or raster).

    Attributes:
        layer_name: The display name returned by :meth:`name`.
        source_uri: The source string returned by :meth:`source`.
        layer_type: ``"vector"`` or ``"raster"``.
    """

    def __init__(
        self,
        name: str,
        source: str = "",
        layer_type: str = "vector",
        fields: list[dict[str, Any]] | None = None,
        extent: tuple[float, float, float, float] | None = None,
    ) -> None:
        self.layer_name = name
        self.source_uri = source
        self.layer_type = layer_type
        self._fields = list(fields) if fields else []
        self._selected: list[dict[str, Any]] = []
        self._visible = True
        self._extent = tuple(extent) if extent else None
        self._opacity = 1.0

    def name(self) -> str:
        return self.layer_name

    def source(self) -> str:
        return self.source_uri

    def type(self) -> str:
        return self.layer_type

    def extent(self) -> Optional[tuple[float, float, float, float]]:
        """Return the layer's extent ``(west, south, east, north)``.

        ``QgsMapLayer.extent()`` returns a ``QgsRectangle``; the mock
        returns a 4-tuple matching the test stubs. Tests that need to
        exercise the ``setExtent`` + ``refresh`` zoom path can pass an
        explicit extent into the mock's constructor; the default
        (``None``) lets tools fall back to the iface-driven zoom path.
        """
        return self._extent

    def isValid(self) -> bool:
        return True

    def fields(self) -> list[dict[str, Any]]:
        return list(self._fields)

    def selectedFeatures(self) -> list[dict[str, Any]]:
        return list(self._selected)

    def selectedFeatureCount(self) -> int:
        return len(self._selected)

    def featureCount(self) -> int:
        return len(self._selected)

    def selectByExpression(self, expression: str, behavior: Any = None) -> None:
        self._selected = [{"id": 1, "expression": expression, "behavior": behavior}]

    def removeSelection(self) -> None:
        self._selected = []

    def boundingBoxOfSelected(self) -> Optional[tuple[float, float, float, float]]:
        if not self._selected:
            return None
        return self._extent

    def geometryType(self) -> str:
        return "unknown"

    def crs(self) -> Any:
        return None

    def id(self) -> str:
        return self.layer_name

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self._visible = bool(value)

    def opacity(self) -> float:
        return self._opacity

    def setOpacity(self, opacity: float) -> None:
        self._opacity = float(opacity)


class MockQGISCanvas:
    """Stand-in for ``QgsMapCanvas``.

    Tracks a numeric scale and an extent tuple so tests can assert on
    zoom/pan calls.
    """

    def __init__(self) -> None:
        self.scale_value: float = 1000.0
        self.extent_value: tuple[float, float, float, float] = (
            -180.0,
            -90.0,
            180.0,
            90.0,
        )
        self.refresh_count: int = 0

    def zoomIn(self) -> None:
        self.scale_value /= 2

    def zoomOut(self) -> None:
        self.scale_value *= 2

    def zoomByFactor(self, factor: float) -> None:
        self.scale_value *= factor

    def setExtent(self, extent: tuple[float, float, float, float]) -> None:
        self.extent_value = tuple(extent)  # type: ignore[assignment]

    def setCenter(self, point: Any) -> None:
        try:
            x = point.x()
            y = point.y()
        except Exception:
            x, y = point
        west, south, east, north = self.extent_value
        width = east - west
        height = north - south
        self.extent_value = (
            float(x) - width / 2,
            float(y) - height / 2,
            float(x) + width / 2,
            float(y) + height / 2,
        )

    def zoomScale(self, scale: float) -> None:
        self.scale_value = float(scale)

    def extent(self) -> tuple[float, float, float, float]:
        return self.extent_value

    def zoomToFullExtent(self) -> None:
        self.extent_value = (-180.0, -90.0, 180.0, 90.0)

    def refresh(self) -> None:
        self.refresh_count += 1

    def scale(self) -> float:
        return self.scale_value


class MockQGISProject:
    """Stand-in for ``QgsProject``.

    Stores layers in a dict keyed by layer name. ``addMapLayer`` and
    ``removeMapLayer`` accept either a :class:`MockQGISLayer` or a layer
    name string.
    """

    def __init__(self) -> None:
        self._layers: dict[str, MockQGISLayer] = {}
        self.saved_path: Optional[str] = None

    def addMapLayer(self, layer: MockQGISLayer) -> MockQGISLayer:
        self._layers[layer.name()] = layer
        return layer

    def removeMapLayer(self, key: Any) -> None:
        if isinstance(key, MockQGISLayer):
            self._layers.pop(key.name(), None)
        else:
            self._layers.pop(key, None)

    def mapLayers(self) -> dict[str, MockQGISLayer]:
        return dict(self._layers)

    def mapLayersByName(self, name: str) -> list[MockQGISLayer]:
        return [layer for layer in self._layers.values() if layer.name() == name]

    def write(self, path: str | None = None) -> bool:
        self.saved_path = path or ""
        return True

    @classmethod
    def instance(cls) -> "MockQGISProject":
        return cls()


class _MockMessageBar:
    def pushMessage(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        """No-op message-bar push, matching ``QgsMessageBar`` shape."""
        return None


class MockQGISIface:
    """Stand-in for ``qgis.utils.iface``.

    Holds a :class:`MockQGISCanvas` and an active layer reference, plus
    the ``addVectorLayer`` / ``addRasterLayer`` shortcuts that the QGIS
    Python console exposes.
    """

    def __init__(self, project: Optional[MockQGISProject] = None) -> None:
        self._canvas = MockQGISCanvas()
        self._active: Optional[MockQGISLayer] = None
        self._project = project or MockQGISProject()

    def mapCanvas(self) -> MockQGISCanvas:
        return self._canvas

    def activeLayer(self) -> Optional[MockQGISLayer]:
        return self._active

    def setActiveLayer(self, layer: MockQGISLayer) -> None:
        self._active = layer

    def addVectorLayer(
        self, path: str, name: str, provider: str = "ogr"
    ) -> MockQGISLayer:
        layer = MockQGISLayer(name, source=path, layer_type="vector")
        self._project.addMapLayer(layer)
        return layer

    def addRasterLayer(self, path: str, name: str) -> MockQGISLayer:
        layer = MockQGISLayer(name, source=path, layer_type="raster")
        self._project.addMapLayer(layer)
        return layer

    def messageBar(self) -> _MockMessageBar:
        return _MockMessageBar()

    def project(self) -> MockQGISProject:
        return self._project

    def zoomFull(self) -> None:
        self._canvas.zoomToFullExtent()

    def zoomToActiveLayer(self) -> None:
        if self._active is not None:
            self._canvas.refresh()


__all__ = [
    "MockLeafmap",
    "MockAnymap",
    "MockQGISIface",
    "MockQGISProject",
    "MockQGISLayer",
    "MockQGISCanvas",
]
