"""Normalize map camera state for leafmap MapLibre and similar widgets."""

from __future__ import annotations

from typing import Any


def deep_plain(obj: Any) -> Any:
    """Convert nested MapLibre view objects to JSON-friendly dicts/lists.

    Handles ``view_state`` shapes where ``center`` / ``bounds._sw`` use
    ``lng`` / ``lat`` attributes or dicts.
    """

    if obj is None:
        return None
    if isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): deep_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [deep_plain(x) for x in obj]
    lng = getattr(obj, "lng", None)
    lat = getattr(obj, "lat", None)
    if lng is not None and lat is not None:
        return {"lng": lng, "lat": lat}
    if hasattr(obj, "__dict__") and type(obj).__module__ != "builtins":
        try:
            return deep_plain(vars(obj))
        except TypeError:
            pass
    return str(obj)


def map_state_from_widget(m: Any) -> dict[str, Any]:
    """Build the payload for ``get_map_state`` tools.

    Prefer **MapLibre** ``view_state`` (``leafmap.maplibregl.Map``), which
    stores camera state as ``center`` (``lng``/``lat``), ``zoom``, ``bounds``
    (``_sw`` / ``_ne``), ``bearing``, ``pitch``.

    Falls back to legacy ``center`` / ``zoom`` / ``_bounds`` / ``_style`` for
    mocks and ipyleaflet-like maps.
    """

    layers = getattr(m, "layers", None) or []
    layer_count = len(layers) if isinstance(layers, list) else 0
    basemap = getattr(m, "_style", None)

    # MapLibre widgets often start with an empty ``view_state`` on the Python
    # side until frontend sync. Use map_options as a deterministic fallback.
    vs = getattr(m, "view_state", None)
    normalized = deep_plain(vs)
    nc = normalized if isinstance(normalized, dict) else {}
    map_options = deep_plain(getattr(m, "map_options", None))
    mo = map_options if isinstance(map_options, dict) else {}

    center = nc.get("center")
    if center is None:
        mo_center = mo.get("center")
        if isinstance(mo_center, (list, tuple)) and len(mo_center) >= 2:
            center = {"lng": mo_center[0], "lat": mo_center[1]}
        elif isinstance(mo_center, dict):
            center = mo_center

    zoom = nc.get("zoom", mo.get("zoom"))
    bounds = nc.get("bounds")
    bearing = nc.get("bearing", mo.get("bearing"))
    pitch = nc.get("pitch", mo.get("pitch"))
    if basemap is None:
        basemap = mo.get("style")

    if any(v is not None for v in (center, zoom, bounds, bearing, pitch)) or mo:
        return {
            "view_state": normalized,
            "map_options": map_options,
            "center": center,
            "zoom": zoom,
            "bounds": bounds,
            "bearing": bearing,
            "pitch": pitch,
            "basemap": basemap,
            "layer_count": layer_count,
        }

    return {
        "center": list(getattr(m, "center", []) or []),
        "zoom": getattr(m, "zoom", None),
        "bounds": getattr(m, "_bounds", None),
        "bearing": getattr(m, "bearing", None),
        "pitch": getattr(m, "pitch", None),
        "basemap": basemap,
        "layer_count": layer_count,
    }
