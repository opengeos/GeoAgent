"""Runtime context for GeoAgent tools and prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class GeoAgentContext:
    """Scalar and handle fields for a GeoAgent run.

    Live objects (map widgets, QGIS iface) are normally captured via closure
    in tool factories; this dataclass holds references for prompts and tools
    that read context explicitly.

    Attributes:
        map_obj: leafmap.Map, anymap.Map, or similar.
        qgis_iface: QGIS ``QgisInterface`` when inside QGIS.
        qgis_project: Optional ``QgsProject`` (tools may fall back to instance).
        workdir: Working directory for downloads and exports.
        current_layer: Optional layer name hint for follow-up commands.
        user_preferences: Free-form preferences for prompts/tools.
        metadata: Arbitrary JSON-serializable metadata for integrations.
    """

    map_obj: Optional[Any] = None
    qgis_iface: Optional[Any] = None
    qgis_project: Optional[Any] = None
    workdir: Path = field(default_factory=Path.cwd)
    current_layer: Optional[str] = None
    user_preferences: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_overrides(self, **kwargs: Any) -> "GeoAgentContext":
        """Return a copy with selected fields replaced."""
        from dataclasses import replace

        return replace(self, **kwargs)
