"""Runtime context object passed to GeoAgent runs.

The :class:`GeoAgentContext` is a thin dataclass that carries non-LLM-visible
runtime state into tool invocations: the live map widget, the QGIS interface
and project handles, the working directory, and user preferences. Live mutable
objects (the map widget, the QGIS iface) are typically captured via closure in
the tool factories under :mod:`geoagent.tools`; LLM-visible scalars (workdir,
current layer, user_preferences) are exposed to deepagents as a typed
``context_schema`` and accessed by tools through LangChain's ``ToolRuntime``
injection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class GeoAgentContext:
    """Runtime context for a GeoAgent invocation.

    Attributes:
        map_obj: A live interactive map instance (e.g. ``leafmap.Map`` or
            ``anymap.Map``). Captured by closure in :func:`leafmap_tools` /
            :func:`anymap_tools`; not serialised across the LLM boundary.
        qgis_iface: The QGIS ``QgisInterface`` (typically ``qgis.utils.iface``)
            when running inside a QGIS Python environment.
        qgis_project: The active ``QgsProject`` handle. Optional even when
            ``qgis_iface`` is provided; tools fall back to
            ``QgsProject.instance()``.
        workdir: Working directory used for downloads, exports, and other I/O
            tools. Defaults to the current working directory.
        current_layer: Optional name of the layer the user most recently
            referenced. The coordinator may set this so subsequent commands
            ("zoom to it", "buffer the selected layer") have a target.
        user_preferences: Free-form user-level preferences (e.g. preferred
            basemap, default colormap). Tools and prompts may consult this
            dict to tailor outputs.
    """

    map_obj: Optional[Any] = None
    qgis_iface: Optional[Any] = None
    qgis_project: Optional[Any] = None
    workdir: Path = field(default_factory=Path.cwd)
    current_layer: Optional[str] = None
    user_preferences: dict[str, Any] = field(default_factory=dict)

    def with_overrides(self, **kwargs: Any) -> "GeoAgentContext":
        """Return a copy of this context with selected fields overridden.

        Args:
            **kwargs: Field overrides; unknown fields raise ``TypeError``.

        Returns:
            A new :class:`GeoAgentContext` instance.
        """
        from dataclasses import replace

        return replace(self, **kwargs)
