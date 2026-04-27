"""DeepAgents subagent specs for GeoAgent.

Each module here exports either a constant ``*_SUBAGENT`` dict or a
factory function returning one. The dicts conform to the
:class:`deepagents.SubAgent` TypedDict (``name``, ``description``,
``system_prompt``, ``tools``, optional ``model`` / ``middleware`` /
``interrupt_on``).

The :func:`default_subagents` helper in :mod:`geoagent.agents.coordinator`
assembles the active list based on the runtime
:class:`geoagent.core.context.GeoAgentContext`: planner / data /
analysis / context always run; mapping is added when a live map is
present; qgis when an iface is present; geoai and earthdata when their
optional packages are importable.
"""

from .analysis import analysis_subagent
from .context import context_subagent
from .coordinator import default_subagents
from .data import data_subagent
from .earthdata import earthdata_subagent
from .geoai import geoai_subagent
from .mapping import mapping_subagent
from .planner import PLANNER_SUBAGENT
from .qgis import qgis_subagent

__all__ = [
    "default_subagents",
    "PLANNER_SUBAGENT",
    "data_subagent",
    "analysis_subagent",
    "context_subagent",
    "mapping_subagent",
    "qgis_subagent",
    "geoai_subagent",
    "earthdata_subagent",
]
