"""Ensure QGIS-bound tools execute on the invoking thread (sync path)."""

from __future__ import annotations

import threading

from geoagent.testing import MockQGISIface, MockQGISProject
from geoagent.tools.qgis import qgis_tools


def test_qgis_tool_runs_on_calling_thread() -> None:
    """Verify that qgis tool runs on calling thread."""
    iface = MockQGISIface()
    project = MockQGISProject()
    tools = {t.tool_name: t for t in qgis_tools(iface, project)}
    main = threading.current_thread()

    box: list[threading.Thread] = []

    def run() -> None:
        """Run the worker body."""
        tools["zoom_in"]()
        box.append(threading.current_thread())

    run()
    assert box[0] is main
