"""NASA OPERA GeoAgent example for the QGIS Python console.

Run this from QGIS after installing GeoAgent in QGIS's Python environment.
Progress appears in the QGIS message bar and in the Log Messages panel under
``GeoAgent NASA OPERA``.
"""

from geoagent.tools.nasa_opera import (
    nasa_opera_tools,
    submit_nasa_opera_search_task,
)
from qgis.utils import iface  # type: ignore[import-not-found]

tools = {tool.tool_name: tool for tool in nasa_opera_tools(iface)}
print("OPERA tools:", sorted(tools))
print("OPERA datasets:", tools["get_available_datasets"]())


task = submit_nasa_opera_search_task(
    iface,
    dataset="OPERA_L3_DSWX-HLS_V1",
    bbox="-95.5,29.5,-95.0,30.0",
    start_date="2024-01-01",
    end_date="2024-01-31",
    max_results=5,
    display_footprints=True,
)
print("Submitted task:", task.description())


# Natural-language chat is intentionally disabled for NASA OPERA inside QGIS.
# Use direct tools or submit_nasa_opera_search_task(...) so QGIS task/thread
# ownership remains explicit.
