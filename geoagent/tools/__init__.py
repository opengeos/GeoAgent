"""Environment-specific tool factories."""

from geoagent.tools.anymap import anymap_tools
from geoagent.tools.gee_data_catalogs import gee_data_catalogs_tools
from geoagent.tools.images import image_generation_tools
from geoagent.tools.leafmap import leafmap_tools
from geoagent.tools.nasa_earthdata import earthdata_tools
from geoagent.tools.nasa_opera import nasa_opera_tools
from geoagent.tools.qgis import qgis_tools
from geoagent.tools.stac import stac_tools
from geoagent.tools.timelapse import timelapse_tools
from geoagent.tools.vantor import vantor_tools
from geoagent.tools.whitebox import whitebox_tools

__all__ = [
    "anymap_tools",
    "earthdata_tools",
    "gee_data_catalogs_tools",
    "image_generation_tools",
    "leafmap_tools",
    "nasa_opera_tools",
    "qgis_tools",
    "stac_tools",
    "timelapse_tools",
    "vantor_tools",
    "whitebox_tools",
]
