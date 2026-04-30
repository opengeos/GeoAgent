"""OpenGeoAgent QGIS plugin."""

from .open_geoagent import OpenGeoAgent


def classFactory(iface):
    """Load OpenGeoAgent plugin class."""
    return OpenGeoAgent(iface)
