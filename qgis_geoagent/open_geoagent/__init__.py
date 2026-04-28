"""OpenGeoAgent QGIS plugin."""

from .deps_manager import ensure_venv_packages_available

# Add isolated dependency site-packages to sys.path before importing GeoAgent.
ensure_venv_packages_available()

from .open_geoagent import OpenGeoAgent  # noqa: E402


def classFactory(iface):
    """Load OpenGeoAgent plugin class."""
    return OpenGeoAgent(iface)
