"""GeoAgent - An AI agent for geospatial data analysis and visualization."""

__author__ = """Qiusheng Wu"""
__email__ = "giswqs@gmail.com"
__version__ = "0.0.1"

from geoagent.core.llm import get_llm, get_default_llm

try:
    from geoagent.core.agent import GeoAgent
except ImportError:
    GeoAgent = None

__all__ = ["GeoAgent", "get_llm", "get_default_llm"]
