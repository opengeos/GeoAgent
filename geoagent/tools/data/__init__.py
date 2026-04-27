"""Data manipulation tools (raster, vector, DuckDB spatial SQL, viz).

Each submodule exposes a ``*_tools()`` factory that returns a list of
LangChain ``BaseTool`` instances stamped with GeoAgent metadata. All wrap
existing implementations from :mod:`geoagent.core.tools` to preserve
behaviour while adding the new category and confirmation metadata.
"""

__all__: list[str] = []
