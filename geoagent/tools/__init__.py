"""Tool adapter modules for GeoAgent.

Each submodule exposes a factory function (e.g. :func:`leafmap_tools`,
:func:`qgis_tools`) that returns a list of LangChain ``BaseTool`` objects
bound to a live runtime resource via closure. Modules import safely on
systems where the underlying optional package is missing; the factories
return an empty list in that case so callers can rely on:

    tools = leafmap_tools(m) + qgis_tools(iface) + stac_tools()
    create_geo_agent(tools=tools, ...)

without guarding each call.
"""

__all__: list[str] = []
