# Decorators

The `@geo_tool` decorator wraps LangChain's `@tool` to stamp GeoAgent metadata (category, confirmation requirement, required Python packages, context keys) onto each tool. The metadata is read by the registry and the safety / confirmation bridge.

::: geoagent.core.decorators
