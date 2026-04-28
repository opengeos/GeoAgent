# Tools

Interactive adapters live under **`geoagent.tools`**:

- `leafmap_tools`, `anymap_tools`, `qgis_tools` — bound to live map / QGIS instances.
- Optional stubs: `stac`, `geoai`, `earthengine`, `nasa_earthdata` (expand in future releases).

Use **`@geo_tool`** ([`geoagent.core.decorators`](decorators.md)) so tools register Strands-compatible metadata for safety hooks.
