# STAC Catalogs

GeoAgent includes a registry of pre-configured STAC catalog endpoints.

## Built-in Catalogs

| Catalog | URL | Auth Required |
|---------|-----|---------------|
| Earth Search (AWS) | `earth-search.aws.element84.com/v1` | No |
| Planetary Computer | `planetarycomputer.microsoft.com/api/stac/v1` | No (signing optional) |
| USGS Landsat | `landsatlook.usgs.gov/stac-server` | No |
| NASA CMR-STAC | `cmr.earthdata.nasa.gov/stac` | Yes (`NASA_EARTHDATA_TOKEN`) |

## Catalog Registry

::: geoagent.catalogs.registry
