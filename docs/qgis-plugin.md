# QGIS Plugin

[![QGIS Plugin](https://img.shields.io/badge/QGIS-Plugin-green.svg)](https://plugins.qgis.org/plugins/open_geoagent)

OpenGeoAgent is the QGIS plugin interface for GeoAgent. It adds a dockable,
project-aware AI assistant to QGIS and connects it to GeoAgent's QGIS tool
surface.

![OpenGeoAgent QGIS plugin](https://github.com/user-attachments/assets/3047e39c-ad0a-4d77-a822-9597da539775)

## Install

Install the plugin from the QGIS Plugin Manager by searching for
**OpenGeoAgent**. For local development, install or symlink the plugin directory
from `qgis_geoagent/open_geoagent` into your QGIS plugin profile.

Open the plugin from the QGIS toolbar or plugin menu, then use
**Settings > Dependencies** to install GeoAgent and provider clients into the
plugin-managed environment. QGIS itself remains provided by your desktop QGIS
installation.

## Provider Setup

OpenGeoAgent supports the same provider families as GeoAgent:

- Bedrock
- OpenAI
- ChatGPT/Codex OAuth
- Anthropic
- Google Gemini
- Ollama
- LiteLLM

Use the settings panel to configure API keys, hosts, model defaults, and
dependency status. Vision features require a provider and model that support
image inputs.

## Chat With QGIS

The chat dock is aware of the current QGIS project and active layer. It can
inspect project state, summarize layers, navigate the map canvas, add vector,
raster, and XYZ tile layers, select features, run QGIS Processing algorithms,
open attribute tables, and save projects.

Examples:

- "Summarize all layers in this project."
- "Zoom to the active layer and describe its CRS and extent."
- "Select parcels where population is greater than 10000."
- "Add this raster and set its opacity to 60 percent."
- "Use a false color composite for the active NAIP layer."

## Images And Screenshots

OpenGeoAgent supports image attachments in chat messages. You can paste images
from the clipboard directly into the chat input, or use the screenshot menu to
attach visual context from QGIS.

Screenshot options include:

- capture the map canvas;
- select a region on the map canvas;
- capture the QGIS window;
- select a region on the screen.

Attached images appear as thumbnails in the chat. Click an image to open a
larger preview, then use the save action or context menu to export it.

## PyQGIS Fallback

When a request needs QGIS API functionality that is not covered by a dedicated
GeoAgent tool, the agent can use the confirmation-gated `run_pyqgis_script`
fallback. The script runs in the QGIS GUI context with access to `iface`,
`project`, `canvas`, and `active_layer`.

This is useful for tasks such as raster band renderer changes, labeling
updates, layer tree adjustments, and other PyQGIS operations. The plugin asks
for confirmation before running the script.

Use **Copy Script** to copy the PyQGIS code that produced the result. The
copied snippet includes a QGIS-console-ready preamble so it can be inspected,
shared, or rerun.

## Copy And Review

The chat dock includes actions for copying the Markdown transcript and the most
recent executed PyQGIS script. These are intended for reproducibility,
debugging, and sharing workflows outside the plugin.

## Safety

OpenGeoAgent uses GeoAgent's confirmation hook for destructive, persistent, or
long-running operations. Actions such as deleting layers, saving projects,
running processing jobs, and executing fallback PyQGIS scripts require user
approval before they run.
