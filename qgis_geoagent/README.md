# OpenGeoAgent QGIS Plugin

OpenGeoAgent is a QGIS plugin that exposes GeoAgent's QGIS tool surface through
a dockable chatbot interface. It can answer questions about the active QGIS
project and use GeoAgent tools for layer inspection, map navigation, loading
data, selection, processing, and project operations.

The plugin is based on the `qgis-plugin-template` structure and keeps the same
development conveniences: local install scripts, packaging scripts, one-click
dependency installation with `uv`, an isolated virtual environment, and a
GitHub update checker.

## Features

- Dockable OpenGeoAgent chat panel with Ctrl+Enter sending
- Voice prompt recording with automatic OpenAI speech-to-text transcription into
  the chat editor
- Up/Down prompt history while the prompt editor is focused
- Built-in sample prompt picker for common QGIS workflows
- Agent modes for General QGIS, WhiteboxTools, NASA Earthdata, NASA OPERA,
  GEE Data Catalogs, and STAC guidance workflows
- Tool permission profiles, defaulting to trusted auto-approval
- Project-scoped chat history with Markdown import/export
- Compact jobs panel for active and completed GeoAgent requests
- Provider and model controls for Bedrock, OpenAI, ChatGPT/Codex OAuth, Anthropic, Google Gemini, Ollama, and LiteLLM
- Settings panel for model defaults, API keys, hosts, AWS region, provider
  smoke tests, and redacted diagnostics export
- Dependency installer that installs core provider packages or selected
  workflow packages into `~/.open_geoagent/`
- QGIS 3.28+ and QGIS 4 compatible plugin structure
- Local install and packaging scripts for development and release

## Structure

```text
qgis_geoagent/
├── open_geoagent/
│   ├── __init__.py
│   ├── open_geoagent.py
│   ├── metadata.txt
│   ├── deps_manager.py
│   ├── uv_manager.py
│   ├── dialogs/
│   │   ├── chat_dock.py
│   │   ├── settings_dock.py
│   │   └── update_checker.py
│   └── icons/
├── install.py
├── install.sh
├── package_plugin.py
├── package_plugin.sh
└── tests/
```

## Install For Development

From this directory:

```bash
python install.py
```

or:

```bash
./install.sh
```

Then restart QGIS and enable `OpenGeoAgent` in **Plugins > Manage and Install
Plugins...**.

## Dependencies

Open the OpenGeoAgent settings panel and use the **Dependencies** tab to install
required Python packages. Choose the dependency set for the workflow you need:
Core Providers, WhiteboxTools, NASA Earthdata/OPERA, GEE Data Catalogs, STAC, or
All. The installer creates an isolated environment under `~/.open_geoagent/` and
adds its site-packages directory when the plugin loads.

Manual fallback:

```bash
pip install "GeoAgent[providers]>=1.4.1"
pip install "GeoAgent[stac]>=1.4.1"
pip install "GeoAgent[whitebox]>=1.4.1"
pip install "GeoAgent[earthdata,nasa-opera]>=1.4.1"
pip install "GeoAgent[earthengine]>=1.4.1"
```

GEE Data Catalogs mode also expects the `gee_data_catalogs` Python module from
the GEE Data Catalogs QGIS plugin/runtime. That module is not currently a PyPI
package, so the OpenGeoAgent dependency installer only installs the PyPI-backed
Earth Engine dependencies (`earthengine-api` and `geemap`).

OpenGeoAgent requires the QGIS Python runtime to be Python 3.11 or newer,
matching GeoAgent's core package requirement. Older QGIS Python runtimes show a
clear installer error before any virtual environment is created.

## Provider Configuration

Use the **Model** tab or the controls at the top of the chat dock to choose a
provider and model. API keys and host settings are stored in QGIS settings and
applied to the current QGIS process before each chat request:

- OpenAI: `OPENAI_API_KEY`
- OpenAI organization/project targeting: optional `OPENAI_ORG_ID` and
  `OPENAI_PROJECT_ID`
- Voice transcription: `OPENAI_API_KEY` is required even when the chat provider
  is set to ChatGPT/Codex OAuth. Voice input sends recorded audio to the OpenAI
  transcription API and may incur API costs. If no key is configured, recording
  is blocked and OpenGeoAgent shows a setup warning. Choose the transcription
  model in **Settings > Model > Voice Transcription**. The default is
  `gpt-4o-mini-transcribe`; `OPENAI_TRANSCRIPTION_MODEL` is used as a fallback
  when no QGIS setting has been saved.
- Image generation: `OPENAI_API_KEY` is required for direct requests such as
  "generate a cat image." ChatGPT/Codex OAuth can still power chat, but the
  `generate_image` tool calls the OpenAI Images API. The default image model
  is `gpt-image-2`; choose `gpt-image-1` in Settings > Model > Image
  Generation when you want the lower-cost fallback model.
- ChatGPT/Codex OAuth: choose `openai-codex` and click **Login with ChatGPT**
  in the Model tab. Headless use can set `OPENAI_CODEX_ACCESS_TOKEN`.
- Anthropic: `ANTHROPIC_API_KEY`
- Google Gemini: `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- Bedrock: `AWS_REGION` plus the normal AWS credential chain
- Ollama: `OLLAMA_HOST`
- LiteLLM: `LITELLM_API_KEY` and optional `LITELLM_BASE_URL`

Default models:

| Provider | Default model |
| --- | --- |
| Bedrock | `us.anthropic.claude-sonnet-4-6` |
| OpenAI | `gpt-5.5` |
| ChatGPT/Codex OAuth | `gpt-5.5` |
| Anthropic | `claude-sonnet-4-6` |
| Google Gemini | `gemini-3.1-pro-preview` |
| Ollama | `qwen3.5:4b` |
| LiteLLM | `openai/gpt-5.5` |

## Chat Workflow

- Type a prompt and press **Ctrl+Enter** to send it.
- Click the microphone button, speak, then click it again to stop. OpenGeoAgent
  records through QGIS/Qt multimedia support, shows a live input-level
  indicator, transcribes the audio with the OpenAI transcription API, and
  inserts the text into the prompt editor for review. The default mic shortcut
  is **Ctrl+Alt+Space** and can be changed in **Settings > Model > Voice
  Transcription**. The shortcut is handled while keyboard focus is inside the
  chat dock.
- Choose an **Agent mode** for the workflow you want. STAC mode can list
  collections, search STAC items, inspect assets, and add concrete raster asset
  URLs to QGIS when QGIS can load them directly.
- Choose a **Permissions** profile. The default **Trusted auto-approve** profile
  exposes and approves confirmation-gated tools. Select narrower profiles when
  you want to restrict edits, processing, script execution, or long-running
  workflow tools.
- **Stream output** is enabled by default and shows model text as it arrives.
- Press **Up** or **Down** in the prompt editor to cycle through previous prompts.
- Choose a prompt from **Sample prompts...** and click **Insert** to load it into
  the editor before sending.
- Use **Export**, **Import**, and **Copy Markdown** to move project chat history
  between sessions. The plugin stores chat history per QGIS project.
- The **Jobs** panel records submitted requests with status, mode, prompt,
  elapsed time, and tool summary. Completed jobs can be selected and rerun.

## Diagnostics

The Settings panel includes:

- **Test Provider** for a tiny no-tool model smoke test using the selected
  provider and model.
- **Copy Diagnostics** and **Save Diagnostics** for redacted JSON containing
  plugin version, installed GeoAgent version, QGIS/Python details, isolated
  environment paths, uv status, dependency status by workflow, selected model
  settings, credential presence booleans, and latest installer/provider-test
  status.

Diagnostics never include raw API keys, OAuth tokens, or passwords.

## Key Functionality

OpenGeoAgent builds on GeoAgent's QGIS-safe tools. It can:

- Inspect project layers, active layer metadata, fields, extents, CRS, feature
  counts, selected features, opacity, and visibility.
- Navigate the QGIS canvas by zooming, centering, setting scale, zooming to
  layers, zooming to extents, and zooming to selected features.
- Add vector, raster, and XYZ tile layers.
- Manage layers by removing them or changing visibility and opacity.
- Select features with QGIS expressions and clear selections.
- Run QGIS Processing algorithms.
- Open attribute tables, refresh the canvas, and save projects when approved.

## Sample Prompts

```text
Summarize the current QGIS project layers, CRS, extents, and feature counts.
Zoom to the active layer and describe what it contains.
List visible layers and identify any layers with no features or invalid data sources.
Add an OpenStreetMap basemap and zoom to the project extent.
Inspect the active vector layer fields and suggest useful styling or labeling options.
Select features in the active layer where population is greater than 100000, then zoom to the selected features.
Run a buffer around the active layer by 1000 meters and add the output to the project.
Create a concise map QA checklist for this project before I export it.
```

## Package

Create a QGIS plugin zip:

```bash
python package_plugin.py
```

or:

```bash
./package_plugin.sh
```

The default package name is `open_geoagent-{version}.zip`.
