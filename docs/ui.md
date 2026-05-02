# Web UI

GeoAgent includes a Solara-based browser workspace for map-bound chat. The UI
creates a persistent live map for the current browser session, binds it to a
GeoAgent, and lets you control the provider, model, fast mode, and confirmation
policy from the page.

## Quick Start

Install the UI dependency plus at least one web map backend:

```bash
pip install "GeoAgent[ui,anymap,openai]"
```

`anymap` is preferred. If it is not installed, the UI tries `leafmap`:

```bash
pip install "GeoAgent[ui,leafmap,openai]"
```

Launch the UI:

```bash
geoagent ui
```

Or run Solara directly:

```bash
solara run geoagent/ui/pages
```

## Workspace

The first screen is the chat-and-map workspace. It includes:

- a persistent interactive map where layers accumulate across prompts;
- provider and model controls for OpenAI, ChatGPT/Codex OAuth, Anthropic,
  Google Gemini, Bedrock, LiteLLM, and Ollama;
- a fast-mode toggle for lower-latency map-control prompts;
- an auto-approve toggle for confirmation-required tools;
- chat history for the current browser session;
- executed tool names, cancelled tool names, and compact tool-call results.

The MVP uses non-streaming `agent.chat(...)` calls. Provider credentials are
still configured through the same environment variables used by the Python API,
such as `OPENAI_API_KEY`, `OPENAI_CODEX_ACCESS_TOKEN`, `ANTHROPIC_API_KEY`,
`GEMINI_API_KEY`, `LITELLM_API_KEY`, `OLLAMA_HOST`, or AWS credentials for
Bedrock.

## Safety

The web UI denies confirmation-required tools by default. This means requests
that remove layers, clear layers, save maps, or run other gated actions will be
cancelled unless you enable **Auto-approve confirmation tools**.

Use auto-approve only for trusted sessions. It applies to the current UI
session and allows GeoAgent to execute tools marked as confirmation-required,
destructive, or long-running.

## Python API

You can also launch the UI programmatically:

```python
from geoagent.ui import launch_ui

launch_ui()
```

## Module Reference

::: geoagent.ui.launch_ui
