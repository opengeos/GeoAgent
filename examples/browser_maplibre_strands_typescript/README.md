# Browser MapLibre Strands TypeScript Example

This example runs a Strands TypeScript agent directly inside the browser. The
agent tools call the live `maplibregl.Map` instance in the Vite app, so no
Python WebSocket backend is required.

## Run

```bash
cd examples/browser_maplibre_strands_typescript
npm ci
npm run dev
```

Open the Vite URL, usually <http://127.0.0.1:5173>. Choose a provider, enter
that provider's API key in the panel, and send a prompt.

Supported providers:

- OpenAI Responses
- OpenAI Chat Completions
- Anthropic
- Google Gemini

The app passes API keys directly to browser clients, including
`dangerouslyAllowBrowser: true` for OpenAI and Anthropic. Use this only for local
development or behind a trusted model proxy.

API keys you type into the panel are persisted in the browser's `sessionStorage`
so the page can rebuild the agent on reload. They live in the page origin until
the tab is closed. The optional `MapLibre JS` toggle (which exposes the
`run_maplibre_script` tool) executes arbitrary JavaScript in that same page
context, so a script can read `sessionStorage` and any other page state. Treat
the JS toggle as a highly trusted, local-only escape hatch and leave it off when
running prompts you do not control. The toggle defaults to off.

`openai-codex` and Bedrock are intentionally not listed in this browser-only
example:

- `openai-codex` uses the ChatGPT/Codex OAuth flow, not an OpenAI API key. The
  QGIS plugin and `geoagent codex login` run a localhost OAuth callback, store
  refreshable tokens outside the web page, and set `OPENAI_CODEX_ACCESS_TOKEN`
  plus the Codex backend URL before creating the model.
- Bedrock normally uses AWS SigV4 credentials from the AWS credential chain, not
  a simple browser API key. Signing AWS requests in a static browser app would
  expose credentials and is usually blocked by the same security boundaries that
  make a backend/proxy the right place for Bedrock.

Use the Python WebSocket example or the QGIS plugin when you need
`openai-codex` or Bedrock. Keep this browser-only example for direct browser SDK
providers.

Default model values mirror the direct browser-compatible entries from the QGIS
OpenGeoAgent plugin:

| Provider | Default model |
| --- | --- |
| OpenAI Responses | `gpt-5.5` |
| OpenAI Chat Completions | `gpt-5.5` |
| Anthropic | `claude-sonnet-4-6` |
| Google Gemini | `gemini-3.1-pro-preview` |

## Prompt Examples

```text
Add a red marker for Knoxville and zoom to it.
```

```text
Fly to Seattle at zoom level 11.
```

```text
Change the basemap to dark, then get the current map state.
```

```text
Change to globe projection.
```

```text
Add the GeoJSON URL https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json as US counties.
```

```text
Hide the US counties layer, then show it again.
```

```text
What features are visible at the center of the current map?
```

Layer removal requires enabling the `Layer removal` toggle:

```text
Remove the US counties layer.
```

Generated MapLibre JavaScript is available through the `run_maplibre_script`
tool. The `MapLibre JS` toggle defaults to off; enable it only for trusted local
sessions where you want the agent to fall back to writing JavaScript when no
dedicated tool fits:

```text
Tilt the map to pitch 75 and rotate it slightly.
```

## Scripts

```bash
npm run dev
npm run build
npm run typecheck
```
