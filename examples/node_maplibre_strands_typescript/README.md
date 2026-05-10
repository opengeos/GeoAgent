# Node MapLibre Strands TypeScript Example

This example runs a Node.js TypeScript backend and a MapLibre browser client.
The browser executes map commands, but model authentication stays in Node so the
page never receives model API keys, ChatGPT/Codex OAuth tokens, or AWS
credentials.

Supported providers match the GeoAgent/QGIS provider ids:

- OpenAI Codex (`openai-codex`) through the ChatGPT/Codex OAuth token used by
  GeoAgent and the Codex CLI flow.
- OpenAI (`openai`) through `OPENAI_API_KEY`.
- Anthropic (`anthropic`) through `ANTHROPIC_API_KEY`.
- Google Gemini (`gemini`) through `GEMINI_API_KEY` or `GOOGLE_API_KEY`.
- Amazon Bedrock (`bedrock`) through the AWS SDK default credential chain.
- LiteLLM (`litellm`) through an OpenAI-compatible LiteLLM proxy.
- Ollama (`ollama`) through Ollama's OpenAI-compatible local endpoint.

## Run

Install dependencies:

```bash
cd examples/node_maplibre_strands_typescript
npm install
```

For OpenAI Codex, log in once:

```bash
npm run codex:login
npm run dev
```

Open <http://127.0.0.1:8765>, connect to the default WebSocket URL, keep
`OpenAI Codex` selected, and send a prompt.

For OpenAI, Anthropic, or Gemini, set the provider API key in the backend
environment before starting the server:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=...
npm run dev
```

For Bedrock, configure AWS credentials and region first:

```bash
export AWS_PROFILE=your-profile
export AWS_REGION=us-east-1
npm run dev
```

Then open <http://127.0.0.1:8765>, connect, choose `Amazon Bedrock`, and send a
prompt. The Bedrock model defaults to `us.anthropic.claude-sonnet-4-6` and can
be changed in the model field or with `BEDROCK_MODEL`.

For LiteLLM, point the backend at an OpenAI-compatible proxy:

```bash
export LITELLM_BASE_URL=http://127.0.0.1:4000/v1
export LITELLM_API_KEY=...
npm run dev
```

For Ollama, start Ollama locally and set `OLLAMA_HOST` if it is not using the
default `http://127.0.0.1:11434`:

```bash
ollama pull qwen3.5:4b
npm run dev
```

To stop the server, press `Ctrl+C` in the terminal. Or kill the process from another terminal:

```bash
pkill -f "tsx src/server/main.ts"
```

## Authentication

Codex auth resolution:

- `OPENAI_CODEX_ACCESS_TOKEN` is used when set.
- Otherwise the server loads the GeoAgent token file from
  `GEOAGENT_CODEX_TOKEN_FILE` or `~/.config/geoagent/openai_codex_oauth.json`.
- Expired stored tokens are refreshed with the stored `refresh_token`.
- `OPENAI_CODEX_BASE_URL` overrides the default
  `https://chatgpt.com/backend-api/codex`.

Bedrock auth resolution uses the AWS SDK default credential chain, including
environment variables, shared config files, SSO/profile credentials, and role
providers supported by the SDK.

Other provider environment variables:

| Provider  | Required or optional environment                                               |
| --------- | ------------------------------------------------------------------------------ |
| OpenAI    | `OPENAI_API_KEY`, optional `OPENAI_MODEL`                                      |
| Anthropic | `ANTHROPIC_API_KEY`, optional `ANTHROPIC_MODEL`                                |
| Gemini    | `GEMINI_API_KEY` or `GOOGLE_API_KEY`, optional `GEMINI_MODEL` / `GOOGLE_MODEL` |
| LiteLLM   | optional `LITELLM_BASE_URL`, `LITELLM_API_KEY`, `LITELLM_MODEL`                |
| Ollama    | optional `OLLAMA_HOST`, `OLLAMA_MODEL`, `OLLAMA_API_KEY`                       |

## Tool Safety

Layer removal tools are not exposed by default. Enable them only for trusted
local sessions:

```bash
npm run dev -- --allow-destructive
```

Generated MapLibre JavaScript is also disabled by default. Enable it only for
trusted local prompts:

```bash
npm run dev -- --allow-browser-code
```

Both flags can be used together.

## Prompt Examples

```text
Add a red marker for Knoxville and zoom to it.
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

Layer removal with `--allow-destructive`:

```text
Remove the US counties layer.
```

Generated MapLibre JavaScript with `--allow-browser-code`:

```text
Tilt the map to pitch 75 and rotate it slightly.
```

## Scripts

```bash
npm run dev
npm run build
npm start
npm run typecheck
npm run codex:login
```
