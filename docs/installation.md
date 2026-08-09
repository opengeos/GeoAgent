# Installation

```bash
pip install GeoAgent
```

Install optional stacks with extras, for example:

```bash
pip install "GeoAgent[leafmap,openai]"
pip install "GeoAgent[qgis]"   # marker extra; QGIS itself is system-installed
```

Development install:

```bash
git clone https://github.com/opengeos/GeoAgent.git
cd GeoAgent
pip install -e ".[dev]"
```

Configure API keys via environment variables (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY` or `GOOGLE_API_KEY`,
`LITELLM_API_KEY`, `OPENROUTER_API_KEY`, AWS credentials for Bedrock,
`OLLAMA_HOST`, `OPENAI_COMPATIBLE_BASE_URL`, `VLLM_BASE_URL`, `VLLM_MODEL_ID`,
etc.). See `GeoAgentConfig` in `geoagent.core.config`.

To use any OpenAI-compatible server (llama.cpp, LM Studio, Text Generation
WebUI, vLLM), install `GeoAgent[openai-compatible]` or `GeoAgent[providers]`
and set `OPENAI_COMPATIBLE_BASE_URL` to the server's `/v1` URL plus
`OPENAI_COMPATIBLE_MODEL`.

The dedicated vLLM provider needs `GeoAgent[vllm]`, a separately running vLLM
server, and vLLM tool calling enabled on the server. `strands-vllm` pins
`openai<2.0`, which conflicts with the `openai>=2.0` that the default
`openai-codex` provider requires, so `GeoAgent[vllm]` is not part of
`GeoAgent[providers]` or `GeoAgent[all]` and cannot share an environment with
them. Prefer `openai-compatible` unless you specifically need `strands-vllm`.
