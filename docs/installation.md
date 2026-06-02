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
`OLLAMA_HOST`, `VLLM_BASE_URL`, `VLLM_MODEL_ID`, etc.). See `GeoAgentConfig`
in `geoagent.core.config`.

For vLLM, install `GeoAgent[vllm]` or `GeoAgent[providers]`, run a vLLM server
separately, and enable vLLM tool calling on the server when using GeoAgent
tools.
