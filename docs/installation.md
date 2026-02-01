# Installation

## Stable Release

To install GeoAgent, run this command in your terminal:

```bash
pip install geoagent
```

This installs the core package with OpenAI support.

## Optional Dependencies

### Additional LLM Providers

To use Anthropic Claude or Google Gemini:

```bash
pip install "geoagent[llm]"
```

### Local LLM with Ollama

For offline or air-gapped environments:

```bash
pip install "geoagent[ollama]"
```

Make sure [Ollama](https://ollama.ai) is installed and running locally.

### Web UI

For Streamlit or Solara web interfaces:

```bash
pip install "geoagent[ui]"
```

### Everything

To install all optional dependencies:

```bash
pip install "geoagent[all]"
```

## From Source

To install the latest development version:

```bash
pip install git+https://github.com/opengeos/GeoAgent
```

Or clone and install in editable mode:

```bash
git clone https://github.com/opengeos/GeoAgent.git
cd GeoAgent
pip install -e ".[all]"
```

## LLM Configuration

GeoAgent needs an LLM provider to function. Set one of these environment variables:

```bash
# OpenAI (default)
export OPENAI_API_KEY="sk-..."

# Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."

# Google Gemini
export GOOGLE_API_KEY="AI..."
```

For Ollama, no API key is needed. Just make sure the Ollama server is running:

```bash
ollama serve
ollama pull llama3.1
```

GeoAgent automatically detects available providers and uses the first one found (in order: OpenAI, Anthropic, Google, Ollama).
