# Usage

## ChatGPT/Codex Login

For notebooks and Python scripts, run the browser login once:

```bash
geoagent codex login
```

or start the same flow from Jupyter:

```python
from geoagent import login_openai_codex

login_openai_codex()
```

GeoAgent saves the refresh token in your user config directory, exports
`OPENAI_CODEX_ACCESS_TOKEN` for the current process, and refreshes the stored
login automatically when `provider="openai-codex"` is used later.

```python
from geoagent import GeoAgent, for_leafmap, GeoAgentConfig

# Minimal (provider from environment)
agent = GeoAgent()

# Explicit config
agent = GeoAgent(config=GeoAgentConfig(provider="openai", model="gpt-5.5"))

# Bound to a live map
agent = for_leafmap(m)

resp = agent.chat("Your prompt")
print(resp.answer_text)
```

For token-by-token output, use the async streaming sibling:

```python
import asyncio

async def main():
    async for event in agent.stream_chat("Your prompt"):
        if "data" in event:
            print(event["data"], end="", flush=True)

asyncio.run(main())
```

Use `fast=True` with `GeoAgent(..., fast=True)` or `for_leafmap(m, fast=True)` for a smaller prompt and fewer tools.

Access the underlying Strands agent as `agent.strands_agent`.
