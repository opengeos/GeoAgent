# Usage

```python
from geoagent import GeoAgent, for_leafmap, GeoAgentConfig

# Minimal (provider from environment)
agent = GeoAgent()

# Explicit config
agent = GeoAgent(config=GeoAgentConfig(provider="openai", model="gpt-5.4-mini"))

# Bound to a live map
agent = for_leafmap(m)

resp = agent.chat("Your prompt")
print(resp.answer_text)
```

Use `fast=True` with `GeoAgent(..., fast=True)` or `for_leafmap(m, fast=True)` for a smaller prompt and fewer tools.

Access the underlying Strands agent as `agent.strands_agent`.
